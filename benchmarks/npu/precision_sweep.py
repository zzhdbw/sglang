#!/usr/bin/env python3
"""Run Nemotron-3-Diarization on NPU in fp32/fp16/mixed/amp modes."""

from __future__ import annotations

import argparse
import json
import math
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch_npu

from sglang_omni.models.nemotron_diarization.backend import (
    load_diarization_model,
    probabilities_to_segments,
)
from sglang_omni.models.nemotron_diarization.model import SpeakerCache


def mixed_attention_forward(self, hidden_states, rotary, mask):
    qkv = (
        self.w_qkv(hidden_states.half())
        .view(1, hidden_states.shape[1], 3, 8, 64)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)
    q, k = rotary(q, k)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    # Softmax stays FP32.
    scores = scores.float()
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(torch.float32).min)
    attention_weights = torch.softmax(scores, dim=-1).to(v.dtype)
    attention_output = torch.matmul(attention_weights, v)
    attention_output = (
        attention_output.transpose(1, 2)
        .contiguous()
        .view(1, hidden_states.shape[1], 512)
    )
    return self.out_proj(attention_output).float()


def mixed_ffn_forward(self, hidden_states):
    return self.net(hidden_states.half()).float()


def configure_mixed_precision(model):
    """Attention/MLP weights+matmuls in FP16; norms/softmax FP32."""
    for layer in model.encoder.layers:
        attn = layer.attn
        attn.w_qkv.to(torch.float16)
        attn.out_proj.to(torch.float16)
        attn.forward = types.MethodType(mixed_attention_forward, attn)

        ffn = layer.ffn
        ffn.net[0].to(torch.float16)
        ffn.net[3].to(torch.float16)
        ffn.forward = types.MethodType(mixed_ffn_forward, ffn)


def run_file(model, audio_path: Path, mode: str) -> dict:
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        raise SystemExit(f"Expected 16 kHz, got {sample_rate}")

    waveform = torch.as_tensor(audio, dtype=torch.float32, device="npu:0")[None, :]
    torch.npu.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        features, frame_count = model.preprocessor(waveform)
        features = features[:, :, :frame_count]
        if mode == "fp16":
            features = features.to(torch.float16)
        cache_size, fifo_size, chunk_size, right_context, update_period = model.profile
        state = SpeakerCache(
            features,
            cache_size=cache_size,
            fifo_size=fifo_size,
            update_period=update_period,
        )
        outputs = []
        autocast_ctx = (
            torch.npu.amp.autocast(dtype=torch.float16)
            if mode == "amp"
            else torch.autocast("cpu", enabled=False)
        )
        for start_frame in range(0, frame_count, chunk_size * 8):
            end_frame = min(start_frame + chunk_size * 8, frame_count)
            right_frames = min(right_context * 8, frame_count - end_frame)
            with autocast_ctx:
                predictions = model.forward_chunk(
                    features[:, :, start_frame : end_frame + right_frames],
                    state,
                    right_frames,
                )[:, : end_frame - start_frame]
            outputs.append(predictions[0].float().cpu().numpy())
        torch.npu.synchronize()
        probabilities = np.concatenate(outputs)
        # Match the duration exactly; a final feature-stacking pad can leave a
        # fraction of a frame beyond the waveform duration.
        expected_frames = int(round((len(audio) / sample_rate) * 100))
        probabilities = probabilities[:expected_frames]
        result = probabilities_to_segments(
            probabilities, duration_s=len(audio) / sample_rate
        )
    elapsed = time.perf_counter() - started
    return {
        "duration": len(audio) / sample_rate,
        "segments": [
            {"start": s.start, "end": s.end, "speaker": s.speaker}
            for s in result.segments
        ],
        "runtime_s": elapsed,
        "rtf": elapsed / (len(audio) / sample_rate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    parser.add_argument("--mode", choices=("fp32", "fp16", "mixed", "amp"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-path",
        default="/workspace/models/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo",
    )
    args = parser.parse_args()

    torch.npu.set_compile_mode(jit_compile=False)
    model = load_diarization_model(args.model_path, device=torch.device("npu:0"), profile="offline")
    model.eval()
    if args.mode == "fp16":
        model.half()
        # The host STFT/mel featurizer must stay FP32.
        model.preprocessor.float()
    elif args.mode == "mixed":
        configure_mixed_precision(model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    summary = []
    warmup = Path("/tmp/jfk.wav")
    if warmup.exists():
        run_file(model, warmup, args.mode)
    for wav in sorted(data_dir.glob("*.wav")):
        result = run_file(model, wav, args.mode)
        result["file"] = wav.stem
        (output_dir / f"{wav.stem}.json").write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )
        summary.append(
            {
                "file": wav.stem,
                "duration_s": round(result["duration"], 3),
                "runtime_s": round(result["runtime_s"], 4),
                "rtf": round(result["rtf"], 6),
                "segments": len(result["segments"]),
            }
        )
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    (output_dir / "summary.json").write_text(
        json.dumps({"mode": args.mode, "files": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
