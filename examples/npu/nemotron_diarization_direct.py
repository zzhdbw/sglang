#!/usr/bin/env python3
"""Run Nemotron-3-Diarization directly through SGLang-Omni on an Ascend NPU."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import soundfile as sf
import torch
import torch_npu  # noqa: F401

from sglang_omni.models.nemotron_diarization.backend import (
    load_diarization_model,
    probabilities_to_segments,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="16 kHz mono WAV/FLAC file")
    parser.add_argument(
        "--model-path",
        default="/workspace/models/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--profile", choices=("offline", "low_latency"), default="offline")
    args = parser.parse_args()

    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if audio.ndim != 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        raise SystemExit(
            f"Expected 16 kHz audio, got {sample_rate} Hz. Convert with: "
            f"ffmpeg -i {args.audio} -ac 1 -ar 16000 -c:a pcm_s16le out.wav"
        )

    device = torch.device(args.device)
    start = time.perf_counter()
    model = load_diarization_model(
        args.model_path, device=device, profile=args.profile
    )
    print(f"model loaded in {time.perf_counter() - start:.2f}s", flush=True)

    waveform = torch.as_tensor(audio, dtype=torch.float32, device=device)[None, :]
    chunks: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for predictions, done in model.iter_chunks(waveform):
            if device.type == "npu":
                torch.npu.synchronize()
            chunks.append(predictions[0].float().cpu().numpy())
            print(
                f"chunk={len(chunks)} frames={predictions.shape[1]} done={done} "
                f"elapsed={time.perf_counter() - start:.3f}s",
                flush=True,
            )

    result = probabilities_to_segments(
        np.concatenate(chunks), duration_s=len(audio) / 16000
    )
    print(json.dumps(
        {
            "duration": result.duration,
            "segments": [
                {"start": s.start, "end": s.end, "speaker": s.speaker}
                for s in result.segments
            ],
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
