#!/usr/bin/env python3
"""Measure preprocessing, NPU forward, D2H, and postprocessing by phase."""

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
from sglang_omni.models.nemotron_diarization.speaker_cache import SpeakerCache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument(
        "--model-path",
        default="/workspace/models/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo",
    )
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    device = torch.device(args.device)

    load_t0 = time.perf_counter()
    model = load_diarization_model(args.model_path, device=device, profile="offline")
    load_s = time.perf_counter() - load_t0

    audio_t0 = time.perf_counter()
    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        raise SystemExit(f"Expected 16 kHz, got {sample_rate}")
    audio_load_s = time.perf_counter() - audio_t0

    waveform = torch.as_tensor(audio, dtype=torch.float32, device=device)[None, :]
    if device.type == "npu":
        torch.npu.synchronize()
    with torch.inference_mode():
        pre_t0 = time.perf_counter()
        features, frame_count = model.preprocessor(waveform)
        if device.type == "npu":
            torch.npu.synchronize()
        preprocess_s = time.perf_counter() - pre_t0
        features = features[:, :, :frame_count]

        cache_size, fifo_size, chunk_size, right_context, update_period = model.profile
        state = SpeakerCache(
            features,
            cache_size=cache_size,
            fifo_size=fifo_size,
            update_period=update_period,
        )
        outputs: list[np.ndarray] = []
        npu_event_s = 0.0
        enqueue_s = 0.0
        d2h_s = 0.0
        chunk_wall_s = 0.0
        chunk_count = 0
        t_all = time.perf_counter()
        for start_frame in range(0, frame_count, chunk_size * 8):
            end_frame = min(start_frame + chunk_size * 8, frame_count)
            right_frames = min(right_context * 8, frame_count - end_frame)
            t0 = time.perf_counter()
            if device.type == "npu":
                start_event = torch.npu.Event(enable_timing=True)
                end_event = torch.npu.Event(enable_timing=True)
                start_event.record()
            predictions = model.forward_chunk(
                features[:, :, start_frame : end_frame + right_frames],
                state,
                right_frames,
            )
            if device.type == "npu":
                end_event.record()
            enqueue_s += time.perf_counter() - t0
            t1 = time.perf_counter()
            outputs.append(predictions[0].float().cpu().numpy())
            d2h_s += time.perf_counter() - t1
            if device.type == "npu":
                torch.npu.synchronize()
                npu_event_s += start_event.elapsed_time(end_event) / 1000.0
            chunk_wall_s += time.perf_counter() - t0
            chunk_count += 1
        if device.type == "npu":
            torch.npu.synchronize()
        total_loop_s = time.perf_counter() - t_all

    t0 = time.perf_counter()
    result = probabilities_to_segments(
        np.concatenate(outputs), duration_s=len(audio) / sample_rate
    )
    postprocess_s = time.perf_counter() - t0

    print(json.dumps(
        {
            "audio_s": len(audio) / sample_rate,
            "frame_count_10ms": frame_count,
            "chunks": chunk_count,
            "model_load_s": round(load_s, 4),
            "audio_load_s": round(audio_load_s, 4),
            "cpu_preprocess_s": round(preprocess_s, 4),
            "chunk_loop_s": round(total_loop_s, 4),
            "npu_forward_events_s": round(npu_event_s, 4),
            "cpu_enqueue_s": round(enqueue_s, 4),
            "d2h_copy_s": round(d2h_s, 4),
            "cpu_postprocess_s": round(postprocess_s, 4),
            "chunk_wall_sum_s": round(chunk_wall_s, 4),
            "segments": len(result.segments),
            "preprocess_rtf": round(preprocess_s / (len(audio) / sample_rate), 6),
            "chunk_loop_rtf": round(total_loop_s / (len(audio) / sample_rate), 6),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
