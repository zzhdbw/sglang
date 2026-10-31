#!/usr/bin/env python3
"""Run the same Nemotron-3-Diarization checkpoint with HF Transformers on CPU."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

MODEL_PATH = "/workspace/models/Nemotron-3-Diarization"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=96)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForAudioFrameClassification.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()

    start = time.perf_counter()
    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        raise SystemExit(f"Expected 16 kHz audio, got {sample_rate}")
    print(f"audio_loaded seconds={len(audio)/sample_rate:.3f} t={time.perf_counter()-start:.2f}", flush=True)

    start = time.perf_counter()
    inputs = processor(audio, sampling_rate=sample_rate)
    print(
        f"features={tuple(inputs['input_features'].shape)} "
        f"processor_t={time.perf_counter()-start:.2f}",
        flush=True,
    )

    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs).logits
    inference_t = time.perf_counter() - start
    print(f"inference_t={inference_t:.2f} logits={tuple(logits.shape)}", flush=True)

    start = time.perf_counter()
    segments = processor.extract_speaker_dict(
        logits, inputs.get("attention_mask")
    )[0]
    normalized = [
        {
            "start": float(segment["Start"]),
            "end": float(segment["End"]),
            "speaker": str(segment["Speaker"]),
        }
        for segment in segments
    ]
    duration = len(audio) / sample_rate
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "duration": duration,
                "segments": normalized,
                "metrics": {
                    "processor_t": round(inference_t, 3),
                    "inference_t": round(inference_t, 3),
                    "postprocess_t": round(time.perf_counter() - start, 3),
                    "total_audio_s": duration,
                    "rtf": round(inference_t / duration, 6),
                },
            },
            handle,
            ensure_ascii=False,
        )
    print(f"wrote {args.output} segments={len(normalized)}", flush=True)


if __name__ == "__main__":
    main()
