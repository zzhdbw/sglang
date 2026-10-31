#!/usr/bin/env python3
"""Prototype chunked CPU feature producer overlapped with NPU inference."""

from __future__ import annotations

import argparse
import json
import threading
import time
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


class ChunkedFeatureProducer:
    def __init__(self, waveform: torch.Tensor, featurizer, device: torch.device) -> None:
        self.waveform = waveform
        self.featurizer = featurizer
        self.device = device
        self.frame_count = waveform.shape[1] // 160
        self.cpu_features = torch.empty(
            (1, 128, self.frame_count), dtype=torch.float32
        )
        self.npu_features = torch.empty(
            (1, 128, self.frame_count), dtype=torch.float32, device=device
        )
        self.produced = 0
        self.copied = 0
        self.error: BaseException | None = None
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            waveform = self.waveform
            window = self.featurizer.window.cpu()
            fb = self.featurizer.fb.cpu()
            emphasized = torch.cat(
                [waveform[:, :1], waveform[:, 1:] - 0.97 * waveform[:, :-1]], dim=1
            )
            padded = torch.nn.functional.pad(emphasized, (256, 256))
            chunk_frames = 3040  # 30.4 s: one full offline inference window
            for start in range(0, self.frame_count, chunk_frames):
                end = min(start + chunk_frames, self.frame_count)
                # center=False over the manually padded signal is numerically
                # identical to the original center=True full-file STFT.
                begin_sample = start * 160
                length = (end - start - 1) * 160 + 512
                x = padded[:, begin_sample : begin_sample + length]
                spectrum = torch.stft(
                    x,
                    n_fft=512,
                    hop_length=160,
                    win_length=400,
                    window=window,
                    center=False,
                    return_complex=True,
                )
                magnitude = torch.sqrt(torch.view_as_real(spectrum).pow(2).sum(-1))
                features = torch.log(torch.matmul(fb, magnitude.pow(2)) + 2**-24)
                assert features.shape[-1] == end - start, (
                    start,
                    end,
                    features.shape,
                )
                self.cpu_features[:, :, start:end].copy_(features)
                with self.cond:
                    self.produced = end
                    self.cond.notify_all()
        except BaseException as exc:
            with self.cond:
                self.error = exc
                self.cond.notify_all()

    def wait_until(self, frame: int) -> None:
        with self.cond:
            while self.produced < frame and self.error is None:
                self.cond.wait()
            if self.error is not None:
                raise self.error
        target = min(frame, self.frame_count)
        if target > self.copied:
            self.npu_features[:, :, self.copied : target].copy_(
                self.cpu_features[:, :, self.copied : target]
            )
            self.copied = target

    def join(self) -> None:
        self.thread.join()
        if self.error is not None:
            raise self.error


def run_baseline(model, waveform, duration_s):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = []
        for predictions, done in model.iter_chunks(waveform):
            outputs.append(predictions[0].float().cpu().numpy())
    torch.npu.synchronize()
    elapsed = time.perf_counter() - t0
    probabilities = np.concatenate(outputs)
    expected = int(round(duration_s * 100))
    result = probabilities_to_segments(probabilities[:expected], duration_s=duration_s)
    return elapsed, result


def run_overlap(model, waveform, duration_s):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    producer = ChunkedFeatureProducer(
        waveform.cpu(), model.preprocessor.featurizer, waveform.device
    )
    producer.start()
    cache_size, fifo_size, chunk_size, right_context, update_period = model.profile
    with torch.inference_mode():
        state = SpeakerCache(
            producer.npu_features,
            cache_size=cache_size,
            fifo_size=fifo_size,
            update_period=update_period,
        )
        outputs = []
        try:
            for start_frame in range(0, producer.frame_count, chunk_size * 8):
                end_frame = min(start_frame + chunk_size * 8, producer.frame_count)
                right_frames = min(right_context * 8, producer.frame_count - end_frame)
                producer.wait_until(end_frame + right_frames)
                predictions = model.forward_chunk(
                    producer.npu_features[
                        :, :, start_frame : end_frame + right_frames
                    ],
                    state,
                    right_frames,
                )[:, : end_frame - start_frame]
                outputs.append(predictions[0].float().cpu().numpy())
        finally:
            producer.join()
    torch.npu.synchronize()
    elapsed = time.perf_counter() - t0
    probabilities = np.concatenate(outputs)
    expected = int(round(duration_s * 100))
    result = probabilities_to_segments(probabilities[:expected], duration_s=duration_s)
    return elapsed, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument(
        "--model-path",
        default="/workspace/models/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo",
    )
    parser.add_argument("--output", default="/tmp/pipeline_overlap_result.json")
    args = parser.parse_args()

    torch.npu.set_compile_mode(jit_compile=False)
    model = load_diarization_model(
        args.model_path, device=torch.device("npu:0"), profile="offline"
    )
    model.eval()
    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sample_rate == 16000
    duration_s = len(audio) / sample_rate
    waveform = torch.as_tensor(audio, dtype=torch.float32, device="npu:0")[None, :]

    # warmup
    warm = waveform[:, : 16000 * 11]
    run_baseline(model, warm, 11.0)
    torch.npu.synchronize()

    base_s, base_result = run_baseline(model, waveform, duration_s)
    overlap_s, overlap_result = run_overlap(model, waveform, duration_s)

    base_segments = [
        {"start": s.start, "end": s.end, "speaker": s.speaker}
        for s in base_result.segments
    ]
    overlap_segments = [
        {"start": s.start, "end": s.end, "speaker": s.speaker}
        for s in overlap_result.segments
    ]
    result = {
        "audio_s": duration_s,
        "baseline_s": base_s,
        "overlap_s": overlap_s,
        "speedup": base_s / overlap_s,
        "segments_equal": base_segments == overlap_segments,
        "baseline_segments": len(base_segments),
        "overlap_segments": len(overlap_segments),
    }
    Path(args.output).write_text(
        json.dumps(
            {"summary": result, "segments": overlap_segments},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
