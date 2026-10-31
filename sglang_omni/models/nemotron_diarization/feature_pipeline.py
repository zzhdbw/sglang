# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CPU feature producer used to overlap STFT/mel with NPU inference."""

from __future__ import annotations

import threading

import torch


class ChunkedFeatureProducer:
    """Compute STFT/mel in chunks while the main thread runs NPU windows.

    The original featurizer uses ``center=True``.  Applying the same 256-sample
    zero padding manually and running ``center=False`` over frame-aligned
    chunks is numerically identical, while allowing the producer to publish
    complete feature ranges incrementally.
    """

    def __init__(
        self,
        waveform: torch.Tensor,
        featurizer: torch.nn.Module,
        device: torch.device,
        *,
        chunk_frames: int = 3040,
    ) -> None:
        self.waveform = waveform
        self.featurizer = featurizer
        self.device = device
        self.chunk_frames = chunk_frames
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
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            # The service constructs these tensors while the request thread is
            # inside torch.inference_mode().  The producer thread needs the
            # same mode for its in-place writes to those inference tensors.
            with torch.inference_mode():
                window = self.featurizer.window.cpu()
                fb = self.featurizer.fb.cpu()
                emphasized = torch.cat(
                    [
                        self.waveform[:, :1],
                        self.waveform[:, 1:] - 0.97 * self.waveform[:, :-1],
                    ],
                    dim=1,
                )
                padded = torch.nn.functional.pad(emphasized, (256, 256))
                for start in range(0, self.frame_count, self.chunk_frames):
                    if self.stop_event.is_set():
                        return
                    end = min(start + self.chunk_frames, self.frame_count)
                    begin_sample = start * 160
                    length = (end - start - 1) * 160 + 512
                    chunk = padded[:, begin_sample : begin_sample + length]
                    spectrum = torch.stft(
                        chunk,
                        n_fft=512,
                        hop_length=160,
                        win_length=400,
                        window=window,
                        center=False,
                        return_complex=True,
                    )
                    magnitude = torch.sqrt(
                        torch.view_as_real(spectrum).pow(2).sum(-1)
                    )
                    features = torch.log(
                        torch.matmul(fb, magnitude.pow(2)) + 2**-24
                    )
                    if features.shape[-1] != end - start:
                        raise RuntimeError(
                            "chunked featurizer produced an unexpected frame count"
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
        """Wait for features through ``frame`` and publish new frames to NPU."""
        with self.cond:
            while self.produced < frame and self.error is None:
                if self.stop_event.is_set():
                    return
                self.cond.wait()
            if self.error is not None:
                raise self.error
        target = min(frame, self.frame_count)
        if target > self.copied:
            self.npu_features[:, :, self.copied : target].copy_(
                self.cpu_features[:, :, self.copied : target]
            )
            self.copied = target

    def close(self) -> None:
        self.stop_event.set()
        with self.cond:
            self.cond.notify_all()
        self.thread.join(timeout=10.0)
        if self.error is not None:
            raise self.error
