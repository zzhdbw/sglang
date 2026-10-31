# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference-only Nemotron 3 Sortformer.

Adapted from NVIDIA-NeMo/Speech revision
2c1a2f91d64566b5d391b83df42f9ab4cd810adb: transformer_encoder.py,
sortformer_diar_models.py, features.py, subsampling.py and multi_head_attention.py.
Omni changes: fixed checkpoint architecture, single-recording inference, plain
PyTorch modules, and explicit request-local cache ownership.
"""

from collections import OrderedDict
from collections.abc import Generator
from threading import local

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.nemotron_diarization.speaker_cache import SpeakerCache
from sglang_omni.utils.device import has_device_streams

# The upstream CUDA path uses a Triton rotary kernel and flex_attention.  Their
# generated kernels are not available on Ascend 310P, so this fork uses a
# numerically equivalent eager PyTorch path.  The model is small (100M params)
# and windows are short, making eager attention practical on NPU.
MASK_BLOCK_FRAMES = 128


class Featurizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # note (Ryan Stewart): Restored from the checkpoint, for a centered rather than causal STFT.
        self.register_buffer("window", torch.empty(400))
        self.register_buffer("fb", torch.empty(1, 128, 257))

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
        device = waveform.device
        if device.type == "npu":
            # torch.stft emits complex tensors, which Ascend 310P/torch_npu does
            # not support.  The feature extractor is only a small STFT, so run it
            # on the host and move the real-valued mel features to the NPU.
            waveform = waveform.cpu()
            window = self.window.cpu()
            fb = self.fb.cpu()
        else:
            window = self.window
            fb = self.fb

        frame_count = waveform.shape[1] // 160
        emphasized = torch.cat(
            [waveform[:, :1], waveform[:, 1:] - 0.97 * waveform[:, :-1]], dim=1
        )
        spectrum = torch.stft(
            emphasized,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        # note (Ryan Stewart): Preserve the reference's sqrt/square rounding.
        magnitude = torch.sqrt(torch.view_as_real(spectrum).pow(2).sum(-1))
        features = torch.log(torch.matmul(fb, magnitude.pow(2)) + 2**-24)
        features[:, :, frame_count:] = 0
        if device.type == "npu":
            features = features.to(device)
        return features, frame_count


class Preprocessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.featurizer = Featurizer()

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
        return self.featurizer(waveform)


class FeatureStacking(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8 * 128, 512, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        frames = features.transpose(1, 2)
        frames = F.pad(frames, (0, 0, 0, -frames.shape[1] % 8))
        return self.proj(frames.reshape(1, -1, 8 * 128))


class RotaryEmbedding(nn.Module):
    def __init__(self, max_frames: int) -> None:
        super().__init__()
        # note (Richard Wang): CPU tables preserve device-independent rounding.
        frequencies = 1.0 / (
            10000 ** (torch.arange(0, 64, 2, device="cpu").float() / 64)
        )
        angles = torch.outer(
            torch.arange(max_frames, device="cpu").float(), frequencies
        )
        angles = torch.cat([angles, angles], dim=-1)
        cos, sin = torch.empty_like(angles), torch.empty_like(angles)
        for angle, cos_row, sin_row in zip(angles, cos, sin):
            torch.cos(angle, out=cos_row)
            torch.sin(angle, out=sin_row)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply NeOX rotation with separate products and addition."""
        frames = q.shape[2]
        assert frames <= len(self.cos), f"{frames} frames exceed the rotary table"
        cos = self.cos[:frames].to(device=q.device, dtype=q.dtype)[None, None]
        sin = self.sin[:frames].to(device=q.device, dtype=q.dtype)[None, None]

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            half = x.shape[-1] // 2
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w_qkv = nn.Linear(512, 1536, bias=False)
        self.out_proj = nn.Linear(512, 512)

    def forward(
        self, hidden_states: torch.Tensor, rotary: RotaryEmbedding, mask: torch.Tensor
    ) -> torch.Tensor:
        qkv = (
            self.w_qkv(hidden_states)
            .view(1, hidden_states.shape[1], 3, 8, 64)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = rotary(q, k)

        # Eager full attention.  The padding mask is shaped (1, frame_count) and
        # broadcasts over query rows, batch and heads.
        scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        return self.out_proj(
            attention_output.transpose(1, 2)
            .contiguous()
            .view(1, hidden_states.shape[1], 512)
        )


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # note (Ryan Stewart): Identity holds the checkpoint's training-dropout index.
        self.net = nn.Sequential(
            nn.Linear(512, 2048), nn.GELU(), nn.Identity(), nn.Linear(2048, 512)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(512)
        self.attn = Attention()
        self.norm2 = nn.LayerNorm(512)
        self.ffn = FeedForward()

    def forward(
        self, hidden_states: torch.Tensor, rotary: RotaryEmbedding, mask: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), rotary, mask
        )
        return hidden_states + self.ffn(self.norm2(hidden_states))


class TransformerEncoder(nn.Module):
    def __init__(self, max_frames: int) -> None:
        super().__init__()
        self.pre_encode = FeatureStacking()
        self.pos_enc = RotaryEmbedding(max_frames)
        self.embed_norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(31)])
        self.final_norm = nn.LayerNorm(512)
        # note (Richard Wang): Per-thread masks stay on that worker thread's stream.
        self.thread_masks = local()

    def attention_mask(self, frame_count: int, valid_length: int) -> torch.Tensor:
        masks = getattr(self.thread_masks, "masks", None)
        if masks is None:
            masks = self.thread_masks.masks = OrderedDict()
        else:
            pass
        mask_key = (frame_count, valid_length, str(self.final_norm.weight.device))
        if mask_key in masks:
            masks.move_to_end(mask_key)
            return masks[mask_key]
        else:
            pass
        device = self.final_norm.weight.device
        # Only keys beyond valid_length are padding.  A (1, frame_count) mask
        # broadcasts across all query rows and attention heads.
        mask = torch.arange(frame_count, device=device)[None, :] < valid_length
        masks[mask_key] = mask
        if len(masks) > 128:
            masks.popitem(last=False)
        else:
            pass
        return mask

    def forward(self, embeddings: torch.Tensor, valid_length: int) -> torch.Tensor:
        mask = self.attention_mask(embeddings.shape[1], valid_length)
        hidden_states = self.embed_norm(embeddings)
        for layer in self.layers:
            hidden_states = layer(hidden_states, self.pos_enc, mask)
        return self.final_norm(hidden_states)


class SortformerModules(nn.Module):
    def __init__(
        self,
        *,
        fc_d_model: int = 512,
        tf_d_model: int = 192,
        speakers: int = 8,
        upsample: int = 8,
    ) -> None:
        super().__init__()
        self.tf_d_model = tf_d_model
        self.speakers = speakers
        self.upsample = upsample
        # note (Ryan Stewart): Unused legacy head that strict loading needs from the archive.
        self.hidden_to_spks = nn.Linear(2 * tf_d_model, speakers)
        self.first_hidden_to_hidden = nn.Linear(tf_d_model, tf_d_model)
        self.single_hidden_to_spks = nn.Linear(tf_d_model, speakers)
        self.encoder_proj = nn.Linear(fc_d_model, tf_d_model)
        self.subpixel_upsample = nn.Conv1d(
            tf_d_model, tf_d_model * upsample, 3, padding=1
        )
        self.learnable_sil_emb = nn.Parameter(torch.empty(fc_d_model))

    def forward(
        self, embeddings: torch.Tensor, valid_length: int | torch.Tensor
    ) -> torch.Tensor:
        """valid_length is an int, or per-row lengths shaped (batch, 1)."""
        hidden = self.encoder_proj(embeddings)
        hidden = self.subpixel_upsample(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = hidden.reshape(embeddings.shape[0], -1, self.tf_d_model)
        hidden = F.relu(self.first_hidden_to_hidden(F.relu(hidden)))
        probabilities = torch.sigmoid(self.single_hidden_to_spks(hidden))
        mask = (
            torch.arange(embeddings.shape[1], device=embeddings.device) < valid_length
        )
        return probabilities * mask.repeat_interleave(self.upsample, dim=-1)[..., None]


class NemotronDiarizationModel(nn.Module):
    def __init__(self, *, profile: tuple[int, int, int, int, int]) -> None:
        super().__init__()
        self.preprocessor = Preprocessor()
        self.encoder = TransformerEncoder(sum(profile[:4]))
        self.sortformer_modules = SortformerModules()
        self.profile = profile

    def iter_chunks(
        self, waveform: torch.Tensor
    ) -> Generator[tuple[torch.Tensor, bool], None, None]:
        """Yield each upload window's 10 ms probabilities and whether it is the last."""
        features, frame_count = self.preprocessor(waveform)
        del waveform
        features = features[:, :, :frame_count]
        cache_size, fifo_size, chunk_size, right_context, update_period = self.profile
        state = SpeakerCache(
            features,
            cache_size=cache_size,
            fifo_size=fifo_size,
            update_period=update_period,
        )
        if frame_count == 0:
            yield features.new_empty(1, 0, 8), True
        else:
            pass
        for start_frame in range(0, frame_count, chunk_size * 8):
            if has_device_streams(features.device):
                # The next window may run on another worker stream.
                device_module = torch.get_device_module(features.device)
                stream = device_module.current_stream(features.device)
                for tensor in (features, state.cache, state.fifo, state.cache_preds):
                    if hasattr(tensor, "record_stream"):
                        tensor.record_stream(stream)
            else:
                pass
            end_frame = min(start_frame + chunk_size * 8, frame_count)
            right_frames = min(right_context * 8, frame_count - end_frame)
            yield (
                self.forward_chunk(
                    features[:, :, start_frame : end_frame + right_frames],
                    state,
                    right_frames,
                )[:, : end_frame - start_frame],
                end_frame == frame_count,
            )

    def forward_chunk(
        self, features: torch.Tensor, state: SpeakerCache, right_frames: int
    ) -> torch.Tensor:
        """Advance one recording's cache, with right context in 10 ms frames."""
        chunk = self.encoder.pre_encode(features)
        prefix_length = state.cache.shape[1] + state.fifo.shape[1]
        embeddings = torch.cat([state.cache, state.fifo, chunk], dim=1)
        valid_length = prefix_length + (features.shape[2] + 7) // 8
        encoded = self.encoder(embeddings, valid_length)
        predictions = self.sortformer_modules(encoded, valid_length)
        low_resolution = F.avg_pool1d(predictions.transpose(1, 2), 8, 8).transpose(1, 2)
        kept_frames = chunk.shape[1] - (right_frames + 7) // 8
        output = predictions[:, prefix_length * 8 : (prefix_length + kept_frames) * 8]
        state.update(
            chunk[:, :kept_frames],
            low_resolution,
            self.sortformer_modules.learnable_sil_emb,
        )
        return output
