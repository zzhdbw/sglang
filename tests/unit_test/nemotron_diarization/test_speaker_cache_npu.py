# SPDX-License-Identifier: Apache-2.0
"""CPU parity for the NPU-safe speaker-cache compression path."""

from __future__ import annotations

import math

import torch

from sglang_omni.models.nemotron_diarization.speaker_cache import SpeakerCache


def reference_compress(
    cache: torch.Tensor,
    cache_predictions: torch.Tensor,
    silence_embedding: torch.Tensor,
    cache_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The upstream indexed-assignment implementation, suitable for CPU tests."""
    cache = cache.clone()
    cache_predictions = cache_predictions.clone()
    batch, _, speakers = cache_predictions.shape
    log_probs = torch.log(torch.clamp(cache_predictions, min=0.25))
    log_other = torch.log(torch.clamp(1.0 - cache_predictions, min=0.25))
    scores = log_probs - log_other + log_other.sum(dim=2, keepdim=True) - math.log(0.5)
    scores = torch.where(cache_predictions > 0.5, scores, float("-inf"))
    frames_per_speaker = cache_size // speakers - 1
    positive = scores > 0
    disable = (
        (~positive)
        * (cache_predictions > 0.5)
        * (positive.sum(dim=1, keepdim=True) >= math.floor(frames_per_speaker * 0.5))
    )
    scores = torch.where(disable, float("-inf"), scores)
    scores[:, cache_size:] += 0.05
    for boosted_frames, boost_scale in [
        (math.floor(frames_per_speaker * 0.75), 2),
        (math.floor(frames_per_speaker * 1.5), 1),
    ]:
        indices = torch.topk(scores, boosted_frames, dim=1, sorted=False).indices
        scores[
            torch.arange(batch)[:, None, None],
            indices,
            torch.arange(speakers)[None, None, :],
        ] -= boost_scale * math.log(0.5)
    scores = torch.cat(
        [scores, scores.new_full((batch, 1, speakers), float("inf"))], dim=1
    )
    frame_count = scores.shape[1]
    values, indices = torch.topk(
        scores.permute(0, 2, 1).reshape(batch, -1), cache_size, sorted=False
    )
    indices = torch.where(values != float("-inf"), indices, 99999).sort(dim=1).values
    disabled = indices == 99999
    indices = indices.remainder(frame_count)
    disabled |= indices >= frame_count - 1
    indices = indices.masked_fill(disabled, 0)
    embeddings = torch.gather(
        cache, 1, indices.unsqueeze(-1).expand(-1, -1, cache.shape[2])
    )
    cache_predictions = torch.gather(
        cache_predictions, 1, indices.unsqueeze(-1).expand(-1, -1, speakers)
    )
    cache = torch.where(
        disabled.unsqueeze(-1), silence_embedding[None, None, :], embeddings
    )
    cache_predictions = torch.where(disabled.unsqueeze(-1), 0.0, cache_predictions)
    return cache, cache_predictions


def test_compress_matches_upstream_indexed_boost_semantics() -> None:
    torch.manual_seed(7)
    cache_size = 32
    state = SpeakerCache(
        torch.empty(1, 0, 8),
        cache_size=cache_size,
        fifo_size=4,
        update_period=2,
        width=8,
        speakers=8,
    )
    state.cache = torch.randn(1, 40, 8)
    state.cache_preds = torch.rand(1, 40, 8)
    state.compressed = True
    silence = torch.randn(8)

    expected_cache, expected_preds = reference_compress(
        state.cache, state.cache_preds, silence, cache_size
    )
    state.compress(silence)

    torch.testing.assert_close(state.cache, expected_cache)
    torch.testing.assert_close(state.cache_preds, expected_preds)
