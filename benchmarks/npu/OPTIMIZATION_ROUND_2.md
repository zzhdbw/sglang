# Second optimization round: Nemotron-3-Diarization on 310P

Baseline: FP32, skip-mask full-window optimization, `OMP_NUM_THREADS=16`.

## Results summary

| Optimization | Speed result | Accuracy result | Decision |
| --- | ---: | ---: | --- |
| Fused `npu_prompt_flash_attention` | attention microbench 0.284 vs 0.290 ms (~2%) | FP16-only op; not integrated | Not adopted |
| Batch D2H copy | no measurable change | exact | Reverted |
| CPU STFT / NPU overlap | service 51.6 min: 11.6 s → **9.7 s** | **segments bit-identical** | **Adopted** |
| `max_concurrency=2` | 2-request wall 23.1 s → **20.4 s** (1.13×) | exact | Optional for batch |
| Attention-only FP16 | 350.5× realtime vs 309.3× (1.13×) | DER0 +2.70 pp | Rejected |
| MLP-only FP16 | 331.5× vs 309.3× (1.07×) | DER0 -1.20 pp aggregate, but `daily-03` +3.69 pp | Optional/risky |
| Mixed attention+MLP FP16 | 355.1× vs 309.3× (1.15×) | DER0 +0.27 pp, `AIPC` +3.37 pp | Not adopted |
| torch_npu AMP | 377.6× vs 309.3× (1.22×) | DER0 +0.96 pp | Not adopted |
| Whole-model FP16 | 441.8× vs 309.3× (1.43×) | DER0 +2.93 pp | Rejected |

## CPU/NPU overlap implementation

The 310P STFT fallback is CPU-only.  The new
`ChunkedFeatureProducer` computes frame-aligned STFT/mel chunks in a
background thread while the main thread runs NPU inference.

- Manual zero-padding plus `center=False` is numerically identical to the
  original `center=True` full-file STFT.
- The producer publishes only newly completed feature frames to a
  pre-allocated NPU buffer.
- The producer thread runs inside `torch.inference_mode()` so in-place writes
  to inference tensors remain valid in the serving path.

Measured:

| Workload | Before overlap | After overlap | Speedup |
| --- | ---: | ---: | ---: |
| Direct model, 51.6 min | 9.95 s | **8.31 s** | 1.20× |
| Service, 51.6 min warm | ~11.6 s | **9.7 s** | 1.20× |
| 8-file service total | 48.67 s | **40.99 s** | 1.19× |

All 8 service outputs were compared against the previous service: every
`segments` list was **exactly identical**, so DER is unchanged:
DER0 = 34.8009%, DER25 = 30.2840%.

## Fused attention notes

- `npu_fusion_attention` fails on 310P: `FlashAttentionScore` has no binary.
- `npu_prompt_flash_attention` works but only accepts FP16 on Atlas inference.
- For the model's actual attention shape (B=1, N=8, S=512, D=64), fused
  attention measured 0.284 ms vs 0.290 ms eager, only ~2%.
- Using it would force attention to FP16, whose accuracy regression was
  already measured at +2.70 pp DER0 for attention-only FP16.
- Therefore fused attention is not worth adopting on this hardware.

## Batch D2H notes

Accumulating all output tensors on NPU and doing one `.cpu()` at the end was
tested.  It produced identical outputs but no measurable speedup because the
stage scheduler still synchronizes the stream after each chunk.  The change was
reverted to keep the simpler path.

## Concurrency note

With the overlap implementation and `max_concurrency=2`, two concurrent long
requests completed in 20.41 s of wall time versus 23.13 s sequentially
(1.13× aggregate).  Individual request latency increased from 9.80/13.34 s to
16.55/20.39 s.  This is useful only for batch throughput, not low-latency
serving; production remains `max_concurrency=1`.

## Current production configuration

- FP32
- Skip-mask full-window attention
- Chunked CPU STFT / NPU overlap
- `OMP_NUM_THREADS=16`
- `max_concurrency=1`
- Single 310P3 die
- 51.6-minute service time: **~9.7 s**
- 8-file service total: **~41.0 s**
- Aggregate DER: **34.8009%** collar=0, **30.2840%** collar=0.25

## Can STFT run on the NPU?

Direct eager STFT/FFT is not available on 310P:

```text
torch.stft / torch.fft.rfft / torch.fft.fft
RuntimeError: Current settings do not support Complex dtype.
```

All three fail because these APIs produce complex tensors and the current
310P torch_npu eager backend does not support complex dtype.

A real-valued DFT implemented as NPU matmuls is possible:

```text
real = frames @ cos_table
imag = frames @ sin_table
magnitude = sqrt(real^2 + imag^2)
mel = magnitude^2 @ mel_filter
```

Measured on a 111.8-second audio sample:

| Implementation | Time |
| --- | ---: |
| CPU STFT/Mel, OMP=16 | 72.7 ms |
| NPU real-DFT matmul, excluding H2D | 297.1 ms |

The NPU real-DFT path is roughly 4× slower and would consume the NPU cycles
that are already the main bottleneck.  With the current CPU/NPU overlap, host
STFT is mostly hidden, so moving STFT to the NPU is not beneficial.

## Current concurrency

Single 310P3, 250 ms upload / 250 ms interval, 60-second sessions, ACK
backpressure:

| Streams | Result |
| ---: | --- |
| 12 | stable |
| 14 | stable, no errors, p95 ACK lag ~0.90 s |
| 15 | queue-full / connection failures begin |

Recommended operating point: **12 realtime streams per die**.
Measured hard limit: **14 realtime streams per die**.

For offline batch requests, the production service uses
`max_concurrency=1`.  A test instance with `max_concurrency=2` completed two
long requests in 20.41 s versus 23.13 s sequential (1.13× aggregate), but
single-request latency increased substantially.  The recommended defaults are:

- low latency: `max_concurrency=1`
- throughput-oriented batch: `max_concurrency=2`

With all eight 310P3 dies dedicated and one service per die, the extrapolated
realtime capacity is about **96 recommended / 112 hard-limit streams**.
