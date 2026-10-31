# Precision sweep: Nemotron-3-Diarization on 310P

## Setup

- Hardware: 1× Ascend 310P3
- Audio: 8 HK-DS3 meetings, 12,845.6 s total
- Direct NPU inference, `OMP_NUM_THREADS=16`
- Baseline: FP32, skip-mask full-window optimization
- DER: pyannote, full-file UEM, optimal speaker mapping, overlap included

Modes:

1. `fp16`: whole model weights and activations FP16; host STFT stays FP32.
2. `mixed`: only attention and MLP linear/MatMul paths FP16. LayerNorm,
   attention Softmax, pre-encoder, final norm, and sortformer stay FP32.
3. `amp`: `torch.npu.amp.autocast(dtype=torch.float16)` around model forward;
   weights stay FP32.

## Speed

| Mode | Total 8-file runtime | Realtime factor | vs FP32 |
| --- | ---: | ---: | ---: |
| FP32 baseline | 41.53 s | 309.3× | 1.00× |
| FP16 whole model | **29.07 s** | **441.8×** | **1.43×** |
| Mixed attention/MLP | 36.18 s | 355.1× | 1.15× |
| torch_npu AMP | 34.02 s | 377.6× | 1.22× |

## Accuracy

| Mode | DER collar=0 | Δ vs FP32 | DER collar=0.25 | Δ vs FP32 |
| --- | ---: | ---: | ---: | ---: |
| FP32 baseline | 34.801% | — | 30.284% | — |
| FP16 whole model | 37.732% | **+2.931 pp** | 33.472% | **+3.188 pp** |
| Mixed attention/MLP | 35.075% | **+0.274 pp** | 30.693% | **+0.409 pp** |
| torch_npu AMP | 35.765% | **+0.964 pp** | 31.341% | **+1.057 pp** |

## Per-file risk

FP16 whole-model is not acceptable for this workload:

- `project-meeting`: +6.38 pp
- `AIPC`: +6.27 pp
- `daily-03`: +10.29 pp

Mixed is the best aggregate trade-off, but not uniformly safe:

- `AIPC`: +3.37 pp
- `evening-meeting`: -1.02 pp
- other files: within roughly ±0.2 pp

AMP is intermediate:

- `AIPC`: +2.30 pp
- `voice-mark-train`: +1.41 pp
- `project-meeting`: +0.90 pp

All three alter speaker/boundary decisions enough that a strict per-file
no-degradation requirement is not met. If the acceptance criterion is
aggregate DER, mixed is the best candidate.

## Recommendation

- **Best speed/accuracy trade-off: manual mixed attention/MLP FP16**
  - +14.8% direct model throughput
  - +0.27 pp aggregate DER0
  - +0.41 pp aggregate DER25
- **Fastest but not acceptable: whole-model FP16**
  - +43% throughput but +2.9 pp aggregate DER0
- **AMP: middle ground**
  - +22% throughput, +0.96 pp aggregate DER0
- Production service remains FP32 + skip-mask for now, because mixed still has
  a 3.37 pp regression on `AIPC` and needs either a safer mixed recipe or a
  per-domain accuracy check.

## Artifacts

- Benchmark script: `benchmarks/npu/precision_sweep.py`
- Outputs: `/tmp/precision_eval/{fp32,fp16,mixed,amp}/`
- Scoring summaries: `/tmp/precision_eval/summary_{fp16,mixed,amp}.json`
