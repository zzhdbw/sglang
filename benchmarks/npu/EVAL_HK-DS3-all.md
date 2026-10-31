# Evaluation: HK-DS3 first batch, NPU service vs CPU Transformers

## Setup

- Dataset directory: `HK-DS3-第一批-V1.0-0924/HK-DS3-wav16k`
- 8 WAV files with matching RTTM references
- Total audio: 12,845.552 s (3 h 34 min 6 s)
- NPU: 1× Ascend 310P3, FP32, current SGLang-Omni service
- CPU baseline: `transformers==5.22.1`,
  `AutoModelForAudioFrameClassification`, FP32, eager attention, 32 threads
- Scoring: `pyannote.metrics.DiarizationErrorRate`, overlap included,
  full-file UEM, optimal speaker mapping

## Per-file results

`collar=0`, DER percentages. `delta = NPU - CPU`; positive means NPU worse.

| File | Duration min | CPU DER0 | NPU DER0 | Delta pp | CPU DER25 | NPU DER25 | CPU time s | NPU time s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-09-07-project-meeting | 51.61 | 24.906 | 25.403 | +0.497 | 21.359 | 21.828 | 61.09 | 19.96 | 3.06× |
| 2026-09-08-evening-meeting | 69.80 | 41.276 | 42.436 | +1.161 | 37.300 | 38.200 | 109.50 | 29.51 | 3.71× |
| 2026-09-09-AIPC | 35.60 | 36.009 | 24.073 | -11.936 | 30.625 | 17.885 | 69.72 | 14.74 | 4.73× |
| 2026-09-11-voice-mark-train | 38.29 | 63.231 | 63.230 | -0.001 | 60.034 | 60.029 | 53.39 | 15.84 | 3.37× |
| 2026-09-17-daily-03 | 3.78 | 20.046 | 19.451 | -0.596 | 17.786 | 17.173 | 4.64 | 1.53 | 3.04× |
| 2026-09-17-discuss-04-hkvision-mic | 6.36 | 29.355 | 29.310 | -0.045 | 19.686 | 19.606 | 7.80 | 2.51 | 3.10× |
| 2026-09-17-product-02 | 4.03 | 4.482 | 4.486 | +0.005 | 3.289 | 3.279 | 4.43 | 1.44 | 3.07× |
| 2026-09-17-project-01 | 4.62 | 12.851 | 12.965 | +0.114 | 6.410 | 6.567 | 5.68 | 1.72 | 3.31× |

## Aggregate

| Metric | CPU Transformers | NPU service | NPU - CPU |
| --- | ---: | ---: | ---: |
| DER collar=0 | 36.386% | 34.801% | **-1.585 pp** |
| DER collar=0.25 | 32.002% | 30.284% | **-1.718 pp** |
| False alarm collar=0 | 14.194% | 14.278% | +0.085 pp |
| Missed detection collar=0 | 13.553% | 13.413% | -0.140 pp |
| Confusion collar=0 | 8.639% | 7.109% | -1.530 pp |
| Total CPU time | 316.23 s | — | — |
| Total NPU service time | — | 87.26 s | **3.62× faster** |
| Aggregate RTF | 0.02462 | 0.00679 | — |
| Real-time throughput | 40.6× | 147.2× | — |

## Interpretation

- Across all 8 files, the NPU service is **not worse overall**.  Its aggregate
  DER is 1.59 pp lower at collar=0 and 1.72 pp lower at collar=0.25.
- The result is not uniform:
  - `2026-09-08-evening-meeting`: NPU is 1.16 pp worse.
  - `2026-09-07-project-meeting`: NPU is 0.50 pp worse.
  - `2026-09-09-AIPC`: NPU is 11.94 pp better.
  - The other five files are within roughly ±0.6 pp, mostly equal.
- Direct NPU-vs-CPU output disagreement is larger than the ground-truth DER gap
  (for example 19.46% direct DER on AIPC), showing that the two paths produce
  different boundaries/speaker mappings while landing at similar or better
  aggregate quality.
- Some difference is expected from SGLang-Omni's native segment decoding vs
  HuggingFace `extract_speaker_dict`; this evaluation compares the complete
  serving path, not just NPU arithmetic.

## Artifacts

- Per-file service JSON: `/tmp/eval_hkds3/service/`
- Per-file CPU JSON: `/tmp/eval_hkds3/cpu/`
- Full machine-readable summary: `/tmp/eval_hkds3/summary_all.json`
- Dataset itself is not copied into the repository.
