# Evaluation: NPU service vs CPU Transformers

## Input

- Audio: `2026-09-07-project-meeting.wav`
- Reference: `2026-09-07-project-meeting.rttm`
- Duration: 3096.6 s (51.61 min), 16 kHz mono PCM16
- Reference segments: 890
- Reference speaker labels: 9 (includes `speaker1` / `speaker2`; model supports 8)
- Reference speech duration: 2408.026 s

## Runtimes

| Path | Hardware | Precision | Processing time | RTF | Speed |
| --- | --- | --- | ---: | ---: | ---: |
| SGLang-Omni HTTP service | 1× Ascend 310P3 | FP32 | 21.85 s | 0.00706 | 141.7× realtime |
| HF Transformers | CPU, 96 threads requested | FP32 | 1145.14 s model forward | 0.3698 | 2.70× realtime |

NPU service speedup over CPU Transformers: **52.4×**.

The CPU baseline used `transformers==5.22.1`,
`AutoModelForAudioFrameClassification`, `attn_implementation="eager"`, FP32.
The NPU service used the current SGLang-Omni server on port 18086.

## DER vs ground truth

Pyannote `DiarizationErrorRate`, overlap included.

### collar = 0 s

| Metric | CPU Transformers | NPU service | NPU - CPU |
| --- | ---: | ---: | ---: |
| DER | **24.906%** | **25.403%** | **+0.497 pp** |
| False alarm | 8.235% | 7.505% | -0.730 pp |
| Missed detection | 11.789% | 12.163% | +0.374 pp |
| Confusion | 4.882% | 5.736% | +0.854 pp |
| Correct | 83.329% | 82.101% | -1.228 pp |

Relative DER change: **+1.996%**.

### collar = 0.25 s

| Metric | CPU Transformers | NPU service | NPU - CPU |
| --- | ---: | ---: | ---: |
| DER | **21.359%** | **21.828%** | **+0.469 pp** |
| False alarm | 6.464% | 5.802% | -0.662 pp |
| Missed detection | 10.428% | 10.706% | +0.278 pp |
| Confusion | 4.467% | 5.321% | +0.854 pp |

## Direct hypothesis agreement

Treating the CPU output as reference and the NPU service output as hypothesis:

| Collar | DER between the two hypotheses |
| ---: | ---: |
| 0 s | 6.190% |
| 0.25 s | 3.577% |

The two runtimes differ in exact boundaries and speaker channel assignment, but
they make errors of similar total magnitude relative to ground truth.

## Conclusion

On this file:

- NPU service is **not catastrophically degraded**.
- Absolute DER is **0.47–0.50 percentage points worse** than the CPU
  Transformers baseline.
- This is a **~2% relative DER increase**, which is small but not zero.
- The NPU service is **52.4× faster** than CPU inference in this test.

The remaining gap may come from a combination of NPU numeric behavior and the
different post-processing paths (SGLang-Omni native segment decoding vs HF
`extract_speaker_dict`), so it should not be attributed to NPU arithmetic alone.

For a final quality decision, run the same comparison on several files from the
dataset and check whether the ~0.5 pp gap is systematic.
