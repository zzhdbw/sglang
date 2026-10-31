# Profile: Nemotron-3-Diarization on Ascend 310P

## Test input

- `2026-09-07-project-meeting.wav`
- 3096.6 s (51.61 min), 16 kHz mono PCM16
- Model: Nemotron-3-Diarization, FP32
- One Ascend 310P3
- Offline profile: 27.2 s scored per internal step

## End-to-end and communication

Built-in request profiler, current service (`OMP_NUM_THREADS=1`):

| Boundary | Time |
| --- | ---: |
| HTTP request total | 19.928 s |
| Coordinator admission → terminal response | 19.381 s |
| Stage input received → stage complete | 18.825 s |
| Outside stage | **1.103 s (5.5%)** |
| Coordinator admission → stage input | 0.552 s |
| HTTP pre-admission / upload parse | 0.547 s |
| Stage complete → terminal response | 0.003 s |

Communication/queueing is therefore **not the main bottleneck**: roughly
5.5% of end-to-end time for this 51-minute file.

The same breakdown for a smaller 111.8 s request gave a 661.7 ms stage and
about 51 ms outside the stage (~7.2%).

## Within the diarization stage

### CPU preprocessor

The full-file STFT/mel extractor runs on CPU because 310P torch_npu cannot
produce the required complex STFT tensor.

Measured with the same model/audio:

| Setting | CPU preprocessing time | RTF |
| --- | ---: | ---: |
| `OMP_NUM_THREADS=1` | **8.71 s** | 0.00281 |
| `OMP_NUM_THREADS=16` | **1.69 s** | 0.00055 |

With the current default `OMP_NUM_THREADS=1`, this is roughly **44% of the
18.8 s stage time**.  It is the largest avoidable CPU bottleneck.

### NPU kernels

A 111.8 s request was profiled with the torch/NPU profiler.  For that request:

- Stage time: 661.7 ms
- CPU preprocess: 253.4 ms
- NPU device-busy union: 388.4 ms (~59% of stage)
- CPU postprocess: 4.1 ms

Top NPU kernels by device time:

| Kernel | Device time |
| --- | ---: |
| `MaskedFill` | 92.1 ms |
| `SoftmaxV2` | 66.9 ms |
| `Cast` | 40.6 ms |
| `TransData` | 22.2 ms |
| `Gelu` | 21.5 ms |
| `Mul` | 17.0 ms |
| `TensorMove` | 16.6 ms |

The NPU cost is dominated by eager attention (`MaskedFill` + `Softmax`),
casts, and layout transforms rather than a single large matmul kernel.

### Approximate split for the 51-minute request

Current default (`OMP_NUM_THREADS=1`):

| Component | Approx. time | Share |
| --- | ---: | ---: |
| CPU STFT/mel preprocessing | 8.7 s | ~44% of stage |
| NPU forward/kernels | 10–11 s | ~52–55% of stage |
| CPU postprocess | 0.06 s | <1% |
| HTTP/coordinator communication | 1.1 s | ~5.5% of total |

The percentages overlap slightly because of measurement boundaries.

## Effect of increasing CPU threads

Launching a second service instance on another 310P die with:

```bash
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
```

gave:

| Configuration | Warm HTTP time, 51.6 min | Stage time | CPU preprocess |
| --- | ---: | ---: | ---: |
| `OMP_NUM_THREADS=1` | 19.93 s | 18.83 s | 8.71 s |
| `OMP_NUM_THREADS=16` | **14.48 s** | 13.38 s | 1.69 s |

This is a **1.38× end-to-end improvement** and reduces the bottleneck to:

- NPU compute: ~10–11 s, now dominant
- CPU preprocessing: 1.69 s
- communication: ~1.1 s
- per-chunk CPU dispatch/D2H/misc: remainder

## Skip-mask optimization result

The attention implementation originally always ran:

```python
scores = scores.masked_fill(~mask, min_value)
```

For a full window, `valid_length == frame_count` and the mask is all true, so
this operation is redundant.  The implementation now passes `mask=None` in
that case and retains the masked path only for partial windows.

Measured on the same 8 HK-DS3 meetings:

- Generated diarization segments are **bit-for-bit identical** to the old
  service outputs on all 8 files.
- Therefore DER is **exactly unchanged**:
  - aggregate NPU DER collar=0: 34.8009%
  - aggregate NPU DER collar=0.25: 30.2840%
- 51.6-minute request: 14.84 s → **11.87 s** (1.25× faster, 260.8× realtime).
- 11-second warm request: 62.81 ms → **60.15 ms**.
- Live 250 ms / 250 ms: stable concurrency increased from 12 to **14**
  streams per die (15 starts failing).

## Conclusion

- **Communication is not the bottleneck**: about 5–7%.
- **Current default service is both CPU- and NPU-bound**:
  - CPU full-file STFT preprocessing is single-threaded and costs ~8.7 s;
  - NPU forward costs ~10–11 s, dominated by attention mask/softmax/cast/layout kernels.
- **After setting `OMP_NUM_THREADS=16`**, CPU preprocessing drops 5.1× and NPU
  compute becomes the clear dominant bottleneck.
- Best next optimization targets:
  1. Keep `OMP_NUM_THREADS`/`MKL_NUM_THREADS` at 8–16 for the CPU STFT stage.
  2. Optimize/fuse the NPU attention path (`MaskedFill`, `Softmax`, casts,
     layout transforms).
  3. Communication is lower priority.

## Reproduce

```bash
# Request-level timeline
curl -X POST http://127.0.0.1:18086/start_request_profile \
  -H 'Content-Type: application/json' \
  -d '{"run_id":"diar-profile","event_dir":"/tmp/diar_profile/events"}'
# send audio
curl -X POST http://127.0.0.1:18086/stop_request_profile \
  -H 'Content-Type: application/json' -d '{"run_id":"diar-profile"}'
.venv/bin/python -m sglang_omni.profiler /tmp/diar_profile/events --format table

# CPU/NPU phase timing
.venv/bin/python benchmarks/npu/profile_diarization_phases.py \
  /path/to/meeting.wav

# Optional torch/NPU trace
curl -X POST http://127.0.0.1:18086/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"run_id":"diar-torch","trace_path_template":"/tmp/diar_torch/trace","event_dir":"/tmp/diar_torch/events","enable_torch":true}'
```
