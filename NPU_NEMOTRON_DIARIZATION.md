# SGLang-Omni on Ascend 300I Duo / 310P3

This repository is a fork of SGLang-Omni v0.2.0 that ports the native
[Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
pipeline to Ascend NPU.  It was validated end-to-end on 4× Atlas 300I Duo
cards (8× Ascend 310P3 dies) with the `vllm-ascend:v0.23.0-310p-openeuler`
container.

## What changed

Upstream SGLang-Omni already implements the model and its HTTP/WebSocket API,
but its diarization path was CUDA-only.  This fork changes:

1. `models/nemotron_diarization/backend.py`
   - accepts `device.type == "npu"`;
   - selects `torch.npu.set_compile_mode(jit_compile=False)` before model
     loading.  Without this, streaming windows with changing prefix lengths
     can trigger multi-second to multi-minute operator recompilations.
2. `models/nemotron_diarization/model.py`
   - replaces the CUDA Triton RoPE kernel with an equivalent eager PyTorch
     half-split rotation;
   - replaces `flex_attention` with dense eager attention and a boolean
     padding mask;
   - runs the small STFT/mel extractor on CPU and moves the real-valued
     features to NPU, because torch_npu on 310P does not support complex
     STFT output;
   - uses generic device streams instead of hard-coded `torch.cuda.*` where
     needed.
3. `models/nemotron_diarization/stages.py`
   - uses `torch.get_device_module(device)` for NPU/CUDA streams and events.
4. `models/nemotron_diarization/streaming.py`
   - keeps cache tensors alive on the active accelerator stream generically.
5. `models/nemotron_diarization/speaker_cache.py`
   - replaces the three-dimensional advanced-index assignment in speaker-cache
     compression, which fails in the 310P AI CPU `Index`/`ScatterElements`
     kernels, with an equivalent broadcast-mask + `torch.where` selection;
   - this makes recordings longer than one offline window work reliably.
6. Added NPU config, install/run scripts, and verification clients.

## Validated environment

- Hardware: 8× Ascend 310P3 (4× Atlas 300I Duo), 43.24 GiB per die
- Container: `quay.nju.edu.cn/ascend/vllm-ascend:v0.23.0-310p-openeuler`
- CANN: 9.1.0
- Python: 3.12
- torch: 2.10.0
- torch_npu: 2.10.0.post4
- transformers: 5.19.0 (isolated venv)
- SGLang: 0.5.22 (imported only; the diarization stage does not use an LLM
  engine)
- Model precision: FP32

## Install

Run inside a container that already has working torch_npu/NPU devices:

```bash
cd /workspace/code/zzh/code/sglang/sglang-omni-npu
bash scripts/npu/install_nemotron_diarization_npu.sh
```

The installer creates `.venv` with `--system-site-packages`, so the vendor
`torch`/`torch_npu` pair is reused and no CUDA wheel replaces it.

Verify the NPU:

```bash
.venv/bin/python - <<'PY'
import torch, torch_npu
print(torch.__version__, torch_npu.__version__)
print(torch.npu.is_available(), torch.npu.device_count())
PY
```

## Direct model smoke test

```bash
.venv/bin/python examples/npu/nemotron_diarization_direct.py \
  /tmp/jfk.wav \
  --model-path /workspace/models/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo \
  --device npu:0
```

Expected shape of the API result:

```json
{
  "duration": 11.0,
  "segments": [
    {"start": 0.28, "end": 2.28, "speaker": "speaker_0"},
    {"start": 3.27, "end": 4.57, "speaker": "speaker_0"},
    {"start": 5.36, "end": 10.64, "speaker": "speaker_0"}
  ]
}
```

The first inference in a new process performs CANN operator initialization and
is slower.  Subsequent requests are much faster; see the benchmark section.

## Start the server

```bash
MODEL_PATH=/workspace/models/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo \
PORT=18080 \
bash scripts/npu/serve_nemotron_diarization_npu.sh
```

Check readiness:

```bash
curl http://127.0.0.1:18080/health
```

Uploaded diarization:

```bash
curl -X POST http://127.0.0.1:18080/v1/audio/diarizations \
  -F file=@/tmp/jfk.wav
```

Python client:

```bash
.venv/bin/python examples/npu/nemotron_diarization_http.py /tmp/jfk.wav \
  --url http://127.0.0.1:18080/v1/audio/diarizations
```

Live WebSocket replay:

```bash
.venv/bin/python examples/npu/nemotron_diarization_stream.py /tmp/jfk.wav \
  --url ws://127.0.0.1:18080/v1/audio/diarizations/stream
```

Both endpoints were validated with an 11-second mono 16 kHz PCM16 WAV.

## Tests

```bash
.venv/bin/python -m pytest \
  tests/unit_test/nemotron_diarization \
  tests/unit_test/serve/test_diarizations.py -q
```

Validated result: `47 passed`.

## Throughput benchmark

Measured on one Ascend 310P3 with the warmed `offline` profile through
`POST /v1/audio/diarizations`.  Audio was 16 kHz mono PCM16.

| Audio duration | Requests | Mean latency | RTF | Single-stream throughput |
| --- | ---: | ---: | ---: | ---: |
| 11.0 s | 29 warm requests | 66.85 ms | 0.0061 | 164.6× real time, ~898 req/min |
| 111.8 s | 5 requests | 692.1 ms | 0.0062 | 161.5× real time, ~86.7 req/min |

The first request after server startup was slower (about 1.7 s for the 11 s
file) due to one-time CANN operator initialization.

Concurrent HTTP benchmark with 111.8 s recordings:

| Concurrent clients | Wall time | Aggregate throughput |
| ---: | ---: | ---: |
| 2 | 1.35 s | 165.7× real time |
| 4 | 2.66 s | 167.8× real time |
| 8 | 5.31 s | 168.6× real time |

The default diarization factory has `max_concurrency=1`, so concurrent clients
queue and latency grows roughly linearly.  Aggregate audio throughput stays
near 166–169× real time.  Increase `--diarization.factory.max_concurrency` only
after validating NPU stream concurrency on the target CANN version.

NPU memory used by the diarization stage during the benchmark was about
717 MiB for the worker plus about 100 MiB for the launcher process.

Benchmark caveat: the repeated-speaker JFK sample measures compute throughput,
not diarization quality.  DER/accuracy should be evaluated on representative
multi-speaker recordings.

## Profiling and OMP tuning

The full-file CPU STFT/mel preprocessor is single-threaded by default because
the worker inherits `OMP_NUM_THREADS=1`.  On one 310P, a 51.6-minute meeting
request dropped from **19.93 s** to **14.48 s** after launching the service
with:

```bash
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
```

CPU preprocessing dropped from 8.71 s to 1.69 s.  NPU compute then becomes the
dominant cost.  Communication/coordinator overhead was only about 5.5% of the
end-to-end time.  See
[`benchmarks/npu/PROFILE_NEMOTRON_DIARIZATION_NPU.md`](./benchmarks/npu/PROFILE_NEMOTRON_DIARIZATION_NPU.md)
for the full profile.

### Skip-mask attention optimization

Full-window attention no longer materialises the all-true padding mask or runs
`MaskedFill`; the mask path remains only for padded/partial windows.  On the
same 8-file HK-DS3 evaluation set, the generated diarization segments were
**exactly identical** to the previous service on every file, so DER did not
change.

Throughput changes on one tuned 310P3 (`OMP_NUM_THREADS=16`):

| Workload | Before skip-mask | After skip-mask | Improvement |
| --- | ---: | ---: | ---: |
| 11.0 s upload, warm mean | 62.81 ms | **60.15 ms** | 1.04× |
| 51.6 min upload, mean | 14.84 s | **11.87 s** | **1.25×** |
| 51.6 min realtime factor | 208.7× | **260.8×** | +25% |
| 8-file total, default OMP=1 vs fully optimized | 87.26 s | **48.67 s** | 1.79× combined |

Live 250 ms / 250 ms, 60-second 310P3 test:

| Concurrent streams | Result |
| ---: | --- |
| 14 | stable, no errors, p95 ACK lag 0.90 s |
| 15 | 4 sessions failed (queue full / connection closed) |

The practical live ceiling improved from about **12** to about **14** streams
per 310P3 die.

### Precision sweep

Direct model throughput and aggregate DER over the 8-meeting set:

| Mode | Realtime factor | DER0 | Δ DER0 | DER25 | Δ DER25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 309.3× | 34.801% | — | 30.284% | — |
| FP16 whole model | 441.8× | 37.732% | +2.93 pp | 33.472% | +3.19 pp |
| Mixed attention/MLP FP16 | 355.1× | 35.075% | +0.27 pp | 30.693% | +0.41 pp |
| torch_npu AMP | 377.6× | 35.765% | +0.96 pp | 31.341% | +1.06 pp |

Manual mixed attention/MLP FP16 is the best aggregate speed/accuracy
trade-off, but it regresses `AIPC` by 3.37 pp, so the production service
remains FP32 + skip-mask.  See
[`benchmarks/npu/PRECISION_SWEEP_NEMOTRON_DIARIZATION.md`](./benchmarks/npu/PRECISION_SWEEP_NEMOTRON_DIARIZATION.md).

### Throughput after OMP tuning

Measured on the tuned instance with `OMP_NUM_THREADS=16` and
`MKL_NUM_THREADS=16`:

| Workload | Default OMP=1 | Tuned OMP=16 | Improvement |
| --- | ---: | ---: | ---: |
| 11.0 s upload, warm mean | 66.85 ms | 62.81 ms | 1.06× |
| 51.6 min upload, mean | 19.93 s | **14.84 s** | **1.34×** |
| 51.6 min upload, realtime factor | 155.4× | **208.7×** | +34% |

Live 250 ms / 250 ms concurrency for 60 s:

| Setting | Stable streams per 310P3 | First failing point |
| --- | ---: | ---: |
| Default OMP=1 | 10 | 11+ |
| Tuned OMP=16 | **12** | 13+ |

For 1000 ms / 1000 ms live chunks on the tuned instance, 12 streams completed
60 s without errors (wall 61.36 s, p95 ACK lag 1.36 s).  14 streams completed
but could not keep up (wall 63.96 s, p95 lag 3.35 s).  The practical live
ceiling is therefore about 12 streams per die; 10 streams is the recommended
operating point with latency headroom.

## Live concurrency: 250 ms uploaded every 250 ms

This is the more relevant capacity metric for realtime audio.  Each WebSocket
stream sends one 250 ms PCM16 message every 250 ms, i.e. one full realtime
audio stream per connection.  Tests used one 310P3, `max_concurrency=1`,
`max_live_sessions=64`, and ran for 60 seconds per configuration.

| Concurrent streams | Errors | p95 ack lag | Max ack lag | Result |
| ---: | ---: | ---: | ---: | --- |
| 8 | 0 / 8 | 0.51 s | 0.84 s | stable, recommended |
| 10 | 0 / 10 | 1.00 s | 1.61 s | ceiling, still no drops |
| 11 | 4 / 11 | 1.95 s | 2.24 s | client sessions dropped |
| 12 | 4 / 12 | 1.83 s | 2.23 s | client sessions dropped |

Interpretation for one 310P3 die:

- **Recommended: 8 concurrent realtime streams**
- **Measured ceiling: about 10–11 concurrent realtime streams**
- At 11 streams the run may complete but wall time exceeds the 60 s input
  duration (for example 62–64 s) and p95 ACK lag grows to 2–3.5 s, so it is no
  longer clean realtime.  The fully sustainable point is therefore about 10.
- Chunk size 250/500/1000 ms barely changes this concurrency ceiling because
  the internal 720 ms diarization windows and per-request scheduler dominate.
  Chunk size mainly changes message overhead and latency granularity.
- Above 10, the per-session 8-message (2 s) audio queue starts filling and the
  server eventually closes sessions.
- Raising `max_concurrency` to 4 did not help and made results worse on this
  CANN/torch_npu version; keep `max_concurrency=1` unless separately tuned.

One Atlas 300I Duo card exposes **two** 310P3 dies, so a card-level estimate
with two independent server processes is:

- **Recommended: about 2 × 8 = 16 concurrent realtime streams per card**
- **Measured ceiling: about 2 × 10 = 20 concurrent realtime streams per card**

The full 300I Duo host exposes 8 × 310P3 dies.  With one server process pinned
to each die, the first-order host capacity is therefore:

- **Recommended: about 8 × 8 = 64 concurrent realtime streams**
- **Measured ceiling: about 8 × 10 = 80 concurrent realtime streams**

This assumes all dies are free and each server process is independently
configured with enough `max_live_sessions` and admission slots.  The default
factory value is only `max_live_sessions=8`, so a single default server exposes
8 live sessions.

Run the benchmark with:

```bash
.venv/bin/python benchmarks/npu/live_250ms_concurrency.py /tmp/jfk_10x.wav \
  --url ws://127.0.0.1:18086/v1/audio/diarizations/stream \
  --streams 8 --seconds 60
```

## Best upload chunk size and interval

The WebSocket endpoint accepts at most `32000` bytes per audio message, i.e.
1000 ms of 16 kHz PCM16.

### Single-stream pipeline throughput

Closed-loop means: send one chunk, wait for `audio.ack`, then immediately send
the next chunk.  This removes the offered-rate limit and measures the pipeline:

| Chunk size | 60 s audio wall time | Throughput |
| ---: | ---: | ---: |
| 250 ms | 5.99–6.24 s | 9.6–10.0× real time |
| 500 ms | 5.54–5.84 s | 10.3–10.8× real time |
| 1000 ms | 5.25–5.58 s | 10.8–11.4× real time |

**1000 ms chunks are the best choice for raw throughput.**  They reduce the
number of scheduler/WebSocket round trips by 4× versus 250 ms chunks.

### Fixed send interval

At an offered load of about 10× real time:

| Chunk / interval | Offered load | ACK lag max | Result |
| ---: | ---: | ---: | --- |
| 1000 ms / 100 ms | 10× | 0.24 s | stable |
| 500 ms / 50 ms | 10× | 0.20 s | stable |
| 250 ms / 25 ms | 10× | — | queue overflow, session closed |

For realtime audio, use:

```text
chunk size = 1000 ms
interval   = 1000 ms
```

This minimizes per-message overhead while preserving realtime pacing.
The observed one-die capacity is still about **10 concurrent realtime
streams** (recommended 8 with latency headroom), because the diarization
pipeline's internal 720 ms windows and request scheduler dominate.

If the client is allowed to ingest faster than realtime, use **ACK
backpressure** instead of a fixed interval: send the next 1000 ms chunk only
after the previous `audio.ack`.  On one 310P3 this reaches about **11.3× real
time** for a single stream.

Run the chunk/interval sweep with:

```bash
.venv/bin/python benchmarks/npu/live_chunk_interval.py /tmp/jfk_10x.wav \
  --url ws://127.0.0.1:18086/v1/audio/diarizations/stream \
  --streams 1 --chunk-ms 1000 --interval-ms 0 \
  --audio-seconds-per-stream 60
```

## Known limitations

- This is not an NVIDIA-supported CUDA configuration.  The CUDA path still
  exists but was not performance-tuned here; the fork favors correctness on
  Ascend 310P.
- Only one NPU is used for a diarization server process.  Multiple server
  processes can be bound to different devices for parallelism.  The built-in
  diarization factory defaults to `max_concurrency=1`.
- The STFT feature extractor runs on the host CPU because complex STFT is not
  available on 310P.  The ~100M parameter encoder/head stays on NPU.
- FP32 is the validated precision.  Do not expect BF16 support on 310P.
- The throughput numbers above use repeated single-speaker audio; multi-speaker
  audio with more speaker-turn post-processing may differ slightly.
- The live-concurrency ceiling is dominated by per-250 ms request scheduling and
  queueing on the current single-worker diarization factory, not by raw model
  FLOPs.  Larger deployments should use one process per NPU die.
- No speaker embedding/identification is performed; the model only returns
  anonymous speaker turns.
