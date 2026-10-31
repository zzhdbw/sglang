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
- No speaker embedding/identification is performed; the model only returns
  anonymous speaker turns.
