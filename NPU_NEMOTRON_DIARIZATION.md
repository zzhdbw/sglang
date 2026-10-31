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
5. Added NPU config, install/run scripts, and verification clients.

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
is slower.  Subsequent requests are much faster; after warmup the upload
endpoint returned an 11-second recording in about 0.08–0.09 seconds on the
validated machine.

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

Validated result: `46 passed`.

## Known limitations

- This is not an NVIDIA-supported CUDA configuration.  The CUDA path still
  exists but was not performance-tuned here; the fork favors correctness on
  Ascend 310P.
- Only one NPU is used for a diarization server by default.  Multiple server
  processes can be bound to different devices for parallelism.
- The STFT feature extractor runs on the host CPU because complex STFT is not
  available on 310P.  The ~100M parameter encoder/head stays on NPU.
- FP32 is the validated precision.  Do not expect BF16 support on 310P.
- No speaker embedding/identification is performed; the model only returns
  anonymous speaker turns.
