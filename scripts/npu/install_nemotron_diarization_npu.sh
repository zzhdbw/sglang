#!/usr/bin/env bash
# Install SGLang-Omni's Nemotron-3-Diarization path into an isolated venv.
#
# The base image/environment must already have a working CANN + torch_npu
# stack.  The venv is created with --system-site-packages so it reuses the
# vendor-provided torch/torch_npu wheels instead of downloading CUDA wheels.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-${REPO_ROOT}/.venv}"

echo "=== SGLang-Omni Nemotron-3-Diarization NPU install ==="
echo "repo:  ${REPO_ROOT}"
echo "venv:  ${VENV}"

"${PYTHON}" - <<'PY'
import torch
import torch_npu
print(f"torch:     {torch.__version__}")
print(f"torch_npu: {torch_npu.__version__}")
if not torch.npu.is_available():
    raise SystemExit("torch.npu.is_available() is false; fix the CANN/driver mount first")
print(f"NPU count: {torch.npu.device_count()}")
PY

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${PYTHON}" -m venv --system-site-packages "${VENV}"
fi

"${VENV}/bin/pip" install --upgrade pip
"${VENV}/bin/pip" install -r "${REPO_ROOT}/requirements-npu-diarization.txt"

# SGLang 0.5.22 is imported by SGLang-Omni.  --no-deps prevents pip from
# replacing the vendor torch/torch_npu stack with a CUDA build.
"${VENV}/bin/pip" install --no-deps "sglang==0.5.22"

# SGLang's utility module imports triton even though this NPU diarization path
# does not execute Triton kernels.  Use the already-installed triton-ascend
# when present; otherwise install a lightweight import-compatible wheel.
if ! "${VENV}/bin/python" -c "import triton" >/dev/null 2>&1; then
  "${VENV}/bin/pip" install "triton==3.5.0"
fi

"${VENV}/bin/pip" install --no-deps -e "${REPO_ROOT}"

cat <<MSG

Installed successfully.

Direct model smoke test:
  ${VENV}/bin/python examples/npu/nemotron_diarization_direct.py /path/to/16k-mono.wav

Start the OpenAI-style SGLang-Omni server:
  MODEL_PATH=/path/to/Nemotron-3-Diarization.nemo \\
    scripts/npu/serve_nemotron_diarization_npu.sh

Then call POST /v1/audio/diarizations and WebSocket
/v1/audio/diarizations/stream.
MSG
