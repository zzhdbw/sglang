#!/usr/bin/env bash
# Serve Nemotron-3-Diarization with SGLang-Omni on Ascend NPU.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-${REPO_ROOT}/.venv}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18080}"
PROFILE="${PROFILE:-offline}"
EXTRA_ARGS=("$@")

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "ERROR: model checkpoint not found: ${MODEL_PATH}" >&2
  echo "Set MODEL_PATH=/path/to/Nemotron-3-Diarization.nemo" >&2
  exit 1
fi
if [[ ! -x "${VENV}/bin/sgl-omni" ]]; then
  echo "ERROR: ${VENV}/bin/sgl-omni not found; run scripts/npu/install_nemotron_diarization_npu.sh" >&2
  exit 1
fi

CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "${CONFIG}"' EXIT
cat >"${CONFIG}" <<YAML
config_cls: NemotronDiarizationPipelineConfig
model_path: ${MODEL_PATH}
YAML

echo "Serving ${MODEL_PATH} on ${HOST}:${PORT} (profile=${PROFILE})"
"${VENV}/bin/sgl-omni" serve \
  --config "${CONFIG}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --skip-server-warmup \
  --diarization.factory.profile "${PROFILE}" \
  "${EXTRA_ARGS[@]}"
