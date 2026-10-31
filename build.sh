
uv venv --python=3.12
source .venv/bin/activate

uv pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install torch-npu==2.10.0.post6 --extra-index-url https://ascend.devcloud.huaweicloud.com/pypi/simple/
uv pip install numpy pyyaml decorator attrs psutil scipy

source /usr/local/Ascend/ascend-toolkit/set_env.sh
python - <<'PY'
import torch
import torch_npu
print(torch.__version__)
print(torch_npu.__version__)
print(torch.npu.is_available())
print(torch.npu.device_count())
PY

source /usr/local/Ascend/ascend-toolkit/set_env.sh

python - <<'PY'
import torch
import torch_npu

print(torch.npu.get_device_name(0))

x = torch.randn(2, 3, device="npu")
y = x @ x.T
torch.npu.synchronize()
print(y.cpu())
print("NPU smoke test passed")
PY

uv pip install -r requirements-npu-diarization.txt

uv pip install --no-deps "sglang==0.5.22"
uv pip install --no-deps -e .
uv pip install orjson "triton==3.5.0" torchaudio==2.10


python - <<'PY'
import torch
import torch_npu
import transformers
import sglang
import sglang_omni

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu:", torch.npu.is_available(), torch.npu.device_count())
print("transformers:", transformers.__version__)
print("sglang:", sglang.__version__)
print("sglang_omni:", sglang_omni.__version__)
PY


python examples/npu/nemotron_diarization_direct.py \
  /root/workspace/code/zzh/code/DongniEval/ASR/Dongni-ASR-0_6B/ASCEND/300I-DUO/data/Deepwise_ASR_Benchmark_V1/audio/1000066_70887a6e.wav \
  --model-path /root/workspace/models/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo \
  --device npu:0



# 启动服务
MODEL_PATH=/root/workspace/models/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo \
PORT=18080 \
bash scripts/npu/serve_nemotron_diarization_npu.sh

# 压测
python benchmarks/npu/live_250ms_concurrency.py \
  /root/workspace/code/zzh/code/v0.2.2-npu/ASR-stream-SPK-PY/data/2026-09-07-project-meeting.wav \
  --url ws://127.0.0.1:18080/v1/audio/diarizations/stream \
  --streams 8 \
  --seconds 60
