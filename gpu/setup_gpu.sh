#!/bin/bash
# setup_gpu.sh — install Python deps on a standalone GPU VM (CUDA 12.1+).
#
# Two envs, from the two pinned requirement files at the repo root:
#   the active env  <- requirements.txt        (training: torch, transformers, peft, trl)
#   .venv-infer     <- requirements-infer.txt  (vLLM serving: generate, filter, eval)
# vLLM 0.9.2 fails at import with the transformers that training pins, so the
# two cannot share one env. rebuttal/launch/run_cell.sh uses TRAIN_PY and
# INFER_PY to pick the right one per stage.
#
# Usage: run once from the repo root after SSHing into your GPU VM, inside the
# training venv (python3 -m venv .venv && source .venv/bin/activate).

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== GPU / CUDA check ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== Installing training deps (requirements.txt) ==="
pip install --quiet -r requirements.txt

echo "=== Creating .venv-infer (requirements-infer.txt) ==="
python3 -m venv .venv-infer
.venv-infer/bin/python -m pip install --quiet --upgrade pip
.venv-infer/bin/python -m pip install --quiet -r requirements-infer.txt

echo "=== Version check ==="
python3 -c "
import torch, peft, trl, transformers, accelerate, bitsandbytes
print('train  torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
print('train  transformers:', transformers.__version__, '| peft:', peft.__version__,
      '| trl:', trl.__version__, '| bitsandbytes:', bitsandbytes.__version__)
"
.venv-infer/bin/python -c "
import torch, transformers, vllm
print('infer  vllm:', vllm.__version__, '| torch:', torch.__version__,
      '| transformers:', transformers.__version__)
"

echo "=== setup_gpu.sh done ==="
