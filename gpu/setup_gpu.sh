#!/bin/bash
# setup_gpu.sh — install Python deps on a standalone GPU VM (CUDA 12.1+).
#
# Unlike the TPU setup, we install torch normally from PyPI since CUDA
# is already present. bitsandbytes and autoawq are included for QLoRA support.
#
# Usage: run once after SSHing into your GPU VM, before launching training.

set -euo pipefail

echo "=== GPU / CUDA check ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== Installing deps ==="
pip install --quiet \
    "torch>=2.1.0,<3.0.0" \
    "bodhi-llm[all]==0.1.4" \
    "transformers>=4.50.0,<5.0.0" \
    "timm>=1.0.0,<2.0.0" \
    "pillow>=10.0,<12.0" \
    "pyyaml>=6.0,<7.0" \
    "peft>=0.10.0,<1.0.0" \
    "trl>=0.9.0,<0.12.0" \
    "rich>=13.0,<15.0" \
    "datasets>=2.18.0,<4.0.0" \
    "accelerate>=0.28.0,<2.0.0" \
    "bitsandbytes>=0.43,<0.48" \
    "autoawq>=0.2.0,<0.3.0" \
    "numpy>=1.24,<3.0" \
    "pandas>=2.0,<3.0" \
    "tqdm>=4.65" \
    "matplotlib>=3.7,<4.0"

echo "=== Version check ==="
python3 -c "
import torch, peft, trl, transformers, accelerate, bitsandbytes
print('torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('trl:', trl.__version__)
print('bitsandbytes:', bitsandbytes.__version__)
"

echo "=== setup_gpu.sh done ==="
