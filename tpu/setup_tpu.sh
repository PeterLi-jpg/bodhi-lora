#!/bin/bash
# setup_tpu.sh — install Python deps on a Cloud TPU VM.
#
# TPU VMs come with torch + torch_xla pre-installed and tightly coupled to the
# TPU runtime. We install everything else from requirements.txt but skip
# torch itself (re-installing would break the xla linkage) and skip the
# CUDA-only packages (bitsandbytes, autoawq) that don't exist on TPU.
#
# Usage: run once after SSH-ing into the TPU VM, before launching training.

set -euo pipefail

echo "=== Python / torch_xla version ==="
python3 -c "import torch; import torch_xla; print('torch:', torch.__version__, '| xla:', torch_xla.__version__)"

echo "=== Installing deps (skipping torch / CUDA-only packages) ==="
pip install --quiet \
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
    "numpy>=1.24,<3.0" \
    "pandas>=2.0,<3.0" \
    "tqdm>=4.65" \
    "matplotlib>=3.7,<4.0"

echo "=== Verifying peft + trl import ==="
python3 -c "from peft import LoraConfig; from trl import SFTTrainer; print('peft + trl OK')"

echo "=== setup_tpu.sh done ==="
