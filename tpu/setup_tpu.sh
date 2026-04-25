#!/bin/bash
# setup_tpu.sh — install Python deps on a Cloud TPU VM (tpu-vm-base image).
#
# We use the tpu-vm-base image (plain Ubuntu) and install torch + torch_xla
# from Google's TPU wheel server. This gives us a current torch version (2.5)
# that satisfies requirements.txt and is built against the TPU runtime.
#
# We skip the CUDA-only packages (bitsandbytes, autoawq) — they don't exist
# on TPU and aren't needed (no memory pressure with 1TB+ HBM).
#
# Usage: called automatically by launch_v6e.sh / launch_v4_ondemand.sh

set -euo pipefail

TORCH_VERSION="2.5.0"
TORCH_XLA_VERSION="2.5.0"
TPU_WHEEL_URL="https://storage.googleapis.com/libtpu-releases/index.html"

echo "=== Installing torch ${TORCH_VERSION} + torch_xla ${TORCH_XLA_VERSION} from TPU wheel server ==="
pip install --quiet \
    "torch==${TORCH_VERSION}" \
    "torch_xla[tpu]==${TORCH_XLA_VERSION}" \
    -f "${TPU_WHEEL_URL}"

echo "=== Verifying torch_xla import ==="
python3 -c "import torch; import torch_xla; print('torch:', torch.__version__, '| xla:', torch_xla.__version__)"

echo "=== Installing remaining deps ==="
# Pin torch here too so pip does not silently downgrade it when resolving
# transitive requirements from transformers / trl / peft.
#
# NOTE: ML libraries are pinned to EXACT versions (==).  Reason: this codebase
# has already worked around shape bugs in transformers DynamicLayer, API drift
# in trl SFTTrainer, and accelerate's TPU-mode model placement (see
# train_lora.py SPMD setup).  An upstream patch release between runs can break
# any of those — pinning prevents silent regressions on a 24-hour pipeline.
pip install --quiet \
    "torch==${TORCH_VERSION}" \
    "bodhi-llm[all]==0.1.4" \
    "transformers==4.57.6" \
    "peft==0.19.1" \
    "trl==0.11.4" \
    "accelerate==1.13.0" \
    "datasets>=2.18.0,<4.0.0" \
    "timm>=1.0.0,<2.0.0" \
    "pillow>=10.0,<12.0" \
    "pyyaml>=6.0,<7.0" \
    "jinja2>=3.1.0" \
    "rich>=13.0,<15.0" \
    "numpy>=1.24,<3.0" \
    "pandas>=2.0,<3.0" \
    "tqdm>=4.65" \
    "matplotlib>=3.7,<4.0" \
    -f "${TPU_WHEEL_URL}"

echo "=== Final version check ==="
python3 -c "
import torch, torch_xla, peft, trl, transformers, accelerate
print('torch:', torch.__version__)
print('torch_xla:', torch_xla.__version__)
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('trl:', trl.__version__)
print('accelerate:', accelerate.__version__)
"

echo "=== setup_tpu.sh done ==="
