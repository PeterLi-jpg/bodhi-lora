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

# ── Mount the data disk (if attached) and redirect HF cache there ────────────
# A 300 GB persistent SSD is attached at TPU create time (see launch script's
# DATA_DISKS map).  It survives preemption and avoids the 100 GB boot-disk
# saturation we hit when medgemma + qwen + vllm-docker all coexist.
#
# First boot of a fresh disk: format ext4.
# Subsequent boots: skip format, just mount.
# If no data disk is attached (fallback zone, on-demand vN-X without the map
# entry, etc.), fall through silently and use the boot disk's HF cache.
DATA_DEV=""
for d in /dev/sdb /dev/nvme0n1 /dev/nvme0n2; do
    if [ -b "$d" ] && ! mount | grep -q " on / "; then  # /dev/sda is /
        # check it's not the root partition's parent
        if ! lsblk -no MOUNTPOINT "$d" 2>/dev/null | grep -q "/$"; then
            DATA_DEV="$d"
            break
        fi
    fi
done

if [ -n "$DATA_DEV" ] && [ ! -d /mnt/cache ] || ! mountpoint -q /mnt/cache 2>/dev/null; then
    if [ -n "$DATA_DEV" ]; then
        echo "=== Mounting data disk $DATA_DEV at /mnt/cache ==="
        # Format only if blank (no existing fs).
        if ! sudo blkid "$DATA_DEV" 2>/dev/null | grep -q TYPE=; then
            echo "  formatting $DATA_DEV (first boot of fresh disk)..."
            sudo mkfs.ext4 -F -E lazy_itable_init=0,lazy_journal_init=0 "$DATA_DEV"
        else
            echo "  $DATA_DEV already has a filesystem, skipping format"
        fi
        sudo mkdir -p /mnt/cache
        sudo mount -o discard,defaults "$DATA_DEV" /mnt/cache 2>/dev/null || true
        sudo chown -R "$(whoami)" /mnt/cache
        df -h /mnt/cache | tail -1
    fi
fi

# Point HuggingFace at a roomy filesystem in priority order:
#   1. /mnt/cache (persistent SSD, if attached) — survives preempt
#   2. /dev/shm  (tmpfs, RAM-backed, ~700GB on v6e-8) — wiped on reboot
#                but big enough for medgemma-27b (54G) + qwen-14b (28G)
#                + orbax checkpoint (54G) all at once
#   3. boot disk default — only ~97GB; saturates at Stage 2 grader load
HF_CACHE_ROOT=""
if mountpoint -q /mnt/cache 2>/dev/null; then
    HF_CACHE_ROOT=/mnt/cache
    echo "HF cache: /mnt/cache (persistent SSD, survives preempt)."
elif [ -d /dev/shm ] && [ "$(df -BG /dev/shm | awk 'NR==2 {gsub(/G/,"",$4); print $4}')" -ge 200 ]; then
    # /dev/shm has enough headroom for the worst case (~140GB peak); use it.
    HF_CACHE_ROOT=/dev/shm
    echo "HF cache: /dev/shm (tmpfs, ~$(df -BG /dev/shm | awk 'NR==2 {print $4}') free, wiped on reboot)."
else
    echo "WARNING: no /mnt/cache and /dev/shm too small; falling back to boot disk."
    echo "  Stage 2 grader (qwen-14b, ~28GB) may ENOSPC if boot disk fills."
fi

if [ -n "$HF_CACHE_ROOT" ]; then
    mkdir -p "${HF_CACHE_ROOT}/hf" "${HF_CACHE_ROOT}/transformers"
    export HF_HOME="${HF_CACHE_ROOT}/hf"
    export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/transformers"
    # Persist for subsequent SSH sessions / nohup'd children.
    {
        echo "export HF_HOME=${HF_CACHE_ROOT}/hf"
        echo "export TRANSFORMERS_CACHE=${HF_CACHE_ROOT}/transformers"
    } | sudo tee /etc/profile.d/bohdi-hf-cache.sh > /dev/null
    sudo chmod +x /etc/profile.d/bohdi-hf-cache.sh
fi

# torch_xla 2.7 ships the C++11 ABI wheels (~20% goodput improvement on
# tracing-bound jobs) and includes scan_layers + the fix for the v6e
# fusion-emitter regression in 2.5 (#8591) that hangs Gemma-3 SPMD compile
# for hours.  Manual mark_sharding on Gemma-3 27B was hanging at "0/70 steps"
# for 30+ min with cache stagnant — the blessed path on v6e is FSDPv2 instead
# (xla_fsdp_v2: True), wired in train_lora.py via optimum-tpu's use_fsdp_v2().
TORCH_VERSION="2.7.0"
TORCH_XLA_VERSION="2.7.0"
TPU_WHEEL_URL="https://storage.googleapis.com/libtpu-releases/index.html"

# Resilience flags — files.pythonhosted.org occasionally throws ReadTimeoutError
# on a TPU VM (the Tokyo region's egress to PyPI CDN can be flaky for minutes
# at a time).  Default pip is 5 retries / 15s connect timeout; bumping both so a
# transient network blip doesn't fail setup, abort the launcher, and force a
# whole-VM rebuild.
PIP_FLAGS="--quiet --retries 10 --timeout 120"

echo "=== Installing torch ${TORCH_VERSION} + torch_xla ${TORCH_XLA_VERSION} from TPU wheel server ==="
pip install ${PIP_FLAGS} \
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
pip install ${PIP_FLAGS} \
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

# optimum-tpu provides the FSDPv2 helpers (use_fsdp_v2/ get_fsdp_training_args)
# that wire torch_xla's XLA FSDP v2 into HuggingFace Trainer.  We install it
# without an upstream pin because the API is stable across recent versions and
# the package is small (pure-Python wrappers around torch_xla).  --no-deps
# keeps it from yanking transformers/torch back to its own pinned versions.
echo "=== Installing optimum-tpu (FSDPv2 helpers) ==="
pip install ${PIP_FLAGS} --no-deps "optimum-tpu>=0.2.0"

# JAX stack for the vendored MaxText baseline (third_party/maxtext).
# Stage 3 is the only stage that uses MaxText, and it runs as its own process,
# so jax and torch_xla don't try to claim TPU chips simultaneously.
#
# Note: third_party/maxtext's pinned tpu-requirements.txt floors are
# jax>=0.9.2 / flax>=0.12.6 / orbax-checkpoint>=0.11.36 / optax>=0.2.8,
# but those reflect an unreleased pre-0.7 JAX series not yet on PyPI
# (PyPI tops out at jax 0.6.x as of 2026-04). We relax the floors here
# to the latest PyPI-available versions that still expose the pjit /
# shard_map / orbax APIs MaxText uses on TPU.
echo "=== Installing JAX stack for MaxText baseline ==="
# `jax[tpu]` pulls libtpu from PyPI directly; no -f flag needed (the
# TPU_WHEEL_URL above is torch_xla's libtpu mirror, a separate distribution).
pip install ${PIP_FLAGS} \
    "jax[tpu]>=0.4.30,<0.7" \
    "flax>=0.10" \
    "orbax-checkpoint>=0.11" \
    "optax>=0.2.4"

echo "=== Pulling vLLM-TPU Docker image ==="
# Inference (Stages 1, 2, 4) runs vLLM inside this container rather than via
# pip install (which installs the CUDA build, not the TPU build).
# Pull here so the first Stage 1 run doesn't stall waiting for a 20 GB download.
sudo docker pull vllm/vllm-tpu:latest

echo "=== Final version check ==="
python3 -c "
import torch, torch_xla, peft, trl, transformers, accelerate
import jax, flax, optax, orbax.checkpoint
print('torch:', torch.__version__)
print('torch_xla:', torch_xla.__version__)
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('trl:', trl.__version__)
print('accelerate:', accelerate.__version__)
print('jax:', jax.__version__)
print('flax:', flax.__version__)
print('optax:', optax.__version__)
print('orbax-checkpoint:', orbax.checkpoint.__version__)
"

echo "=== setup_tpu.sh done ==="
