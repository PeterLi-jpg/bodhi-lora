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

# ── Provision Python 3.11 in a venv ──────────────────────────────────────────
# v6e TPU VMs run Ubuntu 22.04, where python3.11 is in the standard apt
# universe. The vendored MaxText (third_party/maxtext) was built for py3.11+,
# so its tpu-requirements.txt floors (etils 1.14+, ml-collections 1.1+,
# jaxtyping 0.3.9+, psutil 7.2+, etc.) only resolve cleanly on 3.11. Running
# the rest of this script through ${PIP} / ${PY} keeps every install + import
# check inside the same interpreter.
echo "=== Installing Python 3.11 (Ubuntu 22.04 universe) ==="
sudo apt-get update -qq
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

if [ ! -d ~/.venv-py311 ]; then
    python3.11 -m venv ~/.venv-py311
fi
PY=~/.venv-py311/bin/python
PIP=~/.venv-py311/bin/pip

# Persist venv activation for non-login shells (subsequent SSH sessions and
# nohup'd children that the launchers spawn). Mirrors the bohdi-hf-cache.sh
# pattern below.
{
    echo "export PATH=\$HOME/.venv-py311/bin:\$PATH"
    echo "export VIRTUAL_ENV=\$HOME/.venv-py311"
} | sudo tee /etc/profile.d/bohdi-venv.sh > /dev/null
sudo chmod +x /etc/profile.d/bohdi-venv.sh

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
CANDIDATES="/dev/sdb /dev/nvme0n1 /dev/nvme0n2"
echo "=== Detecting data disk (candidates: ${CANDIDATES}) ==="
for d in $CANDIDATES; do
    if [ ! -b "$d" ]; then
        echo "  ${d}: not present, skipping"
        continue
    fi
    # Skip if this device (or any of its partitions) hosts the root filesystem.
    # The previous check used `mount | grep -q " on / "`, which always matched
    # the root mount line regardless of $d, so the loop never selected anything.
    if lsblk -no MOUNTPOINT "$d" 2>/dev/null | grep -q "^/$"; then
        echo "  ${d}: hosts root filesystem, skipping"
        continue
    fi
    DATA_DEV="$d"
    echo "  ${d}: selected as data device"
    break
done

if [ -n "$DATA_DEV" ]; then
    echo "=== Data disk detection: DATA_DEV=${DATA_DEV} ==="
else
    echo "=== Data disk detection: DATA_DEV=(none, using boot disk + /dev/shm) ==="
fi

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

# v6e VMs ship with an old setuptools that crashes when source-building
# packages with newer metadata (canonicalize_version() got an unexpected
# keyword argument 'strip_trailing_zero'). v11 hit this when omegaconf
# pulled antlr4-python3-runtime as a sdist. Upgrade pip + setuptools +
# wheel before any other pip install so source-builds don't blow up.
echo "=== Upgrading pip / setuptools / wheel ==="
${PIP} install ${PIP_FLAGS} -U pip setuptools wheel

echo "=== Installing torch ${TORCH_VERSION} + torch_xla ${TORCH_XLA_VERSION} from TPU wheel server ==="
${PIP} install ${PIP_FLAGS} \
    "torch==${TORCH_VERSION}" \
    "torch_xla[tpu]==${TORCH_XLA_VERSION}" \
    -f "${TPU_WHEEL_URL}"

echo "=== Verifying torch_xla import ==="
${PY} -c "import torch; import torch_xla; print('torch:', torch.__version__, '| xla:', torch_xla.__version__)"

echo "=== Installing remaining deps ==="
# Pin torch here too so pip does not silently downgrade it when resolving
# transitive requirements from transformers / trl / peft.
#
# NOTE: ML libraries are pinned to EXACT versions (==).  Reason: this codebase
# has already worked around shape bugs in transformers DynamicLayer, API drift
# in trl SFTTrainer, and accelerate's TPU-mode model placement (see
# train_lora.py SPMD setup).  An upstream patch release between runs can break
# any of those — pinning prevents silent regressions on a 24-hour pipeline.
${PIP} install ${PIP_FLAGS} \
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
${PIP} install ${PIP_FLAGS} --no-deps "optimum-tpu>=0.2.0"

# JAX stack for the vendored MaxText baseline (third_party/maxtext).
# Stage 3 is the only stage that uses MaxText, and it runs as its own process,
# so jax and torch_xla don't try to claim TPU chips simultaneously.
#
# py3.11 is provisioned at the top of this script via apt + venv, so
# MaxText's tpu-requirements.txt floors (etils 1.14+, ml-collections 1.1+,
# jaxtyping 0.3.9+, psutil 7.2+, chex 0.1.91+, etc.) apply naturally.
#
# We KEEP the jax[tpu]>=0.4.30,<0.7 pin: MaxText's tpu-requirements.txt
# pins jax>=0.9.2 but that's an unreleased pre-0.7 series not yet on PyPI
# (PyPI tops out at jax 0.6.x as of 2026-04). Bumping jax is a separate
# decision once 0.9 ships.
echo "=== Installing JAX stack for MaxText baseline ==="
# `jax[tpu]` pulls libtpu from PyPI directly; no -f flag needed (the
# TPU_WHEEL_URL above is torch_xla's libtpu mirror, a separate distribution).
#
# Beyond the JAX 4-pack, MaxText's runtime modules import several extra
# packages (omegaconf for config dataclasses, etils for path helpers,
# qwix for LoRA, jaxtyping for shape annotations, psutil for memory
# probes, google-cloud-storage for the Orbax converter, chex for tree
# utilities, ml_collections used by maxtext.configs). v9 caught
# omegaconf as the first missing dep at Stage 3a; install the full set
# here so the converter and trainer don't crash on a fresh v6e VM.
#
# flax>=0.11 is required so flax.nnx.Pytree is native (the shim in
# nnx_wrappers.py becomes a no-op).
${PIP} install ${PIP_FLAGS} \
    "jax[tpu]>=0.4.30,<0.7" \
    "flax>=0.11.0" \
    "orbax-checkpoint>=0.11" \
    "optax>=0.2.4" \
    "omegaconf>=2.3.0" \
    "etils[epath]>=1.14.0" \
    "qwix>=0.1.6" \
    "jaxtyping>=0.3.9" \
    "psutil>=7.2.2" \
    "google-cloud-storage>=3.10.1" \
    "chex>=0.1.91" \
    "ml-collections>=1.1.0" \
    "pathwaysutils>=0.1.8" \
    "aqtp>=0.9.0"

echo "=== Final version check ==="
# v9, v10, v11 each died ~70 min into Stage 3a because this check only
# imported the JAX 4-pack — missing deps for the MaxText converter
# weren't surfaced until Stage 3a actually ran. We now import every
# external module the production stages 3a/3b/4/5 actually use, plus
# the MaxText sub-packages so any further dep-tree gap fails here, not
# at smoke-time. Repo path is added so `import maxtext...` resolves
# against the vendored third_party/maxtext.
python3.11 --version
${PY} -c "
import sys, pathlib
repo_root = pathlib.Path.home() / 'bohdi-lora'
sys.path.insert(0, str(repo_root / 'third_party' / 'maxtext' / 'src'))

import torch, torch_xla, peft, trl, transformers, accelerate
import jax, flax, optax, orbax.checkpoint
import omegaconf, etils.epath, ml_collections, jaxtyping, psutil, chex
import google.cloud.storage  # noqa: F401  (Stage 3a Orbax ckpt loader)
import pathwaysutils  # noqa: F401  (transitive: maxtext.utils.elastic_utils)
import aqt.jax.v2.aqt_tensor  # noqa: F401  (transitive: maxtext.layers.initializers)

# Stage 3a / 3b: the MaxText pipeline modules that died on missing deps
# in v9-v11. Force-import them here so any further missing transitive
# dep surfaces in setup, not 70 minutes into Stage 3a.
import maxtext.checkpoint_conversion.to_maxtext  # noqa: F401
import maxtext.configs.pyconfig  # noqa: F401
import maxtext.utils.max_utils  # noqa: F401

# Confirm flax is at least 0.11 so flax.nnx.Pytree is native (no shim
# needed). Higher is fine; we only fail on regressions below the floor.
# Tolerate prerelease suffixes (e.g. '0.11.0rc1') by stripping non-digits.
import re
_parts = re.findall(r'\d+', flax.__version__)
flax_major_minor = tuple(int(x) for x in _parts[:2])
assert flax_major_minor >= (0, 11), f'flax {flax.__version__} < 0.11'

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
print('omegaconf:', omegaconf.__version__)
print('etils:', etils.__version__)
print('ml_collections:', ml_collections.__version__)
print('chex:', chex.__version__)
print('maxtext: importable')
"

# Pull vLLM-TPU image AFTER the import check so a partial-install (e.g.
# next time MaxText pulls a new dep that fails) doesn't burn
# a 20 GB image pull before the failure surfaces.
echo "=== Pulling vLLM-TPU Docker image ==="
# Inference (Stages 1, 2, 4) runs vLLM inside this container rather than via
# pip install (which installs the CUDA build, not the TPU build).
# Pull here so the first Stage 1 run doesn't stall waiting for a 20 GB download.
sudo docker pull vllm/vllm-tpu:latest

echo "=== setup_tpu.sh done ==="
