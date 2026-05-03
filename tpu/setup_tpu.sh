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

# ── torch_xla install gate ──────────────────────────────────────────────────
# torch_xla 2.7 (~3 GB wheel + 5-8 min install) is only used by the legacy
# launchers — launch_5seeds.sh, launch_v6e.sh, launch_v4_ondemand.sh,
# launch_all_seeds.sh, launch_multiseed.sh — which call scripts/train_lora.py.
# The MaxText path (launch_5seeds_maxtext.sh -> train_lora_maxtext.py) is
# pure JAX; Stages 1/2/4 run inside the vLLM-TPU Docker container with its
# own torch internally; Stage 3a (HF -> Orbax converter) and Stage 3b (LoRA
# train) never import torch_xla. We also avoid a libtpu version race where
# torch_xla's libtpu pin and jax[tpu]'s libtpu pin try to coexist.
#
# Default OFF. Legacy launchers export BOHDI_INSTALL_TORCH_XLA=1 before
# calling this script; the MaxText launcher leaves it unset.
INSTALL_TORCH_XLA="${BOHDI_INSTALL_TORCH_XLA:-0}"
echo "=== torch_xla install: $([ "$INSTALL_TORCH_XLA" = "1" ] && echo "YES (legacy path)" || echo "skipped (MaxText path)") ==="

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

# torch_xla 2.7 (legacy path only — see INSTALL_TORCH_XLA gate above):
#   ships C++11 ABI wheels (~20% goodput on tracing-bound jobs) and includes
#   the fix for the v6e fusion-emitter regression in 2.5 (#8591) that hangs
#   Gemma-3 SPMD compile for hours. Manual mark_sharding on Gemma-3 27B was
#   hanging at "0/70 steps" for 30+ min with cache stagnant — the blessed
#   path on v6e is FSDPv2 (xla_fsdp_v2: True), wired in train_lora.py via
#   optimum-tpu's use_fsdp_v2().
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

if [ "$INSTALL_TORCH_XLA" = "1" ]; then
    echo "=== Installing torch ${TORCH_VERSION} + torch_xla ${TORCH_XLA_VERSION} from TPU wheel server (legacy path) ==="
    ${PIP} install ${PIP_FLAGS} \
        "torch==${TORCH_VERSION}" \
        "torch_xla[tpu]==${TORCH_XLA_VERSION}" \
        -f "${TPU_WHEEL_URL}"

    echo "=== Verifying torch_xla import ==="
    ${PY} -c "import torch; import torch_xla; print('torch:', torch.__version__, '| xla:', torch_xla.__version__)"
fi

echo "=== Installing remaining deps ==="
# When INSTALL_TORCH_XLA=1, torch was already installed above from the TPU
# wheel server and the explicit pin below prevents transformers/trl/peft from
# silently downgrading it. When INSTALL_TORCH_XLA=0 (MaxText path), torch
# comes transitively from transformers — Stage 1 (generate_traces.py) imports
# torch directly and Stage 3a (HF -> Orbax converter) loads HF state dicts;
# both work fine with the PyPI CPU build.
#
# NOTE: ML libraries are pinned to EXACT versions (==).  Reason: this codebase
# has already worked around shape bugs in transformers DynamicLayer, API drift
# in trl SFTTrainer, and accelerate's TPU-mode model placement (see
# train_lora.py SPMD setup).  An upstream patch release between runs can break
# any of those — pinning prevents silent regressions on a 24-hour pipeline.
_TORCH_PIN_ARGS=()
if [ "$INSTALL_TORCH_XLA" = "1" ]; then
    _TORCH_PIN_ARGS=("torch==${TORCH_VERSION}" -f "${TPU_WHEEL_URL}")
fi
${PIP} install ${PIP_FLAGS} \
    "${_TORCH_PIN_ARGS[@]}" \
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

# optimum-tpu (legacy path only) provides the FSDPv2 helpers (use_fsdp_v2/
# get_fsdp_training_args) that wire torch_xla's XLA FSDP v2 into HuggingFace
# Trainer. Pure-Python wrappers around torch_xla; useless without it.
# --no-deps keeps it from yanking transformers/torch back to its own pins.
if [ "$INSTALL_TORCH_XLA" = "1" ]; then
    echo "=== Installing optimum-tpu (FSDPv2 helpers, legacy path) ==="
    ${PIP} install ${PIP_FLAGS} --no-deps "optimum-tpu>=0.2.0"
fi

# JAX stack + MaxText runtime deps for third_party/maxtext (vendored).
# Stage 3 is the only stage that uses MaxText, and it runs as its own process,
# so jax and torch_xla don't try to claim TPU chips simultaneously.
#
# Floor versions track MaxText's
#   third_party/maxtext/src/dependencies/requirements/requirements_decoupled_jax_0_7.1.txt
# (the file's name is misleading — it's the latest "decoupled jax" recipe and
# its actual jax pin is the current stable, 0.10.x as of 2026-05). On py3.11
# every floor below resolves cleanly; on py3.10 several do not, which is why
# we provision py3.11 at the top of this script.
#
# v9-v17 each died on a different missing transitive dep — flax.nnx.Pytree,
# tokamax (vendored splash-attention kernels), tiktoken (HF tokenizer fallback),
# sympy (rope helpers in maxtext.layers.engram), tensorflow_text /
# tensorflow_datasets (input pipeline), grain (dataset sharding), tensorstore
# (orbax storage backend), safetensors / sentencepiece (tokenizer loading).
# Pre-flight pip resolution + `import maxtext.*` on a clean py3.11 venv showed
# all of them are reachable from `import maxtext.checkpoint_conversion.to_maxtext`,
# so install them all here. Adding a dep is dramatically cheaper than another
# 70-min smoke crash.
echo "=== Installing JAX stack + MaxText runtime deps ==="
# `jax[tpu]` pulls libtpu from PyPI directly; no -f flag needed (the
# TPU_WHEEL_URL above is torch_xla's libtpu mirror, a separate distribution).
${PIP} install ${PIP_FLAGS} \
    "absl_py>=2.3.1" \
    "jax[tpu]>=0.10.0,<0.11" \
    "jaxlib>=0.10.0" \
    "flax>=0.12.7" \
    "orbax-checkpoint>=0.11.25" \
    "optax>=0.2.6" \
    "chex>=0.1.91" \
    "qwix>=0.1.6" \
    "google-tunix" \
    "pathwaysutils>=0.1.8" \
    "aqtp>=0.9.0" \
    "tokamax>=0.0.12" \
    "ml-collections>=1.1.0" \
    "ml_dtypes>=0.5.3" \
    "etils[epath]>=1.14.0" \
    "jaxtyping>=0.3.9" \
    "psutil>=7.2.2" \
    "google-cloud-storage>=3.10.1" \
    "omegaconf>=2.3.0" \
    "tensorstore>=0.1.76" \
    "grain>=0.2.12" \
    "huggingface_hub>=0.35.3" \
    "tiktoken>=0.12.0" \
    "safetensors>=0.6.2" \
    "sentencepiece>=0.2.1" \
    "sympy>=1.12" \
    "evaluate>=0.4.6" \
    "nltk>=3.9.2" \
    "jsonlines>=4.0.0" \
    "tabulate>=0.9.0" \
    "parameterized>=0.9.0" \
    "tensorflow>=2.19.1" \
    "tensorflow_text>=2.19.0" \
    "tensorflow_datasets>=4.9.9"

echo "=== Final version check ==="
# v9-v17 each died ~70 min into Stage 3a because this check only imported
# a narrow slice of MaxText's transitive deps. We now import every
# external module the production stages 3a/3b/4/5 actually use, plus the
# MaxText sub-packages so any further dep-tree gap fails here, not at
# smoke-time. Repo path is added so `import maxtext...` resolves against
# the vendored third_party/maxtext.
python3.11 --version
${PY} -c "
import sys, pathlib
repo_root = pathlib.Path.home() / 'bohdi-lora'
sys.path.insert(0, str(repo_root / 'third_party' / 'maxtext' / 'src'))

# Main ML stack: torch comes from transformers as a transitive dep on the
# MaxText path; torch_xla only when INSTALL_TORCH_XLA=1 (legacy launchers).
# The shell expands \${INSTALL_TORCH_XLA} into '0' or '1' before Python sees it.
import torch, peft, trl, transformers, accelerate
_install_torch_xla = ${INSTALL_TORCH_XLA} == 1
if _install_torch_xla:
    import torch_xla  # noqa: F401

# JAX stack
import jax, flax, optax, orbax.checkpoint, chex
import jaxtyping  # noqa: F401  (maxtext shape annotations)
import ml_dtypes  # noqa: F401  (maxtext dtype helpers)

# MaxText auxiliary deps surfaced by v9-v17
import omegaconf, etils.epath, ml_collections, psutil
import google.cloud.storage  # noqa: F401  (Stage 3a Orbax ckpt loader)
import pathwaysutils  # noqa: F401  (transitive: maxtext.utils.elastic_utils)
import aqt.jax.v2.aqt_tensor  # noqa: F401  (transitive: maxtext.layers.initializers)
import qwix  # noqa: F401  (transitive: maxtext.layers.quantizations)
import tunix  # noqa: F401  (Stage 3 LoRA SFT trainer)
from tunix.sft import peft_trainer  # noqa: F401
from tunix.models import gemma3  # noqa: F401
import tokamax  # noqa: F401  (transitive: maxtext.layers.attention_op splash kernel)
import tensorstore  # noqa: F401  (transitive: orbax storage backend)
import grain  # noqa: F401  (transitive: maxtext.input_pipeline)
import huggingface_hub  # noqa: F401  (transitive: maxtext.input_pipeline.tokenizer)
import tiktoken  # noqa: F401  (transitive: maxtext.input_pipeline.tokenizer)
import safetensors  # noqa: F401  (transitive: orbax HF converter)
import sentencepiece  # noqa: F401  (transitive: maxtext.input_pipeline.tokenizer)
import sympy  # noqa: F401  (transitive: maxtext.layers.engram rope helpers)
import tensorflow  # noqa: F401  (transitive: maxtext.input_pipeline)
import tensorflow_text  # noqa: F401  (transitive: maxtext.input_pipeline.tokenizer)
import tensorflow_datasets  # noqa: F401  (transitive: maxtext.input_pipeline)

# Stage 3a / 3b: the MaxText pipeline modules that died on missing deps
# in v9-v17. Force-import them here so any further missing transitive
# dep surfaces in setup, not 70 minutes into Stage 3a.
import maxtext.checkpoint_conversion.to_maxtext  # noqa: F401
import maxtext.configs.pyconfig  # noqa: F401
import maxtext.utils.max_utils  # noqa: F401
import maxtext.layers.nnx_wrappers  # noqa: F401  (Pytree integration)
import maxtext.layers.attention_op  # noqa: F401  (tokamax splash kernel)
import maxtext.layers.quantizations  # noqa: F401  (qwix sparsity API)

# Confirm jax / flax / chex / qwix versions match MaxText's runtime
# expectations. Tolerate prerelease suffixes by stripping non-digits.
import re
def _mm(v):
    parts = re.findall(r'\d+', v)
    return tuple(int(x) for x in parts[:3])
assert _mm(jax.__version__) >= (0, 10, 0), f'jax {jax.__version__} < 0.10'
assert _mm(flax.__version__) >= (0, 12, 7), f'flax {flax.__version__} < 0.12.7'
assert _mm(chex.__version__) >= (0, 1, 91), f'chex {chex.__version__} < 0.1.91'

print('torch:', torch.__version__)
if _install_torch_xla:
    print('torch_xla:', torch_xla.__version__)
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('trl:', trl.__version__)
print('accelerate:', accelerate.__version__)
print('jax:', jax.__version__)
print('flax:', flax.__version__)
print('optax:', optax.__version__)
print('orbax-checkpoint:', orbax.checkpoint.__version__)
print('chex:', chex.__version__)
print('qwix:', getattr(qwix, '__version__', '?'))
print('tunix:', tunix.__version__)
print('tokamax:', getattr(tokamax, '__version__', '?'))
print('omegaconf:', omegaconf.__version__)
print('etils:', etils.__version__)
print('ml_collections:', ml_collections.__version__)
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
