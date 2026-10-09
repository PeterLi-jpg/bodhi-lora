#!/bin/bash
# Run this once after cloning, inside the training virtualenv (.venv). It also builds
# the vLLM env (.venv-infer) on Linux and downloads HealthBench to the data root.
set -euo pipefail

cd "$(dirname "$0")"

USING_BASE_CONDA=0
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" = "base" ]; then
    USING_BASE_CONDA=1
fi

if [ -z "${VIRTUAL_ENV:-}" ] && { [ -z "${CONDA_PREFIX:-}" ] || [ "$USING_BASE_CONDA" -eq 1 ]; }; then
    echo "WARNING: installing into a shared/base Python environment."
    if [ "$USING_BASE_CONDA" -eq 1 ]; then
        echo "Conda 'base' is active, which commonly causes vague pip conflicts across projects."
    else
        echo "No virtualenv or project-specific conda env is active."
    fi
    echo
    echo "This does not always break the repo, but it is the main source of unclear install errors."
    echo "Recommended clean setup:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  bash setup.sh"
    echo
    echo "Continuing with the current interpreter in 3 seconds..."
    sleep 3
    echo
fi

echo "=== setup ==="
echo "python: $(command -v python)"
python --version

mkdir -p data/raw

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# vLLM serving needs its own env (requirements-infer.txt explains why). vLLM
# wheels are Linux and CUDA only, so a Mac or CPU dev box skips this step.
if [ "$(uname -s)" = "Linux" ] && [ "${SKIP_INFER_ENV:-0}" != "1" ]; then
    echo
    echo "=== .venv-infer (requirements-infer.txt) ==="
    python -m venv .venv-infer
    .venv-infer/bin/python -m pip install --upgrade pip
    .venv-infer/bin/python -m pip install -r requirements-infer.txt
else
    echo "skipping .venv-infer (not Linux, or SKIP_INFER_ENV=1); vLLM stages will not run here"
fi

echo
echo "=== pip check ==="
if ! python -m pip check; then
    echo
    echo "Dependency conflicts were detected in the active environment."
    echo "This repo installs cleanly in a fresh virtualenv. The usual cause is"
    echo "a shared/base environment that already has unrelated packages installed."
    echo
    echo "Recommended fix:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  bash setup.sh"
    exit 1
fi

echo
echo "=== data ==="
python -m bodhi_lora fetch healthbench

echo
echo "ready"
