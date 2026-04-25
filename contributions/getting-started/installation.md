# Installation

Environment setup for Mac, Linux, and Windows.

## Requirements

- Python 3.10, 3.11, or 3.12 (3.11 recommended)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or `pip` + `venv`
- CUDA 12.1+ for GPU runs (Linux only)
- A HuggingFace account with access to gated models (see below)

## Step 1 — Clone the repo

```bash
git clone https://github.com/PeterLi-jpg/bohdi-lora.git
cd bohdi-lora
```

## Step 2 — Create a virtual environment

### Conda (recommended)

```bash
conda create -n bohdi python=3.11 -y
conda activate bohdi
```

### pip + venv (alternative)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac / Linux
.venv\Scripts\activate           # Windows (Command Prompt)
```

## Step 3 — Install dependencies

```bash
bash setup.sh
```

`setup.sh` now:
- checks that a virtualenv or conda env is active
- prints the active Python interpreter
- runs `python -m pip install -r requirements.txt`
- runs `python -m pip check`
- creates the working directories (`logs/`, `data/`, `eval/`, `checkpoints/`)
- downloads the HealthBench files into `data/raw/`

If you run `setup.sh` in system Python or Conda `base`, it will print a loud warning first and continue. That path can still work, but it is the most common reason for unclear dependency conflicts. The recommended path is still a fresh env created just for this repo. Expected macOS message: `Ignoring autoawq` and `Ignoring bitsandbytes` because both are Linux/CUDA-only.

## Step 4 — Set your HuggingFace token

All base models are gated. Accept the terms on each model card while logged in to HuggingFace, then:

```bash
export HF_TOKEN=hf_...
```

Or add it to a local `.env` file (gitignored):

```
HF_TOKEN=hf_...
```

Required model access:

| Model | Purpose |
|---|---|
| [google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it) | Base model (full run) |
| [google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it) | Smoke test |
| [Qwen/Qwen2.5-14B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ) | Grader (full run) |

## Windows

Use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu and follow the Linux steps above. Native Windows is not tested.

## Verify installation

```bash
python scripts/preflight.py
```

This checks HF token validity, model access, GPU availability, and all required packages in under 30 seconds.
