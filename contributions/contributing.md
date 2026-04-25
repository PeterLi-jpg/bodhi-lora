# Contributing to BOHDI-LoRA

This guide covers environment setup on Mac, Linux, and Windows, how to run the pipeline, and how to open a pull request.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Running the Smoke Test](#running-the-smoke-test)
- [Running Tests](#running-tests)
- [Secret Hygiene](#secret-hygiene)
- [Pull Request Process](#pull-request-process)
- [Project Layout](#project-layout)

---

## Environment Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or `pip` + `venv`
- A HuggingFace account with access to [google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it) (accept Google Health AI terms on the model card)
- GPU with ≥ 40 GB VRAM for the full model (A100/H100). For the smoke test, a smaller GPU or CPU is fine.

### Mac / Linux

```bash
git clone https://github.com/PeterLi-jpg/bohdi-lora.git
cd bohdi-lora

# create and activate the conda environment
conda create -n bohdi python=3.11 -y
conda activate bohdi

# install all dependencies
bash setup.sh

# set your HuggingFace token
export HF_TOKEN=hf_...
```

`setup.sh` prints a loud warning if you run it outside an isolated env, and Conda `base` is treated as shared here too because it is a common source of vague dependency conflicts. It also runs `python -m pip check` after install so broken environments fail clearly instead of much later in training.

### Windows

Use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) (Ubuntu) and follow the Mac/Linux steps above. Native Windows is not tested.

Alternatively, use Git Bash with Conda:

```bash
git clone https://github.com/PeterLi-jpg/bohdi-lora.git
cd bohdi-lora
conda create -n bohdi python=3.11 -y
conda activate bohdi
bash setup.sh
export HF_TOKEN=hf_...
```

---

## Running the Smoke Test

The smoke test runs the full pipeline end-to-end on a small model (`gemma-3n-E4B-it`) to catch setup issues before committing cluster time.

```bash
bash smoke.sh
```

Expected runtime: under 10 minutes on a single GPU. If this passes, the full pipeline should work. If you do not have HF access yet and only want a local wiring check, run `SMOKE_MODEL=Qwen/Qwen2.5-0.5B-Instruct bash smoke.sh`.

---

## Running the Full Pipeline

Jobs are submitted as a Slurm dependency chain:

```bash
bash run_all.sh
```

Outputs are archived under `results/<date>_<config>_seed<N>/`. See [reproducibility.md](reproducibility.md) for expected outputs and troubleshooting.

---

## Running Tests

Tests live in `tests/`. They cover pure-Python logic and do not require a GPU.

```bash
# install test dependencies
pip install pytest

# run all tests
pytest tests/ -v
```

Tests that require `matplotlib` are automatically skipped if it is not installed. Pure-logic tests that touch `torch` / `transformers` paths use mocks, so they still run in CI.

To run the full suite against the real ML stack as well, install the complete dependencies first:

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Secret Hygiene

Before opening a PR, run the secret scanner to make sure no tokens are accidentally committed:

```bash
bash scripts/check_no_secrets.sh
```

This scans all tracked files for HuggingFace (`hf_...`), OpenAI (`sk-...`), and AWS (`AKIA...`) token shapes. Keep your `HF_TOKEN` in a local `.env` file (gitignored) and never hard-code it.

---

## Pull Request Process

1. Fork the repo and create a branch: `git checkout -b your-name/short-description`
2. Make your changes, add tests if relevant
3. Run `bash scripts/check_no_secrets.sh` and `pytest tests/ -v`
4. Open a PR against `main` — CI will run automatically (syntax check, shellcheck, YAML validation, pytest)
5. Address any review comments

Keep PRs focused. One issue per PR makes review faster. If you are unsure what to work on, check the [open issues](https://github.com/PeterLi-jpg/bohdi-lora/issues).

---

## Project Layout

```
bohdi-lora/
├── configs/          # Training hyperparameters (full + smoke)
├── data/
│   ├── raw/          # HealthBench eval IDs
│   └── sft/          # Generated and filtered training data
├── eval/             # Evaluation outputs (gitignored)
├── results/          # Per-run archived outputs (gitignored)
├── scripts/          # Generation, filtering, training, and eval scripts
├── slurm/            # SBATCH job scripts for cluster execution
├── tests/            # Pytest test suite
├── smoke.sh          # End-to-end smoke test (<10 min)
├── run_all.sh        # Full pipeline slurm dependency chain
├── setup.sh          # One-time environment setup
├── contributions/    # Docs, setup, and reproducibility guides
├── KNOWN_ISSUES.md
└── requirements.txt
```

---

## Questions

Open a [GitHub Issue](https://github.com/PeterLi-jpg/bohdi-lora/issues) or ping `@PeterLi-jpg` in the group chat.
