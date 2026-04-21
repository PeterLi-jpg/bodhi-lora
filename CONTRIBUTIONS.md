# CONTRIBUTIONS

Setup, reproducibility, and contribution guide for BOHDI-LoRA. Works on Mac, Linux, and Windows (WSL2).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Running the Smoke Test](#running-the-smoke-test)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Expected Outputs](#expected-outputs)
- [Multi-Seed Variance](#multi-seed-variance)
- [Running Tests](#running-tests)
- [Secret Hygiene](#secret-hygiene)
- [Pull Request Process](#pull-request-process)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware
- **Smoke test**: 1 GPU with ≥ 16 GB VRAM (or CPU, slow). A Mac with ≥ 32 GB unified RAM also works.
- **Full run**: 1 GPU with ≥ 80 GB VRAM for MedGemma-27B (H100 or A100-80G). Multi-GPU not required.

### Software
- Python 3.10, 3.11, or 3.12 (3.11 recommended; 3.12 works but CUDA wheel coverage lags)
- CUDA 12.1+ for the GPU run
- Linux for the full cluster run. macOS is fine for smoke/dev.
- Windows: use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) (Ubuntu) and follow the Linux steps.

### HuggingFace access
All models are gated. Accept the terms on each model page while logged into HF:

| Model | Used for |
|---|---|
| `google/medgemma-27b-text-it` | base model (paper target) |
| `google/gemma-3n-E4B-it` | smoke test / local iteration |
| `Qwen/Qwen2.5-14B-Instruct-AWQ` | grader (full run) |
| `Qwen/Qwen2.5-0.5B-Instruct` | grader (smoke, ungated) |

---

## Environment Setup

```bash
git clone https://github.com/PeterLi-jpg/bohdi-lora.git
cd bohdi-lora

# create and activate a conda environment
conda create -n bohdi python=3.11 -y
conda activate bohdi

# install all dependencies and download data
bash setup.sh

# set your HuggingFace token
export HF_TOKEN=hf_...
```

`setup.sh` creates `logs/ data/raw/ data/sft/ eval/ checkpoints/`, runs `pip install -r requirements.txt`, and downloads HealthBench Hard + Full into `data/raw/`. If `pip install` fails on `autoawq` on macOS, that is expected — a platform marker skips it and the smoke test uses an ungated grader instead.

---

## Running the Smoke Test

Run this every time you change pipeline code. It exercises all four stages end-to-end with `gemma-3n-E4B-it` and catches bugs in minutes rather than after a multi-hour cluster job.

```bash
bash smoke.sh
```

Expected runtime: under 10 minutes on a single GPU. Expected outputs:
- `data/sft/smoke/raw_traces.jsonl` — 3 BOHDI traces
- `data/sft/smoke/{train,val}.jsonl` — graded and split traces
- `checkpoints/best/` — LoRA adapter
- `eval/smoke/lora.json` — eval summary with `mean`, `std`, `brier`, `ece`

Do not submit cluster jobs with a failing smoke test.

### Running smoke on a GCP L4 VM (~20 min, ~$0.25)

```bash
ZONE=us-central1-a && NAME=bohdi-smoke
gcloud compute instances create $NAME \
    --zone=$ZONE --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE --boot-disk-size=100GB \
    --metadata="install-nvidia-driver=True"
gcloud compute ssh $NAME --zone=$ZONE
# inside VM:
git clone https://github.com/PeterLi-jpg/bohdi-lora.git && cd bohdi-lora
export HF_TOKEN=hf_...
bash setup.sh && bash smoke.sh &> logs/smoke.log; echo "exit=$?"
# when done, back on laptop:
gcloud compute instances delete $NAME --zone=$ZONE --quiet
```

---

## Running the Full Pipeline

```bash
export HF_TOKEN=hf_...
bash run_all.sh
```

Submits four Slurm jobs as a dependency chain:

| Job | Script | Est. time | Produces |
|---|---|---|---|
| 1 | `slurm/generate_traces.sh` | ~48 h | `data/sft/raw_traces.jsonl` |
| 2 | `slurm/filter_traces.sh` | ~24 h | `data/sft/{train,val}.jsonl` |
| 3 | `slurm/train_lora.sh` | ~12 h | `checkpoints/best/` |
| 4 | `slurm/eval_lora.sh` | ~12 h | per-config eval JSONs + figures |

Outputs are archived under `results/<date>_<config>_seed<N>/`. Monitor with `squeue -u $USER`.

### Optional variants

- **QLoRA / DoRA / rsLoRA**: set `model.quantization: 4bit` or `lora.variant: dora` in `configs/lora_medgemma27b.yaml`
- **Resume interrupted trace generation**: `python scripts/generate_traces.py --resume-from <path>`
- **HealthBench-only generalization**: pass `--exclude-ids data/raw/healthbench_hard.jsonl` to generate_traces.py

---

## Expected Outputs

### Eval JSONs

Four files, one per `{base|lora} x {no_wrapper|bodhi}` configuration:

```json
{
  "config": "lora_no_wrapper",
  "n_examples": 200,
  "mean": 0.XX, "std": 0.XX, "median": 0.XX,
  "brier": 0.XX, "ece": 0.XX,
  "results": [...]
}
```

The headline result is `lora_no_wrapper.mean` vs `base_bodhi.mean` — the LoRA model without a wrapper should match or beat the prompted baseline.

### U-shape stratification

`eval/ushape.json` breaks down per-tier (`easy/medium/hard`) and per-theme (`emergency_referrals`, `hedging`, etc.) failure rates. Figures rendered automatically to `eval/figures/`. To regenerate manually:

```bash
python scripts/eval_ushape.py \
    --eval-jsons eval/base_no_wrapper.json eval/base_bodhi.json \
                 eval/lora_no_wrapper.json eval/lora_bodhi.json \
    --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --bootstrap 1000 --output eval/ushape.json
```

### Reproducibility guarantees

**Deterministic** on the same hardware + software: `random`, `numpy`, `torch` are all seeded to `42`; greedy decoding is used throughout.

**Not deterministic across** different GPU models, CUDA versions, or PyTorch versions. To lock exact reproducibility, capture your environment:

```bash
pip freeze > requirements.lock.txt
```

---

## Multi-Seed Variance

Re-runs filter → train → eval N times with different seeds, reusing the single raw-traces file:

```bash
# default: seeds 42 7 13 99 101
bash scripts/run_multi_seed.sh

# aggregate across seeds
python scripts/aggregate_seeds.py \
    --seed-dirs eval/seed_* \
    --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --output eval/multi_seed_summary.json
```

---

## Running Tests

Tests live in `tests/` and cover pure-Python logic (no GPU required):

```bash
pip install pytest
pytest tests/ -v
```

Tests requiring `matplotlib` are skipped automatically if it is not installed. To run the full suite:

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Secret Hygiene

Before opening a PR, scan for accidentally committed tokens:

```bash
bash scripts/check_no_secrets.sh
```

Keep your `HF_TOKEN` in a local `.env` file (gitignored). Never hard-code it.

---

## Pull Request Process

1. Fork the repo and create a branch: `git checkout -b your-name/short-description`
2. Make your changes and add tests where relevant
3. Run `bash scripts/check_no_secrets.sh` and `pytest tests/ -v`
4. Open a PR against `main` — CI runs automatically (syntax, shellcheck, YAML, pytest)
5. Address review comments

One issue per PR. If unsure what to work on, check the [open issues](https://github.com/PeterLi-jpg/bohdi-lora/issues).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `401 Unauthorized` on HF | Token unset or missing model access | Accept terms on HF model page, re-export `HF_TOKEN` |
| `CUDA out of memory` during training | 27B at bf16 needs ~55 GB for weights alone | Set `model.quantization: 4bit` in config, or use ≥ 80 GB GPU |
| `Could not auto-detect response template` | Non-standard chat template | Extend `find_response_template` in `scripts/train_lora.py` |
| Slurm job runs in wrong directory | `SLURM_SUBMIT_DIR` not set | `export BOHDI_DIR=/path/to/bohdi-lora` before `run_all.sh` |
| `autoawq` install fails on Mac | No CUDA wheel on macOS | Expected — platform marker skips it |
| `ModuleNotFoundError: liger_kernel` | Optional TRL dep | `pip install liger-kernel` (Linux/CUDA only) |

---

## Questions

Open a [GitHub Issue](https://github.com/PeterLi-jpg/bohdi-lora/issues) or ping `@PeterLi-jpg` in the group chat.
