# BOHDI-LoRA

[![CI](https://github.com/PeterLi-jpg/bohdi-lora/actions/workflows/ci.yml/badge.svg)](https://github.com/PeterLi-jpg/bohdi-lora/actions/workflows/ci.yml)

LoRA fine-tuning to internalize [BOHDI](https://github.com/sebasmos/bodhi-llms) epistemic virtues (humility, calibration, abstention) into model weights, replacing the prompt wrapper with weight-level alignment.

## Motivation

LLM overconfidence is reinforced through RLHF on benchmarks that reward confident answers over abstention or clarifying questions. The BOHDI prompt wrapper addresses this at inference time, but the underlying weights still favor overconfident behavior. This project uses SFT with LoRA to internalize BOHDI virtues directly into the model so it behaves humbly **without** the wrapper.

## Base Model

[google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it) — Google's 27B medical Gemma model (text-only variant). Pre-trained on medical data, so the LoRA only needs to teach behavioral virtues rather than medical knowledge. Requires accepting Google's Health AI terms on HuggingFace.

## Training Data

HealthBench Hard (1000 examples) + HealthBench Full (5000 examples) combined = 5000 unique prompts, with 200 held out for evaluation. That gives 4800 prompts for training data generation. These are run through the BOHDI wrapper, graded using the HealthBench rubric grader, and filtered by score — yielding ~2500-3000 high-quality training pairs.

## Quickstart

```bash
git clone https://github.com/PeterLi-jpg/bohdi-lora.git
cd bohdi-lora
bash setup.sh
export HF_TOKEN=hf_...        # gated model access (see reproducibility.md)
bash smoke.sh                  # end-to-end test, <10 min, catches bugs early
```

Then run the full pipeline on your hardware:

```bash
# TPU (Google TRC / Cloud TPU) — recommended, no cost under TRC grant
echo "HF_TOKEN=hf_..." > .env
bash tpu/launch_multiseed.sh

# GPU (any Linux machine with CUDA GPUs)
bash gpu/launch_multiseed.sh

# Slurm cluster
bash run_all.sh
```

See [contributions/reproducibility.md](contributions/reproducibility.md) for full instructions, hardware requirements, expected outputs, and troubleshooting.

## Pipeline

The full pipeline runs 5 stages automatically via the launch scripts:

| Stage | Script | What it does |
|---|---|---|
| 1 | `generate_traces.py` | MedGemma-27B + BOHDI wrapper over 4800 prompts |
| 2 | `filter_traces.py` | Grade with Qwen2.5-14B, keep score ≥ 0.4 |
| 3 | `train_lora.py` | LoRA SFT, one run per seed |
| 4 | `eval_healthbench.py` | 4 configs × 200 holdout prompts per seed |
| 5 | `aggregate_seeds.py` | Cross-seed means, stds, 95% CIs |

To run individual stages manually:

```bash
# 1. Generate BOHDI traces
python scripts/generate_traces.py \
    --model google/medgemma-27b-text-it \
    --datasets healthbench_hard healthbench \
    --output data/sft/raw_traces.jsonl --use-bodhi

# 2. Grade and filter traces
python scripts/filter_traces.py \
    --input data/sft/raw_traces.jsonl \
    --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --output-dir data/sft/

# 3. Train LoRA
accelerate launch --config_file tpu/accelerate_config_v4_32.yaml \
    scripts/train_lora.py --config configs/lora_medgemma27b_tpu.yaml  # TPU
python scripts/train_lora.py --config configs/lora_medgemma27b.yaml   # GPU

# 4. Evaluate
python scripts/eval_healthbench.py \
    --model google/medgemma-27b-text-it \
    --lora-path checkpoints/best \
    --sample-ids data/raw/hard_200_sample_ids.json \
    --output eval/lora_no_wrapper.json
```

## Hardware Paths

Two fully-supported execution paths — same pipeline, different hardware:

### TPU (Google Cloud TRC)

Runs on Google Cloud TPU VMs via `tpu/launch_multiseed.sh`. Requires a [TRC grant](https://sites.research.google/trc/about/). Supported quota types and zones:

| TPU | Chips | Zone | Notes |
|---|---|---|---|
| v4-32 | 32 | us-central2-b | on-demand + spot |
| v6e-64 | 64 | us-east1-d, europe-west4-a | spot |
| v5litepod-64 | 64 | us-central1-a, europe-west4-b | spot |

The launch script tries all zones automatically and retries until capacity is available. No quantization needed on TPU — full bfloat16 fits comfortably across the chips.

```bash
echo "HF_TOKEN=hf_..." > .env
bash tpu/launch_multiseed.sh

# Optional overrides:
SEEDS="42 123" bash tpu/launch_multiseed.sh
MAX_EXAMPLES=100 bash tpu/launch_multiseed.sh   # quick test
LORA_VARIANT=dora bash tpu/launch_multiseed.sh
LORA_R=32 bash tpu/launch_multiseed.sh
```

### GPU (Cloud or local)

Runs locally on any Linux machine with CUDA GPUs via `gpu/launch_multiseed.sh`. Detects GPU count automatically and picks the right accelerate config.

| Config | VRAM needed | When to use |
|---|---|---|
| `lora_medgemma27b.yaml` | ~60-80 GB | 2× A100-40G or 1× A100-80G / H100 |
| `lora_medgemma27b_qlora.yaml` | ~18-22 GB | Single A100-40G or RTX 4090 |

```bash
bash gpu/launch_multiseed.sh

# Optional overrides:
QUANT=4bit bash gpu/launch_multiseed.sh          # QLoRA
LORA_VARIANT=dora bash gpu/launch_multiseed.sh   # DoRA (full-precision only)
LORA_VARIANT=rslora LORA_R=32 bash gpu/launch_multiseed.sh
```

## Evaluation

Four configurations are compared on the 200-sample HealthBench Hard holdout:

| Configuration | HB-Hard Score | Brier* | ECE* |
|---|---|---|---|
| Base model | | | |
| Base + BOHDI wrapper | | | |
| **LoRA model (no wrapper)** | | | |
| LoRA + BOHDI wrapper | | | |

The key result is row 3: does the fine-tuned model exhibit epistemic humility without the prompt wrapper?

*Brier and ECE here measure grader-internal consistency, not model calibration. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) #1.

### U-shape stratified analysis

Per-tier failure rates stratified by rubric complexity (easy/medium/hard) and by theme (`emergency_referrals`, `hedging`, `context_seeking`, ...). Inspired by the Nature Medicine 2026 triage paper ([s41591-026-04297-7](https://www.nature.com/articles/s41591-026-04297-7)) which showed LLM failures concentrate at clinical extremes. The BOHDI hypothesis is that LoRA flattens this U by lifting the tails.

## Repository Structure

```
bohdi-lora/
├── configs/               # Training hyperparameters
│   ├── lora_medgemma27b_tpu.yaml    # TPU (bfloat16, no quantization)
│   ├── lora_medgemma27b.yaml        # GPU full-precision
│   ├── lora_medgemma27b_qlora.yaml  # GPU QLoRA (4-bit)
│   └── lora_gemma_smoke.yaml        # Smoke test only
├── tpu/                   # TPU launch scripts and accelerate configs
│   ├── launch_multiseed.sh          # Full pipeline on Cloud TPU
│   ├── setup_tpu.sh                 # TPU dependency installer
│   └── accelerate_config_*.yaml     # v4-32, v5e-64, v6e-64
├── gpu/                   # GPU launch scripts and accelerate configs
│   ├── launch_multiseed.sh          # Full pipeline on GPU
│   └── accelerate_config_*.yaml     # 1-GPU and multi-GPU DDP
├── scripts/               # Generation, filtering, training, eval, analysis
├── slurm/                 # SBATCH job scripts (cluster alternative)
├── contributions/         # Reproducibility guide and contribution docs
├── data/
│   ├── raw/               # HealthBench eval IDs
│   └── sft/               # Generated and filtered training data
├── eval/                  # Evaluation outputs
├── tests/                 # Pytest test suite (no GPU required)
├── smoke.sh               # End-to-end smoke test (<10 min)
├── run_all.sh             # Full pipeline dependency chain (slurm)
├── KNOWN_ISSUES.md
└── requirements.txt
```

## Hygiene

Run `bash scripts/check_no_secrets.sh` before opening a PR if you touched config or environment files. Generated outputs under `logs/`, `checkpoints/`, `eval/`, `data/sft/`, and `results/` are intentionally gitignored.

## References

- [sebasmos/humbleai-healthbench](https://github.com/sebasmos/humbleai-healthbench) — BOHDI evaluation framework on HealthBench
- [sebasmos/bodhi-llms](https://github.com/sebasmos/bodhi-llms) — BOHDI wrapper package (`pip install bodhi-llm`)
- [HealthBench Hard](https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl) — 1000 examples
- [HealthBench Full](https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl) — 5000 examples
