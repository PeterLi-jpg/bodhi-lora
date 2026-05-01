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

1. `python3 -m venv .venv && source .venv/bin/activate` — clean local env
2. `bash setup.sh` — one-time install, `pip check`, and data download
3. `export HF_TOKEN=hf_...` — gated model access
4. `bash smoke.sh` — end-to-end test with `gemma-3n-E4B-it` (<10 min, catches bugs)
5. `bash run_all.sh` — full pipeline on slurm

See [contributions/reproducibility.md](contributions/reproducibility.md) for step-by-step pipeline instructions, expected outputs, and troubleshooting. See [contributions/contributing.md](contributions/contributing.md) for environment setup and PR process. [KNOWN_ISSUES.md](KNOWN_ISSUES.md) tracks open methodological concerns.

`setup.sh` now prints a loud warning if you run it outside an isolated env, or from Conda `base`, because shared/base Python environments are the main source of confusing dependency conflicts.

## Stage 3: PyTorch and MaxText paths

Stage 3 (LoRA fine-tune of MedGemma-27B) has two backends:

- **PyTorch** (default, in tree): `scripts/train_lora.py` driven by `tpu/launch_5seeds.sh`. Used on GPU and as the TPU fallback.
- **MaxText** (JAX-native, forked): `scripts/train_lora_maxtext.py`. Added because PyTorch + `torch_xla` 2.7 + FSDPv2 + Gemma-3-27B hangs on the first `mark_step` for over an hour on TPU. See [docs/maxtext_migration.md](docs/maxtext_migration.md) for why we forked, what changed, the Phase 2 run plan, and the HF-PEFT-adapter contract that keeps Stage 4 unchanged.

Stages 1, 2, 4, and 5 are unchanged regardless of which Stage 3 path runs.

## Hygiene

Run `bash scripts/check_no_secrets.sh` before opening a PR if you touched config or environment files. Generated outputs under `logs/`, `checkpoints/`, `eval/`, `data/sft/`, and `results/` are intentionally gitignored.

## Architecture

The pipeline is five stages. Each stage's output is the next stage's input; intermediate state lives on disk so any stage can be resumed independently.

```
Stage 1 (generate_traces.py)         -> data/sft/raw_traces.jsonl
                                        (~4000 BODHI traces from HealthBench Full minus Hard)

Stage 2 (filter_traces.py)           -> data/sft/seed_<N>/{train,val}.jsonl
                                        (rubric-graded; defensive --exclude-ids drops Hard)

    [preflight: check_dataset_overlap.py - aborts on Hard leakage]

Stage 3 (train_lora.py OR
         train_lora_maxtext.py)      -> checkpoints/seed_<N>/best/
                                        (LoRA r=8, q_proj+v_proj, MedGemma-27B base)

Stage 4 (eval_healthbench.py)        -> eval/seed_<N>/{base,base_bodhi,lora,lora_bodhi}.json
                                        (per-seed bootstrap; 200 of 1K Hard, deterministic per seed)

Stage 5 (aggregate_seeds.py)         -> eval/multi_seed_summary.json
                                        (mean +/- std + 95% CI across 5 seeds)
```

## Hardware requirements

- Stage 1 (BODHI inference): TPU v6e-8 spot (TRC quota free) OR H100 80GB
- Stage 2 (Qwen-14B grader -> now Llama-3.1-8B post-grader-swap): same as Stage 1
- Stage 3 (LoRA train of 27B): TPU v6e-8 (PyTorch+torch_xla OR MaxText)
- Stage 4 (eval): same as Stage 1
- Disk: 100 GB boot + tmpfs `/dev/shm` (~700 GB) for HF cache; or attach a 300 GB SSD
- Memory: ~256 GB host RAM, ~256 GB HBM on v6e-8

## Troubleshooting

- `ENOSPC` on Stage 2 grader: `setup_tpu.sh` redirects HF cache to `/dev/shm` (tmpfs). Verify via `df -h /dev/shm` on the VM.
- vLLM startup timeout: check vllm-tpu Docker logs at `~/vllm_serve_*.log`.
- GCS auth expired: re-run `gcloud auth login` + `gcloud auth application-default login`.
- Capacity errors (gRPC code 8): TRC v6e-8 spot is free but contended. Try eur4a evening (US morning) or us-east1-d.
- Missing `bodhi`/`peft` modules in CI: see `.github/workflows/ci.yml` install line; PR #128 added bodhi-llm.
- Stage 1 trace gen takes ~30h+: that's the BODHI two-pass cost on 4K prompts. Use `MAX_EXAMPLES=100` for smoke.

## Cite this work

Paper in progress. Citation TBD.

## Pipeline

```bash
# first-time setup (isolated env, installs deps, checks for conflicts, downloads data)
python3 -m venv .venv
source .venv/bin/activate
bash setup.sh

# 1. Generate BOHDI traces
python scripts/generate_traces.py \
    --model google/medgemma-27b-text-it \
    --datasets healthbench_hard healthbench \
    --exclude-ids data/raw/hard_200_sample_ids.json \
    --output data/sft/raw_traces.jsonl --use-bodhi

# 2. Grade and filter traces
python scripts/filter_traces.py \
    --input data/sft/raw_traces.jsonl \
    --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --output-dir data/sft/

# 3. Train LoRA
python scripts/train_lora.py --config configs/lora_medgemma27b.yaml

# 4. Evaluate
python scripts/eval_healthbench.py \
    --model google/medgemma-27b-text-it \
    --lora-path checkpoints/best \
    --sample-ids data/raw/hard_200_sample_ids.json \
    --output eval/lora_no_wrapper.json
```

Slurm scripts for cluster execution are in `slurm/`. Update the `cd` path and submit with `sbatch`.

## Evaluation

Four configurations are compared on the 200-sample HealthBench Hard holdout. Eval JSONs now report two confidence families:

- `brier_model_calibration` / `ece_model_calibration`: use a model-derived confidence proxy, the geometric mean next-token probability of the emitted response
- `brier_grader_consistency` / `ece_grader_consistency`: legacy grader-derived proxies kept for backward comparison

Lower is better for all four.

| Configuration | HB-Hard Score | Brier (model) | ECE (model) |
|---|---|---|---|
| Base model | | | |
| Base + BOHDI wrapper | | | |
| **LoRA model (no wrapper)** | | | |
| LoRA + BOHDI wrapper | | | |

The key result is row 3: does the fine-tuned model exhibit epistemic humility without the prompt wrapper?

### U-shape stratified analysis

Per-tier failure rates stratified by rubric complexity (easy/medium/hard) and by theme (`emergency_referrals`, `hedging`, `context_seeking`, ...). Inspired by the Nature Medicine 2026 triage paper ([s41591-026-04297-7](https://www.nature.com/articles/s41591-026-04297-7)) which showed LLM failures concentrate at clinical extremes. The BOHDI hypothesis is that LoRA flattens this U by lifting the tails. Produced post-hoc by `scripts/eval_ushape.py`; output at `eval/ushape.json`.

## Repository Structure

```
bohdi-lora/
├── configs/          # Training hyperparameters (full + smoke)
├── data/
│   ├── raw/          # HealthBench eval IDs
│   └── sft/          # Generated and filtered training data
├── eval/             # Evaluation outputs
├── scripts/          # Generation, filtering, training, and eval scripts
├── slurm/            # SBATCH job scripts
├── contributions/    # Reproducibility guide and contribution docs
├── tests/            # Pytest test suite (no GPU required)
├── smoke.sh          # End-to-end smoke test (<10 min)
├── run_all.sh        # Full pipeline dependency chain (slurm)
├── KNOWN_ISSUES.md
└── requirements.txt
```

## References

- [sebasmos/humbleai-healthbench](https://github.com/sebasmos/humbleai-healthbench) — BOHDI evaluation framework on HealthBench
- [sebasmos/bodhi-llms](https://github.com/sebasmos/bodhi-llms) — BOHDI wrapper package (`pip install bodhi-llm`)
- [HealthBench Hard](https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl) — 1000 examples
- [HealthBench Full](https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl) — 5000 examples
