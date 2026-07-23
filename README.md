# BOHDI-LoRA

[![CI](https://github.com/REDACTED-FOR-ANONYMOUS-REVIEW/bohdi-lora/actions/workflows/ci.yml/badge.svg)](https://github.com/REDACTED-FOR-ANONYMOUS-REVIEW/bohdi-lora/actions/workflows/ci.yml)

LoRA fine-tuning to internalize [BOHDI](https://github.com/REDACTED/bodhi-llms) epistemic virtues (humility, calibration, abstention) into model weights, replacing the prompt wrapper with weight-level alignment.

## Motivation

LLM overconfidence is reinforced through RLHF on benchmarks that reward confident answers over abstention or clarifying questions. The BOHDI prompt wrapper addresses this at inference time, but the underlying weights still favor overconfident behavior. This project uses SFT with LoRA to internalize BOHDI virtues directly into the model so it behaves humbly **without** the wrapper.

## Base Model

[google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it) — Google's 27B medical Gemma model (text-only variant). Pre-trained on medical data, so the LoRA only needs to teach behavioral virtues rather than medical knowledge. Requires accepting Google's Health AI terms on HuggingFace.

## Training Data

HealthBench Hard (1000 examples) is a strict subset of HealthBench Full (5000 examples). To eliminate any train/eval leakage, *all* 1000 Hard prompts are excluded from training, leaving ~4000 non-Hard prompts in the training pool. These are run through the BOHDI wrapper, graded using the HealthBench rubric grader, and filtered by score — yielding ~2500-3000 high-quality training pairs.

Evaluation uses one of two paths:

- **Per-seed bootstrap (default for multi-seed runs):** `scripts/make_bootstrap_eval_ids.py` draws an independent 200-prompt subset of the 1K Hard for each seed and writes it to `data/raw/hard_seed_<SEED>.json`. Because all 1K Hard are held out from training, every draw is honestly out-of-sample. Scores are reported as mean ± std across seeds.
- **Fixed 200-prompt holdout (legacy):** `data/raw/hard_200_sample_ids.json` is the original fixed eval set, used by `slurm/eval_lora.sh` when `SEED` is unset. Still supported for one-off evals.

## Quickstart

1. `python3 -m venv .venv && source .venv/bin/activate` — clean local env
2. `bash setup.sh` — one-time install, `pip check`, and data download
3. `export HF_TOKEN=hf_...` — gated model access
4. `bash smoke.sh` — end-to-end test with `gemma-3n-E4B-it` (<10 min, catches bugs)
5. `bash run_all.sh` — full pipeline on slurm

See [contributions/reproducibility.md](contributions/reproducibility.md) for step-by-step pipeline instructions, expected outputs, and troubleshooting. See [contributions/contributing.md](contributions/contributing.md) for environment setup and PR process. [KNOWN_ISSUES.md](KNOWN_ISSUES.md) tracks open methodological concerns.

`setup.sh` now prints a loud warning if you run it outside an isolated env, or from Conda `base`, because shared/base Python environments are the main source of confusing dependency conflicts.

## GPU-only pipeline (rebuttal branch)

This branch is a single, clean, GPU-only pipeline. All TPU / MaxText / tunix / Modal
code from `main` has been removed here. Stage 3 LoRA SFT is `scripts/train_lora.py`
(QLoRA for 24-27B, full-precision LoRA for 7-8B; FlashAttention-2/SDPA, TF32, fused AdamW).

To reproduce the rebuttal generality experiments (extra base models + benchmarks), see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) and `rebuttal/launch/run_cell.sh`.

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

Stage 3 (train_lora.py)              -> checkpoints/seed_<N>/best/
                                        (LoRA; GPU: QLoRA for 24-27B, full-precision for 7-8B)

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
#    --exclude-ids drops all 1000 HealthBench Hard prompts (Hard is a strict
#    subset of Full) plus the legacy 200-prompt holdout, leaving ~4000 non-Hard
#    training prompts.
python scripts/generate_traces.py \
    --model google/medgemma-27b-text-it \
    --datasets healthbench_hard healthbench \
    --exclude-ids data/raw/healthbench_hard.jsonl data/raw/hard_200_sample_ids.json \
    --output data/sft/raw_traces.jsonl --use-bodhi

# 2. Grade and filter traces
#    filter_traces.py defensively re-applies --exclude-ids so any stale rows
#    in raw_traces.jsonl (e.g. from a pre-#60 resume file) are dropped here.
python scripts/filter_traces.py \
    --input data/sft/raw_traces.jsonl \
    --healthbench-data data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --exclude-ids data/raw/healthbench_hard.jsonl data/raw/hard_200_sample_ids.json \
    --output-dir data/sft/

# 2b. Preflight leakage gate (fail-fast before spending GPU time on training)
python scripts/check_dataset_overlap.py \
    --train-jsonl data/sft/train.jsonl \
    --tag-overlap

# 3. Train LoRA
python scripts/train_lora.py --config configs/lora_medgemma27b.yaml

# 4. Evaluate
#    For multi-seed runs, generate the per-seed eval set first (deterministic
#    in SEED) and pass it via --sample-ids:
#        python scripts/make_bootstrap_eval_ids.py --seed 0 \
#            --output data/raw/hard_seed_0.json
#    For one-off evals, the legacy fixed 200-prompt holdout is still supported.
python scripts/eval_healthbench.py \
    --model google/medgemma-27b-text-it \
    --lora-path checkpoints/best \
    --sample-ids data/raw/hard_200_sample_ids.json \
    --output eval/lora_no_wrapper.json
```

Slurm scripts for cluster execution are in `slurm/`. Update the `cd` path and submit with `sbatch`.

## Known limitation: contaminated cached raw_traces

The pre-existing `gs://bohdi-runs-tokyo-micron/raw_traces.jsonl` was generated before the issue #60 fix and still contains all 1K HealthBench Hard prompts. `filter_traces.py`'s defensive `--exclude-ids` drops them at Stage 2, so `train.jsonl` ends up clean — but for a fully clean rebuild, delete the GCS file and let Stage 1 regenerate (~40h H100). All current smoke / production runs resume from the contaminated raw_traces and rely on the Stage 2 filter to keep training honest.

## Evaluation

Four configurations are compared on a 200-prompt HealthBench Hard eval set (per-seed bootstrap or legacy fixed holdout — see [Training Data](#training-data)). Eval JSONs now report two confidence families:

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

- [REDACTED/humbleai-healthbench](https://github.com/REDACTED/humbleai-healthbench) — BOHDI evaluation framework on HealthBench
- [REDACTED/bodhi-llms](https://github.com/REDACTED/bodhi-llms) — BOHDI wrapper package (`pip install bodhi-llm`)
- [HealthBench Hard](https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl) — 1000 examples
- [HealthBench Full](https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl) — 5000 examples
