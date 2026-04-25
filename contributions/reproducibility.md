# Reproducibility Guide

Step-by-step instructions to reproduce every experiment in this repo. If a command is not written below, do not invent one — open an issue.

## 1. Prerequisites

### Hardware

Two fully-supported paths — pick one:

**TPU (recommended if you have TRC access)**
- Smoke test: any machine with gcloud installed and HF_TOKEN set
- Full run: Cloud TPU v4-32, v6e-64, or v5litepod-64 via TRC grant
- No VRAM constraints — full bfloat16, no quantization needed
- See section 4a (TPU) for setup

**GPU**
- Smoke test: 1 GPU with ≥16 GB VRAM (or CPU, slow)
- Full run (full-precision LoRA): ≥80 GB VRAM — H100 or A100-80G, or 2× A100-40G
- Full run (QLoRA): ≥18 GB VRAM — single A100-40G or RTX 4090
- See section 4b (GPU) for setup

### Software
- Python 3.10 or 3.11 (tested on 3.11; 3.12 works but CUDA wheel coverage lags)
- CUDA 12.1+ for the GPU run
- Linux for the full run (slurm + autoawq). macOS is fine for smoke/dev (autoawq is skipped via platform marker).

### Hugging Face access
Accept the terms on each model page while logged into HF, then set `HF_TOKEN`:

| Model | Used for | URL |
|---|---|---|
| `google/medgemma-27b-text-it` | base model (paper target) | https://huggingface.co/google/medgemma-27b-text-it |
| `google/gemma-3n-E4B-it` | smoke / local iteration | https://huggingface.co/google/gemma-3n-E4B-it |
| `google/gemma-3n-E2B-it` | fallback if E4B OOMs | https://huggingface.co/google/gemma-3n-E2B-it |
| `Qwen/Qwen2.5-14B-Instruct` | grader — full pipeline (GPU + TPU) | https://huggingface.co/Qwen/Qwen2.5-14B-Instruct |
| `Qwen/Qwen2.5-0.5B-Instruct` | grader (smoke) | ungated |

Note: the grader is `Qwen2.5-14B-Instruct` (bfloat16, not AWQ) so it runs on both GPU and TPU. AWQ requires CUDA and cannot run on TPU.

```bash
export HF_TOKEN=hf_...
```

## 2. Setup (one-time)

```bash
git clone https://github.com/PeterLi-jpg/bohdi-lora.git
cd bohdi-lora
python3 -m venv .venv
source .venv/bin/activate
bash setup.sh
```

`setup.sh`:
1. Prints a loud warning if you install into a base/shared Python, including Conda `base`
2. Creates `logs/ data/raw/ data/sft/ eval/ checkpoints/`
3. Runs `python -m pip install -r requirements.txt`
4. Runs `python -m pip check`
5. Downloads HealthBench Hard + Full + Consensus into `data/raw/`

Expected macOS install messages:
- `Ignoring autoawq` — expected, Linux/CUDA only
- `Ignoring bitsandbytes` — expected, Linux/CUDA only

If `pip check` fails after install, the active environment already contains unrelated packages with incompatible constraints. Use a fresh virtualenv or project-specific conda env and rerun `bash setup.sh`.

## 4a. Full pipeline — TPU (Google TRC)

Requires a [TRC grant](https://sites.research.google/trc/about/) with Cloud TPU quota. The script handles VM creation, dependency setup, all 5 pipeline stages, and VM deletion automatically.

```bash
# store your HF token in .env (gitignored)
echo "HF_TOKEN=hf_..." > .env

# launch — tries all TRC zones until capacity is available
bash tpu/launch_multiseed.sh
```

The script tries TRC-granted zones in this order, retrying every 3 minutes:

| Priority | TPU | Chips | Zone |
|---|---|---|---|
| 1 | v4-32 on-demand | 32 | us-central2-b |
| 2 | v4-32 spot | 32 | us-central2-b |
| 3 | v6e-64 spot | 64 | europe-west4-a |
| 4 | v6e-64 spot | 64 | us-east1-d |
| 5 | v5litepod-64 spot | 64 | us-central1-a |
| 6 | v5litepod-64 spot | 64 | europe-west4-b |

**Expected time once VM is acquired:** ~6-7 hours total (generation dominates at ~2-3 hours).

**Outputs** land in `./results/` locally:
```
results/
  seed_42/{checkpoints/, eval/, figures/}
  seed_123/{checkpoints/, eval/, figures/}
  seed_456/{checkpoints/, eval/, figures/}
  multi_seed_summary.json
```

**Optional overrides:**
```bash
SEEDS="42 123" bash tpu/launch_multiseed.sh          # fewer seeds
MAX_EXAMPLES=100 bash tpu/launch_multiseed.sh        # quick test
LORA_VARIANT=dora bash tpu/launch_multiseed.sh       # DoRA variant
LORA_VARIANT=rslora LORA_R=32 bash tpu/launch_multiseed.sh
```

Note: `QUANT=4bit` will error on TPU (bitsandbytes is CUDA-only). Use the default full-precision bfloat16 on TPU.

## 4b. Full pipeline — GPU

On any Linux machine with CUDA GPUs:

```bash
# store HF token
export HF_TOKEN=hf_...

# auto-detects GPU count, picks 1-GPU or multi-GPU accelerate config
bash gpu/launch_multiseed.sh
```

Stages 1 and 2 (generate + filter) are skipped if their output files already exist, so interrupted runs resume from the right stage.

**Optional overrides:**
```bash
QUANT=4bit bash gpu/launch_multiseed.sh              # QLoRA (saves VRAM)
LORA_VARIANT=dora bash gpu/launch_multiseed.sh       # DoRA
CONFIG=configs/lora_medgemma27b_qlora.yaml bash gpu/launch_multiseed.sh
```

## 3a. Smoke test on GCP (recommended — ~20 min, ~$0.25)

Cheapest path: a single L4 GPU VM. Kill it the moment smoke passes.

```bash
# one-time: enable Compute Engine if you haven't
gcloud services enable compute.googleapis.com

# pick any GCP zone that has L4s: us-central1-a, us-east4-a, europe-west4-a
ZONE=us-central1-a
NAME=bohdi-smoke

# spin up the VM (Deep Learning VM image has CUDA + drivers preinstalled)
gcloud compute instances create $NAME \
    --zone=$ZONE \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=100GB \
    --metadata="install-nvidia-driver=True"

# wait ~90s for first boot, then SSH in
gcloud compute ssh $NAME --zone=$ZONE

# --- inside the VM ---
git clone https://github.com/PeterLi-jpg/bohdi-lora.git
cd bohdi-lora
python3 -m venv .venv
source .venv/bin/activate
export HF_TOKEN=hf_...      # paste yours; gemma-3n-E4B-it is gated
bash setup.sh
# Run smoke and write to a log. DO NOT use `bash smoke.sh | tee`: piping
# to tee makes the pipeline's exit code tee's (always 0), so smoke failures
# appear to pass. Use redirection or PIPESTATUS instead.
bash smoke.sh &> logs/smoke.log; echo "smoke exit=$?"
# (to watch progress from another terminal: tail -f logs/smoke.log)

# --- back on your laptop when done ---
gcloud compute instances delete $NAME --zone=$ZONE --quiet
```

If smoke passes, the pipeline is wired correctly. Scale up to a bigger GPU + real config when you're ready for a paper run.

## 3b. Smoke test on a local Mac (≥32 GB unified RAM)

Run this FIRST, every time you change any pipeline code. It exercises all four stages (generate, filter, train, eval) end-to-end with `gemma-3n-E4B-it` so bugs surface in minutes instead of after a 20-minute slurm queue. Per Felipe, 4B is the floor for meaningful BOHDI signal; E2B works too but its reasoning is often too weak to produce traces worth training on. Scale to MedGemma-27B once local runs show promise.

```bash
# default: 3 examples, ~10 min. For a meatier smoke:
N_EXAMPLES=10 bash smoke.sh
```

Validated on an M4 Max / 64 GB on 2026-04-16 — full pipeline ran in ~40 min at N_EXAMPLES=10. `bash smoke.sh` starts with a preflight check (scripts/preflight.py) that fails in ~10s if HF_TOKEN is wrong, a gated model isn't accessible, or a dep is missing, so env problems surface before trace generation.

If you only want a completely ungated local wiring check, override the model explicitly:

```bash
SMOKE_MODEL=Qwen/Qwen2.5-0.5B-Instruct bash smoke.sh
```

Expected outputs:
- `data/sft/smoke/raw_traces.jsonl` — 3 BOHDI traces
- `data/sft/smoke/{train,val}.jsonl` — split of the graded traces
- `checkpoints/best/` — LoRA adapter saved by SFTTrainer
- `eval/smoke/lora.json` — eval summary with `mean`, `std`, `brier_model_calibration`, `ece_model_calibration`, and the legacy `*_grader_consistency` fields

If any stage errors, fix it before moving to the full pipeline. Do not submit slurm jobs with a broken smoke test.

## 4c. Full pipeline — Slurm cluster

On a slurm cluster, from the repo root:

```bash
export HF_TOKEN=hf_...
# optional: override if BOHDI_DIR should differ from $SLURM_SUBMIT_DIR
# export BOHDI_DIR=/path/to/bohdi-lora

bash run_all.sh
```

`run_all.sh` archives train and eval artifacts under `results/<date>_<config>_seed<N>/`.

This submits four jobs as a dependency chain (`--dependency=afterok`):

| Job | Script | Time | Produces |
|---|---|---|---|
| 1 | `slurm/generate_traces.sh` | ~48 h | `data/sft/raw_traces.jsonl` |
| 2 | `slurm/filter_traces.sh`  | ~24 h | `data/sft/{train,val}.jsonl` |
| 3 | `slurm/train_lora.sh`     | ~12 h | `checkpoints/best/` |
| 4 | `slurm/eval_lora.sh`      | ~12 h | `eval/{base,lora}_{no_wrapper,bodhi}.json` |

Monitor:
```bash
squeue -u $USER
tail -f logs/generate_traces_<jobid>.out
```

### Running one stage at a time

Each stage can be submitted alone; see the commands inside the matching `slurm/*.sh`. The most common reason to do this is to resume from an interrupted generate step — `generate_traces.py` supports `--resume-from <path>` and skips prompt_ids it already produced.

## 5. Expected outputs

### Stage 1 — generate
`data/sft/raw_traces.jsonl` — one JSON object per line, each with:
```json
{
  "prompt_id": "...",
  "messages": [...],
  "response": "...",
  "tags": [...],
  "source_dataset": "healthbench_hard" | "healthbench",
  "model": "google/medgemma-27b-text-it",
  "bodhi": true
}
```
Expect ~4800 lines after excluding the 200-sample eval holdout.

### Stage 2 — filter
`data/sft/{train,val}.jsonl` — same shape plus a `"grade"` field:
```json
"grade": {
  "overall_score": 0.73,
  "tag_scores": {"accuracy": 0.8, "safety": 0.5},
  "criteria_results": [...]
}
```
Note: `overall_score` can go **negative** when negative-point rubric items are "met" (i.e., the response did the bad thing). The `--min-score 0.4` default filters these out. With the default threshold, expect 2500–3000 kept traces; 10% of them go to val.

### Stage 3 — train
`checkpoints/best/` — LoRA adapter weights + tokenizer. `trainer_state.json` has the loss curve. Training is seeded (`seed: 42` in the config); reruns with the same data produce bit-identical weights on the same hardware.

### Stage 4 — eval
Five JSON files in `eval/`:
- Four per-config files, one per `{base|lora} x {no_wrapper|bodhi}` combo
- `eval/ushape.json` — post-hoc stratified analysis (see below)

Each per-config file contains:
```json
{
  "config": "...",
  "n_examples": 200,
  "mean": 0.XX, "std": 0.XX, "median": 0.XX,
  "brier_model_calibration": 0.XX,
  "ece_model_calibration": 0.XX,
  "brier_grader_consistency": 0.XX,
  "ece_grader_consistency": 0.XX,
  "results": [...]
}
```
The headline comparison is `lora_no_wrapper.mean` vs `base_bodhi.mean` — we want the LoRA model (no wrapper) to match or beat the base model with the wrapper.

### Stage 4b — U-shape stratification (`eval/ushape.json`)

Inspired by the Nature Medicine 2026 triage paper (s41591-026-04297-7). Post-hoc aggregation that re-uses the four per-config eval outputs and HealthBench metadata — no extra model inference. Produced automatically by `slurm/eval_lora.sh` and `smoke.sh`; can be re-run manually:

```bash
python scripts/eval_ushape.py \
    --eval-jsons eval/base_no_wrapper.json eval/base_bodhi.json \
                 eval/lora_no_wrapper.json eval/lora_bodhi.json \
    --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --output eval/ushape.json
```

Structure:
- `thresholds` — the pos-points quartile cutoffs used to tier examples
- `configs.<name>.by_tier` — per-tier (`easy`/`medium`/`hard`) stats: `n`, `mean`, `median`, `min`, `max`, `fail_rate` (fraction scoring below `--fail-threshold`, default 0.4)
- `configs.<name>.by_theme` — same stats broken down by HealthBench theme, e.g. `emergency_referrals`, `hedging`, `context_seeking`

The figures are rendered automatically by `scripts/plot_ushape.py`, which runs at the end of `slurm/eval_lora.sh` and `smoke.sh`. Output in `eval/figures/`:

**Coarse (3-tier, from `ushape.json`):**
- `u_curve.png` — mean score across `easy | medium | hard`, one line per config. Good for slide decks but straight-segment, not a curve.
- `u_fail.png` — same x-axis, failure rate on y-axis.
- `theme_fail.png` — grouped bar chart, themes on x-axis, fail rate per config. `emergency_referrals` and `hedging` are highlighted (BOHDI's humility claim lives there).

**Smooth (10-bin, from per-example eval JSONs):**
- `u_curve_smooth.png` — mean score vs continuous difficulty (pos-points sum), 10 equal-frequency bins, quadratic fit overlay. Actual curve, not 3 line segments.
- `u_fail_smooth.png` — same but fail rate. Expect the classic U on `base_no_wrapper`.
- `u_scatter.png` — 4-panel scatter, all 200 examples per config with the quadratic fit and the fail-threshold line. Shows the raw data distribution.

To regenerate plots without rerunning eval:
```bash
# coarse only
python scripts/plot_ushape.py --input eval/ushape.json --out-dir eval/figures

# coarse + smooth + scatter
python scripts/plot_ushape.py \
    --input eval/ushape.json \
    --eval-jsons eval/base_no_wrapper.json eval/base_bodhi.json \
                 eval/lora_no_wrapper.json eval/lora_bodhi.json \
    --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --n-bins 10 --out-dir eval/figures
```

### Bootstrap 95% CIs

`eval_ushape.py --bootstrap 1000` resamples the 200-prompt holdout 1000 times with replacement and attaches a 95% CI to every mean and fail rate reported, for overall + per-tier + per-theme. The CIs render as shaded bands on line plots and as error bars on the per-theme bar chart. Seed with `--bootstrap-seed` so CIs are reproducible (default 42).

```bash
python scripts/eval_ushape.py \
    --eval-jsons eval/*.json \
    --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --bootstrap 1000 \
    --output eval/ushape.json
```

## 5c. Multi-seed variance (optional but recommended)

For variance bars on the training step, `scripts/run_multi_seed.sh` re-runs `filter → train → eval` N times with different seeds, reusing the single raw-traces file from stage 1. Trace generation is the expensive stage, so this costs roughly `N * (filter + train + eval)` hours, not `N * full pipeline`.

```bash
# default: seeds 42 7 13 99 101
bash scripts/run_multi_seed.sh

# override:
SEEDS="42 7 13 99 101 1 2 3 4 5" bash scripts/run_multi_seed.sh
```

Aggregate across seeds into `eval/multi_seed_summary.json`:

```bash
python scripts/aggregate_seeds.py \
    --seed-dirs eval/seed_* \
    --healthbench data/raw/healthbench_hard.jsonl data/raw/healthbench.jsonl \
    --output eval/multi_seed_summary.json
```

Reports across-seed mean ± std per config per metric. Percentile CIs added automatically when `N >= 5`.

## 5d. HealthBench-only generalization experiment

Tests whether training without any HealthBench Hard prompts still transfers to the Hard holdout. First verify the splits don't overlap:

```bash
python scripts/check_dataset_overlap.py
```

HealthBench Hard is a strict subset of HealthBench full, so to exclude all Hard prompts from training (not just the 200 eval prompts), pass both files to `--exclude-ids`:

```bash
python scripts/generate_traces.py \
    --model google/medgemma-27b-text-it \
    --datasets healthbench \
    --exclude-ids data/raw/healthbench_hard.jsonl data/raw/hard_200_sample_ids.json \
    --output data/sft/raw_traces_healthbench_only.jsonl \
    --use-bodhi
```

Then run filter / train / eval as usual against this trace file.

## 5e. Optional: QLoRA, DoRA, rsLoRA

`configs/lora_medgemma27b.yaml` accepts:

- `model.quantization`: `null` (default), `4bit` (QLoRA, NF4 + bf16 compute), or `8bit`
- `lora.variant`: `standard` (default), `dora`, or `rslora`

Defaults match the existing pipeline, so current configs behave identically. QLoRA requires `bitsandbytes`, which is Linux/CUDA-only (already platform-gated in `requirements.txt`). DoRA is incompatible with quantization and will error out if combined.

## 6. Reproducibility guarantees

**Deterministic under the same hardware + software**:
- `random`, `numpy`, `torch` seeded to `42` (configurable via `seed:` in the training YAML)
- `SFTConfig(seed=42, data_seed=42)`
- Greedy decoding (`do_sample=False`) in trace generation and eval

**Not deterministic across**:
- Different GPU models (kernel non-determinism — cuDNN/cuBLAS)
- Different CUDA / PyTorch versions (numerical rounding differences)
- Different numbers of GPUs (batch ordering changes)

If you need bit-exact reproducibility, lock: GPU model, CUDA version, PyTorch version, and the output of `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `401 Unauthorized` on HF | `HF_TOKEN` unset or missing access | Accept terms on HF model page, re-export token |
| `CUDA out of memory` on train | 27B at bf16 is ~55 GB just for weights | Set `model.quantization: 4bit` in the config (QLoRA, requires `bitsandbytes`) or use a ≥80 GB GPU |
| `Could not auto-detect response template` | Tokenizer's chat template is non-standard | Printed suffix shows the last 50 chars — extend `find_response_template` in `scripts/train_lora.py` if needed |
| Slurm job runs in wrong dir | `SLURM_SUBMIT_DIR` not set (manual `sbatch` from unusual location) | Set `export BOHDI_DIR=/path/to/bohdi-lora` before `run_all.sh` |
| `autoawq` install fails on Mac | Expected — no CUDA wheel | Platform marker skips it; smoke test uses ungated Qwen 0.5B grader |
| `ModuleNotFoundError: liger_kernel` on import trl | Optional TRL dep | `pip install liger-kernel` (Linux/CUDA only) |
| `model.quantization=4bit` error on TPU | bitsandbytes is CUDA-only | Remove `QUANT=4bit`; use full bfloat16 on TPU (plenty of HBM) |
| `There is no more capacity` on TPU create | Zone temporarily full | Script retries all TRC zones automatically; try again later or wait |
| `IN_USE_ADDRESSES limit` on v6e | Transient Google error | Script skips to next zone; resolves on its own |
| `device_map="auto"` error on TPU | XLA doesn't support CUDA device mapping | Should not happen with current scripts; file an issue if it does |

## 8. Minimal environment freeze

After `pip install -r requirements.txt`, capture the exact versions for later:
```bash
pip freeze > requirements.lock.txt
```
Commit `requirements.lock.txt` alongside your results so reviewers can reproduce the env.

## 9. Hygiene

Before opening a PR, run:

```bash
bash scripts/check_no_secrets.sh
```

This scans tracked files for common Hugging Face, OpenAI, and AWS token patterns. Generated outputs under `logs/`, `checkpoints/`, `eval/`, `data/sft/`, and `results/` are gitignored on purpose.
