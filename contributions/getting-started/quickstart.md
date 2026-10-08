# Quickstart

Finish [installation.md](installation.md) first.

## Smoke test

```bash
bash smoke.sh
```

This runs generate, filter, train and eval on `google/gemma-3n-E4B-it` with 3 examples,
in under 10 minutes on one GPU. Without access to gated models, use an ungated model:

```bash
SMOKE_MODEL=Qwen/Qwen2.5-0.5B-Instruct bash smoke.sh
```

For a larger sample, set `N_EXAMPLES=20`. Do not pipe the run into `tee`: the pipe
returns the exit code of `tee`, so a failed smoke test looks like a pass. Redirect to a
file instead:

```bash
mkdir -p logs && bash smoke.sh > logs/smoke.log 2>&1; echo "smoke exit=$?"
```

Expected outputs:

```
data/sft/smoke/raw_traces.jsonl   3 traces
data/sft/smoke/train.jsonl        graded training examples
data/sft/smoke/val.jsonl          validation split
eval/smoke/lora.json              eval summary
```

Fix any failing stage before you start a real run.

## Real runs

One added cell, or the whole grid, is in [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md)
section 5. On a slurm cluster, `bash run_all.sh` submits the stages as a dependency chain
using the scripts in `slurm/`.

## One stage alone

Evaluate an adapter that already exists:

```bash
python scripts/eval_healthbench.py \
    --model google/medgemma-27b-text-it \
    --lora-path checkpoints/best \
    --sample-ids data/raw/hard_200_sample_ids.json \
    --output eval/lora_no_wrapper.json
```

Resume an interrupted generation with `scripts/generate_traces.py --resume-from <file>`;
it skips the prompt ids it already wrote.
