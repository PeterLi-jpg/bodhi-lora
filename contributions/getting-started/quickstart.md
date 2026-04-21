# Quickstart

Get the pipeline running in under 10 minutes using the smoke test.

## Prerequisites

Complete [installation.md](installation.md) first.

## Run the smoke test

```bash
export HF_TOKEN=hf_...
bash smoke.sh
```

This runs the full four-stage pipeline (generate → filter → train → eval) on `gemma-3n-E4B-it` with 3 examples. Expected runtime: under 10 minutes on a single GPU.

Expected outputs:

```
data/sft/smoke/raw_traces.jsonl     # 3 BOHDI traces
data/sft/smoke/train.jsonl          # graded training examples
data/sft/smoke/val.jsonl            # validation split
checkpoints/best/                   # LoRA adapter weights
eval/smoke/lora.json                # eval scores (mean, brier, ece)
```

If all four stages complete without error, your environment is correctly set up.

## Run on a larger sample

```bash
N_EXAMPLES=20 bash smoke.sh
```

## Run the full pipeline (cluster)

Once smoke passes, submit the full Slurm job chain:

```bash
bash run_all.sh
```

See [reproducibility.md](../reproducibility.md) for expected runtimes, outputs, and how to monitor jobs.

## Run the eval only (no training)

To evaluate a checkpoint that already exists:

```bash
python scripts/eval_healthbench.py \
    --model google/medgemma-27b-text-it \
    --lora-path checkpoints/best \
    --sample-ids data/raw/hard_200_sample_ids.json \
    --output eval/lora_no_wrapper.json
```

## Troubleshooting

See the [Troubleshooting](../reproducibility.md#7-troubleshooting) section in `reproducibility.md` for common errors.
