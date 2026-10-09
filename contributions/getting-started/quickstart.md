# Quickstart

Finish [installation.md](installation.md) first.

## Smoke test

`smoke.sh` and its GPU smoke config were removed on 2026-10-08. The smoke cell
`configs/cells/tiny-rubric__tiny-random-llama__bodhi.yaml` replaces them: it runs every
stage on the CPU with a tiny public model.

## Real runs

One added cell, or the whole grid, is in [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md)
section 5.

## One stage alone

Evaluate an adapter that already exists:

```bash
python -m bodhi_lora.pipeline.eval \
    --model google/medgemma-27b-text-it \
    --lora-path checkpoints/best \
    --sample-ids data/raw/hard_200_sample_ids.json \
    --output eval/lora_no_wrapper.json
```

Resume an interrupted generation with `bodhi_lora/pipeline/traces.py --resume-from <file>`;
it skips the prompt ids it already wrote.
