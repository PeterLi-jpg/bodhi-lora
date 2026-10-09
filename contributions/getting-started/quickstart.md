# Quickstart

Finish [installation.md](installation.md) first.

## Smoke test

```bash
make smoke
```

This runs the smoke cell `configs/cells/tiny-rubric__tiny-random-llama__bodhi.yaml` on
the CPU, in about two minutes: every stage of a cell, with the stub engine in place of
vLLM and a tiny public model (`hf-internal-testing/tiny-random-LlamaForCausalLM`) trained
for real. It needs no GPU and no token. The outputs go to `outputs/smoke/` (ignored by
git). A failing stage stops the run and prints the end of its log.

## Real runs

One added cell, or the whole grid, is in [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md)
section 5.

## One stage alone

Every stage is a module with the same command line, so one stage of one seed can run by
hand, for example the eval of one condition:

```bash
python -m bodhi_lora.pipeline.eval --cell <cell id> --seed 42 --condition lora+bodhi
```

A rerun of the traces or filter stage resumes: it skips the prompts it already wrote or
graded. `python -m bodhi_lora run` skips every stage whose manifest says complete.
