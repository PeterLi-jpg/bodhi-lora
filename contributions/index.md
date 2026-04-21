# BOHDI-LoRA Docs

LoRA fine-tuning to internalize BOHDI epistemic virtues (humility, calibration, abstention) directly into model weights, replacing the inference-time prompt wrapper.

## What is this?

Standard RLHF training rewards confident answers, which reinforces overconfidence in medical LLMs. The [BOHDI wrapper](https://github.com/sebasmos/bodhi-llms) fixes this at inference time, but the underlying weights still behave poorly without it. This project uses supervised fine-tuning with LoRA to bake the virtues into the weights — so the model behaves humbly **without** any wrapper.

## Contents

| File / Folder | What it covers |
|---|---|
| [getting-started/installation.md](getting-started/installation.md) | Environment setup on Mac, Linux, and Windows |
| [getting-started/quickstart.md](getting-started/quickstart.md) | Run your first smoke test in under 10 minutes |
| [reproducibility.md](reproducibility.md) | Full pipeline, expected outputs, and determinism guarantees |
| [contributing.md](contributing.md) | How to open a PR, run tests, and hygiene checks |

## Base Model

[google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it) — Google's 27B medical Gemma model. Pre-trained on medical data, so LoRA only needs to teach behavioral virtues rather than domain knowledge. Requires accepting Google's Health AI terms on HuggingFace.

## Evaluation

Four configurations compared on a 200-sample HealthBench Hard holdout:

| Configuration | Measures |
|---|---|
| Base model, no wrapper | Baseline overconfidence |
| Base model + BOHDI wrapper | Wrapper benefit |
| **LoRA model, no wrapper** | **Core claim: virtues in weights** |
| LoRA model + BOHDI wrapper | Additive benefit |

Headline metric: does `lora_no_wrapper` match or beat `base_bodhi`?
