# Reuse and licenses

BOHDI-LoRA is deliberately thin. The epistemic wrapper, the benchmark, the base model, and
the LoRA training stacks are all reused; only the pipeline glue (trace generation, rubric
grading/filtering, leakage gates, multi-seed bootstrap eval, and the seven-dimension
behavioral analysis) is genuinely ours. This file records what we reuse so the paper's reuse
statement is exact. Exact Python pins live in `requirements.txt` / `requirements.lock.txt`.

| Component | Source | License | Used by | Version |
|---|---|---|---|---|
| BOHDI epistemic wrapper | https://github.com/sebasmos/bodhi-llms (`pip install bodhi-llm`) | see upstream | Stage 1 `scripts/generate_traces.py` | see `requirements.txt` |
| HealthBench eval framework | https://github.com/sebasmos/humbleai-healthbench | see upstream | Stage 2 grading, Stage 4 eval | see `requirements.txt` |
| HealthBench Hard / Full data | OpenAI simple-evals release | OpenAI terms | Stages 1, 2, 4 | 2025-05-07 / 2025-05-08 |
| MedGemma-27B (text-it) | Hugging Face `google/medgemma-27b-text-it` | Health AI Developer Foundations / Gemma | base model, all stages | HF revision |
| Gemma-3n-E4B (smoke) | Hugging Face `google/gemma-3n-E4B-it` | Gemma | `smoke.sh` end-to-end test | HF revision |
| tunix + qwix | Google (JAX LoRA SFT) | Apache-2.0 | Stage 3 `scripts/train_lora_tunix.py` | see `requirements.txt` |
| MaxText (forked) | https://github.com/google/maxtext, vendored in `third_party/maxtext` | Apache-2.0 | Stage 3 legacy path `scripts/train_lora_maxtext.py` | see `third_party/maxtext/VENDOR.md` |
| transformers + peft + trl + torch_xla | Hugging Face / PyTorch | Apache-2.0 / BSD | Stage 3 PyTorch path `scripts/train_lora.py` | see `requirements.lock.txt` |
| vLLM (vllm-tpu) | https://github.com/vllm-project/vllm | Apache-2.0 | trace generation / eval serving | see `requirements.txt` |
| accelerate | Hugging Face | Apache-2.0 | multi-device launch (PyTorch path) | see `requirements.lock.txt` |

## Grader models (cross-family, asymmetric)

To keep the filter-grader bias out of the reported metrics, the quality filter and the
evaluator come from different model families:

- **Qwen-14B** — quality filter at Stage 2 (which traces enter training).
- **Llama-3.1-8B** — evaluator at Stage 4 (post grader-swap; scores the held-out Hard set).

Both are used through their released Hugging Face checkpoints under their respective licenses.
