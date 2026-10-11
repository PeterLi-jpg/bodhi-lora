# Reuse and licenses

Most parts of this project are reused: the prompt, the benchmarks, the base models, the
graders and the training libraries. The code in this repo is the pipeline around them:
trace generation, rubric grading and filtering, leakage checks, the multi-seed eval, the
seven-behavior grading and the statistics. This file lists what is reused so the
paper's reuse statement is exact. Python pins live in `pyproject.toml` (training and
analysis) and `requirements-infer.txt` (vLLM serving); `legacy/requirements.lock.txt` is
the old lock from the submission era. "See upstream" means we have not copied the license terms here;
read them at the source before you redistribute anything.

## Code

| Component | Source | License | Used by |
|---|---|---|---|
| BODHI prompt (`bodhi-llm`) | https://github.com/sebasmos/bodhi-llms | see upstream | stage 1, `bodhi_lora/pipeline/traces.py` |
| HealthBench grader code | https://github.com/sebasmos/humbleai-healthbench | see upstream | stages 2 and 4 |
| transformers, peft, trl, accelerate | Hugging Face | Apache-2.0 | stage 3 on GPU, `bodhi_lora/pipeline/train.py` |
| bitsandbytes | https://github.com/bitsandbytes-foundation/bitsandbytes | MIT | 4-bit QLoRA |
| vLLM | https://github.com/vllm-project/vllm | Apache-2.0 | generation, grading, eval |
| tunix, qwix | Google | Apache-2.0 | stage 3 of the submitted cell (`legacy/`) |

## Data

| Dataset | Source | License | Used by |
|---|---|---|---|
| HealthBench and HealthBench-Hard | OpenAI simple-evals release (2025-05-07, 2025-05-08) | see upstream | all HealthBench cells |
| ChatDoctor (HealthCareMagic) | Hugging Face `lavita/ChatDoctor-HealthCareMagic-100k` | see upstream | ChatDoctor cells |
| MedQuAD | Hugging Face `lavita/MedQuAD` | see upstream | MedQuAD cell |
| MedQA-USMLE | Hugging Face `GBaker/MedQA-USMLE-4-options` | see upstream | MedQA cell (set aside) |

## Models

| Model | Hugging Face id | License | Role |
|---|---|---|---|
| MedGemma-27B text | `google/medgemma-27b-text-it` | Health AI Developer Foundations terms | base model, submitted cell |
| Mistral-Small-24B | `mistralai/Mistral-Small-24B-Instruct-2501` | Apache-2.0 | base model, added cells |
| Med42-8B | `m42-health/Llama3-Med42-8B` | see upstream (Llama 3 based) | base model, added cells |
| BioMistral-7B | `BioMistral/BioMistral-7B` | see upstream | base model, added cell |
| Qwen2.5-14B-Instruct | `Qwen/Qwen2.5-14B-Instruct` | see upstream | filter grader, stage 2 |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct`, mirror `NousResearch/Meta-Llama-3.1-8B-Instruct` | Llama 3.1 Community License | evaluator, stages 4 and 5 |
| Gemma-3n-E4B | `google/gemma-3n-E4B-it` | Gemma terms | smoke test only |

## Grader families

The filter grader (Qwen) and the evaluator (Llama) come from different families, so
the model that picks the training data does not score the results. Med42-8B is a Llama-3
model, so its cells break this rule. They are marked †, and
`bodhi_lora/pipeline/grade.py` regrades them with a judge from neither family.
