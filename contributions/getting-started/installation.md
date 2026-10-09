# Installation

Linux with CUDA 12.1 or later for real runs. macOS works for the smoke test with a small
model and for the CPU-only tests. On Windows, use WSL2 with Ubuntu.

## 1. Clone

```bash
git clone https://github.com/PeterLi-jpg/bohdi-lora.git
cd bohdi-lora
git config core.hooksPath hooks     # activates the repo hooks, see contributing.md
```

## 2. Training env

Python 3.10, 3.11 or 3.12 (3.11 is tested). Use a fresh env for this repo only:

```bash
python3 -m venv .venv
source .venv/bin/activate
bash setup.sh
```

`setup.sh` warns if you install into a shared or base Python, installs
`requirements.txt`, runs `python -m pip check`, makes the working folders and downloads
HealthBench into `data/raw/`. On macOS, pip prints `Ignoring autoawq` and
`Ignoring bitsandbytes`; both are Linux-only, so this is expected.

## 3. Inference env

`rebuttal/launch/run_cell.sh` runs generation, grading and eval with
`.venv-infer/bin/python`, a second env for vLLM 0.9.2, which needs an older transformers
than training does. `requirements-infer.txt` pins it; the env behind the posted cells was
not frozen (KNOWN_ISSUES). Point
`INFER_PY` and `TRAIN_PY` at other interpreters if your envs live elsewhere.

## 4. Hugging Face access

Accept the terms on each model card, then `export HF_TOKEN=...`, or keep a token in
`~/.cache/huggingface/token`. Never write the token into a tracked file.

| Model | Role |
|---|---|
| `google/medgemma-27b-text-it` | base model, submitted cell |
| `mistralai/Mistral-Small-24B-Instruct-2501` | base model, added cells |
| `m42-health/Llama3-Med42-8B` | base model, added cells |
| `BioMistral/BioMistral-7B` | base model, added cell |
| `Qwen/Qwen2.5-14B-Instruct` | filter grader, stage 2 |
| `meta-llama/Llama-3.1-8B-Instruct` | evaluator, stages 4 and 5 (the added cells use the `NousResearch/Meta-Llama-3.1-8B-Instruct` mirror) |
| `google/gemma-3n-E4B-it` | smoke test |

## 5. Check the install

```bash
python -m bodhi_lora list cells
pytest tests/
```

`list` loads and checks every config in `configs/`.
