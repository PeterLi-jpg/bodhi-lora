# Installation

There are three environments. Pick the one you need.

| Environment | For | Install |
|---|---|---|
| analysis (`.venv`, CPU) | the tables, statistics, figures and tests: `make analyze`, `make numbers`, `make figures`, `make test` (the tests that need the training stack skip) | `pip install -e '.[dev]'` |
| training (`.venv`) | the above, plus LoRA training, the CPU smoke cell (`make smoke`) and every `python -m bodhi_lora run`, dry runs too | `bash setup.sh`, which runs `pip install -e '.[train,gpu,dev]'` |
| inference (`.venv-infer`, Linux and CUDA) | the vLLM stages of a real run: traces, filter, eval, grade, probe, bench | `bash setup.sh` on Linux, from `requirements-infer.txt` |

Python 3.10, 3.11 or 3.12; CI runs 3.11. Real runs need Linux with CUDA 12.1 or later.
macOS works for the analysis environment and the smoke cell. On Windows, use WSL2 with
Ubuntu.

Always install the package editable (`pip install -e`). `bodhi_lora/paths.py` finds
`configs/` and `results/` next to the package folder, so a copy in site-packages finds
neither.

## 1. Clone

```bash
git clone https://github.com/PeterLi-jpg/bohdi-lora.git
cd bohdi-lora
git config core.hooksPath hooks     # activates the repo hooks, see CONTRIBUTING.md
```

## 2. Analysis environment (CPU only)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m bodhi_lora fetch healthbench    # the public HealthBench files, checked against their pins
make analyze numbers figures
```

The pins are in `pyproject.toml`. This is enough to rebuild every table, macro and figure
of the paper from the tracked results. `make paper` also needs `tectonic`. `make test`
passes here, and skips the tests that need the training stack (torch, transformers or
`bodhi-llm`); CI runs them with `'.[train,dev]'`. `python -m bodhi_lora run`, a dry run
too, builds the protocols and so needs `bodhi-llm` from the train extra; here it stops and
names that install.

## 3. Training environment

```bash
python3 -m venv .venv
source .venv/bin/activate
bash setup.sh
```

`setup.sh` warns if you install into a shared or base Python, runs
`pip install -e '.[train,gpu,dev]'`, runs `python -m pip check` and downloads HealthBench
into `data/raw/`. On Linux it also builds the inference environment (section 4). On
macOS, pip prints `Ignoring autoawq` and `Ignoring bitsandbytes`: the `gpu` extra is
Linux-only, so this is expected.

## 4. Inference environment

The runner (`python -m bodhi_lora run`) runs generation, grading and eval with
`.venv-infer/bin/python`, a second environment for vLLM 0.9.2, which cannot import with
the transformers version that training pins. `requirements-infer.txt` pins it; the
environment behind the posted cells was not frozen (KNOWN_ISSUES.md). If your
environments live elsewhere, change `python.train` and `python.infer` in
`configs/project.yaml`.

## 5. Hugging Face access

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
| `hf-internal-testing/tiny-random-LlamaForCausalLM` | smoke cell (`make smoke`), no token needed |

## 6. Check the install

```bash
bodhi-lora list cells     # the same as python -m bodhi_lora list cells
make check
make test
```

`list` loads and checks every record in `configs/`. In the training environment,
`make smoke` then runs the smoke cell on the CPU, every stage, in about two minutes.
