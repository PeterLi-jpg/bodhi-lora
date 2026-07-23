# Reproducibility — rebuttal generality experiments (GPU)

This branch (`rebuttal/generality-experiments`, **not merged to `main`**) is a single,
clean, GPU-only pipeline for the NeurIPS rebuttal: it reproduces the paper's result and
extends it to additional base models and benchmarks to address the generality critique.
All TPU / MaxText / tunix / Modal code from `main` has been removed here.

## 1. Hardware / environment

- 1x NVIDIA H100 80GB per run (queued; the driver parallelizes seeds across whatever
  GPUs are idle and never touches a GPU another job is using).
- CUDA 12.x, Python 3.10-3.12.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch 2.7, transformers 4.57.6, peft, trl, vllm 0.9.x, bodhi-llm
# gated models need HF access: MedGemma-27B-text-it, Mistral-Small-24B-Instruct,
# BioMistral-7B, Qwen2.5-14B-Instruct (filter), Llama-3.1-8B-Instruct (eval).
# A cached token at ~/.cache/huggingface/token (or HF_TOKEN in env) is used automatically.
python -c "import torch,vllm,transformers,peft,trl,bodhi; print('env OK', torch.cuda.is_available())"
```

## 2. Pipeline (same four steps as the paper, GPU end-to-end)

1. **Generate** BODHI CoT traces over the training prompts (`scripts/generate_traces.py`, vLLM).
2. **Grade + filter** with Qwen2.5-14B (`scripts/filter_traces.py`, keep normalized score >= 0.4).
3. **LoRA SFT** (`scripts/train_lora.py`) — QLoRA for 24-27B, full-precision LoRA for 7-8B.
   FlashAttention-2/SDPA, TF32, fused AdamW, batch-longest padding.
4. **Evaluate** the 2x2 design (Base, Base+CoT, LoRA, LoRA+CoT): HealthBench rubric score
   (`scripts/eval_healthbench.py`, Llama-3.1-8B grader) + the 7 epistemic dimensions
   (`scripts/eval_epistemic.py`, benchmark-agnostic). Aggregate across seeds
   (`scripts/aggregate_seeds.py`).

The asymmetric grading design is preserved on every benchmark: filter = Qwen-14B,
evaluator = Llama-3.1-8B (different families).

## 3. Generality grid

| Base model | Family | Config |
|---|---|---|
| google/medgemma-27b-text-it | Gemma (clinical) | `configs/lora_medgemma27b_qlora.yaml` |
| mistralai/Mistral-Small-24B-Instruct-2501 | Mistral (general) | `rebuttal/configs/lora_mistral_small_24b_qlora.yaml` |
| BioMistral/BioMistral-7B | Mistral (clinical) | `rebuttal/configs/lora_biomistral7b.yaml` |

Benchmarks: `healthbench` (HealthBench-Hard), `medqa` (MedQA-USMLE, open-ended, clinician-facing),
`medquad` (NIH consumer-health, patient-facing). MedQA/MedQuAD are converted to HealthBench-format
JSONL with rubrics synthesized from ground truth by `scripts/build_benchmark_jsonl.py`, so the
pipeline runs on them unchanged. Seeds: 42 7 13 99 101.

## 4. Run a cell

```bash
# Mistral-Small-24B on HealthBench, 5 seeds:
MODEL=mistralai/Mistral-Small-24B-Instruct-2501 \
CONFIG=rebuttal/configs/lora_mistral_small_24b_qlora.yaml \
BENCH=healthbench SEEDS="42 7 13 99 101" \
bash rebuttal/launch/run_cell.sh

# BioMistral-7B on MedQA (open-ended):
MODEL=BioMistral/BioMistral-7B \
CONFIG=rebuttal/configs/lora_biomistral7b.yaml \
BENCH=medqa SEEDS="42 7 13 99 101" \
bash rebuttal/launch/run_cell.sh
```

`run_cell.sh` builds the benchmark JSONL if needed, generates + filters once, trains + evaluates
each seed on an idle GPU, and aggregates. Results land in `results_rebuttal/<bench>__<model>/`.

Wall-clock (one queued H100): ~8-13h per 24-27B cell, ~5-6h per 7B cell. Full 3x3 grid ~3-4 days
of compute (longer on a shared queue).

## 5. First-run verification (do NOT trust a full run until these pass)

- **Completion-only masking on Mistral:** `train_lora.py` prints the response template;
  `DataCollatorForCompletionOnlyLM` warns if `[/INST]` isn't found in a tokenized example
  (that would mean loss on prompt tokens). The Mistral configs set `data.response_template: "[/INST]"`.
- **BioMistral chat template:** confirm `AutoTokenizer.apply_chat_template` works (it inherits
  Mistral-Instruct's); if not, set `chat_template` from `mistralai/Mistral-7B-Instruct-v0.1`.
- **New-benchmark schema:** `build_benchmark_jsonl.py` fails loudly if 0 rows are written
  (the HF mirror uses different field names) — adjust `medqa_row` / `medquad_row`.
- **vLLM/torch:** if `pip` can't resolve `vllm` against the pinned torch 2.7, pin the exact vllm
  build for the box's CUDA.

## 6. No-GPU analyses (run against committed `results_modal/`)

```bash
python rebuttal/analyses/mimicry_split.py          # calibration vs mimicry (meta-review W4)
python rebuttal/analyses/interference_mechanism.py # CoT/LoRA interference cause (Reviewer LxMF W4)
```

These reproduce the two rebuttal findings that need no new runs. See `rebuttal/COVERAGE.md`
for the full reviewer-concern -> response map and `rebuttal/PLAN.md` for the run plan.
