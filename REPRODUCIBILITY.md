# Reproducibility

How to rerun every cell in the paper, the config of each cell, how the two pipelines
differ, and what lives outside the repo. To regenerate the paper's numbers from
results that already exist, you need none of this: run `make numbers` (see
[README.md](README.md)).

## 1. The grid

The grid matches the paper's evaluation. Seeds are 42, 7, 13, 99 and 101 in every cell.

| Base model (family) | HealthBench-Hard | ChatDoctor | MedQuAD | MedQA-USMLE |
|---|---|---|---|---|
| MedGemma-27B (Gemma) | submitted cell | | | |
| Mistral-Small-24B (Mistral) | added | added | added | run, set aside |
| Med42-8B † (Llama-3) | added | added | | |
| BioMistral-7B (Mistral) | added, null result | | | |

† Med42-8B shares the evaluator's Llama-3 family, which breaks the paper's
grader-separation rule. Its cells are marked in every table and are not used in any
cross-model statement. `bodhi_lora/pipeline/grade.py` regrades them with a judge
outside the Llama and Qwen families.

MedQA-USMLE was run and set aside: it is a single-best-answer exam with complete
vignettes, so nothing is withheld and asking is not the target behavior. Its only run used
the old prompt preamble that told the model to ask (removed in 035d85f), and seed 42's
epistemic grading found none of its prompts. See [RESULTS.md](RESULTS.md).

Four runs on the GPU host did not finish and are not in the grid: MedGemma-27B on MedQA
and on MedQuAD (folders with `eval_ids.json` only; trace generation failed on 2026-07-27
because the vLLM server did not become healthy), BioMistral-7B on MedQA (a smoke run:
one seed, 40 traces, 3 kept, 3 optimizer steps, 10 eval prompts) and Gemma-2-9B on
HealthBench-Hard (stopped in trace generation on 2026-07-27 while its vLLM server was
starting; it left no folder). `rebuttal/results/cell_recipe.json` records each one's status.

## 2. Per-cell training configs

### Submitted cell: MedGemma-27B

- Trained with tunix and qwix (JAX LoRA) on 8x TPU v6e. Adapters were exported to PEFT
  format. Eval ran with vLLM on one A100-80GB on Modal. The launchers are frozen in
  `legacy/`.
- LoRA rank 8 and alpha 16, from repo evidence: the export step in the tunix launcher
  passes `--r 8 --alpha 16`, and the only committed adapter (commit `607440e`, branch
  `results-temp`) has r=8 and alpha=16. **This is not confirmed** against the
  `adapter_config.json` of the five submitted adapters: the GCS bucket that held them
  returned 404 on 2026-10-08 (section 7). The early committed adapter comes from an HF-path
  run, not from the evaluated adapters.
- The paper's hyperparameter table for this cell lists the full recipe that ran.
- `02ac9a3:configs/lora_medgemma27b.yaml` and `configs/lora_medgemma27b_qlora.yaml` are configs
  for the GPU path (r=16, alpha=32). They did not train the submitted adapters.

### Added cells

All three configs use LoRA r=16, alpha=32, dropout 0.05 on `q,k,v,o,gate,up,down_proj`,
3 epochs, effective batch 16, learning rate 1e-4 with a cosine schedule and 5% warmup,
and maximum sequence length 4,096. The synced `adapter_config.json` and
`trainer_state.json` of all 35 adapters of the seven cells confirm rank, alpha, dropout,
target modules, epochs and the effective batch (`rebuttal/results/cell_recipe.json`). The
evaluated adapter in `ckpt/best/` is the epoch checkpoint with the lowest validation loss
(`load_best_model_at_end` in `bodhi_lora/pipeline/train.py`): epoch 1, 2 or 3, depending on the
cell and seed.

| Config | Base model | Training precision | Eval precision | Batch per device x accumulation |
|---|---|---|---|---|
| `rebuttal/configs/lora_mistral_small_24b_qlora.yaml` | `mistralai/Mistral-Small-24B-Instruct-2501` | 4-bit NF4 QLoRA, bf16 compute | bf16 base | 2 x 8 |
| `rebuttal/configs/lora_med42_8b.yaml` | `m42-health/Llama3-Med42-8B` | bf16 | bf16 | 4 x 4 |
| `rebuttal/configs/lora_biomistral7b.yaml` | `BioMistral/BioMistral-7B` | bf16 | bf16 | 4 x 4 |

The Mistral-Small-24B adapter is trained on a 4-bit base and served on a bf16 base.
The MedGemma cell has no such mismatch.

## 3. Settings that differ between the submitted and added cells

| Setting | Submitted cell | Added cells |
|---|---|---|
| Trainer | tunix on TPU v6e | `bodhi_lora/pipeline/train.py` (transformers, peft, trl) on H100 |
| LoRA | r=8, alpha=16 (from the code; the saved adapters could not be checked) | r=16, alpha=32 (from the saved adapter configs) |
| Eval server | vLLM on Modal A100-80GB, `max_model_len` 4096, `--max-num-seqs` 32 | vLLM 0.9 subprocess on H100, `max_model_len` 8192, `--max-num-seqs` 64 |
| Eval grader | `meta-llama/Llama-3.1-8B-Instruct` | `NousResearch/Meta-Llama-3.1-8B-Instruct`, a mirror of the same model |
| Eval prompts | a new 200-prompt draw from HealthBench-Hard per seed | HealthBench: the fixed `data/raw/hard_200_sample_ids.json` for every seed; other benchmarks: the first 200 prompts of the built JSONL |
| Filter threshold tau | 0.4 | 0.4 in every cell, from the filter logs (`rebuttal/results/cell_recipe.json`). The launcher's rule would re-filter a non-HealthBench cell at the 22nd percentile of the scores when its `train.jsonl` holds under 40% of the raw traces (`1262448:rebuttal/launch/run_cell.sh:160-200`); no cell met it (`refilter_rule_met` in `cell_recipe.json`), and no log shows a second cut |
| What a seed changes | eval draw, train/val split, LoRA init, data order | LoRA init and data order. One filter pass per cell, and fixed eval prompts |
| Teacher traces | generated once with greedy decoding | generated once per cell with greedy decoding |

## 4. Environment

- Training: `requirements.txt` in a `.venv` (torch 2.7.0, transformers 4.57.6,
  peft 0.19.1, trl 0.11.4, bodhi-llm 0.1.4). `bash setup.sh` installs it and runs
  `pip check`.
- Inference: a second env, `.venv-infer`, for vLLM 0.9.2, which needs an older
  transformers. `run_cell.sh` calls `.venv-infer/bin/python` for generation, grading
  and eval. `requirements-infer.txt` pins it. The env that produced the posted cells
  was not frozen (KNOWN_ISSUES).
- Hardware for an added cell: one H100 80GB per seed. `rebuttal/PLAN.md` estimated 8 to
  13 hours per 24-27B cell and 5 to 6 hours per 7-8B cell on one card.
- Gated models need Hugging Face access: MedGemma-27B-text-it, Mistral-Small-24B,
  Llama-3.1-8B (or the mirror above), Qwen2.5-14B-Instruct. Set `HF_TOKEN`, or keep a
  token in `~/.cache/huggingface/token`.

## 5. Run

Run one added cell. The seven paper cells use these MODEL, CONFIG and BENCH values:

```bash
MODEL=mistralai/Mistral-Small-24B-Instruct-2501 \
CONFIG=rebuttal/configs/lora_mistral_small_24b_qlora.yaml \
BENCH=healthbench SEEDS="42 7 13 99 101" \
bash rebuttal/launch/run_cell.sh
```

| MODEL | CONFIG | BENCH |
|---|---|---|
| `mistralai/Mistral-Small-24B-Instruct-2501` | `rebuttal/configs/lora_mistral_small_24b_qlora.yaml` | `healthbench`, `chatdoctor`, `medquad`, `medqa` |
| `m42-health/Llama3-Med42-8B` | `rebuttal/configs/lora_med42_8b.yaml` | `healthbench`, `chatdoctor` |
| `BioMistral/BioMistral-7B` | `rebuttal/configs/lora_biomistral7b.yaml` | `healthbench` |

`run_cell.sh` builds the benchmark JSONL if needed, generates and filters once, trains
and evaluates each seed on a free GPU, and aggregates. Output goes to
`results_rebuttal/<bench>__<model>/`. `rebuttal/launch/run_grid.sh` lists every cell of
the table above. After each run, `bodhi_lora/manifest.py` records the resolved config, git commit
and dirty flag, library versions, model revisions and adapter hash.

Check on the first run of a new base model:

- Mistral masking: the Mistral configs set `data.response_template: "[/INST]"`. The
  collator warns if the template is missing from an example, which would mean loss on
  prompt tokens.
- BioMistral chat template: `apply_chat_template` must work. If not, take the template
  from `mistralai/Mistral-7B-Instruct-v0.1`.
- New benchmarks: `bodhi_lora/data/loaders.py` stops if it writes 0 rows.

The submitted cell needs TPU quota, a GCS bucket and a Modal account, and costs money.
Its launchers in `legacy/` are kept to document what ran, not as the path to use.

## 6. What each stage writes

| Stage | Script | Output |
|---|---|---|
| 1 Generate | `bodhi_lora/pipeline/traces.py` | `raw_traces.jsonl`: prompt_id, messages, response, tags, model |
| 2 Grade and filter | `bodhi_lora/pipeline/filter.py` | `train.jsonl`, `val.jsonl`, `graded.jsonl` with a `grade` field per trace |
| 3 Train | `bodhi_lora/pipeline/train.py` | adapter in `ckpt/best/`, `trainer_state.json` |
| 4 Eval | `bodhi_lora/pipeline/eval.py` | one JSON per condition: per-prompt score, rubric results, parse failures |
| 5 Epistemic grade | `bodhi_lora/grading/epistemic.py` | `epistemic_scores.json`: per-prompt scores on the seven behaviors |
| Aggregate | `scripts/aggregate_seeds.py` | `multi_seed_summary.json`. The copies from the GPU host predate 55f6c7c: they list the seeds as `["eval"]` (a wrong glob) and hold one seed. They are gitignored and unused; `make cells` recomputes from the per-seed files |

The normalized rubric score is `(earned - negative) / (positive - negative)`, so it is
comparable across prompts with different penalty items.

## 7. What lives outside the repo

| Data | Where | How to get it |
|---|---|---|
| HealthBench JSONL | `data/raw/` (ignored) | `bash setup.sh` |
| ChatDoctor, MedQuAD, MedQA JSONL | `data/raw/` (ignored) | `bodhi_lora/data/loaders.py`, at the pinned dataset revisions; its manifest records the sha256 of each file |
| Filtered training traces of the submitted cell | `data/sft_qwen_graded/` (ignored) | regenerate with stages 1 and 2 |
| Submitted adapters and trainer states | were at `gs://bohdi-runs-tokyo-micron/prod-v39-tunix-27b-neurips/`; that bucket returned 404 on 2026-10-08, and the project has no buckets | `legacy/modal_eval.py` copied each seed's `checkpoints/best/` and `train.jsonl`/`val.jsonl` into the Modal volume `bohdi-data-cache` (`/data/seed_<N>/`); that volume may still hold them (BLOCKED.md item 3) |
| Added-cell outputs | `/data/PROJECTS/bohdi/results_rebuttal/` on the shared H100 host; copied on 2026-10-08 to `results_rebuttal/` here and to a local archive outside the repo | the sync command below |
| Added-cell host logs | `/data/PROJECTS/bohdi/logs/` on the host; copied to `logs/host/` (gitignored) | the second command below |
| Returned physician grades | the study coordinator | only de-identified grades go in the repo |

Sync the added cells. This copies JSON only (eval files, summaries, eval ids, adapter
configs, trainer states) and leaves raw generations and weights on the host:

```bash
rsync -av --prune-empty-dirs --include '*/' --include '*.json' --exclude '*' \
    <gpu-host>:/data/PROJECTS/bohdi/results_rebuttal/ results_rebuttal/
```

The copy of 2026-10-08 was complete: 194 of 194 JSON files, 36 of 36 adapter configs and
108 of 108 trainer states. Git tracks the adapter configs, trainer states and eval ids; the
per-seed eval JSON (full responses, about 370 MB) is gitignored. Copy the logs too, then
run `make cells`, which reads both and writes the tracked `generated_values.json`,
`quality_drop.json` and `cell_recipe.json`:

```bash
mkdir -p logs/host
rsync -av <gpu-host>:/data/PROJECTS/bohdi/logs/ logs/host/
make cells
```

## 8. Determinism

Trace generation and eval use greedy decoding. Training seeds `random`, `numpy` and
`torch`, and passes the seed to the trainer. Results still change across GPU models,
CUDA or PyTorch versions, and GPU counts. To compare runs exactly, fix all three and
record them, which `bodhi_lora/manifest.py` does.

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `401 Unauthorized` from Hugging Face | no token, or no access to a gated model | accept the model terms, set `HF_TOKEN` |
| CUDA out of memory in training | 24-27B in bf16 | use the QLoRA config or an 80 GB card |
| `Could not auto-detect response template` | non-standard chat template | set `data.response_template` in the config |
| vLLM hangs at start on a GPU box | the Docker path pulled a TPU image | `run_cell.sh` sets `BODHI_VLLM_MODE=subprocess`; keep it |
| `pip check` fails after install | a shared or base env | use a fresh venv |
