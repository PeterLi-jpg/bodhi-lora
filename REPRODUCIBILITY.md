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
starting; it left no folder on the host). `results/cross_cell/cell_recipe.json` records
each one's status.

## 2. Per-cell training configs

### Submitted cell: MedGemma-27B

- Trained with tunix and qwix (JAX LoRA) on 8x TPU v6e. Adapters were exported to PEFT
  format. Eval ran with vLLM on one A100-80GB on Modal. The launchers are frozen in
  `legacy/`.
- LoRA rank 8 and alpha 16, from repo evidence: the export step in the tunix launcher
  passes `--r 8 --alpha 16`, and the only committed adapter (commit `607440e`, branch
  `results-temp`) has r=8 and alpha=16. **This cannot be confirmed** against the
  `adapter_config.json` of the five submitted adapters: the GCS bucket that held them
  returned 404 on 2026-10-08, and the adapters are treated as not retained (D19,
  section 7). The early committed adapter comes from an HF-path run on 838 traces (the
  2026-04 pilot cell), not from the evaluated adapters.
- The paper's hyperparameter table for this cell lists the full recipe that ran.
- `02ac9a3:configs/lora_medgemma27b.yaml` and `02ac9a3:configs/lora_medgemma27b_qlora.yaml`
  were configs for the GPU path (r=16, alpha=32). They did not train the submitted
  adapters. The cell record `configs/cells/healthbench__medgemma-27b-text__bodhi__2026-05.yaml`
  holds the recipe that did, as far as the repo shows it.

### Added cells

All three recipes use LoRA r=16, alpha=32, dropout 0.05 on `q,k,v,o,gate,up,down_proj`,
3 epochs, effective batch 16, learning rate 1e-4 with a cosine schedule and 5% warmup,
and maximum sequence length 4,096. The synced `adapter_config.json` and
`trainer_state.json` of all 35 adapters of the seven cells confirm rank, alpha, dropout,
target modules, epochs and the effective batch (`results/cross_cell/cell_recipe.json`). The
evaluated adapter (the host's `ckpt/best/`, now `results/*/seed_*/train/`) is the epoch
checkpoint with the lowest validation loss
(`load_best_model_at_end` in `bodhi_lora/pipeline/train.py`): epoch 1, 2 or 3, depending on the
cell and seed.

| Cell records (train block) | Base model | Training precision | Eval precision | Batch per device x accumulation |
|---|---|---|---|---|
| `configs/cells/*__mistral-small-24b__bodhi__2026-07.yaml` | `mistralai/Mistral-Small-24B-Instruct-2501` | 4-bit NF4 QLoRA, bf16 compute | bf16 base | 2 x 8 |
| `configs/cells/*__med42-8b__bodhi__2026-07.yaml` | `m42-health/Llama3-Med42-8B` | bf16 | bf16 | 4 x 4 |
| `configs/cells/healthbench__biomistral-7b__bodhi__2026-07.yaml` | `BioMistral/BioMistral-7B` | bf16 | bf16 | 4 x 4 |

The recipes were the files `61dada9:rebuttal/configs/lora_mistral_small_24b_qlora.yaml`,
`61dada9:rebuttal/configs/lora_med42_8b.yaml` and `61dada9:rebuttal/configs/lora_biomistral7b.yaml`;
the cell records copy every value, and a check on 2026-10-08 compared them field by field.

The Mistral-Small-24B adapter is trained on a 4-bit base and served on a bf16 base.
The MedGemma cell has no such mismatch.

## 3. Settings that differ between the cell sets

`python -m bodhi_lora list cells --settings` prints every setting that differs between
the cell records. The planned column is the proposal of EXPERIMENTS.md E13; the first
author fixes it when he adopts docs/ANALYSIS_PLAN.md.

| Setting | Submitted cell (2026-05) | Added cells (2026-07) | Planned cells |
|---|---|---|---|
| Trainer | tunix on TPU v6e | `bodhi_lora/pipeline/train.py` (transformers, peft, trl) on H100 | the same, through `python -m bodhi_lora run` |
| LoRA | r=8, alpha=16 (from the code; the saved adapters cannot be checked) | r=16, alpha=32 (from the saved adapter configs) | r=16, alpha=32, seven projections; QLoRA 4-bit for 24-27B bases |
| Eval server | vLLM on Modal A100-80GB, `max_model_len` 4096, `--max-num-seqs` 32 | vLLM 0.9 subprocess on H100, `max_model_len` 8192, `--max-num-seqs` 64; the 24B base served in bf16 although trained in 4-bit | vLLM 0.9 subprocess, `max_model_len` 8192; the base served as it was trained |
| Eval grader | `meta-llama/Llama-3.1-8B-Instruct` | `NousResearch/Meta-Llama-3.1-8B-Instruct`, a mirror of the same model | `meta-llama/Llama-3.1-8B-Instruct`, pinned |
| HealthBench training pool | the 4,000 prompts outside Hard | the same | the 3,321 that remain after the physician-labelled holdout |
| Eval prompts | a new 200-prompt draw from HealthBench-Hard per seed | HealthBench: the fixed `02ac9a3:data/raw/hard_200_sample_ids.json` for every seed (now `results/*/shared/inputs/eval_ids.json`); other benchmarks: the first 200 prompts of the built JSONL | a separate random draw of 200 from the eval pool per seed |
| Filter threshold tau | 0.4 | 0.4 in every cell, from the filter logs (`results/cross_cell/cell_recipe.json`). The launcher's rule would re-filter a non-HealthBench cell at the 22nd percentile of the scores when its `train.jsonl` holds under 40% of the raw traces (`1262448:rebuttal/launch/run_cell.sh:160-200`); no cell met it (`refilter_rule_met` in `cell_recipe.json`), and no log shows a second cut | 0.4, the `fixed` tau rule |
| What a seed changes | eval draw, train/val split, LoRA init, data order | LoRA init and data order. One filter pass per cell, and fixed eval prompts | eval draw, train/val split, LoRA init, data order |
| Teacher traces | generated once with greedy decoding | generated once per cell with greedy decoding | the same (E23 tests sampled trace sets) |

## 4. Environment

- Training: `requirements.txt` in a `.venv` (torch 2.7.0, transformers 4.57.6,
  peft 0.19.1, trl 0.11.4, bodhi-llm 0.1.4). `bash setup.sh` installs it and runs
  `pip check`.
- Inference: a second env, `.venv-infer`, for vLLM 0.9.2, which needs an older
  transformers. The runner calls `.venv-infer/bin/python` for generation, grading
  and eval. `requirements-infer.txt` pins it. The env that produced the posted cells
  was not frozen (KNOWN_ISSUES).
- Hardware for a cell: one H100 80GB per seed. Measured on the shared host, a 2026-07
  cell took 4.1 to 15.6 H100-hours: 11.2 for Mistral-Small-24B on HealthBench, 4.5 for
  Med42-8B (docs/RUN_PLAN.md section 3.1 gives every cell and stage).
- Gated models need Hugging Face access: MedGemma-27B-text-it, Mistral-Small-24B,
  Llama-3.1-8B (or the mirror above), Qwen2.5-14B-Instruct. Set `HF_TOKEN`, or keep a
  token in `~/.cache/huggingface/token`.

## 5. Run

Every cell is a record in `configs/cells/`, and `python -m bodhi_lora run` runs it
(`bodhi_lora/pipeline/runner.py`). The runner writes the cell's pools and per-seed eval
ids, then runs the stages in order: the contamination probe, traces, filter (grading),
the leakage gate, and per seed the filter split, training, the eval of each condition,
the grading (the evaluator, then each judge) and, on the seeds in `bench.seeds`, the
timing bench. A stage whose manifest says complete is skipped, so a run resumes where it
stopped. A request too long for a server's context is listed in the stage's manifest
and skipped, since a rerun would fail the same way.

```bash
python -m bodhi_lora run <cell_id> --gpus 0,1,2,3,4              # one seed per GPU
python -m bodhi_lora run <cell_id> --dry-run                     # print the GPU commands; CPU stub for the rest
python -m bodhi_lora run <cell_id> --stages grade --gpus 0       # judge re-grades only, also on a recorded cell
python -m bodhi_lora run <cell_id> --stages grade --judges gemma-2-27b --gpus 0   # one judge of the record
python -m bodhi_lora run <cell_id> --stages bench --seeds 42 --gpus 0   # timing on an idle GPU
python -m bodhi_lora analyze <cell_id> --extract                 # the response tables and statistics
```

No GPU run starts without the first author's go-ahead (D21). docs/RUN_PLAN.md gives the
order, the commands and the cost of every planned run, and docs/ANALYSIS_PLAN.md fixes
how each result is read before it exists.

The model stages run as subprocesses in the two environments of `configs/project.yaml`
(`.venv` for training, `.venv-infer` for vLLM), each pinned to one GPU with its own vLLM
port. The seven 2026-07 cells and the 2026-05 cell are records of past runs: the runner
refuses to run them again, so their outputs never mix with new ones, and a new run of the
same design is a new cell. `61dada9:rebuttal/launch/run_cell.sh` is the launcher the
2026-07 cells ran; their records hold its settings.

Check on the first run of a new base model:

- Masking: a model whose chat template has no assistant header to detect needs
  `chat_template.response_template` in its record (`[/INST]` for the Mistral models).
  The collator warns if the template is missing from an example, which would mean loss
  on prompt tokens.
- BioMistral chat template: `apply_chat_template` must work. If not, take the template
  from `mistralai/Mistral-7B-Instruct-v0.1`.
- New benchmarks: `bodhi_lora/data/loaders.py` stops if it builds 0 rows.

The submitted cell ran on TPU quota, a GCS bucket and a Modal account. Its launchers in
`legacy/` are kept to document what ran, not as a path to use: the project no longer
uses Modal, and the adapters are not retained (D19). A new run on MedGemma-27B is a new
cell (open decision O7).

## 6. What each stage writes

Each stage writes into `results/<cell id>/` (docs/ARCHITECTURE.md section 6), with a
`manifest.json` in every stage folder.

| Stage | Module | Output |
|---|---|---|
| Prepare | `bodhi_lora/pipeline/runner.py` | `shared/inputs/`: pools, eval ids per seed, leakage report |
| Traces | `bodhi_lora/pipeline/traces.py` | `results/<cell>/shared/traces/traces.jsonl`: prompt_id, messages, reply, protocol, calls, tokens |
| Filter | `bodhi_lora/pipeline/filter.py` | `results/<cell>/shared/filter/graded.jsonl` (a `grade` per trace), `kept.jsonl`, and the train/val split (shared or per seed) |
| Train | `bodhi_lora/pipeline/train.py` | `seed_<n>/train/`: adapter, `trainer_state.json`, `checkpoint-<k>/` |
| Eval | `bodhi_lora/pipeline/eval.py` | `results/<cell>/seed_<n>/eval/<condition>.json`: per-prompt reply, rubric score and results, parse failures, calls, tokens |
| Grade | `bodhi_lora/pipeline/grade.py` | `results/<cell>/seed_<n>/grade/epistemic__<evaluator>.json`, and `judge__<judge>.json` per judge |

The normalized rubric score is `(earned - negative) / (positive - negative)`, so it is
comparable across prompts with different penalty items.

## 7. What lives outside the repo

| Data | Where | How to get it |
|---|---|---|
| HealthBench JSONL | `data/raw/` (ignored) | `bash setup.sh` |
| ChatDoctor, MedQuAD, MedQA JSONL | `data/raw/` (ignored) | `bodhi_lora/data/loaders.py`, at the pinned dataset revisions; its manifest records the sha256 of each file |
| Filtered training traces of the submitted cell | the seed-7 split only, `results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_7/filter/train.jsonl` (ignored); the other seeds' traces were on a bucket that returned 404 (results/PROVENANCE.md) | regenerate with stages 1 and 2 |
| Submitted adapters and trainer states | were at `gs://bohdi-runs-tokyo-micron/prod-v39-tunix-27b-neurips/`; that bucket returned 404 on 2026-10-08, and the project has no buckets | not retained (D19); rerun as a new cell to evaluate MedGemma-27B again |
| Added-cell outputs | `/data/PROJECTS/bohdi/results_rebuttal/` on the shared H100 host; copied on 2026-10-08 and moved into the `results/*__2026-07/` cells (eval JSON gitignored), and to a local archive outside the repo | the sync record below |
| Added-cell host logs | `/data/PROJECTS/bohdi/logs/` on the host; copied on 2026-10-08 into each cell's `logs/` folder (gitignored) | the sync record below |
| Returned physician grades | the study coordinator | only de-identified grades go in the repo |

The copy of 2026-10-08 was complete: 194 of 194 JSON files, 36 of 36 adapter configs and
108 of 108 trainer states. It copied JSON only (eval files, summaries, eval ids, adapter
configs, trainer states) and the logs, and left raw generations and weights on the host:

```bash
rsync -av --prune-empty-dirs --include '*/' --include '*.json' --exclude '*' \
    <gpu-host>:/data/PROJECTS/bohdi/results_rebuttal/ <copy>/results_rebuttal/
rsync -av <gpu-host>:/data/PROJECTS/bohdi/logs/ <copy>/logs/host/
```

`scripts/migrate_results.py` then moved each file into its cell folder and checked its
sha256 where it landed (results/PROVENANCE.md). Git tracks the adapter configs, trainer
states, eval ids and the stage manifests; the per-seed eval JSON (full responses, about
370 MB) and the logs are gitignored. `make cells` reads both and writes the tracked
`results/cross_cell/generated_values.json`, `results/cross_cell/quality_drop.json` and
`results/cross_cell/cell_recipe.json`. New runs write the results root directly
(`python -m bodhi_lora run`), so they need no move.

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
| `Could not auto-detect response template` | non-standard chat template | set `chat_template.response_template` in the model record |
| `pip check` fails after install | a shared or base env | use a fresh venv |
