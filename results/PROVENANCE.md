# Provenance of the results root

On 2026-10-08 every result file moved into this one folder (docs/ARCHITECTURE.md section
6), by `scripts/migrate_results.py`. `provenance.json` has one entry per old file: its old
path, its new paths, its sha256 and size, whether git tracked it before and after, and a
note. This file says the same in prose.

## How the move was checked

Before the move, the old locations (the table below) held 635 files. The script hashed each file
before it moved and again where it landed. All 635 hashes match: 546 files moved (one of
them to five places), 88 went to the archive and one README was merged into another file.
No file was deleted. Tracked files moved with `git mv`, so `git log --follow` keeps their
history; ignored files moved on disk.

The archive is `../_archive/foundation_2026-10-08/` next to the repository, outside git.
The local copy of the GPU host's folder, `../_remote_sync/results_rebuttal/`, stayed where
it is. Each of its 338 files has the sha256 of a file that moved here.

## Where each old location went

| Old location | New location | What it held |
|---|---|---|
| `02ac9a3:results_modal/` | `healthbench__medgemma-27b-text__bodhi__2026-05/` | the 2026-05 cell: eval and grade files per seed, physician grading sheets, a cross-seed summary |
| `02ac9a3:results_rebuttal/` (tracked part) and its ignored local copy | the `*__2026-07/` cells | the 2026-07 cells: eval ids, eval and grade files, adapter configs and trainer states |
| `results/seed_42/`, `results/_rescue/` (ignored) | `healthbench__medgemma-27b-text__bodhi__2026-04/` | the 2026-04 pilot: eval files, teacher traces, its train/val split |
| `checkpoints/seed_42/` (ignored) | the pilot's `seed_42/train/` | the pilot adapter and its epoch checkpoints |
| `02ac9a3:rebuttal/results/` | the 2026-05 cell's `analysis/`, and `cross_cell/` | the analyses posted in the discussion |
| `02ac9a3:data/rebuttal_cells/` | `cross_cell/generated_values.json`, `rebuttal/posted_values.json`, `rebuttal/README.md` | the added-cell values, computed and as posted |
| `02ac9a3:data/derived/` | the 2026-05 cell's `seed_7/filter/`, the pilot's `analysis/` | two small score copies the paper reads |
| `02ac9a3:data/irr_instructions/` | the 2026-05 cell's `validation/instructions/` | the rater instructions |
| `data/sft_qwen_graded/` (ignored) | the 2026-05 cell's `seed_7/filter/train.jsonl` | the seed-7 train split |
| `logs/host/` (ignored) | each cell's `logs/` | the GPU host's stage logs |
| `02ac9a3:data/raw/hard_200_sample_ids.json` | `shared/inputs/eval_ids.json` of the three 2026-07 HealthBench cells, the Gemma-2-9B run and the pilot | the fixed eval set they used, one copy each |

Inside a cell, an eval file of condition `<w>_no_wrapper` is now `eval/<w>.json`, and one
of `<w>_<protocol>` is `eval/<w>+<protocol>.json` (for example `base_bodhi.json` became
`eval/base+bodhi.json`). The file contents did not change, so the condition names inside
them stay the old ones. Each `epistemic_scores.json` is now
`grade/epistemic__<evaluator>.json`, with the evaluator's model record id
(`llama-3.1-8b` for the 2026-05 cell, `llama-3.1-8b-nous` for the 2026-07 cells). A 2026-07
adapter's files from `ckpt/best/` are now in `seed_<n>/train/`, and its epoch checkpoints
in `seed_<n>/train/checkpoint-<k>/`.

## What went to the archive

- The host's `multi_seed_summary.json` of each 2026-07 cell. It lists its seeds as
  `["eval"]` (a wrong glob) and holds one seed; the analysis recomputes from the per-seed
  files.
- The pilot's figures and its three analysis outputs (`hedge_rate.json`, `rubric_diff.json`,
  `ushape.json`). No table uses them.
- The adapters of smoke runs in `checkpoints/` (`best/`, `checkpoint-1/`, `qwen_test/`), a
  copy of the pilot's trainer state, and the outputs of the removed smoke script in
  `data/sft/` and `eval/smoke/`.
- Four host logs that belong to several cells or none: the grid driver, the two install
  logs and one re-grade driver log.

## What the analysis replaces

These moved files are analysis outputs, kept byte for byte: `analysis/cross_seed_aggregate.*`,
`analysis/withdrawn/mimicry_split.json`, `analysis/withdrawn/interference_mechanism.json` and
`analysis/withdrawn/second_person_questions.json` of the 2026-05 cell (in `withdrawn/` since
2026-10-11, because no paper number may come from them), the pilot's
`analysis/qwen_grader_scores.json`, the 2026-05 cell's `seed_7/filter/filter_scores.json`,
and `cross_cell/{quality_drop,cell_recipe,generated_values}.json`. Since 2026-10-10 the
paper's numbers come from `make analyze` and `make numbers` (`python -m bodhi_lora
report`), which reproduce these values (ARCHITECTURE section 10.3). The analysis still
reads `filter_scores.json` (the tau table); the others stay as the record of the removed
scripts and of the posted values.

The scripts that wrote these files now read the cell folders. On 2026-10-09, `make
rebuttal` wrote the three 2026-05 analyses byte for byte, and `make cells` wrote
`quality_drop.json` byte for byte. `generated_values.json` and `cell_recipe.json` came out
with the same values; only the folder and log paths they record differ, and
`n_eval_ids` is 200 for the three HealthBench cells, which now hold their eval ids. With
those two files rebuilt, the old `make numbers` wrote the same `numbers.tex` and tables;
`numbers.json` differed only in the input hashes and the recipe paths it copied.

## Facts taken out of the old files

The filter of a 2026-07 cell ran on the GPU host, and its graded and kept traces stayed
there. The tau, the trace counts and the train/val split sizes that its filter log gives
(through `cross_cell/cell_recipe.json`) are in the cell's `shared/filter/manifest.json`.
Every stage folder of a recorded cell has a `manifest.json` that lists the files the move
kept, with their sha256, and says `recorded`, or `not_kept` when the files stayed on the
host or were lost. Each recorded cell also has `cell.yaml`, a copy of its record in
`configs/cells/`.

## Runs with few or no outputs

- `medqa-open-preamble__medgemma-27b-text__bodhi__2026-07` and
  `medquad__medgemma-27b-text__bodhi__2026-07`: trace generation failed on 2026-07-27 (the
  vLLM server did not start). Only `eval_ids.json` and the logs exist.
- `medqa-open-preamble__biomistral-7b__bodhi__2026-07`: a one-seed smoke run on the host
  (40 traces, 3 kept, 3 optimizer steps, 10 eval prompts).
- `healthbench__gemma-2-9b__bodhi__2026-07`: stopped in trace generation on 2026-07-27
  while its vLLM server was starting. It left logs only.
- The 2026-05 cell: its teacher traces, filter outputs and adapters were on TPU VMs and a
  cloud bucket that returned 404 on 2026-10-08, except the seed-7 train split; the adapters
  are treated as not retained.

## Tracked and ignored

The rule in `.gitignore` depends on the kind of file. Tracked: manifests, cell records,
id lists, score copies, adapter configs, trainer states, analysis and validation files.
Ignored: traces and the other JSONL, adapter weights and checkpoint state, raw eval and
grade JSON, logs. One exception waits on the first author (open decision O2): the 2026-05
cell's raw eval and grade JSON stay tracked, as they were before the move.
