# Rebuttal cells: posted and generated values

Two files, with different jobs:

- `generated_values.json` is the source for the paper. `make cells`
  (`rebuttal/analyses/cell_values.py`) computes it from the raw per-seed outputs of every
  cell, synced from the GPU host on 2026-10-08, and `make numbers` reads it. It also lists
  each posted value that differs from the raw outputs (`mismatches`).
- `posted_values.json` is the record of what was posted. It lists every number the authors
  posted on OpenReview about the rebuttal cells, plus the posted analyses of the submitted
  cell, as a hand transcription of `rebuttal/responses/posted/` with a source line for each
  value. No script generated these numbers. Do not overwrite it; `make numbers` reads it
  only for cell metadata (names, families, set-aside flags), for posted numbers the paper
  quotes as posted, and as the fallback for a cell that has no generated values.

The sync found one posted value that the raw outputs do not give: MedQuAD Base hedging
quality, posted as 1.46 in Table 1 and computed as 1.4548 (withdrawals W21). Every other
posted value of the added cells matches at the posted precision.

## Cells

| id | Benchmark | Base model (HF repo) | Status |
|---|---|---|---|
| `healthbench__medgemma-27b` | HealthBench-Hard | `google/medgemma-27b-text-it` | submitted cell |
| `healthbench__mistral-small-24b` | HealthBench-Hard | `mistralai/Mistral-Small-24B-Instruct-2501` | added |
| `healthbench__med42-8b` | HealthBench-Hard | `m42-health/Llama3-Med42-8B` | added, same family as the evaluator |
| `healthbench__biomistral-7b` | HealthBench-Hard | `BioMistral/BioMistral-7B` | added, null result |
| `chatdoctor__mistral-small-24b` | ChatDoctor | `mistralai/Mistral-Small-24B-Instruct-2501` | added |
| `chatdoctor__med42-8b` | ChatDoctor | `m42-health/Llama3-Med42-8B` | added, same family as the evaluator |
| `medquad__mistral-small-24b` | MedQuAD | `mistralai/Mistral-Small-24B-Instruct-2501` | added |
| `medqa__mistral-small-24b` | MedQA | `mistralai/Mistral-Small-24B-Instruct-2501` | added, set aside |

The HF repos come from `rebuttal/configs/*.yaml` and `rebuttal/launch/run_grid.sh`.
The posted text names the models only by short name.

## File shape

The shape follows the schema the stats emitter reads:

- `cells[].conditions` has the keys `base`, `base_cot`, `lora` and `lora_cot`. Each maps
  a metric name to `{"mean": ..., "sd": ...}`. The metric names are `active_inquiry_pct`,
  `context_seeking`, `scope_bounding`, `hedging_quality`, `red_flag`, `uncertainty_ack`,
  `specificity` and `aggregate`.
- A metric that was never posted is left out. An empty condition means nothing was posted
  for it.
- `sd` is the cross-seed SD as posted, or `null` when the post gave none.
- `teacher_active_inquiry_pct` is the same quantity as `base_cot.active_inquiry_pct`. The
  teacher is the base model under the CoT protocol. The posts call it "teacher" and
  "wrapper".
- `filter_clearance_pct` is the share of generated traces that passed the filter, from
  one trace-generation run per cell.

The file adds three things to that schema:

- `cells[].notes`: rounding, conflicts between posts, and gaps for that cell.
- `analyses`: posted numbers that are not per-cell condition means. These are the
  discrimination index, format leakage, second-person questions, the MedQA preamble run,
  the paired tests on the aggregate drop, counts across cells (for example "four of
  seven"), cost, sample sizes and the recipe settings as posted. Each entry has `values`
  and `sources`. Counts written as words in the post ("four of seven") are stored as
  numbers.
- `analyses_note`: one line that says what `analyses` holds.

## How it was transcribed

1. Two independent passes went to temporary files outside the repo. Pass A was typed by
   hand from the posted replies. Pass B parsed the markdown tables and pulled the prose
   numbers by regular expression, and it recorded the line each value came from.
2. A deep diff of the two passes gave two differences. Both were resolved against the text:
   - Pass B missed two source lines for the BioMistral cell
     (`HomM_2_followup_part1.md:7` and `:9`). Both lines restate BioMistral values, so
     they stay.
   - The MedQA set-aside reason now uses the exact posted words.
3. Pass B also found three places where the posts disagree with each other. Each one is
   in the cell's `notes`:
   - Mistral-Small-24B on HealthBench: clearance 62.1 in Table 2, and 62% in two
     earlier posts (rounding).
   - BioMistral-7B: clearance 20.4 in Table 2, and 20% in two earlier posts (rounding).
   - MedQuAD with Mistral-Small-24B: base hedging is 1.46 in Table 1 and 1.45 in two
     earlier posts. A draft gives the unrounded value as 1.455.

Checks run on the committed file:

- Every number appears on at least one of its cited source lines.
- Every count and range posted across cells recomputes from the cell values: the drop
  ranges 0.013 to 0.015 and 0.016 to 0.086, "lower in three, level in two, higher in
  one", "wrapper below base in all six", "LoRA+CoT below LoRA in five of six", "four of
  seven" against the teacher, and the spreads 13.8 and 11.0 in the withdrawn convergence
  claim.
- The submitted cell matches `results_modal/cross_seed_aggregate.json` (8 values).
- The discrimination index and the leakage numbers match `rebuttal/results/mimicry_split.json`
  and `interference_mechanism.json` (41 values, 0 mismatches). Both files are now tracked.
- The Base and LoRA aggregate means in Table 3 match the 4-decimal table in
  `rebuttal/analyses/FINDINGS_quality_drop.md`.

These checks show that the transcription matches the posts and that the posts agree with
each other. They do not show that the posted numbers for the added cells are correct.
Only the raw outputs can show that, and `make cells` now does: one mismatch (W21).

## Where the raw outputs live

The submitted cell's raw outputs are in this repo under `results_modal/seed_*/`.

The raw outputs of the added cells were copied read-only on 2026-10-08 from the shared GPU
host (audit finding PROV-1):

```
mit-critical-data-1:/data/PROJECTS/bohdi/results_rebuttal/<bench>__<tag>/
```

into `results_rebuttal/` (194 JSON files, 36 adapter configs, 108 trainer states). Git
tracks each cell's `eval_ids.json` and every `adapter_config.json` and
`trainer_state.json`. The per-seed eval JSON holds full responses (about 370 MB), so it is
gitignored and stays in the local copy and on the host. The host logs are in `logs/host/`
(113 files, 28 MB, gitignored).

`rebuttal/launch/run_cell.sh:39-40` sets `<bench>` to `healthbench`, `chatdoctor`,
`medquad` or `medqa`. It sets `<tag>` to the HF repo with `/`, `:` and `.` replaced by
`_`, for example `healthbench__mistralai_Mistral-Small-24B-Instruct-2501`. The names on the
host match. Each cell directory holds these files:

- `sft/raw_traces.jsonl`, `sft/graded.jsonl`, `sft/train.jsonl` and `sft/val.jsonl`:
  trace generation and filtering.
- `seed_<N>/eval/{base_no_wrapper,base_bodhi,lora_no_wrapper,lora_bodhi}.json`:
  responses and rubric scores. These four files map to `base`, `base_cot`, `lora` and
  `lora_cot`.
- `seed_<N>/eval/epistemic_scores.json`: the seven-dimension grades.
- `multi_seed_summary.json`: the cross-seed summary. Do not use the copies from the host.
  They were written before commit 55f6c7c, so each one lists its seeds as `["eval"]` (a
  wrong glob) and holds one seed only. They are gitignored, and `cell_values.py`
  recomputes everything from the per-seed files.

The host also holds three folders that are not cells of the paper:
`medqa__google_medgemma-27b-text-it` and `medquad__google_medgemma-27b-text-it` hold only
`eval_ids.json` (trace generation failed on 2026-07-27: the vLLM server did not start), and
`medqa__BioMistral_BioMistral-7B` is a smoke run (one seed, 40 traces, 3 kept, 3 optimizer
steps, 10 eval prompts). `rebuttal/results/cell_recipe.json` records their status.

The filter and eval logs (`filter_<bench>_<tag>.log`, `eval_<bench>_<tag>_s<seed>.log`)
are in `logs/host/`. `rebuttal/analyses/cell_recipe.py` (`make cells`) reads the tau, the
trace counts and the train/val split of each cell from the filter log, checks them against
the trainer states, and writes `rebuttal/results/cell_recipe.json` (PROV-3). Every cell
used tau = 0.4, and no log shows the 22nd-percentile re-filter.

## How to check this file against generated values

`rebuttal/analyses/cell_values.py` (`make cells`) does the check. For each cell it reads
the per-seed eval files, computes every posted metric with the definitions of
`scripts/paper_stats.py`, and writes `generated_values.json` in the same shape. Then it
lists each posted value that differs from the generated one at the posted decimals.

- The metrics are per-seed means over scored responses (epistemic metrics) or over all
  completions (`aggregate`), then the mean and the sample SD over seeds. The posts do not
  say which SD they used. The submitted cell uses the sample SD
  (`scripts/aggregate_cross_seed.py:56`).
- The folder of each cell is `results_rebuttal/<bench>__<tag>/`, as above. To read a cell
  from another folder, pass `--cell-dir <cell id>=<path>`.
- Without the local copy of `results_rebuttal/`, only the submitted cell has data. Its 8
  posted values match `results_modal/`. The script then names the missing cells and writes
  no file, so the tracked `generated_values.json` stays as it is.
- A seed whose epistemic grading failed has no value for the seven dimensions. The MedQA
  cell has one: seed 42's grading found none of its prompts, so its seven-dimension means
  are null and `per_seed` shows the four seeds that have them.
- Each cell also gets `lora_askers_minus_non_askers`: the aggregate score of LoRA responses
  that ask minus those that do not, pooled over seeds (Appendix J).

The filter clearance comes from the filter log, through `cell_recipe.json`. The posted
clearances match the logs (`Kept N/M`).

When a generated value differs from a posted value by more than rounding, the generated
value is the record. Add the difference to `rebuttal/withdrawals.md` as a posted number
that was wrong (W21 is the one the sync found). Keep `posted_values.json` after that. It
stays as the record of what was posted.

## Not in this file

The drafts in `rebuttal/responses/drafts/` hold numbers that were never posted. These are
the full four-condition tables for five metrics in `reviewer_HomM_tables_companion.md`,
and the MedQA aggregate of 0.665 to 0.683 in `reviewer_HomM_followup.md` and `_5k.md`.
They can be a second check on the generated values. They are not part of the public
record.
