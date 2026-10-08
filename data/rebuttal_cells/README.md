# Rebuttal cells: posted values

`posted_values.json` lists every number the authors posted on OpenReview about the
rebuttal cells, plus the posted analyses of the submitted cell. It is a hand
transcription of the posted replies in `rebuttal/responses/posted/`. No script
generated these numbers, and no raw output behind them is in this repo yet.

Use it for two things:

1. As the record of what was claimed in public, with a source line for each value.
2. As the target that generated values must match once the raw outputs are synced.

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
Only the raw outputs can show that.

## Where the raw outputs live

The submitted cell's raw outputs are in this repo under `results_modal/seed_*/`.

The raw outputs of the added cells are on the shared GPU host, and they have not been
synced (audit finding PROV-1):

```
mit-critical-data-1:/data/PROJECTS/bohdi/results_rebuttal/<bench>__<tag>/
```

`rebuttal/launch/run_cell.sh:39-40` sets `<bench>` to `healthbench`, `chatdoctor`,
`medquad` or `medqa`. It sets `<tag>` to the HF repo with `/`, `:` and `.` replaced by
`_`, for example `healthbench__mistralai_Mistral-Small-24B-Instruct-2501`. Confirm the
names on the host. Each cell directory should hold these files:

- `sft/raw_traces.jsonl`, `sft/graded.jsonl`, `sft/train.jsonl` and `sft/val.jsonl`:
  trace generation and filtering.
- `seed_<N>/eval/{base_no_wrapper,base_bodhi,lora_no_wrapper,lora_bodhi}.json`:
  responses and rubric scores. These four files map to `base`, `base_cot`, `lora` and
  `lora_cot`.
- `seed_<N>/eval/epistemic_scores.json`: the seven-dimension grades.
- `multi_seed_summary.json`: the cross-seed summary, if `aggregate_seeds.py` succeeded.
  Do not use the copies on the host. Before commit 55f6c7c, `aggregate_seeds.py` named
  every seed `eval`, so each summary holds one seed only. Run it again after the sync.

The filter and eval logs are under `logs/` on the host
(`filter_<bench>_<tag>.log`, `eval_<bench>_<tag>_s<seed>.log`). The filter log holds the
tau that each cell used, which the audit needs for PROV-3.

## How to check this file against generated values

`rebuttal/analyses/cell_values.py` (`make cells`) does the check. For each cell it reads
the per-seed eval files, computes every posted metric with the definitions of
`scripts/paper_stats.py`, and writes `generated_values.json` in the same shape. Then it
lists each posted value that differs from the generated one at the posted decimals.

- The metrics are per-seed means over scored responses (epistemic metrics) or over all
  completions (`aggregate`), then the mean and the sample SD over seeds. The posts do not
  say which SD they used. The submitted cell uses the sample SD
  (`scripts/aggregate_cross_seed.py:56`).
- The folder of each cell is `results_rebuttal/<bench>__<tag>/`, as above. If a name on
  the host differs, pass `--cell-dir <cell id>=<path>`.
- Until the sync, only the submitted cell has data. Its 8 posted values match
  `results_modal/`. The script names the 7 missing cells.

The script does not compute the filter clearance. Read it from the filter log
(`Kept N/M`), and confirm the denominator against the posted values.

When a generated value differs from a posted value by more than rounding, the generated
value is the record. Add the difference to `rebuttal/withdrawals.md` as a posted number
that was wrong. Keep `posted_values.json` after that. It stays as the record of what was
posted.

## Not in this file

The drafts in `rebuttal/responses/drafts/` hold numbers that were never posted. These are
the full four-condition tables for five metrics in `reviewer_HomM_tables_companion.md`,
and the MedQA aggregate of 0.665 to 0.683 in `reviewer_HomM_followup.md` and `_5k.md`.
They can be a second check on the generated values. They are not part of the public
record.
