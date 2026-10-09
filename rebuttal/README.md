# Rebuttal record: NeurIPS 2026, submission 32818

This folder holds the record of the NeurIPS 2026 discussion. The paper was rejected. The
cells we added during it are records in `configs/cells/` (`*__2026-07`), and their
outputs are in `results/`. Use this folder to see what we posted, what we
withdrew, and what we promised. The ledgers are the files that the resubmission must act on.

## The ledgers

| File | What it holds |
|---|---|
| `withdrawals.md` | Every claim that we withdrew or corrected in the discussion, plus posted claims that the audit or a later check found wrong. Each entry has the claim as made, the correction and the statement that the paper must carry. |
| `commitments.md` | Every promise in the posted replies (C1 to C33), with its source line and a status column for the paper lanes. |
| `posted_values.json` | Every number that we posted about the rebuttal cells, transcribed with source lines. It is the record of what was posted. |
| `../results/cross_cell/generated_values.json` | The same numbers computed from the raw outputs (`make cells`); the paper uses these. The section below lists the one posted value they do not match (W21). |

## Posted replies

`responses/posted/` holds the final version of each reply that we posted on OpenReview, in
the order the posting record gives:

| File | Thread | Opens with | File date |
|---|---|---|---|
| `FzpA_1_rebuttal.md` | FzpA | "Thank you for this careful and constructive review" | 2026-07-28 |
| `FzpA_2_followup.md` | FzpA | "The authors would like to thank you again for these questions" | 2026-08-03 |
| `LxMF_1_rebuttal.md` | LxMF | "Thank you for this thorough review and for the specific suggestions" | 2026-07-29 |
| `HomM_1_rebuttal.md` | HomM | "Thank you for this careful and generous review" | 2026-07-29 |
| `HomM_2_followup_part1.md` | HomM | "Thanks for coming back with specifics" | 2026-08-02 |
| `HomM_3_followup_part2.md` | HomM | "Continued from the previous comment." | 2026-08-02 |
| `HomM_4_consolidation.md` | HomM | "Thank you for the care you have given this throughout" | 2026-08-03 |

How we know these are the posted versions, and where we are not sure:

- **The HomM follow-up, parts 1 and 2.** A rendered copy (`HomM_response_final.docx`) says
  "Two comments, posted in order", with comment 1 at 4,864 characters and comment 2 at
  4,812. The two files have exactly those lengths, and every paragraph and table cell
  matches the copy. The .docx is not in the repo. It is archived in `BODHI/_archive/rebuttal/`
  on the first author's machine. Its text adds nothing to the two files.
- **LxMF and HomM rebuttals.** Git history has an earlier version of each (before commit
  0100b3c) that writes math as MathJax (`$\kappa = 0.35$`, `$n \approx 985$–$998$`). The
  versions here are the later edit (2026-07-29), which writes the same math as plain text
  ("kappa = 0.35", "LoRA minus Base"). Only the math notation differs. The numbers are the
  same. We cannot tell from the repo which notation is on OpenReview.
- **All seven.** Nobody has compared these files byte for byte with OpenReview. That check
  needs an OpenReview login.

## Drafts

`responses/drafts/` holds the text that we did not post, or that a later version replaced.
Drafts are kept as history. Do not cite them as the public record.

| File | What it is |
|---|---|
| `reviewer_HomM_followup.md` | First draft of the HomM follow-up (9,872 characters). The 5,000-character limit made us split it into parts 1 and 2. |
| `reviewer_HomM_followup_5k.md` | Second draft of the same reply. Still over the limit. |
| `reviewer_HomM_tables_companion.md` | Draft companion post with four-condition tables for five metrics. Not in the posted list. |
| `meta_review_comment.md` | Draft reply to the AC meta-review, in one comment. Part 1 and part 2 replace it. |
| `meta_review_part1.md`, `meta_review_part2.md` | Reply to the AC meta-review in two comments (file date 2026-08-04). Not in the posted list for this cleanup. They may have gone to the AC. See "Open" below. |
| `meta_review_ARCHIVED.md` | Early plan to answer each meta-review weakness in its own box. Replaced by the per-reviewer replies. |
| `SUBMIT.md` | Posting guide from 2026-07-28. This README replaces it. |

Some drafts hold numbers that we never posted. The last section below lists them.

## Other contents

| Path | What it is |
|---|---|
| `analyses/mimicry_split.py` | The discrimination index and the asking odds ratios posted to LxMF. |
| `analyses/interference_mechanism.py` | The format-leakage analysis posted to LxMF. |
| `analyses/second_person_questions.py` | The second-person question count posted to FzpA. |
| `analyses/quality_drop.py` | The paired t and Holm test behind "all six drops p < 0.02". Needs the synced cells. |
| `analyses/cell_values.py` | Computes the posted cell values from the synced cells and diffs them against `posted_values.json` (`make cells`). |
| `analyses/peek_results.py` | Quick look at per-seed results on the host. |
| `responses/drafts/FINDINGS_quality_drop.md` | Internal analysis of the aggregate drop (paired t, rubric test). Never posted. |
| `PLAN.md` | The rebuttal plan, now closed. It records what ran and what did not. |
| `COVERAGE.md` | Each reviewer concern, how we answered it, and the open item that remains. |

## Open

- **Raw outputs of the added cells.** Synced on 2026-10-08 (PROV-1). `make cells` diffs the
  cell values against `posted_values.json` (one mismatch, W21), runs the paired t (PROV-2;
  every posted test holds) and reads the per-cell tau and recipe from the host logs (PROV-3).
  The full responses stay gitignored; versioning them is open (KNOWN_ISSUES).
- **Scripts for posted numbers.** Each posted analysis now has a script. Two results differ
  from the posts. The second-person count gives 17.9% against 36.7%, not the posted 18.4%
  against 39.6% (PROV-7). The odds ratios reproduce 1.16, 1.54 and 1.56, but their 95% CIs
  overlap (PROV-10). The outputs are in `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/`.
- **The MedQA cell.** Settled by the sync: the only run is the preamble run, so the cell
  measures instruction following and stays set aside. Table 3 reports its aggregate score
  (W12, C24 done).
- **The meta-review reply.** Confirm on OpenReview whether `meta_review_part1.md` and
  `meta_review_part2.md` were posted to the AC. If they were, move them to
  `responses/posted/` as `AC_1_part1.md` and `AC_2_part2.md` with `git mv`, and add them to
  the table above. Their withdrawals and promises repeat ones that are already in the ledgers.
- **Promised experiments.** C25 to C33 need GPU time, people or data access.
  `commitments.md` says where each one stands (done, built and awaiting GPU, changed or
  deferred). docs/RUN_PLAN.md orders the runs, with their cost, and
  docs/ANALYSIS_PLAN.md fixes how each result is read before it exists. No GPU run starts
  before the first author's go-ahead, and the two remaining raters (C28) are deferred
  (docs/DECISIONS.md D20, D21).

## Posted and generated values

Two files, with different jobs:

- `../results/cross_cell/generated_values.json` is the source for the paper. `make cells`
  (`analyses/cell_values.py`) computes it from the raw per-seed outputs of every cell,
  synced from the GPU host on 2026-10-08, and `make numbers` reads it. It also lists each
  posted value that differs from the raw outputs (`mismatches`).
- `posted_values.json` is the record of what was posted. It lists every number the authors
  posted on OpenReview about the rebuttal cells, plus the posted analyses of the submitted
  cell, as a hand transcription of `responses/posted/` with a source line for each value.
  No script generated these numbers. Do not overwrite it; `make numbers` reads it only for
  cell metadata (names, families, set-aside flags), for posted numbers the paper quotes as
  posted, and as the fallback for a cell that has no generated values.

The sync found one posted value that the raw outputs do not give: MedQuAD Base hedging
quality, posted as 1.46 in Table 1 and computed as 1.4548 (withdrawals W21). Every other
posted value of the added cells matches at the posted precision.

### Cells

| Posted id | Cell record | Status |
|---|---|---|
| `healthbench__medgemma-27b` | `healthbench__medgemma-27b-text__bodhi__2026-05` | submitted cell |
| `healthbench__mistral-small-24b` | `healthbench__mistral-small-24b__bodhi__2026-07` | added |
| `healthbench__med42-8b` | `healthbench__med42-8b__bodhi__2026-07` | added, same family as the evaluator |
| `healthbench__biomistral-7b` | `healthbench__biomistral-7b__bodhi__2026-07` | added, null result |
| `chatdoctor__mistral-small-24b` | `chatdoctor__mistral-small-24b__bodhi__2026-07` | added |
| `chatdoctor__med42-8b` | `chatdoctor__med42-8b__bodhi__2026-07` | added, same family as the evaluator |
| `medquad__mistral-small-24b` | `medquad__mistral-small-24b__bodhi__2026-07` | added |
| `medqa__mistral-small-24b` | `medqa-open-preamble__mistral-small-24b__bodhi__2026-07` | added, set aside |

The cell records give each cell's model repo and settings. The posted text names the
models only by short name.

### File shape

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
- `cells[].notes` holds rounding, conflicts between posts and gaps for that cell;
  `analyses` holds posted numbers that are not per-cell condition means (discrimination
  index, format leakage, second-person questions, the MedQA preamble run, the paired tests
  on the aggregate drop, counts across cells, cost, sample sizes and the recipe settings as
  posted), each with `values` and `sources`; `analyses_note` says what `analyses` holds.

### How it was transcribed

1. Two independent passes went to temporary files outside the repo. Pass A was typed by
   hand from the posted replies. Pass B parsed the markdown tables and pulled the prose
   numbers by regular expression, and it recorded the line each value came from.
2. A deep diff of the two passes gave two differences, both resolved against the text:
   Pass B missed two source lines for the BioMistral cell (`HomM_2_followup_part1.md:7`
   and `:9`, both restating BioMistral values), and the MedQA set-aside reason now uses the
   exact posted words.
3. Pass B also found three places where the posts disagree with each other, each in the
   cell's `notes`: Mistral-Small-24B on HealthBench, clearance 62.1 in Table 2 and 62% in
   two earlier posts (rounding); BioMistral-7B, clearance 20.4 and 20% (rounding); MedQuAD
   with Mistral-Small-24B, base hedging 1.46 in Table 1 and 1.45 in two earlier posts (a
   draft gives 1.455).

Checks run on the file: every number appears on at least one of its cited source lines;
every count and range posted across cells recomputes from the cell values; the submitted
cell matches its `analysis/cross_seed_aggregate.json` (8 values); the discrimination index
and the leakage numbers match its `analysis/mimicry_split.json` and
`analysis/interference_mechanism.json` (41 values, 0 mismatches); the Base and LoRA
aggregate means in Table 3 match the 4-decimal table in
`responses/drafts/FINDINGS_quality_drop.md`. These checks show that the transcription matches the
posts and that the posts agree with each other. Only the raw outputs can show that the
posted numbers are correct, and `make cells` now does: one mismatch (W21).

### Where the raw outputs live

Every cell's outputs are in `results/<cell record>/` (`results/PROVENANCE.md` maps each
old path to its new one). The added cells' outputs were copied read-only on 2026-10-08
from the shared GPU host (audit finding PROV-1), from
`mit-critical-data-1:/data/PROJECTS/bohdi/results_rebuttal/<bench>__<tag>/`. Git tracks
each cell's `shared/inputs/eval_ids.json` and every `adapter_config.json` and
`trainer_state.json` under `seed_<n>/train/`. The per-seed eval and grade JSON holds full
responses (about 370 MB), so it is ignored by git and stays in the local copy and on the
host. The host logs are in each cell's `logs/` folder (ignored).

The 2026-07 launcher (`61dada9:rebuttal/launch/run_cell.sh`) named a cell folder
`<bench>__<tag>`, with `<tag>` the HF repo with `/`, `:` and `.` replaced by `_`. Its eval
files `base_no_wrapper`, `base_bodhi`, `lora_no_wrapper` and `lora_bodhi` are now
`eval/base.json`, `eval/base+bodhi.json`, `eval/lora.json` and `eval/lora+bodhi.json`; its
`epistemic_scores.json` is `grade/epistemic__llama-3.1-8b-nous.json`. The host's
`multi_seed_summary.json` copies were written before commit 55f6c7c, list their seeds as
`["eval"]` (a wrong glob) and hold one seed only; they went to the archive, and
`cell_values.py` recomputes everything from the per-seed files. The teacher traces and
filter outputs (`sft/`) stayed on the host.

Three host folders are not cells of the paper:
`medqa-open-preamble__medgemma-27b-text__bodhi__2026-07` and
`medquad__medgemma-27b-text__bodhi__2026-07` hold only `eval_ids.json` (trace generation
failed on 2026-07-27: the vLLM server did not start), and
`medqa-open-preamble__biomistral-7b__bodhi__2026-07` is a smoke run (one seed, 40 traces,
3 kept, 3 optimizer steps, 10 eval prompts). Its train split held 2 of its 40 traces,
which meets the launcher's re-filter condition, but it ran on 2026-07-23, before the
launcher had that rule (912782d, 2026-07-27). A fourth run left no folder: Gemma-2-9B on
HealthBench-Hard was stopped in trace generation on 2026-07-27 while its vLLM server was
starting; its logs are in `results/healthbench__gemma-2-9b__bodhi__2026-07/logs/`.
`results/cross_cell/cell_recipe.json` records the status of all four (the last under
`runs_without_folder`).

`analyses/cell_recipe.py` (`make cells`) reads the tau, the trace counts and the train/val
split of each cell from its filter log, checks them against the trainer states, and writes
`results/cross_cell/cell_recipe.json` (PROV-3). Every cell used tau = 0.4, and no log shows
the 22nd-percentile re-filter.

### How the generated values are checked

`analyses/cell_values.py` (`make cells`) reads each cell's per-seed eval and grade files,
computes every posted metric with the definitions of `scripts/paper_stats.py`, and writes
`generated_values.json` in the same shape. Then it lists each posted value that differs
from the generated one at the posted decimals.

- The metrics are per-seed means over scored responses (epistemic metrics) or over all
  completions (`aggregate`), then the mean and the sample SD over seeds. The posts do not
  say which SD they used. The submitted cell uses the sample SD.
- Without the local copy of the added cells' raw files, only the submitted cell has data.
  The script then names the missing cells and writes no file, so the tracked
  `generated_values.json` stays as it is.
- A seed whose epistemic grading failed has no value for the seven dimensions. The MedQA
  cell has one: seed 42's grading found none of its prompts, so its seven-dimension means
  are null and `per_seed` shows the four seeds that have them.
- Each cell also gets `lora_askers_minus_non_askers`: the aggregate score of LoRA responses
  that ask minus those that do not, pooled over seeds (Appendix J).

When a generated value differs from a posted value by more than rounding, the generated
value is the record. Add the difference to `withdrawals.md` as a posted number that was
wrong (W21 is the one the sync found). Keep `posted_values.json` after that. It stays as
the record of what was posted.

### Not in these files

The drafts in `responses/drafts/` hold numbers that were never posted: the full
four-condition tables for five metrics in `reviewer_HomM_tables_companion.md`, and the
MedQA aggregate of 0.665 to 0.683 in `reviewer_HomM_followup.md` and `_5k.md`. They can
be a second check on the generated values. They are not part of the public record.
