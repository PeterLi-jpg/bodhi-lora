# Rebuttal record: NeurIPS 2026, submission 32818

This folder holds the record of the NeurIPS 2026 discussion and the code for the cells we
added during it. The paper was rejected. Use this folder to see what we posted, what we
withdrew, and what we promised. The ledgers are the files that the resubmission must act on.

## The ledgers

| File | What it holds |
|---|---|
| `withdrawals.md` | Every claim that we withdrew or corrected in the discussion, plus two posted claims that the audit found wrong. Each entry has the claim as made, the correction and the statement that the paper must carry. |
| `commitments.md` | Every promise in the posted replies (C1 to C33), with its source line and a status column for the paper lanes. |
| `../data/rebuttal_cells/posted_values.json` | Every number that we posted about the rebuttal cells, transcribed with source lines. Its README says how `make cells` checks it against generated values. |

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

Some drafts hold numbers that we never posted. The README in `data/rebuttal_cells/` lists
them.

## Other contents

| Path | What it is |
|---|---|
| `configs/` | LoRA configs for the added base models (Mistral-Small-24B with QLoRA, Med42-8B, BioMistral-7B). |
| `launch/run_cell.sh`, `launch/run_grid.sh` | Driver for one cell, and for the grid on the shared GPU host. |
| `launch/run_baselines.sh` | Runs the two prompt baselines (clarifying questions, verbalized confidence) for a cell (C27). Not run yet. |
| `analyses/mimicry_split.py` | The discrimination index and the asking odds ratios posted to LxMF. |
| `analyses/interference_mechanism.py` | The format-leakage analysis posted to LxMF. |
| `analyses/second_person_questions.py` | The second-person question count posted to FzpA. |
| `analyses/quality_drop.py` | The paired t and Holm test behind "all six drops p < 0.02". Needs the synced cells. |
| `analyses/cell_values.py` | Computes the posted cell values from the synced cells and diffs them against `posted_values.json` (`make cells`). |
| `results/` | Outputs of the analyses above (`make rebuttal`). |
| `analyses/peek_results.py` | Quick look at per-seed results on the host. |
| `analyses/FINDINGS_quality_drop.md` | Internal analysis of the aggregate drop (paired t, rubric test). Never posted. |
| `prior_work_candidates.bib` | Candidate references for the prior-work subsection (C15). |
| `PLAN.md` | The rebuttal plan, now closed. It records what ran and what did not. |
| `COVERAGE.md` | Each reviewer concern, how we answered it, and the open item that remains. |

## Open

- **Raw outputs of the added cells.** They exist only on the shared GPU host, under
  `/data/PROJECTS/bohdi/results_rebuttal/`. Until they are synced, no number for an added
  cell can be checked (audit PROV-1). After the sync, run `make cells`. It diffs the cell
  values against `posted_values.json` and runs the paired t (PROV-2).
- **Scripts for posted numbers.** Each posted analysis now has a script. Two results differ
  from the posts. The second-person count gives 17.9% against 36.7%, not the posted 18.4%
  against 39.6% (PROV-7). The odds ratios reproduce 1.16, 1.54 and 1.56, but their 95% CIs
  overlap (PROV-10). See `results/`.
- **Settings for each cell.** The tau that each cell used, the retention and the holdout IDs
  are in the filter logs on the host (PROV-3). `withdrawals.md` W11 depends on them.
- **The MedQA cell.** Two posted reasons for setting it aside disagree. HomM_4 promised that
  the cell would move into the main evaluation (W12, C24).
- **The meta-review reply.** Confirm on OpenReview whether `meta_review_part1.md` and
  `meta_review_part2.md` were posted to the AC. If they were, move them to
  `responses/posted/` as `AC_1_part1.md` and `AC_2_part2.md` with `git mv`, and add them to
  the table above. Their withdrawals and promises repeat ones that are already in the ledgers.
- **Promised experiments.** C25 to C33 in `commitments.md` need GPU time, people or data
  access. Code for C27 (`launch/run_baselines.sh`) and C29 (`PROTOCOL=ask_first` in
  `launch/run_cell.sh`) exists and passes its dry-run tests. Neither has run.
