# Commitments made in the posted replies

This file lists every promise in the posted OpenReview replies: each "if accepted we will",
each "the camera-ready will" and each "the paper will". The paper was rejected, but these are
the changes that reviewers and the AC saw us promise. A resubmission must keep them or say
why it drops one.

All sources are in `rebuttal/responses/posted/`. The paper lanes fill the Status column. Use
`to-do`, `done (<commit>)` or `dropped (<reason>)`.

Kinds:
- `text`: wording only.
- `figure` or `table`: regenerate from existing data.
- `analysis`: new computation on data that already exists (some of it remote).
- `experiment`: needs new GPU runs or people.

## Claims and framing

| ID | The camera-ready must carry | Kind | Source | Status | Note |
|---|---|---|---|---|---|
| C1 | The contribution as a recipe plus an evaluation decomposition for open-ended, information-seeking clinical Q&A. Say what is not covered: tasks with no missing information to seek, such as closed-form multiple choice, extraction and summarization. | text | `FzpA_1_rebuttal.md:5`, `HomM_1_rebuttal.md:7` | done (99d97ec) | withdrawals W1 |
| C2 | Replace the "Any structured CoT protocol..." sentence with a scoped one that separates what we showed from what we conjecture. | text | `FzpA_1_rebuttal.md:5` | done (b53088a) | The sentence is at submitted `main.tex:251`. The introduction now ends with what we show and a conjecture we have not tested. |
| C3 | Place the recipe against inference-time calibration methods: they act at prompt time, the recipe acts on the weights. | text | `FzpA_1_rebuttal.md:5` | done (99d97ec, 5c2e9b0) | Section 5.1 and its table. |
| C4 | How the decomposition changes across contexts. Active inquiry and context-seeking are informative only where information is missing and the user can supply it. Scope bounding and red-flag identification apply more widely. Use the ChatDoctor against MedQuAD contrast. | text | `FzpA_1_rebuttal.md:7` | done (8a737f9) | |
| C5 | Remove the vague "not statements". Replace them with positive claims that give per-dimension numbers, including accuracy 0.117 → 0.121 and completeness 0.168 → 0.169. | text | `FzpA_1_rebuttal.md:21` | done (99d97ec) | |
| C6 | Name the scope as patient-facing clinical Q&A. The setup section says that the prompts are mostly patient-facing (14 of 1,000 HealthBench-Hard prompts identify a clinician; corrected to the HealthBench tag count, 40 of 1,000, withdrawals W14). | text | `FzpA_2_followup.md:5`, `HomM_4_consolidation.md:3` | done (99d97ec) | withdrawals W2 |
| C7 | The argument that every rubric anchor addresses the person who asks, so no dimension assumes a clinician reader. | text | `FzpA_2_followup.md:9` | done (8a737f9) | |
| C8 | The aggregate-quality trade-off in the abstract, next to the behavioural gain, in place of "quality preserved". | text | `HomM_4_consolidation.md:3` | done (99d97ec) | withdrawals W8 |
| C9 | The behavioural gain and the aggregate cost stated together for every cell. | table | `HomM_4_consolidation.md:3` | done (99d97ec) | Table 3. Intervals for the added cells stay pending until the sync (PROV-1). |
| C10 | A Discussion criterion for when the behavioural gain is worth the aggregate cost, framed by deployment risk. | text | `HomM_4_consolidation.md:3` | done (99d97ec) | |
| C11 | Each interpretation corrected in the discussion appears in its corrected form. The convergence sentence is removed everywhere. | text | `HomM_4_consolidation.md:3`, `HomM_2_followup_part1.md:7` | done (99d97ec) | All of `withdrawals.md` |
| C12 | No claim that the recipe is protocol-agnostic. Only one CoT protocol was tested. | text | `HomM_1_rebuttal.md:27` | done (99d97ec) | |
| C13 | The grader design as a limitation. The cross-family split removes the most direct route to circularity. It does not rule out correlated grading preferences, and it gives no clinical validity. Also state why the graders are not clinically tuned. | text | `HomM_3_followup_part2.md:22`, `HomM_1_rebuttal.md:35` | done (a59cb82) | withdrawals W13. The paper gives the reason as posted in `HomM_3_followup_part2.md:22`: the design rule was family separation, general models as in HealthBench, and a clinically tuned judge is untested. It does not repeat the posted argument that the clinical judges are all Llama-derived, because that argument does not bear on separation from the Qwen filter. |
| C14 | The missing empirical prior-work comparison stated plainly as a limitation and named as the next experiment, not folded into a claim. | text | `HomM_3_followup_part2.md:26`, `HomM_1_rebuttal.md:49` | done (99d97ec) | See C27. |
| C15 | A Discussion subsection, "Comparison to Prior Calibration Approaches". It covers uncertainty prompting (Lin et al., 2022; Tian et al., 2023), STaR (Zelikman et al., 2022) and Constitutional AI (Bai et al., 2022). It has a table that compares inference cost, teacher requirement, behavioural against factual focus, and generality. | text + table | `HomM_1_rebuttal.md:47` | done (5c2e9b0) | Section 5.1, Table 4. Generality is in the text (what each approach was tested on), not a column. No candidate from `rebuttal/prior_work_candidates.bib` was needed. |

## Figures and tables

| ID | The camera-ready must carry | Kind | Source | Status | Note |
|---|---|---|---|---|---|
| C16 | Figure 2 in two panels. Panel A has the 0 to 2 dimensions (uncertainty, context-seeking, red-flag, scope, hedging, specificity). Panel B has the rates (active inquiry, red-flag, scope-bounded, blanket disclaimer). Each panel has its own labelled axis. | figure | `FzpA_1_rebuttal.md:24`, `LxMF_1_rebuttal.md:54` | done (99d97ec) | The two-panel figure moved to the per-seed appendix in 1508893 to keep the body at nine pages, where it is now Figure E.1; Table 2 carries the same values in the body. |
| C17 | A direction arrow on every row of Table 2. Blanket disclaimer is the only row where lower is better, so it is the only down arrow. Red-flag identification is labelled higher-is-better. | table | `FzpA_1_rebuttal.md:22`, `FzpA_1_rebuttal.md:28` | done (43937ed) | |
| C18 | A Table 2 caption note that the red-flag rate does not capture false positives, so a model that flags everything would also score high. | text | `FzpA_1_rebuttal.md:28` | done (43937ed) | |
| C19 | A Table 1 caption that says only the CoT conditions lose prompts. The LoRA row is 200. | text | `LxMF_1_rebuttal.md:7` | done (43937ed) | Use the corrected n (withdrawals W10). |

## Dimensions and grading

| ID | The camera-ready must carry | Kind | Source | Status | Note |
|---|---|---|---|---|---|
| C20 | A clinical implications paragraph. Active inquiry is history-taking before recommending. Red-flag identification is recognising presentations that need escalation. Scope bounding is declining to advise beyond the available evidence. | text | `FzpA_1_rebuttal.md:23` | done (99d97ec, 5c2e9b0) | 5c2e9b0 adds the antidepressant example from `FzpA_2_followup.md:21`, checked against `results_modal/seed_99/`. The posted reply says the adapter asks about current medications and conditions; it does not, so the paper leaves that out. |
| C21 | The origin of the seven dimensions and the excluded candidates (empathy, depth of patient education), stated explicitly. | text | `FzpA_1_rebuttal.md:36` | done (99d97ec) | State the origin as corrected: informal author consensus (withdrawals W3). |
| C22 | A verbatim example for each of the seven dimensions, as listed in the reply. | text | `FzpA_1_rebuttal.md:40` | done (99d97ec, a59cb82) | a59cb82 adds the grader anchor for each dimension, quoted from `scripts/eval_epistemic.py`, and one mapping from the dimensions to the BODHI components (now Table D.1). The posted context-seeking example ("I would need the patient's age and medication history") asks no direct question, so it breaks the anchor; the paper uses a direct question instead. |
| C23 | The grader's hedging anchor quoted: 0 for a bare referral, 2 for hedging calibrated to the case. Also the case-bound score-2 example (cough with fever, normal chest X-ray). | text | `FzpA_2_followup.md:27`, `FzpA_2_followup.md:29` | done (8a737f9, a59cb82) | The prompt quotes both ends but does not tie a score to each end; the grader gave hedging 2 to most flagged blanket disclaimers (`\statBlanketDisclaimerHedgingTwo` of `\statBlanketDisclaimerTotal`). a59cb82 says so instead of the posted "0 for a bare referral". |

## Evaluation scope

| ID | The camera-ready must carry | Kind | Source | Status | Note |
|---|---|---|---|---|---|
| C24 | All seven added cells in the main evaluation, including the BioMistral null and the set-aside MedQA cell. | table | `HomM_4_consolidation.md:3` | done (99d97ec) | Table 3 holds all seven added cells. Values are transcribed from the posts until the sync; the MedQA run is still to be named (withdrawals W12). |
| C25 | Training and inference cost for each configuration, from per-stage timings measured in isolation. Do not use the shared-node timings. | experiment | `FzpA_1_rebuttal.md:64` | to-do | GPU time |

## Experiments still owed

These were promised, or named as outstanding. Each one needs compute, people or credentialed
data access.

| ID | Commitment | Kind | Source | Status | Note |
|---|---|---|---|---|---|
| C26 | A clinical judge as an extra robustness panel next to the primary grader. It does not replace the primary grader. | experiment | `LxMF_1_rebuttal.md:34`, `HomM_1_rebuttal.md:35` | to-do | For the Med42 cells the judge must be neither Llama nor Qwen (audit EVAL-7). `scripts/regrade_cross_family.py` re-grades stored outputs with such a judge (an LLM, not a clinician). Not run. |
| C27 | An empirical head-to-head against an inference-time calibration baseline (Tian et al., 2023, is named). | experiment | `HomM_1_rebuttal.md:49`, `HomM_1_rebuttal.md:59`, `HomM_4_consolidation.md:5`, `LxMF_1_rebuttal.md:82` | to-do | Inference only (audit EVAL-8). Code: `scripts/baselines/prompts.py`, `rebuttal/launch/run_baselines.sh`. Dry-run tested, not run. |
| C28 | Complete the three-physician validation: consensus kappa against the pre-registered 0.6 target. | experiment | `HomM_1_rebuttal.md:59`, `HomM_4_consolidation.md:5`, `FzpA_2_followup.md:11`, `LxMF_1_rebuttal.md:82` | to-do | People (audit EVAL-5, PROV-5) |
| C29 | A second CoT protocol. | experiment | `HomM_1_rebuttal.md:59`, `HomM_4_consolidation.md:5` | to-do | GPU time. Code: `scripts/baselines/ask_first.py`, `PROTOCOL=ask_first` in `rebuttal/launch/run_cell.sh`. Dry-run tested, not run. |
| C30 | A labelled set of prompts with no genuine red flag, to measure over-flagging (the false-positive check on red-flag identification). | experiment | `FzpA_1_rebuttal.md:28`, `FzpA_2_followup.md:23` | to-do | Labelling, then inference |
| C31 | Surface red flags early in the protocol, before the extended reasoning, and test whether this recovers the LoRA+CoT red-flag gap. | experiment | `LxMF_1_rebuttal.md:50` | to-do | Inference |
| C32 | A clinician-posed benchmark (MIMIC or eICU). | experiment | `FzpA_1_rebuttal.md:48` | to-do | Needs credentialed PhysioNet access |
| C33 | The Appendix C harm grading. | experiment | `FzpA_2_followup.md:23` | to-do | Named as outstanding before patient-facing use, not promised for the camera-ready. |

## Offered, not promised for the paper

- An anonymized link to the code and per-cell outputs for the AC, on request
  (`HomM_3_followup_part2.md:32`). This needs the raw outputs in a versioned place first
  (PROV-1).
