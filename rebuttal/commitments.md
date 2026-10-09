# Commitments made in the posted replies

This file lists every promise in the posted OpenReview replies: each "if accepted we will",
each "the camera-ready will" and each "the paper will". The paper was rejected, but these are
the changes that reviewers and the AC saw us promise. A resubmission must keep them or say
why it drops one.

All sources are in `rebuttal/responses/posted/`. The paper lanes fill the Status column. Use
`done (<commit>)`, `built, awaiting GPU (<run>)`, `to build, then GPU (<run>)`,
`deferred (<decision>)`, `changed (<reason>)` or `dropped (<reason>)`. A run id such as R2
refers to `docs/RUN_PLAN.md`; an H id to `docs/ANALYSIS_PLAN.md`, which fixes how the
result is read before it exists.

## Where each commitment stands (2026-10-09)

| State | Commitments | What is left |
|---|---|---|
| done | C1 to C24 | Nothing. The uniform analysis (EXPERIMENTS.md E01) recomputes every number they cite from the response tables. |
| built, awaiting GPU | C25 (manifests and the `bench` stage), C26 (judges in the grade stage), C27 (all three baselines), C29 (`ask-first`), C30 (physician emergency labels and the holdout split), C31 (`red-flags-first`), C32 (the NLM loader for the secondary set) | The first author's go-ahead (D21): RUN_PLAN.md R1, R2, R4, R5, R6, R10; the MediQ and NLM files pinned first. |
| changed | C30 (labels from HealthBench physicians, not new labelling), C32 (open clinician-posed sets in place of MIMIC and eICU) | The paper states the change and why. |
| deferred | C28 (the two remaining raters, D20), C33 (the physician harm review, E28) | The first author. The LLM-graded harm proxy (E06) runs on existing outputs meanwhile. |

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
| C9 | The behavioural gain and the aggregate cost stated together for every cell. | table | `HomM_4_consolidation.md:3` | done (99d97ec, 60c1a54, 38b800e, ab58ee5) | Table 1 of the paper (section 4.1) gives the gain and the cost for every cell with the same seed-level tests, and Tables E.1 and E.2 give all four conditions. Each added cell has a paired t(4) interval computed from its raw outputs, synced on 2026-10-08. |
| C10 | A Discussion criterion for when the behavioural gain is worth the aggregate cost, framed by deployment risk. | text | `HomM_4_consolidation.md:3` | done (99d97ec) | |
| C11 | Each interpretation corrected in the discussion appears in its corrected form. The convergence sentence is removed everywhere. | text | `HomM_4_consolidation.md:3`, `HomM_2_followup_part1.md:7` | done (99d97ec) | All of `withdrawals.md` |
| C12 | No claim that the recipe is protocol-agnostic. Only one CoT protocol was tested. | text | `HomM_1_rebuttal.md:27` | done (99d97ec) | |
| C13 | The grader design as a limitation. The cross-family split removes the most direct route to circularity. It does not rule out correlated grading preferences, and it gives no clinical validity. Also state why the graders are not clinically tuned. | text | `HomM_3_followup_part2.md:22`, `HomM_1_rebuttal.md:35` | done (a59cb82) | withdrawals W13. The paper gives the reason as posted in `HomM_3_followup_part2.md:22`: the design rule was family separation, general models as in HealthBench, and a clinically tuned judge is untested. It does not repeat the posted argument that the clinical judges are all Llama-derived, because that argument does not bear on separation from the Qwen filter. |
| C14 | The missing empirical prior-work comparison stated plainly as a limitation and named as the next experiment, not folded into a claim. | text | `HomM_3_followup_part2.md:26`, `HomM_1_rebuttal.md:49` | done (99d97ec) | See C27. |
| C15 | A Discussion subsection, "Comparison to Prior Calibration Approaches". It covers uncertainty prompting (Lin et al., 2022; Tian et al., 2023), STaR (Zelikman et al., 2022) and Constitutional AI (Bai et al., 2022). It has a table that compares inference cost, teacher requirement, behavioural against factual focus, and generality. | text + table | `HomM_1_rebuttal.md:47` | done (5c2e9b0) | Section 5.1, Table 4. Generality is in the text (what each approach was tested on), not a column. Table 4 and Section 2 now also cover MediQ, AMIE, R-Tuning and linguistic calibration, and Section 2 cites CRAFT-MD, from `paper/prior_work_candidates.bib` (meta-review W6, HomM W4). |

## Figures and tables

| ID | The camera-ready must carry | Kind | Source | Status | Note |
|---|---|---|---|---|---|
| C16 | Figure 2 in two panels. Panel A has the 0 to 2 dimensions (uncertainty, context-seeking, red-flag, scope, hedging, specificity). Panel B has the rates (active inquiry, red-flag, scope-bounded, blanket disclaimer). Each panel has its own labelled axis. | figure | `FzpA_1_rebuttal.md:24`, `LxMF_1_rebuttal.md:54` | done (99d97ec) | The two-panel figure moved to the per-seed appendix in 1508893 to keep the body at nine pages, where it is now Figure E.1; Table 2 carries the same values in the body. |
| C17 | A direction arrow on every row of Table 2. Blanket disclaimer is the only row where lower is better, so it is the only down arrow. Red-flag identification is labelled higher-is-better. | table | `FzpA_1_rebuttal.md:22`, `FzpA_1_rebuttal.md:28` | done (43937ed) | Questions per response now carries a circle for "no better direction", because asking helps only where facts are missing (the Discussion says so); every other row keeps its arrow. |
| C18 | A Table 2 caption note that the red-flag rate does not capture false positives, so a model that flags everything would also score high. | text | `FzpA_1_rebuttal.md:28` | done (43937ed) | Reworded: the grader anchor credits only red flags relevant to the case (Appendix D), so the caption now says that over-flagging has no labelled test, not that flagging everything scores high. |
| C19 | A Table 1 caption that says only the CoT conditions lose prompts. The LoRA row is 200. | text | `LxMF_1_rebuttal.md:7` | done (43937ed) | Use the corrected n (withdrawals W10). |

## Dimensions and grading

| ID | The camera-ready must carry | Kind | Source | Status | Note |
|---|---|---|---|---|---|
| C20 | A clinical implications paragraph. Active inquiry is history-taking before recommending. Red-flag identification is recognising presentations that need escalation. Scope bounding is declining to advise beyond the available evidence. | text | `FzpA_1_rebuttal.md:23` | done (99d97ec, 5c2e9b0) | 5c2e9b0 adds the antidepressant example from `FzpA_2_followup.md:21`, checked against `results_modal/seed_99/`. The posted reply says the adapter asks about current medications and conditions; it does not, so the paper leaves that out. |
| C21 | The origin of the seven dimensions and the excluded candidates (empathy, depth of patient education), stated explicitly. | text | `FzpA_1_rebuttal.md:36` | done (99d97ec) | State the origin as corrected: informal author consensus (withdrawals W3). |
| C22 | A verbatim example for each of the seven dimensions, as listed in the reply. | text | `FzpA_1_rebuttal.md:40` | done (99d97ec, a59cb82) | a59cb82 adds the grader anchor for each dimension, quoted from `bodhi_lora/grading/epistemic.py`, and one mapping from the dimensions to the BODHI components (now Table D.1). The posted context-seeking example ("I would need the patient's age and medication history") asks no direct question, so it breaks the anchor; the paper uses a direct question instead. |
| C23 | The grader's hedging anchor quoted: 0 for a bare referral, 2 for hedging calibrated to the case. Also the case-bound score-2 example (cough with fever, normal chest X-ray). | text | `FzpA_2_followup.md:27`, `FzpA_2_followup.md:29` | done (8a737f9, a59cb82) | The prompt quotes both ends but does not tie a score to each end; the grader gave hedging 2 to most flagged blanket disclaimers (`\statBlanketDisclaimerHedgingTwo` of `\statBlanketDisclaimerTotal`). a59cb82 says so instead of the posted "0 for a bare referral". |

## Evaluation scope

| ID | The camera-ready must carry | Kind | Source | Status | Note |
|---|---|---|---|---|---|
| C24 | All seven added cells in the main evaluation, including the BioMistral null and the set-aside MedQA cell. | table | `HomM_4_consolidation.md:3` | done (60c1a54, 38b800e, ab58ee5) | Table 1 holds the submitted cell and all seven added cells, computed from the raw outputs synced on 2026-10-08. The MedQA row sits last, set aside: its aggregate score uses five seeds, and its dimensions use the four seeds that have dimension grades (seed 42's epistemic grading found none of the MedQA prompts; ANALYSIS_PLAN.md 2.3). The MedQA run is the preamble run (withdrawals W12), and section 4.1, the abstract, the Limitations and Appendix J say why. |
| C25 | Training and inference cost for each configuration, from per-stage timings measured in isolation. Do not use the shared-node timings. | experiment | `FzpA_1_rebuttal.md:64` | built, awaiting GPU (R2) | Every run manifest records wall-clock time, GPU and tokens per stage, and the `bench` stage (`bodhi_lora/pipeline/bench.py`) times 50 prompts per condition on an idle GPU (E08, E20). The 2026-07 host logs give shared-node times only; RUN_PLAN.md section 3.1 uses them for planning, not for the paper. |

## Experiments still owed

These were promised, or named as outstanding. Each one needs compute, people or credentialed
data access.

| ID | Commitment | Kind | Source | Status | Note |
|---|---|---|---|---|---|
| C26 | A clinical judge as an extra robustness panel next to the primary grader. It does not replace the primary grader. | experiment | `LxMF_1_rebuttal.md:34`, `HomM_1_rebuttal.md:35` | built, awaiting GPU (R1, R2, R6) | The grade stage runs each judge a cell record lists (`bodhi_lora/pipeline/grade.py`), and the registry's family rule decides which judges a cell may use. R1 re-grades the Med42 cells with a judge that is neither Llama nor Qwen (EVAL-7); R6 adds the clinical judge `medgemma-27b-text` to six recorded cells of 2026-07, the Med42 cells included, and `phi-4` to the 2026-05 cell. Read by H8. The records name `medgemma-27b-text`, and `phi-4` for the Gemma bases (EXPERIMENTS.md E10). Still named in the Limitations as promised and not run. |
| C27 | An empirical head-to-head against an inference-time calibration baseline (Tian et al., 2023, is named). | experiment | `HomM_1_rebuttal.md:49`, `HomM_1_rebuttal.md:59`, `HomM_4_consolidation.md:5`, `LxMF_1_rebuttal.md:82` | built, awaiting GPU (R2) | Inference only (audit EVAL-8). `clarify` and `verbalized-confidence` (Tian et al., 2023) are protocol records run by `bodhi_lora/protocols/instruction.py`, and `mediq-abstain` by `bodhi_lora/protocols/abstain.py`; all stub-tested (E12). They run as conditions of the new cells, so they share the grader and prompts of the adapter they are compared with. Read by H4. |
| C28 | Complete the three-physician validation: consensus kappa against the pre-registered 0.6 target. | experiment | `HomM_1_rebuttal.md:59`, `HomM_4_consolidation.md:5`, `FzpA_2_followup.md:11`, `LxMF_1_rebuttal.md:82` | deferred (D20) | People (audit EVAL-5, PROV-5). The first author deferred the two remaining raters; BLOCKED.md item 6 has the steps for when they resume. D5 stands meanwhile. |
| C29 | A second CoT protocol. | experiment | `HomM_1_rebuttal.md:59`, `HomM_4_consolidation.md:5` | built, awaiting GPU (R5) | `bodhi_lora/protocols/ask_first.py` and the `ask-first` protocol record, stub-tested. The cell is `healthbench__mistral-small-24b__ask-first` (E16), with the MediQ probe on its adapters. Read by H5. |
| C30 | A labelled set of prompts with no genuine red flag, to measure over-flagging (the false-positive check on red-flag identification). | experiment | `FzpA_1_rebuttal.md:28`, `FzpA_2_followup.md:23` | changed (labels exist); built, awaiting GPU (R4) | No new labelling: HealthBench physicians already labelled 134 prompts non-emergent and 319 emergent or conditionally emergent. The `emergent-physician` label and the `default` split of the HealthBench record keep all of them out of training, and the labelled eval-only cell evaluates the new adapters on them (E18). Read by H7. A post hoc check on the existing HealthBench cells ran on 2026-10-09 (the `red-flag-at-stake` label and the escalation rule, ANALYSIS_PLAN.md section 12); it reads no physician label, and no text cites it until the first author accepts it (BLOCKED.md item 1). |
| C31 | Surface red flags early in the protocol, before the extended reasoning, and test whether this recovers the LoRA+CoT red-flag gap. | experiment | `LxMF_1_rebuttal.md:50` | built, awaiting GPU (R2) | Inference. The `red-flags-first` protocol (BODHI with the red-flag section first) is built and stub-tested; `lora+red-flags-first` and `base+red-flags-first` are conditions of the new cells (E17). Read by H6. Still named in the Limitations as promised and not run. |
| C32 | A clinician-posed benchmark (MIMIC or eICU). | experiment | `FzpA_1_rebuttal.md:48` | changed (open sets in place of MIMIC and eICU); awaiting GPU (R4, R10) | MIMIC and eICU need credentialed PhysioNet access, and D4 makes them future work. In their place: the 322 HealthBench prompts that physicians labelled as written by a health professional (held out of training, R4), and the NLM Clinical Questions Collection (R10) (E19). Read by H11. The paper must say why the promised sets changed. |
| C33 | The Appendix C harm grading. | experiment | `FzpA_2_followup.md:23` | deferred (E28) | Named as outstanding before patient-facing use, not promised for the camera-ready. The physician review waits for the first author. An LLM-graded proxy, the share of responses that meet a negative rubric criterion (E06), runs on existing outputs without a GPU (`analyze`); the judges also grade harm on extent, likelihood and delayed care. Neither is the physician review. |

## Offered, not promised for the paper

- An anonymized link to the code and per-cell outputs for the AC, on request
  (`HomM_3_followup_part2.md:32`). The repo now tracks the per-seed values
  (`results/cross_cell/generated_values.json`), adapter configs and trainer states. The per-seed
  eval JSON with full responses (about 370 MB) is copied from the host but gitignored, so sharing
  it needs a versioned archive first (open decision O3).
