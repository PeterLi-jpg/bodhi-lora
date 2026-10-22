# Reviewer concerns: how each was answered

Each concern from the meta-review (MW), the AC's final justification, the program chairs'
citation check and reviewers FzpA, LxMF and HomM (reviews, follow-ups and final
justifications), with the outcome at the end of the discussion. "Answered in" points to
`responses/posted/`. "Open" points to an ID in `commitments.md` (C) or `withdrawals.md` (W).
`docs/EXPERIMENTS.md` section 5 maps the same concerns to the experiments that answer them.

Outcome values (at the end of the discussion):
- `done`: the work ran and the result was posted.
- `partial`: some of it ran.
- `promised`: text or figure changes promised for the camera-ready.
- `not run`: planned or promised, but not done.

The last column, "Resubmission", says where each concern stands in the resubmission on
2026-10-11, and for the rows that the reruns of 2026-10-19 answered, on 2026-10-19. "built, awaiting GPU" means the code exists and passes its CPU tests, the run
is in `docs/RUN_PLAN.md`, and the paper names it as not run (section 5.3).

## A. Generality

| ID | Source | Concern | Outcome | Answered in | Open | Resubmission |
|---|---|---|---|---|---|---|
| A1 | MW1, FzpA W2 | MedGemma does not fit HealthBench's patient-facing use | done: added Mistral-Small-24B, a general-purpose model | `FzpA_1_rebuttal.md:9-15` | C6 | done: Table 1 and section 4.1 |
| A2 | MW2, HomM W1/Q2, FzpA Q3 | One model and one benchmark, but a "framework" claim | done: seven added cells over three families and three benchmarks; claim narrowed | `FzpA_1_rebuttal.md:5`, `HomM_1_rebuttal.md:7-25` | C1, C24 | done: Table 1; the claim of generality goes no further than the counted cells (section 4.1) |
| A3 | HomM Q2, LxMF | One more clinical model or eval set | done: Med42-8B, ChatDoctor and MedQuAD. The MedQA cell was set aside. | `HomM_1_rebuttal.md:9-21`, `HomM_2_followup_part1.md:11` | W12, C24 | done: Table 1 |
| A4 | HomM W1 | Only one CoT protocol | not run | `HomM_1_rebuttal.md:27` | C12, C29 | done: R5 ran (ask-first); H5 supported, section 4.3 and Table 2; the Limitations name two protocols on one base model |

## B. Calibration or mimicry (MW4, LxMF)

| ID | Source | Concern | Outcome | Answered in | Open | Resubmission |
|---|---|---|---|---|---|---|
| B1 | MW4, LxMF Q2 | Real calibration or surface mimicry? LxMF's split of withholding against self-contained prompts | done: discrimination index on the submitted data | `LxMF_1_rebuttal.md:9-30` | none | answered against the method: the MediQ probe (H3) indicates mimicry in both cells, and E02 detects no rise in any cell (section 4.3, Table 2, Table C.1); the posted index stays withdrawn (W24) |
| B2 | MW4 | Restraint on well-specified prompts | partial: MedQuAD reported; MedQA set aside | `LxMF_1_rebuttal.md:74` | C24 | answered against the method: on the complete MediQ versions the adapter asks more than Base (H3, section 4.3); MedQA stays set aside (W12) |
| B3 | MW4, LxMF Q2 | Asks when information is missing, holds back when it is not | not run: the MediQ probe | none | none | answered against the method: R3 and the ask-first MediQ cell ran; mimicry indicated (H3, section 4.3, Table 2) |

## C. Interference mechanism (LxMF W4)

| ID | Source | Concern | Outcome | Answered in | Open | Resubmission |
|---|---|---|---|---|---|---|
| C1 | LxMF W4 | "Competition" is speculative; it could be truncation or attention dilution | done: format leakage, 54.6% under LoRA+CoT; the mechanism claim was restated | `LxMF_1_rebuttal.md:40-50` | W9, C31 | done: section 4.4 keeps the post hoc leakage account; R2 shows no drop to recover (H6), and the Phi-4 judge reverses the 2026-05 drop, so it is reported as grader-dependent; R7 with the 4,096-token pair is not run |

## D. Grading (MW3, LxMF Q3, HomM W3/Q4)

| ID | Source | Concern | Outcome | Answered in | Open | Resubmission |
|---|---|---|---|---|---|---|
| D1 | LxMF Q3, HomM Q4 | Why a Llama-3.1-8B judge and not a clinical model? | partial: the structural reason was given. The clinical-judge re-grade did not run. | `LxMF_1_rebuttal.md:32-36`, `HomM_1_rebuttal.md:35`, `HomM_3_followup_part2.md:22` | C13, C26 | done: R1 and R6 (shared host) and R2 and R5 (NRP) ran; the judges confirm both H1 rises wherever the evaluator found them, with weak criterion agreement (section 4.3, Table K.2) |
| D2 | MW3, HomM W3 | Claims rest on automatic grading of subjective dimensions | partial: physician validation, 1 of 3 raters, kappa 0.35 against a 0.6 target | `HomM_1_rebuttal.md:37`, `FzpA_2_followup.md:9` | C28 | deferred (D20): one of three raters (section 3.3, Appendix B) |

## E. Prior work (MW6, HomM W4/Q3)

| ID | Source | Concern | Outcome | Answered in | Open | Resubmission |
|---|---|---|---|---|---|---|
| E1 | MW6, HomM W4 | No comparison to prior literature | promised: a Discussion subsection with a comparison table | `HomM_1_rebuttal.md:43-49`, `HomM_3_followup_part2.md:26` | C14, C15 | done: section 5.1 and Table L.1, with what each approach was tested on |
| E2 | HomM Q3 | No empirical comparison to calibration methods | not run | `HomM_1_rebuttal.md:49` | C27 | done: three prompt baselines in R2 (H4); two ask more than the adapter, and the adapter has the higher rubric score (section 4.3, section 5.1) |

## F. Clarity (MW5, FzpA W1/W3/Q1/Q2, LxMF)

| ID | Source | Concern | Outcome | Answered in | Open | Resubmission |
|---|---|---|---|---|---|---|
| F1 | FzpA W1, MW2 | "Framework" overstated | promised | `FzpA_1_rebuttal.md:5-7` | W1, C1 to C4 | done (C1 to C4) |
| F2 | FzpA W3, LxMF | Figure 2 mixes the 0 to 2 scale and percentages | promised: two panels | `FzpA_1_rebuttal.md:24`, `LxMF_1_rebuttal.md:54` | C16 | done: Figure E.1 in two panels (C16) |
| F3 | FzpA W3 | Table 2 red-flag arrow | promised: arrow on every row, caption note | `FzpA_1_rebuttal.md:22-28` | C17, C18 | done (C17, C18) |
| F4 | FzpA W3 | Vague "not statements" | promised | `FzpA_1_rebuttal.md:21` | C5 | done (C5) |
| F5 | FzpA W3 | Clinical implications missing | promised | `FzpA_1_rebuttal.md:23` | C20 | done (C20) |
| F6 | FzpA Q1 | Where the seven dimensions came from | answered, then corrected: informal author consensus, no formal method | `FzpA_1_rebuttal.md:30-36`, `FzpA_2_followup.md:9` | W3, C21 | done (C21) |
| F7 | FzpA Q2 | How each dimension looks in text | promised: a verbatim example per dimension and the hedging anchor | `FzpA_1_rebuttal.md:40`, `FzpA_2_followup.md:23-27` | C22, C23 | done (C22, C23) |
| F8 | LxMF Q1 | Table 1 n of 200, 191 and 192 | answered: CoT conditions lose prompts. The posted pooled n range is wrong for the CoT conditions. | `LxMF_1_rebuttal.md:5-7` | W10, C19 | done (C19, W10) |

## G. Cost (FzpA)

| ID | Source | Concern | Outcome | Answered in | Open | Resubmission |
|---|---|---|---|---|---|---|
| G1 | FzpA | Training and inference cost across setups | promised: cost per configuration, timed in isolation | `FzpA_1_rebuttal.md:62-64` | C25 | done: the R2 and R5 benches ran (E20); as served (unmerged, 4-bit base) the adapter is not faster than the two-pass protocol; merged weights not timed (section 4.3, Appendix K) |

## H. Framing

| ID | Source | Concern | Outcome | Answered in | Open | Resubmission |
|---|---|---|---|---|---|---|
| H1 | HomM W2 | "The authors are likely already pursuing extensions" | answered | `HomM_1_rebuttal.md:1` | none | no change needed |
| H2 | HomM | "Workshop, not conference" | answered; HomM then listed the consolidation points | `HomM_1_rebuttal.md:51-59`, `HomM_3_followup_part2.md:28-30`, `HomM_4_consolidation.md` | C8 to C11 | done (C8 to C11) |

## I. Follow-ups, final justifications and the citation check

| ID | Source | Concern | Outcome | Answered in | Open | Resubmission |
|---|---|---|---|---|---|---|
| I1 | FzpA follow-up (W1, W2, Q1) | Criteria of clinical decision support applied to patient-facing tools; the formal grounding and validation of the criteria | answered: "clinical decision support" was the wrong term; the grounding is the cited literature; the selection was informal author consensus | `FzpA_2_followup.md:3-9` | W2, W3, C6, C7, C21 | done: the setting is patient-facing question answering (section 1, C6), and section 3 says the seven came from informal author consensus with no formal method (C21); no physician graded the dimensions, which stays deferred (E27, D20) |
| I2 | FzpA follow-up (W3) | Specific clinical implications; walk through how the recipe is used | answered: the antidepressant example and the cost of the wrapper; the recipe as four steps | `FzpA_2_followup.md:15-21` | C10, C20 | done: "What these behaviors mean clinically" and "The trade-off" in section 5 (C10, C20), with the criterion's measurable parts in Table C.1 (E07); the method section gives the four steps |
| I3 | FzpA follow-up (Q2) | A concrete "qualified statement" | answered: the grader's anchor and a case-bound example | `FzpA_2_followup.md:23-27` | C23 | changed (C23): the paper quotes the anchor as the grader states it |
| I4 | FzpA final justification | The use case is not scoped; the rebuttal added so much that the paper needs a full new review | final, no reply possible | none | W2, C6, C11 | done for the scope: patient-facing question answering throughout (C6, D4); the resubmission is that full review, with every claim made in the discussion corrected or withdrawn (C11, `withdrawals.md`) and an analysis plan fixed before any new run (P1) |
| I5 | HomM follow-up (W1) | What "BioMistral-7B is a scope condition" means | withdrawn as wording: the cell is where the recipe stopped working | `HomM_2_followup_part1.md:5` | W5 | done (W5); the teacher-or-student test is E21, R9, built, awaiting GPU |
| I6 | HomM follow-up (W1) | The convergence argument is not supported | withdrawn | `HomM_2_followup_part1.md:7-9` | W4, C11 | done: the sentence is gone (W4); E01 reports the spreads with no claim |
| I7 | HomM follow-up (W1) | The tables behind the prose values | done: three tables posted | `HomM_2_followup_part1.md:11-41`, `HomM_3_followup_part2.md:3-18` | W6, W7, C9, C24 | done: Table 1 and the per-cell appendix tables come from the uniform analysis of every cell (E01) |
| I8 | HomM follow-up (W3, Q4) | Clinically tuned against general judges belongs in the discussion as a limitation | answered: agreed | `HomM_3_followup_part2.md:20-22` | C13, C26 | done: the Limitations say a clinically tuned evaluator is untested (C13); the judge panel ran (Table K.2) |
| I9 | HomM follow-up (W4, Q3) | An empirical comparison with one or two of the closest published methods | answered: a justified scope limit; the comparison is promised | `HomM_3_followup_part2.md:24-26` | C14, C27 | done: three prompt baselines, Tian et al. (2023) among them (H4, section 4.3, section 5.1) |
| I10 | HomM final justification | Consistent aggregate drops, and when the behavioral gain justifies them | final, no reply possible | `HomM_4_consolidation.md:3` | C8, C9, C10 | done: the trade-off is in the abstract and Table 1 for every cell (C8, C9); the criterion and its measurable parts are in section 5 and Table C.1 (C10, E07); the axes of the drop are in Table H.2 (E05) |
| I11 | HomM final justification | Claims, terms and readings corrected or withdrawn during the discussion | final, no reply possible | `HomM_4_consolidation.md:3` | C11 | done: `withdrawals.md` W1 to W25; the paper carries only the corrected forms (C11) |
| I12 | AC final justification (MR-F) | Limited baselines, narrow teaching protocols and base models; the discussion changed the scope; a full new review is needed | final | none | C24, C27, C29 | partial: baselines (R2) and a second protocol (R5) ran at one set of settings (section 4.3); the clinical base at those settings (R7) is running on NRP (D30): its five adapters are trained, its evals run now |
| I13 | Program chairs' citation check | 0 of 22 references flagged | no action needed | none | none | done: the check ran again on all 45 references of the resubmission on 2026-10-11, and one title was fixed (EXPERIMENTS.md P2) |
