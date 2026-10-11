# Reviewer concerns: how each was answered

Each concern from the meta-review (MW) and from reviewers FzpA, LxMF and HomM, with the
outcome at the end of the discussion. "Answered in" points to `responses/posted/`. "Open"
points to an ID in `commitments.md` (C) or `withdrawals.md` (W).

Outcome values (at the end of the discussion):
- `done`: the work ran and the result was posted.
- `partial`: some of it ran.
- `promised`: text or figure changes promised for the camera-ready.
- `not run`: planned or promised, but not done.

The last column, "Resubmission", says where each concern stands in the resubmission on
2026-10-11. "built, awaiting GPU" means the code exists and passes its CPU tests, the run
is in `docs/RUN_PLAN.md`, and the paper names it as not run (section 5.3).

## A. Generality

| ID | Source | Concern | Outcome | Answered in | Open | Resubmission |
|---|---|---|---|---|---|---|
| A1 | MW1, FzpA W2 | MedGemma does not fit HealthBench's patient-facing use | done: added Mistral-Small-24B, a general-purpose model | `FzpA_1_rebuttal.md:9-15` | C6 | done: Table 1 and section 4.1 |
| A2 | MW2, HomM W1/Q2, FzpA Q3 | One model and one benchmark, but a "framework" claim | done: seven added cells over three families and three benchmarks; claim narrowed | `FzpA_1_rebuttal.md:5`, `HomM_1_rebuttal.md:7-25` | C1, C24 | done: Table 1; the claim of generality goes no further than the counted cells (section 4.1) |
| A3 | HomM Q2, LxMF | One more clinical model or eval set | done: Med42-8B, ChatDoctor and MedQuAD. The MedQA cell was set aside. | `HomM_1_rebuttal.md:9-21`, `HomM_2_followup_part1.md:11` | W12, C24 | done: Table 1 |
| A4 | HomM W1 | Only one CoT protocol | not run | `HomM_1_rebuttal.md:27` | C12, C29 | built, awaiting GPU (R5); one protocol stated in the Limitations |

## B. Calibration or mimicry (MW4, LxMF)

| ID | Source | Concern | Outcome | Answered in | Open | Resubmission |
|---|---|---|---|---|---|---|
| B1 | MW4, LxMF Q2 | Real calibration or surface mimicry? LxMF's split of withholding against self-contained prompts | done: discrimination index on the submitted data | `LxMF_1_rebuttal.md:9-30` | none | withdrawn (W24): the posted index used a keyword labelling. The theme-tag index and the physician-label test E02 (section 4.4, Table C.1) detect no rise in any cell |
| B2 | MW4 | Restraint on well-specified prompts | partial: MedQuAD reported; MedQA set aside | `LxMF_1_rebuttal.md:74` | C24 | as at the discussion; MedQA stays set aside (W12) |
| B3 | MW4, LxMF Q2 | Asks when information is missing, holds back when it is not | not run: the MediQ probe | none | none | built, awaiting GPU (R3); the MediQ file needs a download go-ahead first |

## C. Interference mechanism (LxMF W4)

| ID | Source | Concern | Outcome | Answered in | Open | Resubmission |
|---|---|---|---|---|---|---|
| C1 | LxMF W4 | "Competition" is speculative; it could be truncation or attention dilution | done: format leakage, 54.6% under LoRA+CoT; the mechanism claim was restated | `LxMF_1_rebuttal.md:40-50` | W9, C31 | done: section 4.3, leak rule now the seven headers only (W25), and truncation named as a cause it cannot rule out, because the outputs did not record the finish reason (the new runs record it, E13); red flags first built, awaiting GPU (R2, R7) |

## D. Grading (MW3, LxMF Q3, HomM W3/Q4)

| ID | Source | Concern | Outcome | Answered in | Open | Resubmission |
|---|---|---|---|---|---|---|
| D1 | LxMF Q3, HomM Q4 | Why a Llama-3.1-8B judge and not a clinical model? | partial: the structural reason was given. The clinical-judge re-grade did not run. | `LxMF_1_rebuttal.md:32-36`, `HomM_1_rebuttal.md:35`, `HomM_3_followup_part2.md:22` | C13, C26 | built, awaiting GPU (R1, R6); the Limitations say a clinically tuned grader is untested |
| D2 | MW3, HomM W3 | Claims rest on automatic grading of subjective dimensions | partial: physician validation, 1 of 3 raters, kappa 0.35 against a 0.6 target | `HomM_1_rebuttal.md:37`, `FzpA_2_followup.md:9` | C28 | deferred (D20): one of three raters (section 3.3, Appendix B) |

## E. Prior work (MW6, HomM W4/Q3)

| ID | Source | Concern | Outcome | Answered in | Open | Resubmission |
|---|---|---|---|---|---|---|
| E1 | MW6, HomM W4 | No comparison to prior literature | promised: a Discussion subsection with a comparison table | `HomM_1_rebuttal.md:43-49`, `HomM_3_followup_part2.md:26` | C14, C15 | done: section 5.1 and Table K.1, with what each approach was tested on |
| E2 | HomM Q3 | No empirical comparison to calibration methods | not run | `HomM_1_rebuttal.md:49` | C27 | built, awaiting GPU (R2); section 5.1 says it is a positioning, not a measurement |

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
| G1 | FzpA | Training and inference cost across setups | promised: cost per configuration, timed in isolation | `FzpA_1_rebuttal.md:62-64` | C25 | built, awaiting GPU (R2 bench); section 4.4 says the shared-host times do not show the adapter faster |

## H. Framing

| ID | Source | Concern | Outcome | Answered in | Open | Resubmission |
|---|---|---|---|---|---|---|
| H1 | HomM W2 | "The authors are likely already pursuing extensions" | answered | `HomM_1_rebuttal.md:1` | none | no change needed |
| H2 | HomM | "Workshop, not conference" | answered; HomM then listed the consolidation points | `HomM_1_rebuttal.md:51-59`, `HomM_3_followup_part2.md:28-30`, `HomM_4_consolidation.md` | C8 to C11 | done (C8 to C11) |
