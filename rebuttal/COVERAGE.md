# Rebuttal coverage matrix — every reviewer concern -> response

Goal: NOTHING uncovered. Every distinct concern from the meta-review (MW) and Reviewers
FzpA / LxMF / HomM is listed with its response, artifact, and whether it needs a GPU run.
Status: [done] local artifact exists; [build] code/text to write; [run] needs a cluster run.

## A. Generality (the central critique)
| ID | Source | Concern | Response | Status |
|---|---|---|---|---|
| A1 | MW1, FzpA W2 | MedGemma not aligned with HealthBench's patient-facing use | add **Mistral-Small-24B-Instruct** (general-domain) as a base model | [run] |
| A2 | MW2, HomM W1/Q2, FzpA Q3 | one model / one benchmark vs "framework" | 3x3 grid: {MedGemma-27B, Mistral-Small-24B, BioMistral-7B} x {HealthBench, MedQA-open, MedQuAD} | [run] |
| A3 | HomM Q2, LxMF | one additional clinical eval / model | **BioMistral-7B** (2nd clinical model) + **MedQA-open** (clinician-facing benchmark) | [run] |
| A4 | HomM W1 | one CoT protocol | run a **BODHI ablation variant** (same pipeline, `_bodhi_ablation.py`) on one cell to show protocol-robustness; reframe the "framework" claim (see F1) | [run/build] |

## B. Calibration vs. mimicry (MW4, LxMF W2)
| ID | Source | Concern | Response | Status |
|---|---|---|---|---|
| B1 | MW4, LxMF Q2 | is it real calibration or surface-template mimicry? LxMF's exact test: missing-info vs answerable split | `rebuttal/analyses/mimicry_split.py`: Base does NOT reliably discriminate; Wrapper & LoRA DO (theme-only LoRA discrimination +13.2pp, CI [+4.1,+21.7]). Adapter reproduces wrapper targeting. | [done] |
| B2 | MW4 | restraint on well-specified prompts | MedQA-open + MedQuAD are complete-info -> show the adapter does NOT over-ask there (anti-mimicry) | [run] |
| B3 | MW4, LxMF Q2 | asks when info missing, restrains when not | **MediQ probe**: paired complete-vs-incomplete same cases; grade active-inquiry on the split | [run, eval-only] |

## C. Interference mechanism (LxMF W4)
| ID | Source | Concern | Response | Status |
|---|---|---|---|---|
| C1 | LxMF W4 | CoT/LoRA competition is speculative; could be truncation / attention dilution | `rebuttal/analyses/interference_mechanism.py`: drop is 54.6% Pass-1-format leakage; non-leaked hold at 1.73 (>LoRA); red-flag RISES with length within non-leaked -> refutes dilution | [done] |

## D. Grading robustness / clinical judges (MW3, LxMF Q3, HomM W3/Q4)
| ID | Source | Concern | Response | Status |
|---|---|---|---|---|
| D1 | LxMF Q3, HomM Q4 | why Llama-8B judge, not clinical models (Meditron3/Aloe/Med42)? | **clinical-judge re-grade** of epistemic dims with Med42-v2 (or Meditron3); report agreement vs Llama | [run] |
| D2 | MW3, HomM W3 | claims lean on automatic grading of subjective dims | surface the **physician IRR** already in `results_modal/irr/` (`compute_irr_kappa.py`); report updated kappa | [build] |

## E. Prior-work comparison / calibration baselines (MW6, HomM W4/Q3)
| ID | Source | Concern | Response | Status |
|---|---|---|---|---|
| E1 | MW6, HomM W4 | no comparison to prior literature | related-work positioning + an empirical baseline | [build/run] |
| E2 | HomM Q3 | compare vs other calibration / uncertainty-eliciting protocols | inference-time baselines on the eval set: verbalized-confidence (Lin/Tian) + a plain "ask clarifying questions" system prompt; compare to BODHI-LoRA | [run, inference-only] |

## F. Clarity / presentation (MW5, FzpA W1/W3/Q1/Q2, LxMF clarity, LxMF Q1)
| ID | Source | Concern | Response | Status |
|---|---|---|---|---|
| F1 | FzpA W1, MW2 | overstated "framework" claim | scope it to open-ended info-seeking Q&A; add nuance on which problem types fit + how the decomposition shifts across contexts; new experiments convert the claim from asserted to demonstrated | [build-text] |
| F2 | FzpA W3c, LxMF clarity | Figure 2 mixes 0-2 and 0-100 scales | split into two panels (0-2 dims vs rates) | [build-fig] |
| F3 | FzpA W3b | Table 2 red-flag: missing a down-arrow? | clarify caption: red-flag identification is higher-is-better (naming warning signs); only blanket-disclaimer is lower-is-better | [build-text] |
| F4 | FzpA W3d | vague "not statements" ("aggregate is not the primary claim") | rewrite with concrete per-dimension detail | [build-text] |
| F5 | FzpA W3a | discussion lacks specificity on how behaviors improve responses + clinical implications | add per-dimension clinical-implication prose | [build-text] |
| F6 | FzpA Q1 | where did the 7 dims come from (BODHI is 5)? clinician feedback? | map BODHI's 5 (B/O/D/H/I) -> 7 eval dims; note clinician co-authors (Nakitanda, Kasimbazi, issue #73 clinical leads) informed them | [build-text] |
| F7 | FzpA Q2 | how is each epistemic value operationalized in text? | surface the epistemic grader's rubric anchors per dim + verbatim text examples | [build-text] |
| F8 | LxMF Q1 | Table 1 n = 200/191/192 differ | explain: BODHI 2-pass exceeds the 4096-token cap ~4-5%; LoRA has 100% response rate | [build-text] |

## G. Compute comparison (FzpA strength #4 — an implicit ask)
| ID | Source | Concern | Response | Status |
|---|---|---|---|---|
| G1 | FzpA S4 | more explicit training-time + inference-time comparison across setups | table from `scripts/latency_benchmark.py`: train time (LoRA) + inference tokens/latency for Base, Base+CoT, LoRA, LoRA+CoT | [run] |

## H. Framing / limitations already partly in the paper
| ID | Source | Concern | Response | Status |
|---|---|---|---|---|
| H1 | HomM W2 | "authors likely already pursuing extensions" | yes — deliver models+benchmarks+human-eval; say so | [build-text] |
| H2 | HomM overall | "workshop not conference" | rebut with the expanded 3x3 + baselines + human IRR + mechanism analyses | [build-text] |

Anything a reviewer raised that is NOT in a row above is a gap — none known as of this pass.
