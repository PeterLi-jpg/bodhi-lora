# Rebuttal plan: closed

The NeurIPS 2026 discussion for submission 32818 ("Ask Before You Answer") is over, and the
paper was rejected. The reviewer ratings were FzpA 3, LxMF 4 and HomM 2. The AC meta-review
listed six weaknesses. This file records what the plan was and what happened. For the record
and the open items, see `README.md`, `withdrawals.md` and `commitments.md`.

Git history keeps the earlier plan, with its hardware notes, time estimates and setup
checklists.

## Meta-review weaknesses: what we did

| # | Weakness | What we did | Answered in | Still open |
|---|---|---|---|---|
| W1 | MedGemma does not fit HealthBench's use case | Added a general-purpose base (Mistral-Small-24B) and a clinical base from another family (Med42-8B). Named the scope as patient-facing. | `FzpA_1_rebuttal.md`, `FzpA_2_followup.md` | C6, C7 |
| W2 | One pipeline, but a "general framework" claim | Narrowed the claim. Ran seven added cells: six reported, one set aside. | `FzpA_1_rebuttal.md`, `HomM_1_rebuttal.md`, `HomM_2_followup_part1.md` | C1, C2, C24, C29 |
| W3 | Automated grading is limited | Reported the partial physician validation (1 of 3 raters, kappa 0.35). The clinical-judge re-grade did not run. | `LxMF_1_rebuttal.md`, `HomM_1_rebuttal.md`, `HomM_3_followup_part2.md` | C13, C26, C28 |
| W4 | Calibration or mimicry? | Ran LxMF's test on the submitted data: the discrimination index. | `LxMF_1_rebuttal.md` | none |
| W5 | Discussion and context lack clarity | Promised text, figure and table fixes. | `FzpA_1_rebuttal.md`, `LxMF_1_rebuttal.md` | C5, C16 to C23 |
| W6 | No comparison with prior work | Positioning only. The baseline did not run. | `HomM_1_rebuttal.md`, `HomM_3_followup_part2.md` | C14, C15, C27 |

All reply files are in `responses/posted/`.

## What ran

Five seeds per cell. The filter is Qwen2.5-14B-Instruct and the evaluator is
Llama-3.1-8B-Instruct.

| Benchmark | Base model | Result as posted |
|---|---|---|
| HealthBench-Hard | MedGemma-27B | the submitted cell |
| HealthBench-Hard | Mistral-Small-24B | active inquiry rises |
| HealthBench-Hard | Med42-8B | active inquiry rises |
| HealthBench-Hard | BioMistral-7B | null |
| ChatDoctor | Mistral-Small-24B | active inquiry rises |
| ChatDoctor | Med42-8B | active inquiry rises |
| MedQuAD | Mistral-Small-24B | active inquiry rises |
| MedQA | Mistral-Small-24B | set aside (see `withdrawals.md` W12) |

The posted numbers are in `data/rebuttal_cells/posted_values.json`. The raw outputs are
still on the shared GPU host (audit PROV-1).

## Planned but not run

- The MediQ paired complete-against-incomplete probe.
- MedicationQA, MedDialog and MedMCQA.
- MedGemma and BioMistral on MedQA or MedQuAD.
- A second CoT protocol, or a BODHI ablation variant (C29).
- The clinical-judge re-grade (C26).
- `analyses/prior_work_baseline.py`, the inference-time calibration baseline (C27).
- The cost table from `scripts/latency_benchmark.py` (C25).

## Rules that still hold

- **Grader families.** The filter is Qwen and the evaluator is Llama-3. A new base model must
  avoid both families, or the asymmetric-grading argument breaks. The Med42-8B cells break
  this rule (Llama-3 base, Llama-3 evaluator). A re-grade of those cells needs a judge from
  neither family (audit EVAL-7).
- **Seeds.** 42, 7, 13, 99 and 101, the same as the paper.
