# Claims ledger

Convention 1 in [CONVENTIONS.md](CONVENTIONS.md). One row per claim that the abstract,
the contributions or a posted reply has made. The ledger was seeded on 2026-10-07 from
the audit and [DECISIONS.md](DECISIONS.md).

Columns:

- **Status**: live (the paper makes it), restated (the paper makes a narrower claim) or
  withdrawn (the paper must not make it).
- **Evidence**: the result file and the script that produce the number. The number
  itself comes from `make numbers`; it is not copied here.
- **Paper commit**: the commit that brought `paper/main.tex` in line with the status.
  "pending" means the paper text has not been changed yet.

| # | Claim as first made | Status | Evidence | Basis | Paper commit |
|---|---|---|---|---|---|
| 1 | The adapter raises active inquiry and context-seeking without the prompt at inference | live, LLM-graded | `results_modal/seed_*/epistemic_scores.json`, `make numbers` | D5, D7 | pending |
| 2 | The behavior shift holds across base models and benchmarks | restated: holds in the added cells except BioMistral-7B; Med42-8B cells are not used for cross-model statements | `data/rebuttal_cells/posted_values.json` until the sync | D6, D12 | pending |
| 3 | Aggregate clinical quality is preserved; non-inferiority; no degradation | withdrawn: report LoRA minus Base per cell with a t(4) interval and state the trade-off | `results_modal/`, added-cell data, `make numbers` | D2; EVAL-1, PROV-8 | pending |
| 4 | Aggregate score "essentially unchanged" in the added cells | withdrawn: it fell in all six reported added cells | `rebuttal/analyses/FINDINGS_quality_drop.md` | D2; CLAIM-1 | pending |
| 5 | The aggregate drop is a rubric artifact (asking is penalized) | withdrawn: the drop persists where neither model asks | `rebuttal/analyses/FINDINGS_quality_drop.md` section 2 | CLAIM-1 | pending |
| 6 | BioMistral-7B shows a "scope condition" of the recipe | withdrawn: reported as a null result with an untested hypothesis | added-cell data | D12; CLAIM-1 | pending |
| 7 | Four of five added cells match or exceed the teacher | restated: state the count with the metric named | added-cell data, `make numbers` | CLAIM-1 | pending |
| 8 | Blanket disclaimers are eliminated | restated: counts per condition with a two-sided Fisher exact p, in the results only | `results_modal/seed_*/epistemic_scores.json` | D10; CLAIM-8 | pending |
| 9 | CoT and LoRA compete for the same behavioral capacity | withdrawn: replaced by the format-leakage account, labeled post hoc | `rebuttal/analyses/interference_mechanism.py` | D11 | pending |
| 10 | The method distills clinical epistemic calibration | withdrawn: calibration is not measured | none | D3; CLAIM-7 | pending |
| 11 | The target is clinical decision support; the dimensions were shaped by practising physician co-authors | withdrawn: the scope is patient-facing open-ended QA; the dimensions came from informal author consensus | none | D4; CLAIM-3 | pending |
| 12 | Physicians validated the LLM grader | withdrawn: 1 of 3 raters returned, agreement below the 0.6 target | returned grades (blocked), `scripts/compute_irr_kappa.py` | D5; EVAL-5 | pending |
| 13 | The adapter is statistically indistinguishable from the wrapper at lower cost | withdrawn as an equivalence claim; the output-length difference can be reported as measured | `results_modal/`, `make numbers` | D2; CLAIM-9 | pending |
| 14 | Cross-family grading prevents filter-grader bias | restated: it separates the families; it does not rule out correlated preferences, and Med42-8B breaks it | design, `scripts/regrade_cross_family.py` | D6; CLAIM-9 | pending |
| 15 | Length analysis rules out verbosity | withdrawn unless a direct test is added | `results_modal/`, `make numbers` | CLAIM-9 | pending |
| 16 | A general framework for epistemic behaviors | restated: a recipe and a behavior decomposition for patient-facing open-ended QA | all of the above | D4; CLAIM-2 | pending |
