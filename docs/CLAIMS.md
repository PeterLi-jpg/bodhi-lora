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
| 1 | The adapter raises active inquiry and context-seeking without the prompt at inference | live, LLM-graded | `results_modal/seed_*/epistemic_scores.json`, `make numbers` | D5, D7 | 99d97ec |
| 2 | The behavior shift holds across base models and benchmarks | restated: holds in the added cells except BioMistral-7B; Med42-8B cells are not used for cross-model statements | `data/rebuttal_cells/generated_values.json` (`make cells` from the raw outputs synced 2026-10-08) | D6, D12 | 99d97ec, 38b800e |
| 3 | Aggregate clinical quality is preserved; non-inferiority; no degradation | withdrawn: report LoRA minus Base per cell with a t(4) interval and state the trade-off | `results_modal/`, added-cell data, `make numbers` | D2; EVAL-1, PROV-8 | 99d97ec |
| 4 | Aggregate score "essentially unchanged" in the added cells | withdrawn: it fell in all six added cells that are not set aside, each drop significant after Holm | `rebuttal/results/quality_drop.json`, `data/rebuttal_cells/generated_values.json` (raw outputs) | D2; CLAIM-1 | 99d97ec, 38b800e |
| 5 | The aggregate drop is a rubric artifact (asking is penalized) | withdrawn: in the added cells outside the evaluator's family, LoRA's responses that ask do not score consistently lower than those that answer (higher in 3 of 4 cells, near zero in BioMistral-7B, where asking did not rise and the score still fell) | `data/rebuttal_cells/generated_values.json` (`lora_askers_minus_non_askers`); `rebuttal/analyses/FINDINGS_quality_drop.md` section 2 (ad hoc code) | CLAIM-1 | 99d97ec, 38b800e |
| 6 | BioMistral-7B shows a "scope condition" of the recipe | withdrawn: reported as a null result with an untested hypothesis | added-cell data | D12; CLAIM-1 | 99d97ec |
| 7 | Four of five added cells match or exceed the teacher | restated: state the count with the metric named | added-cell data, `make numbers` | CLAIM-1 | 99d97ec |
| 8 | Blanket disclaimers are eliminated | restated: counts per condition with a two-sided Fisher exact p, in the results only | `results_modal/seed_*/epistemic_scores.json` | D10; CLAIM-8 | 99d97ec |
| 9 | CoT and LoRA compete for the same behavioral capacity | withdrawn: replaced by the format-leakage account, labeled post hoc | `rebuttal/analyses/interference_mechanism.py` | D11 | 99d97ec |
| 10 | The method distills clinical epistemic calibration | withdrawn: calibration is not measured | none | D3; CLAIM-7 | 99d97ec |
| 11 | The target is clinical decision support; the dimensions were shaped by practising physician co-authors | withdrawn: the scope is patient-facing open-ended QA; the dimensions came from informal author consensus | none | D4; CLAIM-3 | 99d97ec |
| 12 | Physicians validated the LLM grader | withdrawn: 1 of 3 raters returned, agreement below the 0.6 target | returned grades (blocked), `scripts/compute_irr_kappa.py` | D5; EVAL-5 | 99d97ec |
| 17 | The physician check (kappa 0.35) bears on the seven behavior anchors | withdrawn: the check grades HealthBench rubric criteria only; no physician graded the seven dimensions | `scripts/make_irr_excel.py`, `scripts/compute_irr_kappa.py` | D5 | a59cb82 |
| 18 | Context-seeking 2 requires a direct question; bare referral scores 0 on hedging | restated: the grader often departs from both anchors, and the paper reports the rates | `results_modal/seed_*/epistemic_scores.json`, `make numbers` | D5 | a59cb82 |
| 13 | The adapter is statistically indistinguishable from the wrapper at lower cost | withdrawn as an equivalence claim; the output-length difference can be reported as measured | `results_modal/`, `make numbers` | D2; CLAIM-9 | 99d97ec |
| 14 | Cross-family grading prevents filter-grader bias | restated: it separates the families; it does not rule out correlated preferences, and Med42-8B breaks it | design, `bodhi_lora/pipeline/grade.py` | D6; CLAIM-9 | 99d97ec |
| 15 | Length analysis rules out verbosity | withdrawn unless a direct test is added | `results_modal/`, `make numbers` | CLAIM-9 | 99d97ec |
| 16 | A general framework for epistemic behaviors | restated: a recipe and a behavior decomposition for patient-facing open-ended QA | all of the above | D4; CLAIM-2 | 99d97ec |
| 19 | The MedQA cell's aggregate score rises (`HomM_2_followup_part1.md:11`) | restated: it rises in the only MedQA run, which used a prompt that told the model to ask, so the cell measures instruction following and is set aside | `data/rebuttal_cells/generated_values.json`, `rebuttal/results/quality_drop.json`; withdrawals W12 | D12; PROV-1 | 38b800e |
| 20 | The added cells re-ran the pipeline unchanged, every hyperparameter fixed | withdrawn: they used another stack and recipe (r=16, alpha=32, seven modules, effective batch 16, 3 epochs, best-validation checkpoint); tau was 0.4 in every cell | `rebuttal/results/cell_recipe.json` (host filter logs, synced adapter and trainer files); withdrawals W11 | D1; PROV-3 | 38b800e |
