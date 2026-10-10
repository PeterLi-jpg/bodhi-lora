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

New runs can change these rows. [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) section 11 says, before
any result exists, which result changes which claim and how. When a planned result lands,
the row changes by that rule, with the commit of the result in the Evidence column, and
the paper follows in the same pull request.

| # | Claim as first made | Status | Evidence | Basis | Paper commit |
|---|---|---|---|---|---|
| 1 | The adapter raises active inquiry and context-seeking without the prompt at inference | live, LLM-graded | `results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_*/grade/epistemic__llama-3.1-8b.json`, `make numbers` | D5, D7 | 99d97ec |
| 2 | The behavior shift holds across base models and benchmarks | restated: active inquiry and context-seeking both rise (Holm p < 0.05) in every cell outside the evaluator's family that is not set aside, the submitted cell included, except BioMistral-7B; every cell gets the same seed-level tests; the Med42-8B and MedQA cells are reported but left out of every count | `results/cross_cell/generated_values.json` (`make cells` from the raw outputs synced 2026-10-08), `uniform_section` of `scripts/paper_stats.py` | D6, D12, D22 | 99d97ec, 38b800e, ab58ee5 |
| 3 | Aggregate clinical quality is preserved; non-inferiority; no degradation | withdrawn: report LoRA minus Base per cell with a t(4) interval and state the trade-off | `results/healthbench__medgemma-27b-text__bodhi__2026-05/`, added-cell data, `make numbers` | D2; EVAL-1, PROV-8 | 99d97ec |
| 4 | Aggregate score "essentially unchanged" in the added cells | withdrawn: it fell in all six added cells that are not set aside, each drop significant after Holm | `results/cross_cell/quality_drop.json`, `results/cross_cell/generated_values.json` (raw outputs) | D2; CLAIM-1 | 99d97ec, 38b800e |
| 5 | The aggregate drop is a rubric artifact (asking is penalized) | withdrawn: in the added cells outside the evaluator's family, LoRA's responses that ask do not score consistently lower than those that answer (higher in 3 of 4 cells, near zero in BioMistral-7B, where asking did not rise and the score still fell) | `results/cross_cell/generated_values.json` (`lora_askers_minus_non_askers`); `rebuttal/responses/drafts/FINDINGS_quality_drop.md` section 2 (ad hoc code) | CLAIM-1 | 99d97ec, 38b800e |
| 6 | BioMistral-7B shows a "scope condition" of the recipe | withdrawn: reported as a null result with an untested hypothesis | added-cell data | D12; CLAIM-1 | 99d97ec |
| 7 | Four of five added cells match or exceed the teacher | restated: state the count with the metric named | added-cell data, `make numbers` | CLAIM-1 | 99d97ec |
| 8 | Blanket disclaimers are eliminated | restated: counts per condition with a two-sided Fisher exact p, in the results only | `results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_*/grade/epistemic__llama-3.1-8b.json` | D10; CLAIM-8 | 99d97ec |
| 9 | CoT and LoRA compete for the same behavioral capacity | withdrawn: replaced by the format-leakage account, labeled post hoc, for the submitted cell; since 2026-10-10 a reply leaks when it repeats two of the seven Pass 1 section headers, and the posted nine-marker rule, with two free words, is withdrawn (withdrawals W25) | `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json` (`leakage`), the `analysis-format` rule of `configs/protocols/bodhi.yaml`; `rebuttal/analyses/interference_mechanism.py` is only the record of the posted rule | D11; CONVENTIONS rule 15 | 99d97ec, 62b8179 |
| 10 | The method distills clinical epistemic calibration | withdrawn: calibration is not measured | none | D3; CLAIM-7 | 99d97ec |
| 11 | The target is clinical decision support; the dimensions were shaped by practising physician co-authors | withdrawn: the scope is patient-facing open-ended QA; the dimensions came from informal author consensus | none | D4; CLAIM-3 | 99d97ec |
| 12 | Physicians validated the LLM grader | withdrawn: 1 of 3 raters returned, agreement below the 0.6 target | returned grades (blocked), `scripts/compute_irr_kappa.py` | D5; EVAL-5 | 99d97ec |
| 17 | The physician check (kappa 0.35) bears on the seven behavior anchors | withdrawn: the check grades HealthBench rubric criteria only; no physician graded the seven dimensions | `scripts/make_irr_excel.py`, `scripts/compute_irr_kappa.py` | D5 | a59cb82 |
| 18 | Context-seeking 2 requires a direct question; bare referral scores 0 on hedging | restated: the grader often departs from both anchors, and the paper reports the rates | `results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_*/grade/epistemic__llama-3.1-8b.json`, `make numbers` | D5 | a59cb82 |
| 13 | The adapter is statistically indistinguishable from the wrapper at lower cost | withdrawn as an equivalence claim; the output-length difference can be reported as measured | `results/healthbench__medgemma-27b-text__bodhi__2026-05/`, `make numbers` | D2; CLAIM-9 | 99d97ec |
| 14 | Cross-family grading prevents filter-grader bias | restated: it separates the families; it does not rule out correlated preferences, and Med42-8B breaks it | design, `bodhi_lora/pipeline/grade.py` | D6; CLAIM-9 | 99d97ec |
| 15 | Length analysis rules out verbosity | withdrawn unless a direct test is added | `results/healthbench__medgemma-27b-text__bodhi__2026-05/`, `make numbers` | CLAIM-9 | 99d97ec |
| 16 | A general framework for epistemic behaviors | restated: a recipe and a behavior decomposition for patient-facing open-ended QA | all of the above | D4; CLAIM-2 | 99d97ec |
| 19 | The MedQA cell's aggregate score rises (`HomM_2_followup_part1.md:11`) | restated: it rises in the only MedQA run, which used a prompt that told the model to ask, so the cell measures instruction following and is set aside | `results/cross_cell/generated_values.json`, `results/cross_cell/quality_drop.json`; withdrawals W12 | D12; PROV-1 | 38b800e |
| 20 | The added cells re-ran the pipeline unchanged, every hyperparameter fixed | withdrawn: they used another stack and recipe (r=16, alpha=32, seven modules, effective batch 16, 3 epochs, best-validation checkpoint); tau was 0.4 in every cell | `results/cross_cell/cell_recipe.json` (host filter logs, synced adapter and trainer files); withdrawals W11 | D1; PROV-3 | 38b800e |
| 21 | The red-flag drop under LoRA+BODHI does not repeat in the added cells (new in the paper, 2026-10-09) | live, LLM-graded: in no added cell outside the evaluator's family is red-flag identification lower under LoRA+BODHI than under LoRA (p < 0.05) | `results/cross_cell/generated_values.json`, `uniform_section` of `scripts/paper_stats.py` | D6, D22 | ab58ee5, 07de88d |
| 22 | The adapter needs about 58% fewer output tokens than the protocol (`FzpA_2_followup.md:19`) | withdrawn: no file supports the value; the two-pass conditions stored only their final response, so their tokens cannot be counted; the paper gives the engine's counts for Base and LoRA | `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json` (`cost`); withdrawals W22 | no audit ID (found 2026-10-09) | ab58ee5 |
| 23 | Without the grader, the base asks the reader a question in 18.4% of responses and the adapter in 39.6% (`FzpA_2_followup.md:13`) | withdrawn: a keyword rule (a sentence with "?" and "you" or "your") cannot decide whether a reply asks; the evaluator's active-inquiry grade is the measure, and a judge cross-check with a hand-label gate is planned (ANALYSIS_PLAN.md section 12, 2026-10-10) | `rebuttal/analyses/second_person_questions.py` (record only); withdrawals W23 | CONVENTIONS rule 15; PROV-7 | 62b8179 |
| 24 | The adapter discriminates information-withholding prompts (+10.5 points) and the base does not (+2.1) (`LxMF_1_rebuttal.md:3`, Table R1) | withdrawn: the labelling matched keywords in rubric text; the theme-tag labelling stays, post hoc, and its LoRA minus Base interval includes zero (claim of selectivity already withdrawn, W15) | `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json` (`discrimination.active_inquiry.needs-info-theme`); `rebuttal/analyses/mimicry_split.py` (record only); withdrawals W24 | D9; CONVENTIONS rule 15 | 62b8179 |
| 25 | 14 of 1,000 HealthBench-Hard prompts identify the speaker as a clinician (`FzpA_1_rebuttal.md:46`, `FzpA_2_followup.md:5`) | withdrawn: a keyword count; the committed keyword rule's 4 of 667 is withdrawn with it. HealthBench's own `physician_agreed_category:health-professional` tag replaces it, a lower bound because few prompts carry the tag | `data/raw/healthbench/hard.jsonl` (tags), `speaker_tags` of `scripts/paper_stats.py`; `rebuttal/analyses/mimicry_split.py` (record only); withdrawals W14 | CONVENTIONS rule 15 | 62b8179 |
