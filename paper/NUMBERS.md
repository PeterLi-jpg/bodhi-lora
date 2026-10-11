# Number map for paper/main.tex

Every number and claim in the abstract, the introduction and the results of `paper/main.tex`,
with the macro that prints it, the function of `bodhi_lora/report/` that writes the macro
(`cell_macros` and `detail_tables` in `cell.py`; `cross_table`, `cell_detail_tables`,
`group_macros` and `count_macros` in `cross.py`), and the input. `make numbers` runs
`python -m bodhi_lora report`, which rebuilds every macro from tracked files; nothing in this
table is typed into the paper by hand unless the Kind column says so. No number comes from a
keyword rule (docs/CONVENTIONS.md rule 15).

The paper-level claim ledger, with the status of each claim made in the submission and the
replies, is `docs/CLAIMS.md`. This file is the number-level map for the current text.

Kinds:

- **computed**: the macro comes from committed data in this repository: each cell's
  `analysis/stats.json`, which `make analyze` writes from the cell's tracked response
  tables, and the cross-cell statistics.
- **posted**: the macro comes from `rebuttal/posted_values.json`, a transcription of the
  OpenReview replies, and the paper quotes it as what was posted.
- **design**: a setting of the experiment (seeds, prompts per seed, context length, tau,
  LoRA rank). Its source is the config or launcher named in the Source column.
- **unverified**: typed in the paper; the output that produced it is not in the repository.

Inputs: `MS` is `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json`
(the submitted cell), `CS` is the added cells' `results/*/analysis/stats.json` with the groups
and counts of `results/cross_cell/stats.json` (`configs/report.yaml` names them), `FM` is the
filter manifests and adapter files of each cell as `stats.json` reads them (`filter`,
`recipe`), and `PV` is `rebuttal/posted_values.json`. Until 2026-10-10 the added cells' values
came from `results/cross_cell/generated_values.json` and `results/cross_cell/cell_recipe.json`
through `503b975:scripts/paper_stats.py`; `tests/test_analysis_equivalence.py` checks that the
analysis reproduces both.

## Abstract

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| 5 seeds, 667 unique prompts | `\statHBMedGemmaMaySeeds`, `\statHBMedGemmaMayPrompts` | `cell_macros` | MS | computed |
| Active inquiry 17.5% to 45.6%, +28.0 points, 95% CI [+22.9, +33.2] | `\statHBMedGemmaMayActiveInquiryBase`, `\statHBMedGemmaMayActiveInquiryLoRA`, `\statHBMedGemmaMayActiveInquiryLoRAvsBaseDiff`, `...CI` | `cell_macros`, `cell_macros` | MS | computed |
| No detected change in aggregate score, +0.004 [-0.011, +0.020] | `\statHBMedGemmaMayRubricScoreLoRAvsBaseDiff`, `...CI` | `cell_macros` | MS | computed |
| 8 cells, each with the same seed-level tests | `\statCrossCells` | `count_macros` | CS | computed |
| 5 cells outside the evaluator's family and not set aside; active inquiry and context-seeking both rise (Holm p < 0.05) in 4 | `\statGroupOutsideFamilyCells`, `\statGroupOutsideFamilyAskRose` | `count_macros` | CS (per-seed means; `paired_test` and `holm` of `bodhi_lora/analysis/stats.py`) | computed |
| Those 4 cells cover 2 base models and 3 benchmarks (also section 4.1, where the paper says its claim of generality goes no further) | `\statGroupOutsideFamilyAskRoseModels`, `\statGroupOutsideFamilyAskRoseBenchmarks` | `count_macros` | CS, with the `base_model` and `benchmark` of each cell in PV | computed |
| No HealthBench-Hard cell shows selective asking: LoRA minus Base on the theme-tag asking index is not detectably above zero in any of them | none in the abstract; section 4.4 and Table H.6 give the values | `cell_macros`, `cross_table` | `results/healthbench__*/analysis/stats.json` (`discrimination.active_inquiry.needs-info-theme.contrasts`) | computed (post hoc theme-tag label, D9; read from the four intervals) |
| Aggregate falls in all 4 on each cell's fixed prompt set, by 0.014 to 0.086, largest paired t p = 0.016; with prompts resampled, clear of zero in 2 | `\statGroupAddedOutsideFamilyRubricScoreLoRAvsBaseMeasured`, `\statGroupAddedOutsideFamilyRubricScoreLoRAvsBaseAbsMin`, `\statGroupAddedOutsideFamilyRubricScoreLoRAvsBaseAbsMax`, `\statGroupAddedOutsideFamilyRubricScoreLoRAvsBasePMax`, `\statGroupAddedOutsideFamilyAggregateFellBoot` | `group_macros`, `count_macros` | CS (`contrasts.lora - base.metrics.rubric_score`: `p` and `diff_ci95`, the prompt-clustered bootstrap) | computed; the seed-level t covers training only where every seed has the same prompts (ANALYSIS_PLAN.md section 2.1) |
| Adding CoT to the adapter changes red-flag identification by -0.161 [-0.262, -0.059] on the 0-2 score, Holm-adjusted p = 0.058 (the rate, -20.6 points, is a secondary measure stated in section 4.3) | `\statHBMedGemmaMayRedFlagLoRABodhivsLoRADiff`, `...CI`, `...PHolm` | `cell_macros` | MS | computed |
| Format leakage in 53.2% of the graded LoRA+BODHI answers (at least 2 of the 7 Pass 1 headers, each at the start of a line) | `\statHBMedGemmaMayLeakBodhiAnalysisFormatLoRABodhi` | `cell_macros` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json` (`leakage.leak_bodhi__analysis-format`) | computed (post hoc; the posted 54.6% is withdrawn, W25) |
| The red-flag drop does not repeat in the added cells | none in the abstract; section 4.3 gives the counts | `count_macros` | CS | computed (see the section 4.3 rows) |
| The 2 Med42-8B cells share the evaluator's family and the MedQA cell's prompt told the model to ask, so the counts leave them out | `\statGroupFamilyWaiverCells` | `count_macros` | the `same_family_as_evaluator` and `set_aside` flags of each cell in PV | design (D6; the MedQA run identity: withdrawals W12) |
| Physician agreement kappa 0.35, target 0.6 set in advance for the filter grader, on rubric criteria only; no physician graded the seven dimensions | `\statHBMedGemmaMayKappaEvalLlamaAll`; 0.6 is the target in the plan of 2026-04-29 | `cell_macros` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/kappa_results.json`, computed by `make kappa` from the returned rater file, which carries a name and is kept outside the repo; row design in `scripts/make_irr_excel.py` | computed from an input outside the repo (audit NUM-2): the committed `IRR-clinicians/` sheets are blank, so the value cannot be recomputed from the repo |
| BioMistral-7B is a null result on the seven dimensions; its blanket disclaimers rise from 4.6% to 8.2% (section 4.1) | `\statHBBioMistralJulDimsUpLoRAvsBase`, `\statHBBioMistralJulDimsDownLoRAvsBase`, `\statHBBioMistralJulBlanketDisclaimerPct<Cond>`, `\statHBBioMistralJulBlanketDisclaimerFisherPLoRA` | `cell_macros` | CS (`blanket_disclaimers`) | computed |

## Introduction

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| The two passes overran the context on 4.3% of prompts | `\statHBMedGemmaMayCompletionDropPctBaseBodhi` | `cell_macros` | MS | computed |
| 40 of 1,000 Hard prompts tagged as written by a health professional; 63 carry the tag (lower bound) | `\statHBMedGemmaMayLabelClinicianUserPopPos`, `\statHBMedGemmaMayLabelClinicianUserPopLabelled`, `\statHBMedGemmaMayPopulationPrompts` | `cell_macros`, `cell_macros` | `data/raw/healthbench/hard.jsonl` (`physician_agreed_category` tags) | computed; HealthBench is a public download |
| Posted 14 of 1,000 (Appendix H), from a keyword audit; withdrawn, and no keyword count replaces it | `\statPostedClinicianPrompts` | `count_macros` | PV | posted (withdrawals W14) |
| 8 cells, 7 of them added; active inquiry and context-seeking rise in 4 of the 5 cells with a base model outside the evaluator's family and no prompt that cues asking; aggregate falls in all 4 added cells among them | `\statCrossCells`, `\statGroupAddedCells`, `\statGroupOutsideFamilyAskRose`, `\statGroupOutsideFamilyCells`, `\statGroupAddedOutsideFamilyRubricScoreLoRAvsBaseMeasured` | `count_macros`, `count_macros` | CS | computed |
| BODHI expansion "Balanced, Open-minded, Diagnostic, Humble, Inquisitive" | none | | the cited BMJ paper (docs/DECISIONS.md, D15 note) | design |
| Pass 1 has seven fields: task type, audience, best assessment, key uncertainties, questions to ask, red flags, safe recommendations | none | | `bodhi/prompts.py` in bodhi-llm 0.1.4 (pinned in `pyproject.toml`) | design |

## Method and evaluation

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| 4,000 HealthBench prompts outside the Hard set | `\statHBMedGemmaMayFilterRaw` | `cell_macros` | constant `FILTER_POOL` (5,000 HealthBench minus 1,000 Hard; legacy launcher) | design |
| Seed-7 training split: 2,816 traces, 0 in the Hard set | `\statHBMedGemmaMayFilterTrainSplit`, `\statHBMedGemmaMayFilterTrainOverlap` | `cell_macros` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_7/filter/filter_scores.json` (scores copied from the ignored `seed_7/filter/train.jsonl` of the same cell), `data/raw/healthbench/hard.jsonl` | computed; HealthBench is a public download |
| tau = 0.4, r = 8, alpha = 16, 300 steps, 64 examples per step, 2,048 tokens | none | | `legacy/configs/lora_medgemma27b_tunix.yaml`, `legacy/tpu/launch_5seeds_tunix.sh` (Table hparams comments) | design; not checked against the saved adapters, because the GCS bucket returned 404 on 2026-10-08 (D1, D19) |
| Added cells: r = 16, alpha = 32 (section 3.3, Table F.2, Appendix J) | `\statGroupAddedRecipeRank`, `\statGroupAddedRecipeAlpha` | `group_macros` | FM (the synced `adapter_config.json` of all 35 adapters) | computed |
| 1,000 draws per condition cover 667 unique prompts; 271 repeat | `\statHBMedGemmaMayPromptDraws`, `\statHBMedGemmaMayPrompts`, `\statHBMedGemmaMayPromptsRepeated` | `cell_macros` | MS | computed |
| Epistemic parse failures 0.20% to 1.49% per condition | `\statHBMedGemmaMayEpistemicParseFailPctBase`, `\statHBMedGemmaMayEpistemicParseFailPctMax` | `cell_macros` | MS | computed |
| HealthBench criterion failures at most 0.087% per run | `\statHBMedGemmaMayRubricParseFailPctMaxRun` | `cell_macros` | MS | computed |
| 1 of 3 physicians; kappa 0.35 [0.28, 0.43] on 509 rows from 10 prompts | `\statHBMedGemmaMayKappaRaters`, `\statHBMedGemmaMayKappaEvalLlamaAll`, `\statHBMedGemmaMayKappaEvalLlamaAllCI`, `\statHBMedGemmaMayKappaEvalLlamaAllN`, `\statHBMedGemmaMayKappaEvalLlamaAllPrompts` | `cell_macros` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/kappa_results.json` | computed from an input outside the repo (see the abstract row) |
| 4,096-token evaluation context, greedy decoding, A100-80GB | none | | `legacy/modal_eval.py` (max_model_len patch, GPU type) | design |
| Context-seeking 2 without a direct question: 74.4% of Base 2s, 47.2% of LoRA 2s | `\statHBMedGemmaMayContextSeekingTwoNoQuestionPctBase`, `\statHBMedGemmaMayContextSeekingTwoNoQuestionPctLoRA` | `cell_macros` | MS | computed |
| Mapping of the seven dimensions to the BODHI components (Table bodhi-map) | none | | names in the cited BMJ paper; Pass 1 fields in bodhi-llm 0.1.4 | design (our reading) |
| General graders as in HealthBench (GPT-4.1 grader) | none | | `healthbench2025` | design |
| Hedging 2 given to 4 of the 5 flagged blanket disclaimers (Appendix D) | `\statHBMedGemmaMayBlanketDisclaimerFlaggedHedgingTwo`, `\statHBMedGemmaMayBlanketDisclaimerFlagged` | `blanket_disclaimers` | MS | computed |

## Results: the MedGemma-27B cell (section 4.2; its Tables E.3 and E.4 and Figure E.1 are in Appendix E)

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| Table E.3 (score, t interval, pooled n, n per seed) | `tab_main_healthbench.tex` | `detail_tables` | MS | computed |
| Table E.4 (seven dimensions, rates, scored n) | `tab_epistemic.tex` | `detail_tables` | MS | computed |
| Figure E.1 (two panels) | `paper/figures/epistemic_bar.pdf` | `scripts/plot_epistemic_figures.py` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/cross_seed_aggregate.json` | computed |
| Active inquiry, context-seeking, red flag, scope: diff, CI, Holm p; the four effects keep Holm-adjusted p <= 0.0065 | `\stat<Dim>LoRAvsBaseDiff`, `...CI`, `...PHolm`, `\statHBMedGemmaMayScopeBoundingLoRAvsBasePHolm` (the largest of the four) | `cell_macros` | MS | computed |
| Output tokens per response from the serving engine, Base and LoRA (Appendix H; section 4.4 prints the same values through the report macros) | `\statHBMedGemmaMayTokensEngineBase`, `\statHBMedGemmaMayTokensEngineLoRA` | `cell_macros` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json` (`cost`, written by `make analyze`) | computed |
| Questions per response 0.45 to 1.16 | `\statHBMedGemmaMayQuestionsBase`, `\statHBMedGemmaMayQuestionsLoRA` | `cell_macros` | MS | computed |
| Hedging and specificity significant before but not after Holm | `\statHBMedGemmaMayHedgingQualityLoRAvsBaseDiff`, `\statHBMedGemmaMaySpecificityLoRAvsBaseDiff`, `...PHolm` | `cell_macros` | MS | computed |
| No detected change in uncertainty acknowledgment | `\statHBMedGemmaMayUncertaintyAckLoRAvsBaseDiff`, `...CI` | `cell_macros` | MS | computed |
| Pooled d 0.63, 0.44, 0.20, 0.22 (descriptive) | `\stat<Dim>LoRAvsBaseD` | `cell_macros` | MS | computed |
| Worst case keeps the four effects at p <= 0.035 | `\statHBMedGemmaMayScopeBoundingLoRAvsBaseWorstCaseP` (the largest of the four) | `cell_macros` | MS | computed |
| LoRA vs teacher on active inquiry: -2.4 [-5.7, +1.0] | `\statHBMedGemmaMayActiveInquiryLoRAvsBaseBodhiDiff`, `...CI`, `...CILow` | `cell_macros` | MS | computed |
| Teacher gave no answer on 4.3% of prompts | `\statHBMedGemmaMayCompletionDropPctBaseBodhi` | `cell_macros` | MS | computed |
| Aggregate +0.004 [-0.011, +0.020], p = 0.48; the lower end leaves room for a small drop | `\statHBMedGemmaMayRubricScoreLoRAvsBaseDiff`, `...CI`, `...P` | `cell_macros` | MS | computed |
| Rubric axes: accuracy, completeness, communication, instruction following | `\statAxis<Axis><Cond>` | `axis_scores` | MS | computed |
| Blanket disclaimers 5/998 vs 0/985, Fisher p = 0.062 | `\statHBMedGemmaMayBlanketDisclaimerCountBase`, `\statHBMedGemmaMayScoredBase`, `\statHBMedGemmaMayBlanketDisclaimerCountLoRA`, `\statHBMedGemmaMayScoredLoRA`, `\statHBMedGemmaMayBlanketDisclaimerFisherPLoRA` | `blanket_disclaimers`, `cell_macros` | MS | computed |
| LoRA answers +183 characters [-21, +387] | `\statHBMedGemmaMayCharsLoRAvsBaseDiff`, `...CI` | `cell_macros` | MS | computed |
| Active inquiry rises in every length quartile | `tab_length_quartiles.tex` | `cell_macros` | MS | computed |

## Results: every cell (section 4.1, Table 1; Tables E.1 and E.2 in Appendix E)

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| Table 1, every row (the MedGemma-27B cell first; its per-seed means must equal the ones computed from MS, or the run stops) | `tab_cross_cell.tex` | `count_macros`, `cross_table` | CS (per-seed means; `paired_test` and `holm` of `bodhi_lora/analysis/stats.py`) | computed |
| LoRA - Base with half its 95% t interval for every cell; Holm over the seven dimensions; paired t for the aggregate | `tab_cross_cell.tex` | `count_macros` | CS (per-seed means; `paired_test` and `holm` of `bodhi_lora/analysis/stats.py`) | computed |
| Tables E.1 and E.2: every cell in the four conditions, LoRA - Base and LoRA+BODHI - LoRA | `tab_cell_detail_1.tex`, `tab_cell_detail_2.tex` | `count_macros`, `cell_detail_tables` | CS (per-seed means; `paired_test` and `holm` of `bodhi_lora/analysis/stats.py`) | computed |
| Red flag rises in 4 and scope bounding in 4 of the 5 cells outside the family; a dimension falls in 2 (uncertainty acknowledgment on ChatDoctor, specificity on MedQuAD) | `\statGroupOutsideFamilyRedFlagRose`, `\statGroupOutsideFamilyScopeRose`, `\statGroupOutsideFamilyDimFell`, `\statCDMistralJulUncertaintyAckLoRAvsBaseDiff`, `\statMQMistralJulSpecificityLoRAvsBaseDiff` | `count_macros` | CS (per-seed means; `paired_test` and `holm` of `bodhi_lora/analysis/stats.py`) | computed |
| BioMistral-7B: no dimension rises or falls with Holm p < 0.05 | `\statHBBioMistralJulDimsUpLoRAvsBase`, `\statHBBioMistralJulDimsDownLoRAvsBase` | `count_macros` | CS (per-seed means; `paired_test` and `holm` of `bodhi_lora/analysis/stats.py`) | computed |
| BioMistral-7B: 11.7% to 10.1%, teacher 14.5%, filter kept 20.4% | `\statHBBioMistralJul<...>` | `count_macros` | CS; the filter share from FM | computed |
| Adapter at or above its teacher in 3 of 5 cells (submitted included) | `\statGroupOutsideFamilyAtOrAboveTeacher`, `\statGroupOutsideFamilyCells` | `count_macros` | MS, CS | computed |
| Aggregate falls in all 4 added cells outside the evaluator's family, 0.014 to 0.086; the tests cover training, not prompt sampling; with prompts resampled the drop is clear of zero in 2 (Table 1, prompt-bootstrap column) | `\statGroupAddedOutsideFamilyRubricScoreLoRAvsBaseMeasured`, `\statGroupAddedOutsideFamilyRubricScoreLoRAvsBaseAbsMin`, `\statGroupAddedOutsideFamilyRubricScoreLoRAvsBaseAbsMax`, `\statGroupAddedOutsideFamilyAggregateFellBoot`; Table 1 | `group_macros`, `count_macros`, `cross_table` | CS | computed |
| In no cell outside the evaluator's family do the LoRA responses that ask score detectably lower than those that do not (askers gap; self-selected, descriptive) | none in the text; Table H.6 gives the gap and its bootstrap interval for every cell | `cross_table` | `results/*/analysis/stats.json` (`askers_gap.lora`) | computed (read from the five intervals; withdrawals W7) |
| Table 1 and the cells' statistics are one source: Table 1, the per-cell tables and every macro come from `results/*/analysis/stats.json`; the per-seed means the paper printed before 2026-10-10 (from `results/cross_cell/generated_values.json`) equal these | none | `tests/test_analysis_equivalence.py` | CS | computed (more than 1,000 per-seed means) |
| Largest p of the four drops 0.016; each survives Holm over the 6 added cells not set aside (largest p_Holm 0.016), as posted | `\statGroupAddedOutsideFamilyRubricScoreLoRAvsBasePMax`, `\statGroupAddedKeptRubricScoreLoRAvsBasePHolmMax`, `\statGroupAddedKeptCells` | `count_macros` | CS | computed (`results/cross_cell/quality_drop.json` gives the same tests with Holm over 7 cells) |
| MedQA: Base asked in 99.0% to 100.0% of responses per graded seed; aggregate +0.018 [+0.003, +0.033]; the posted rise | `\statMedQAPreMistralJulActiveInquiryBaseSeedMin`, `...SeedMax`, `\statMedQAPreMistralJulRubricScoreLoRAvsBaseDiff`, `...DiffCI` | `count_macros` | CS | computed |
| MedQA dimensions in Table 1 use the 4 seeds with dimension grades | `\statMedQAPreMistralJulSeedsGraded` | `count_macros` | CS (`per_seed`) | computed |
| Recipe and evaluation differences of the added cells | none | | the train and eval blocks of `configs/cells/*__2026-07.yaml` (from `61dada9:rebuttal/configs/` and `61dada9:rebuttal/launch/run_cell.sh`), comments above Table hparams-cells | design |

## Results: CoT on the adapted model (section 4.3)

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| Active inquiry 80.0% under LoRA+BODHI | `\statHBMedGemmaMayActiveInquiryLoRABodhi` | `cell_macros` | MS | computed |
| Red flag 1.67 to 1.51, -0.161 [-0.262, -0.059], p = 0.012, Holm 0.058 | `\statHBMedGemmaMayRedFlagLoRA`, `\statHBMedGemmaMayRedFlagLoRABodhi`, `\statHBMedGemmaMayRedFlagLoRABodhivsLoRADiff`, `...CI`, `...P`, `...PHolm` | `cell_macros`, `cell_macros` | MS | computed |
| Red-flag rate -20.6 points [-29.8, -11.4], p = 0.0034 | `\statHBMedGemmaMayRedFlagRateLoRABodhivsLoRADiff`, `...CI`, `...P` | `cell_macros` | MS | computed |
| Against Base: no detected difference on the score, lower on the rate | `\statHBMedGemmaMayRedFlagLoRABodhivsBaseDiff`, `...CI`, `\statHBMedGemmaMayRedFlagRateLoRABodhivsBaseDiff`, `...CI` | `cell_macros` | MS | computed |
| Aggregate LoRA+BODHI minus LoRA -0.008 [-0.020, +0.004] | `\statHBMedGemmaMayRubricScoreLoRABodhivsLoRADiff`, `...CI` | `cell_macros` | MS | computed |
| Leakage 53.2% against 0.0% to 0.8%: at least 2 of the 7 Pass 1 section headers, the `analysis-format` rule of `configs/protocols/bodhi.yaml` | `\statHBMedGemmaMayLeakBodhiAnalysisFormat<Cond>` | `cell_macros` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json` | computed (post hoc) |
| Posted nine-marker rule (Appendix H): 54.6%, two of its markers free words; withdrawn | `\statPostedLeakagePctLoRABodhi` | `count_macros` | PV | posted (withdrawals W25) |
| Red flag 1.32 (n = 494) leaked, 1.73 (n = 434) not leaked | `\statHBMedGemmaMayLeakBodhiAnalysisFormatLoRABodhiRedFlagLeaked`, `...RedFlagClean`, `...LoRABodhiN`, `...LoRABodhiNClean` | `cell_macros` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json` | computed (post hoc) |
| The drop does not repeat in the added cells: LoRA+BODHI below LoRA (p < 0.05) in 0 of the 4 added cells outside the evaluator's family, above it (Holm p < 0.05) in 1 | `\statGroupAddedOutsideFamilyRedFlagProtocolFell`, `\statGroupAddedOutsideFamilyRedFlagProtocolRose`, `\statGroupAddedOutsideFamilyCells` | `count_macros` | CS (per-seed means; `paired_test` and `holm` of `bodhi_lora/analysis/stats.py`) | computed |
| The format still leaks in the added HealthBench cells: 41.1% of LoRA+BODHI responses with Mistral-Small-24B, 50.1% with Med42-8B; almost never with BioMistral-7B, on ChatDoctor or on MedQuAD (0.0% to 0.9%, Table H.6) | `\statHBMistralJulLeakBodhiAnalysisFormatLoRABodhi`, `\statHBMedFortyTwoJulLeakBodhiAnalysisFormatLoRABodhi` | `cell_macros`, `cross_table` | `results/*/analysis/stats.json` (`leakage.leak_bodhi__analysis-format`) | computed (post hoc, D11) |
| With Mistral-Small-24B, leaked responses score 1.41 on red-flag identification and the others 1.81, yet the condition mean does not fall below LoRA; with Med42-8B the two groups score almost the same (1.09 and 1.09) | `\statHBMistralJulLeakBodhiAnalysisFormatLoRABodhiRedFlagLeaked`, `...RedFlagClean`, `\statHBMedFortyTwoJulLeakBodhiAnalysisFormatLoRABodhiRedFlagLeaked`, `...RedFlagClean` | `cell_macros` | `results/*/analysis/stats.json` | computed (post hoc); "does not fall below LoRA" reads the cell's LoRA+BODHI minus LoRA contrast in Table E.1 |

## Results: post hoc checks in every cell (section 4.4; Table H.6 in Appendix H)

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| Table H.6, every row and column | `tab_cross_cell_checks.tex` | `cross_table`, columns from `cross_table` of `configs/report.yaml` | `results/*/analysis/stats.json` | computed (post hoc, descriptive) |
| Outside HealthBench's emergency-referrals theme, LoRA raises the red-flag rate by +6.2 points [+3.3, +9.1] with MedGemma-27B and by +39.6 with Mistral-Small-24B (36.4% to 76.0%); not a rate of false flags | `\statHBMedGemmaMayRedFlagRateEmergencyThemeNegRateLoRAvsBaseDiff`, `...CI`, `\statHBMistralJulRedFlagRateEmergencyThemeNegRateLoRAvsBaseDiff`, `\statHBMistralJulRedFlagRateEmergencyThemeNegRate<Cond>` | `cell_macros` | `results/healthbench__*/analysis/stats.json` (`discrimination.red_flag_rate.emergency-theme`, seed-level test) | computed (post hoc `emergency-theme` label, ANALYSIS_PLAN.md section 12, 2026-10-10 and the paper entry of the same day; the test of over-flagging stays H7) |
| Asking index, LoRA minus Base: +10.0 [-0.5, +20.1] with MedGemma-27B, +7.6 [-9.2, +23.4] with Mistral-Small-24B, -13.0 [-26.6, -0.3] with BioMistral-7B; no cell shows that the adapter learned when to ask | `\stat<Cell>ActiveInquiryNeedsInfoThemeIndexLoRAvsBaseDiff`, `...CI` for HBMedGemmaMay, HBMistralJul, HBBioMistralJul | `cell_macros` | `results/healthbench__*/analysis/stats.json` (bootstrap over prompts) | computed (post hoc theme-tag label, D9) |
| One generation pass against two; the two-pass conditions stored only their final response; LoRA responses have more output tokens than Base in every cell except ChatDoctor with Med42-8B (781 against 753 with MedGemma-27B) | `\statHBMedGemmaMayTokensEngineLoRA`, `\statHBMedGemmaMayTokensEngineBase` | `cell_macros` | `results/*/analysis/stats.json` (`cost.conditions.<cond>.tokens_engine_mean`) | computed; "every cell except" reads the token columns of Table H.6 |

## Discussion

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| Aggregate falls by 0.014 to 0.086 in the added cells outside the family | `\statGroupAddedOutsideFamilyRubricScoreLoRAvsBaseAbsMin`, `\statGroupAddedOutsideFamilyRubricScoreLoRAvsBaseAbsMax` | `count_macros` | CS | computed |
| Future work (section 5.3): experiments built and checked on the CPU, not run, among them three prompt baselines, a second protocol, a clinical judge on every cell but the set-aside MedQA cell (its check of the asking grade is specified and not built, RUN_PLAN.md R17), per-stage timing on an idle GPU (R2, C25), an over-flagging test and a rerun of the probe with its outputs kept; the physician-label discrimination index needs no run, only the plan's adoption. The appendix names no unrun experiment except by a pointer to section 5.3 | none | | the planned cell records in `configs/cells/`, the `clarify`, `verbalized-confidence` and `mediq-abstain` records in `configs/protocols/`, docs/ANALYSIS_PLAN.md, docs/RUN_PLAN.md | design |
| Antidepressant example: Base asks nothing, LoRA asks about duration, intensity, sleep, appetite, energy, concentration and self-harm, and gives a crisis line | none (qualitative) | | `results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_99/eval/{base,lora}.json` and `grade/epistemic__llama-3.1-8b.json`, prompt `2c915b71` | computed (read by hand; quoted in Appendix K) |
| Table 2: prior approaches compared on what they change, inference cost, what they need and their target | none | | the cited papers | design (positioning, not a measurement) |
| Second protocol implemented, not run (section 5.3) | none | | `bodhi_lora/protocols/ask_first.py` | design |
| The adapter also asks more on prompts outside the context-seeking theme (17.0% to 43.2%); its asking index does not rise detectably over Base in any HealthBench cell (section 4.4); in Appendix H, the MedGemma-27B gain +10.0 [-0.5, +20.1], the adapter's own index +13.2 [+4.1, +21.7], Base's +3.2 [-4.2, +11.2]; the posted index (+10.5, +2.1) and its rubric-keyword labelling withdrawn | `\statHBMedGemmaMayActiveInquiryNeedsInfoThemeNegRate<Cond>`, `\statHBMedGemmaMayActiveInquiryNeedsInfoThemeIndexLoRAvsBaseDiff`, `...CI`, `\statHBMedGemmaMayActiveInquiryNeedsInfoThemeIndex<Cond>`, `...CI`, `...PosPrompts`, `...NegPrompts`, `\statPostedDiscrimination<Cond>` | `cell_macros`, `count_macros` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json` (`discrimination.active_inquiry.needs-info-theme`); PV | computed (post hoc theme-tag label; withdrawals W15, W24) |
| Posted second-person question rate 18.4% and 39.6% (Appendix H), a keyword rule; withdrawn, and no count of this kind replaces it | `\statPostedSecondPersonBase`, `\statPostedSecondPersonLoRA` | `count_macros` | PV | posted (withdrawals W23) |
| Prior work compared: MediQ, CRAFT-MD, AMIE, R-Tuning, linguistic calibration (section 2, Table 2) | none | | the cited papers; venues checked against Crossref and arXiv on 2026-10-08 | design (positioning) |

## Appendix numbers that are not computed here

| Claim | Where | Kind | What would settle it |
|---|---|---|---|
| Contamination probe: 200 sampled, 105 long enough, 0 exact matches, 0 overlap | Appendix A | unverified | Re-run `bodhi_lora/pipeline/probe.py` on the base model (GPU) and commit the output |
| Filter kept 3,129 of 4,000 traces (tau table pool column) | Table tau-calibration | unverified (constant `FILTER_KEPT`) | The filter logs of the submitted run (PROV-11) |
| Figure 1 labels: 4,000 prompts, 2,816 train in seed 7, r = 8, 250 to 300 steps, 200 of 1,000 prompts | Figure 1 | computed for the counts (`paper/figures/pipeline_standalone.tex` inputs `generated/numbers.tex`); design for r and steps | `make figures` rebuilds the PDF |
| Every cell on every dimension in the four conditions (Tables E.1 and E.2), MedQuAD specificity 1.71, 1.47, 1.33 | Appendix E, `tab_cell_detail_2.tex`; Appendix J text | computed (CS; `\statMQMistralJul<Dim><Cond>`; withdrawals W20, W21) | Settled by the sync |
| Per-cell tau, graded, kept, train, val, steps, evaluated epoch (Table J.1); tau = 0.4 in every cell, no re-filter | Appendix J, `tab_cell_filter.tex`, `\statGroupAddedRecipeTau` | computed (FM, from `logs/host/filter_*.log` and the synced trainer states) | Settled by the sync (PROV-3) |
| Table F.2: rank, alpha, dropout, target modules, epochs, batch per device and accumulation of the added cells | `\statGroupAddedRecipe*`, `\statGroupAddedRecipeEpochs`, `\statGroupAddedRecipeEffectiveBatch`, `\statGroupAdded<Model>RecipeBatchPerDevice`, `\statGroupAdded<Model>RecipeGradAccum` | computed (FM, from the synced adapter and trainer files) | Settled by the sync; learning rate, schedule, sequence length and precision stay design values from the train blocks of `configs/cells/*__2026-07.yaml` |
| Askers minus non-askers: positive in 3 of 4 cells, -0.002 in BioMistral-7B; the bootstrap interval lies above zero on ChatDoctor and MedQuAD and includes zero in the other cells outside the family; Med42-8B -0.036 and -0.025, with intervals below zero | Appendix J, `\statGroupAddedOutsideFamilyAskersGapLoRAUp`, `\statCell<Cell>AskersGap` (`count_macros`), `\stat<Cell>AskersGapLoRACI` (`cell_macros`) | computed (CS, `lora_askers_minus_non_askers`; intervals from `results/*/analysis/stats.json`, `askers_gap.lora.gap_ci95`) | Settled by the sync |
| Cross-seed SD of Base active inquiry in the added cells, 0.3 to 2.1 points | Appendix J, `\statGroupAddedBaseActiveInquirySDMin`, `\statGroupAddedBaseActiveInquirySDMax` (`count_macros`) | computed (CS) | Settled by the sync |
| Posted: the adapter needs about 58% fewer output tokens than the protocol; withdrawn | Appendix H, `\statPostedFewerTokensPct` | posted (PV; withdrawals W22) | Nothing: the two-pass conditions stored only their final response |
| Incomplete host runs: BioMistral-7B on MedQA, 1 seed, 40 traces, 3 steps, 10 prompts; MedGemma-27B on MedQA and MedQuAD and Gemma-2-9B on HealthBench-Hard stopped in trace generation | Appendix J, `\statRunMedQABioMistral<...>` | computed (FM) for the counts; the four runs from `cell_recipe.json` (`incomplete` in `numbers.json`) | |
| Second grader on seed 42 (r = 0.70, rho = 0.68, 43 shared prompts) | Appendix H | computed (`grader_comparisons` in `results/cross_cell/stats.json`, from the pilot's and the 2026-05 cell's seed-42 response tables) | Corrected 2026-10-10: the earlier r = 0.42 and rho = 0.35 compared the pilot's stored score, earned points over positive points, with the normalized score of Eq. 1; with one formula on both sides r is 0.70 |
| Plan specified in the repository on 2026-04-29 (not an outside registry), grader switch 2026-05-01 | Abstract, section 3.4, Appendix B | design | commits aa7eec4 (`aa7eec4:RESULTS.md` section 7.B) and 1376aec; the paper omits the hashes because they find the public repo (withdrawals W19) |
