# Claims ledger for paper/main.tex

Every number and claim in the abstract, the introduction and the results of `paper/main.tex`,
with the macro that prints it, the function in `scripts/paper_stats.py` (or, for a
`\statHBMedGemmaMay...` macro, in `bodhi_lora/report/`, which `make numbers` runs) that
writes the macro, and the input file. No number comes from a keyword rule
(docs/CONVENTIONS.md rule 15). `make numbers` rebuilds every macro; nothing in this table is typed into
the paper by hand unless the Kind column says so.

The paper-level claim ledger, with the status of each claim made in the submission and the
replies, is `docs/CLAIMS.md`. This file is the number-level map for the current text.

Kinds:

- **computed**: the macro comes from committed data in this repository. For the added cells
  that data is `results/cross_cell/generated_values.json` and the cells' `analysis/` files,
  which `make cells` computes from the raw outputs synced from the GPU host on 2026-10-08
  (the per-seed eval JSON and the host logs are gitignored; `make numbers` needs only the
  tracked files).
- **computed, untracked input**: the macro comes from a file that git ignores. The only
  such input left is `data/raw/healthbench/hard.jsonl`, a public download (`python -m bodhi_lora fetch healthbench`).
  `make numbers` stops with a message when it is absent.
- **posted**: the macro comes from `rebuttal/posted_values.json`, a transcription
  of the OpenReview replies, and the paper quotes it as what was posted. Since the sync no
  added-cell result is of this kind.
- **design**: a setting of the experiment (seeds, prompts per seed, context length, tau,
  LoRA rank). Its source is the config or launcher named in the Source column.
- **unverified**: typed in the paper; the output that produced it is not in the repository.

Inputs: `RM` is `results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_*/` (the four `eval/<condition>.json` files and
`grade/epistemic__llama-3.1-8b.json`), `PV` is `rebuttal/posted_values.json`, `GV` is
`results/cross_cell/generated_values.json`, `CR` is `results/cross_cell/cell_recipe.json`.
Before the move to the results root these were `results_modal/`, `data/rebuttal_cells/` and
`rebuttal/results/`; the source comments in `numbers.tex` still use those names, and
`results/provenance.json` maps them.

## Abstract

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| 5 seeds, 667 unique prompts | `\statNSeeds`, `\statUniquePrompts` | `counts_and_failures` | RM | computed |
| Active inquiry 17.5% to 45.6%, +28.0 points, 95% CI [+22.9, +33.2] | `\statActiveInquiryBase`, `\statActiveInquiryLoRA`, `\statActiveInquiryLoRAvsBaseDiff`, `...CI` | `describe_conditions`, `compare_conditions` | RM | computed |
| No detected change in aggregate score, +0.004 [-0.011, +0.020] | `\statHealthBenchLoRAvsBaseDiff`, `...CI` | `compare_conditions` | RM | computed |
| 8 cells, each with the same seed-level tests | `\statCellsAll` | `uniform_section` | GV | computed |
| 5 cells outside the evaluator's family and not set aside; active inquiry and context-seeking both rise (Holm p < 0.05) in 4 | `\statCellsOutside`, `\statCellsOutsideAskRose` | `uniform_section` | GV (per-seed means; `paired_test` and `holm` of `bodhi_lora/analysis/stats.py`) | computed |
| Aggregate falls in all 4, by 0.014 to 0.086, largest paired t p = 0.016 | `\statCellsAddedWithAggregate`, `\statCellsAddedDropMin`, `\statCellsAddedDropMax`, `\statCellsAddedDropPMax` | `generality_section` | GV | computed |
| Adding CoT to the adapter changes red-flag identification by -0.161 [-0.262, -0.059] on the 0-2 score, Holm-adjusted p = 0.058 (the rate, -20.6 points, is a secondary measure stated in section 4.3) | `\statRedFlagLoRACoTvsLoRADiff`, `...CI`, `...PHolm` | `compare_conditions` | RM | computed |
| Format leakage in 53.2% of LoRA+BODHI answers (at least 2 of the 7 Pass 1 headers) | `\statHBMedGemmaMayLeakBodhiAnalysisFormatLoRABodhi` | `bodhi_lora.report.cell_macros` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json` (`leakage.leak_bodhi__analysis-format`) | computed (post hoc; the posted 54.6% is withdrawn, W25) |
| The red-flag drop does not repeat in the added cells | none in the abstract; section 4.3 gives the counts | `uniform_section` | GV | computed (see the section 4.3 rows) |
| The 2 Med42-8B cells share the evaluator's family and the MedQA cell's prompt told the model to ask, so the counts leave them out | `\statCellsSameFamily` | `uniform_section` | the `same_family_as_evaluator` and `set_aside` flags of each cell in PV | design (D6; the MedQA run identity: withdrawals W12) |
| Physician agreement kappa 0.35, target 0.6 set in advance for the filter grader, on rubric criteria only; no physician graded the seven dimensions | `\statKappaAll`; 0.6 is the target in the plan of 2026-04-29 | `kappa_section` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/kappa_results.json`, computed by `make kappa` from the returned rater file, which carries a name and is kept outside the repo; row design in `scripts/make_irr_excel.py` | computed from an input outside the repo (audit NUM-2): the committed `IRR-clinicians/` sheets are blank, so the value cannot be recomputed from the repo |
| BioMistral-7B is a null result | none in the abstract | | GV (see the section 4.1 rows) | computed |

## Introduction

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| The two passes overran the context on 4.3% of prompts | `\statCompletionDropPctBaseCoT` | `counts_and_failures` | RM | computed |
| 40 of 1,000 Hard prompts tagged as written by a health professional; 63 carry the tag (lower bound) | `\statHardHealthProfessional`, `\statHardSpeakerTagged`, `\statHardPrompts` | `speaker_tags`, `difficulty_range` | `data/raw/healthbench/hard.jsonl` (`physician_agreed_category` tags) | computed; HealthBench is a public download |
| Posted 14 of 1,000 (Appendix H), from a keyword audit; withdrawn, and no keyword count replaces it | `\statPostedClinicianPrompts` | `generality_section` | PV | posted (withdrawals W14) |
| 8 cells, 7 of them added; active inquiry and context-seeking rise in 4 of the 5 cells outside the evaluator's family; aggregate falls in all 4 added cells among them | `\statCellsAll`, `\statCellsAdded`, `\statCellsOutsideAskRose`, `\statCellsOutside`, `\statCellsAddedWithAggregate` | `uniform_section`, `generality_section` | GV | computed |
| BODHI expansion "Balanced, Open-minded, Diagnostic, Humble, Inquisitive" | none | | the cited BMJ paper (docs/DECISIONS.md, D15 note) | design |
| Pass 1 has seven fields: task type, audience, best assessment, key uncertainties, questions to ask, red flags, safe recommendations | none | | `bodhi/prompts.py` in bodhi-llm 0.1.4 (pinned in `requirements.txt`) | design |

## Method and evaluation

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| 4,000 HealthBench prompts outside the Hard set | `\statFilterPool` | `tau_calibration` | constant `FILTER_POOL` (5,000 HealthBench minus 1,000 Hard; legacy launcher) | design |
| Seed-7 training split: 2,816 traces, 0 in the Hard set | `\statFilterTrainNSeedSeven`, `\statTrainHardOverlapSeedSeven` | `tau_calibration` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_7/filter/filter_scores.json` (scores copied from the ignored `seed_7/filter/train.jsonl` of the same cell), `data/raw/healthbench/hard.jsonl` | computed; HealthBench is a public download |
| tau = 0.4, r = 8, alpha = 16, 300 steps, 64 examples per step, 2,048 tokens | none | | `legacy/configs/lora_medgemma27b_tunix.yaml`, `legacy/tpu/launch_5seeds_tunix.sh` (Table hparams comments) | design; not checked against the saved adapters, because the GCS bucket returned 404 on 2026-10-08 (D1, D19) |
| Added cells: r = 16, alpha = 32 (section 3.3, Table F.2, Appendix J) | `\statCellsLoRARank`, `\statCellsLoRAAlpha` | `recipe_macros` | CR (the synced `adapter_config.json` of all 35 adapters) | computed |
| 1,000 draws per condition cover 667 unique prompts; 271 repeat | `\statPromptEvaluations`, `\statUniquePrompts`, `\statRepeatedPrompts` | `counts_and_failures` | RM | computed |
| Epistemic parse failures 0.20% to 1.49% per condition | `\statEpistemicParseFailPctBase`, `\statEpistemicParseFailPctMax` | `counts_and_failures` | RM | computed |
| HealthBench criterion failures at most 0.087% per run | `\statHealthBenchParseFailPctMaxRun` | `counts_and_failures` | RM | computed |
| 1 of 3 physicians; kappa 0.35 [0.28, 0.43] on 509 rows from 10 prompts | `\statKappaRaters`, `\statKappaAll`, `\statKappaAllCI`, `\statKappaAllN`, `\statKappaPrompts` | `kappa_section` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/kappa_results.json` | computed from an input outside the repo (see the abstract row) |
| 4,096-token evaluation context, greedy decoding, A100-80GB | none | | `legacy/modal_eval.py` (max_model_len patch, GPU type) | design |
| Context-seeking 2 without a direct question: 74.4% of Base 2s, 47.2% of LoRA 2s | `\statContextSeekingTwoNoQuestionPctBase`, `\statContextSeekingTwoNoQuestionPctLoRA` | `context_seeking_anchor` | RM | computed |
| Mapping of the seven dimensions to the BODHI components (Table bodhi-map) | none | | names in the cited BMJ paper; Pass 1 fields in bodhi-llm 0.1.4 | design (our reading) |
| General graders as in HealthBench (GPT-4.1 grader) | none | | `healthbench2025` | design |
| Hedging 2 given to 4 of the 5 flagged blanket disclaimers (Appendix D) | `\statBlanketDisclaimerHedgingTwo`, `\statBlanketDisclaimerTotal` | `blanket_disclaimers` | RM | computed |

## Results: the MedGemma-27B cell (section 4.2; its Tables E.3 and E.4 and Figure E.1 are in Appendix E)

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| Table E.3 (score, t interval, pooled n, n per seed) | `tab_main_healthbench.tex` | `build_tables` | RM | computed |
| Table E.4 (seven dimensions, rates, scored n) | `tab_epistemic.tex` | `build_tables` | RM | computed |
| Figure E.1 (two panels) | `paper/figures/epistemic_bar.pdf` | `scripts/plot_epistemic_figures.py` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/cross_seed_aggregate.json` | computed |
| Active inquiry, context-seeking, red flag, scope: diff, CI, Holm p; the four effects keep Holm-adjusted p <= 0.0065 | `\stat<Dim>LoRAvsBaseDiff`, `...CI`, `...PHolm`, `\statScopeBoundingLoRAvsBasePHolm` (the largest of the four) | `compare_conditions` | RM | computed |
| Output tokens per response from the serving engine, Base and LoRA (also Appendix H) | `\statTokensEngineBase`, `\statTokensEngineLoRA` | `cost_section` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json` (`cost`, written by `make analyze`) | computed |
| Questions per response 0.45 to 1.16 | `\statQuestionsBase`, `\statQuestionsLoRA` | `describe_conditions` | RM | computed |
| Hedging and specificity significant before but not after Holm | `\statHedgingQualityLoRAvsBaseDiff`, `\statSpecificityLoRAvsBaseDiff`, `...PHolm` | `compare_conditions` | RM | computed |
| No detected change in uncertainty acknowledgment | `\statUncertaintyAckLoRAvsBaseDiff`, `...CI` | `compare_conditions` | RM | computed |
| Pooled d 0.63, 0.44, 0.20, 0.22 (descriptive) | `\stat<Dim>LoRAvsBaseD` | `compare_conditions` | RM | computed |
| Worst case keeps the four effects at p <= 0.035 | `\statScopeBoundingLoRAvsBaseWorstCaseP` (the largest of the four) | `worst_case` | RM | computed |
| LoRA vs teacher on active inquiry: -2.4 [-5.7, +1.0] | `\statActiveInquiryLoRAvsBaseCoTDiff`, `...CI`, `...CILow` | `compare_conditions` | RM | computed |
| Teacher gave no answer on 4.3% of prompts | `\statCompletionDropPctBaseCoT` | `counts_and_failures` | RM | computed |
| Aggregate +0.004 [-0.011, +0.020], p = 0.48 | `\statHealthBenchLoRAvsBaseDiff`, `...CI`, `...P`, `...CILow` | `compare_conditions` | RM | computed |
| Rubric axes: accuracy, completeness, communication, instruction following | `\statAxis<Axis><Cond>` | `axis_scores` | RM | computed |
| Blanket disclaimers 5/998 vs 0/985, Fisher p = 0.062 | `\statBlanketDisclaimerCountBase`, `\statNScoredBase`, `\statBlanketDisclaimerCountLoRA`, `\statNScoredLoRA`, `\statBlanketDisclaimerFisherPLoRA` | `blanket_disclaimers`, `counts_and_failures` | RM | computed |
| LoRA answers +183 characters [-21, +387] | `\statResponseCharsLoRAvsBaseDiff`, `...CI` | `compare_conditions` | RM | computed |
| Active inquiry rises in every length quartile | `tab_length_quartiles.tex` | `length_quartiles` | RM | computed |

## Results: every cell (section 4.1, Table 1; Tables E.1 and E.2 in Appendix E)

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| Table 1, every row (the MedGemma-27B cell first; its per-seed means must equal the ones computed from RM, or the run stops) | `tab_cross_cell.tex` | `uniform_section`, `cross_cell_table` | GV (per-seed means; `paired_test` and `holm` of `bodhi_lora/analysis/stats.py`) | computed |
| LoRA - Base with half its 95% t interval for every cell; Holm over the seven dimensions; paired t for the aggregate | `tab_cross_cell.tex` | `uniform_section` | GV (per-seed means; `paired_test` and `holm` of `bodhi_lora/analysis/stats.py`) | computed |
| Tables E.1 and E.2: every cell in the four conditions, LoRA - Base and LoRA+BODHI - LoRA | `tab_cell_detail_1.tex`, `tab_cell_detail_2.tex` | `uniform_section`, `cell_detail_tables` | GV (per-seed means; `paired_test` and `holm` of `bodhi_lora/analysis/stats.py`) | computed |
| Red flag rises in 4 and scope bounding in 4 of the 5 cells outside the family; a dimension falls in 2 (uncertainty acknowledgment on ChatDoctor, specificity on MedQuAD) | `\statCellsOutsideRedFlagRose`, `\statCellsOutsideScopeRose`, `\statCellsOutsideDimFell`, `\statCellChatDoctorMistralUncertaintyAckLoRAvsBaseDiff`, `\statCellMedQuADMistralSpecificityLoRAvsBaseDiff` | `uniform_section` | GV (per-seed means; `paired_test` and `holm` of `bodhi_lora/analysis/stats.py`) | computed |
| BioMistral-7B: no dimension rises or falls with Holm p < 0.05 | `\statCellHBBioMistralDimsRose`, `\statCellHBBioMistralDimsFell` | `uniform_section` | GV (per-seed means; `paired_test` and `holm` of `bodhi_lora/analysis/stats.py`) | computed |
| BioMistral-7B: 11.7% to 10.1%, teacher 14.5%, filter kept 20.4% | `\statCellHBBioMistral<...>` | `generality_section` | GV; the filter share from CR | computed |
| Adapter at or above its teacher in 3 of 5 cells (submitted included) | `\statCellsAtOrAboveTeacher`, `\statCellsCompared` | `generality_section` | RM, GV | computed |
| Aggregate falls in all 4 added cells outside the evaluator's family, 0.014 to 0.086 | `\statCellsAddedWithAggregate`, `\statCellsAddedDropMin`, `\statCellsAddedDropMax` | `generality_section` | GV | computed |
| Largest p of the four drops 0.016; each survives Holm over the 6 added cells not set aside (largest p_Holm 0.016), as posted | `\statCellsAddedDropPMax`, `\statCellsAddedDropPHolmMax`, `\statCellsHolmFamily` | `generality_section` | GV | computed (`results/cross_cell/quality_drop.json` gives the same tests with Holm over 7 cells) |
| MedQA: Base asked in 99.0% to 100.0% of responses per graded seed; aggregate +0.018 [+0.003, +0.033]; the posted rise | `\statCellMedQAMistralBaseActiveInquirySeedMin`, `...SeedMax`, `\statCellMedQAMistralAggregateDiff`, `...DiffCI` | `generality_section` | GV | computed |
| MedQA dimensions in Table 1 use the 4 seeds with dimension grades | `\statCellMedQAMistralEpistemicSeeds` | `generality_section` | GV (`per_seed`) | computed |
| Recipe and evaluation differences of the added cells | none | | the train and eval blocks of `configs/cells/*__2026-07.yaml` (from `61dada9:rebuttal/configs/` and `61dada9:rebuttal/launch/run_cell.sh`), comments above Table hparams-cells | design |

## Results: CoT on the adapted model (section 4.3)

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| Active inquiry 80.0% under LoRA+BODHI | `\statActiveInquiryLoRACoT` | `describe_conditions` | RM | computed |
| Red flag 1.67 to 1.51, -0.161 [-0.262, -0.059], p = 0.012, Holm 0.058 | `\statRedFlagLoRA`, `\statRedFlagLoRACoT`, `\statRedFlagLoRACoTvsLoRADiff`, `...CI`, `...P`, `...PHolm` | `describe_conditions`, `compare_conditions` | RM | computed |
| Red-flag rate -20.6 points [-29.8, -11.4], p = 0.0034 | `\statRedFlagRateLoRACoTvsLoRADiff`, `...CI`, `...P` | `compare_conditions` | RM | computed |
| Against Base: no detected difference on the score, lower on the rate | `\statRedFlagLoRACoTvsBaseDiff`, `...CI`, `\statRedFlagRateLoRACoTvsBaseDiff`, `...CI` | `compare_conditions` | RM | computed |
| Aggregate LoRA+BODHI minus LoRA -0.008 [-0.020, +0.004] | `\statHealthBenchLoRACoTvsLoRADiff`, `...CI` | `compare_conditions` | RM | computed |
| Leakage 53.2% against 0.0% to 0.8%: at least 2 of the 7 Pass 1 section headers, the `analysis-format` rule of `configs/protocols/bodhi.yaml` | `\statHBMedGemmaMayLeakBodhiAnalysisFormat<Cond>` | `bodhi_lora.report.cell_macros` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json` | computed (post hoc) |
| Posted nine-marker rule (Appendix H): 54.6%, two of its markers free words; withdrawn | `\statPostedLeakagePctLoRACoT` | `generality_section` | PV | posted (withdrawals W25) |
| Red flag 1.32 (n = 494) leaked, 1.73 (n = 434) not leaked | `\statHBMedGemmaMayLeakBodhiAnalysisFormatLoRABodhiRedFlagLeaked`, `...RedFlagClean`, `...LoRABodhiN`, `...LoRABodhiNClean` | `bodhi_lora.report.cell_macros` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json` | computed (post hoc) |
| The drop does not repeat in the added cells: LoRA+BODHI below LoRA (p < 0.05) in 0 of the 4 added cells outside the evaluator's family, above it (Holm p < 0.05) in 1 | `\statCellsAddedRedFlagCoTFell`, `\statCellsAddedRedFlagCoTRose`, `\statCellsAddedOutside` | `uniform_section` | GV (per-seed means; `paired_test` and `holm` of `bodhi_lora/analysis/stats.py`) | computed |

## Discussion

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| Aggregate falls by 0.014 to 0.086 in the added cells outside the family | `\statCellsAddedDropMin`, `\statCellsAddedDropMax` | `generality_section` | GV | computed |
| Future work (section 5.3): experiments built and checked on the CPU, not run, among them three prompt baselines, a second protocol, a clinical judge and an over-flagging test | none | | the planned cell records in `configs/cells/`, the `clarify`, `verbalized-confidence` and `mediq-abstain` records in `configs/protocols/`, docs/EXPERIMENTS.md, docs/RUN_PLAN.md | design |
| Antidepressant example: Base asks nothing, LoRA asks about duration, intensity, sleep, appetite, energy, concentration and self-harm, and gives a crisis line | none (qualitative) | | `results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_99/eval/{base,lora}.json` and `grade/epistemic__llama-3.1-8b.json`, prompt `2c915b71` | computed (read by hand; quoted in Appendix K) |
| Table 2: prior approaches compared on what they change, inference cost, what they need and their target | none | | the cited papers | design (positioning, not a measurement) |
| Second protocol implemented, not run (section 5.3) | none | | `bodhi_lora/protocols/ask_first.py` | design |
| The adapter also asks more on prompts outside the context-seeking theme (17.0% to 43.2%); its gain in discrimination over Base includes zero (+10.0 [-0.5, +20.1]); the adapter's own index +13.2 [+4.1, +21.7], Base's +3.2 [-4.2, +11.2]; the posted index (+10.5, +2.1) and its rubric-keyword labelling withdrawn | `\statHBMedGemmaMayActiveInquiryNeedsInfoThemeNegRate<Cond>`, `\statHBMedGemmaMayActiveInquiryNeedsInfoThemeIndexLoRAvsBaseDiff`, `...CI`, `\statHBMedGemmaMayActiveInquiryNeedsInfoThemeIndex<Cond>`, `...CI`, `...PosPrompts`, `...NegPrompts`, `\statPostedDiscrimination<Cond>` | `bodhi_lora.report.cell_macros`, `generality_section` | `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json` (`discrimination.active_inquiry.needs-info-theme`); PV | computed (post hoc theme-tag label; withdrawals W15, W24) |
| Posted second-person question rate 18.4% and 39.6% (Appendix H), a keyword rule; withdrawn, and no count of this kind replaces it | `\statPostedSecondPersonBase`, `\statPostedSecondPersonLoRA` | `generality_section` | PV | posted (withdrawals W23) |
| Prior work compared: MediQ, CRAFT-MD, AMIE, R-Tuning, linguistic calibration (section 2, Table 2) | none | | the cited papers; venues checked against Crossref and arXiv on 2026-10-08 | design (positioning) |

## Appendix numbers that are not computed here

| Claim | Where | Kind | What would settle it |
|---|---|---|---|
| Contamination probe: 200 sampled, 105 long enough, 0 exact matches, 0 overlap | Appendix A | unverified | Re-run `bodhi_lora/pipeline/probe.py` on the base model (GPU) and commit the output |
| Filter kept 3,129 of 4,000 traces (tau table pool column) | Table tau-calibration | unverified (constant `FILTER_KEPT`) | The filter logs of the submitted run (PROV-11) |
| Figure 1 labels: 4,000 prompts, 2,816 train in seed 7, r = 8, 250 to 300 steps, 200 of 1,000 prompts | Figure 1 | computed for the counts (`paper/figures/pipeline_standalone.tex` inputs `generated/numbers.tex`); design for r and steps | `make figures` rebuilds the PDF |
| Every cell on every dimension in the four conditions (Tables E.1 and E.2), MedQuAD specificity 1.71, 1.47, 1.33 | Appendix E, `tab_cell_detail_2.tex`; Appendix J text | computed (GV; `\statCellMedQuADMistral<Cond><Dim>`; withdrawals W20, W21) | Settled by the sync |
| Per-cell tau, graded, kept, train, val, steps, evaluated epoch (Table J.1); tau = 0.4 in every cell, no re-filter | Appendix J, `tab_cell_filter.tex`, `\statCellsTau`, `\statCellsRefilterLines` | computed (CR, from `logs/host/filter_*.log` and the synced trainer states) | Settled by the sync (PROV-3) |
| Table F.2: rank, alpha, dropout, target modules, epochs, batch per device and accumulation of the added cells | `\statCellsLoRA*`, `\statCellsTrainEpochs`, `\statCellsEffectiveBatch`, `\statCellsBatchPerDevice*`, `\statCellsGradAccum*` | computed (CR, from the synced adapter and trainer files) | Settled by the sync; learning rate, schedule, sequence length and precision stay design values from the train blocks of `configs/cells/*__2026-07.yaml` |
| Askers minus non-askers: positive in 3 of 4 cells, -0.002 in BioMistral-7B; Med42-8B -0.036 and -0.025 | Appendix J, `\statCellsAddedAskersGapPositive`, `\statCell<Cell>AskersGap` (`uniform_section`) | computed (GV, `lora_askers_minus_non_askers`) | Settled by the sync |
| Cross-seed SD of Base active inquiry in the added cells, 0.3 to 2.1 points | Appendix J, `\statCellsAddedBaseAISDMin`, `\statCellsAddedBaseAISDMax` (`uniform_section`) | computed (GV) | Settled by the sync |
| Posted: the adapter needs about 58% fewer output tokens than the protocol; withdrawn | Appendix H, `\statPostedFewerTokensPct` | posted (PV; withdrawals W22) | Nothing: the two-pass conditions stored only their final response |
| Incomplete host runs: BioMistral-7B on MedQA, 1 seed, 40 traces, 3 steps, 10 prompts; MedGemma-27B on MedQA and MedQuAD and Gemma-2-9B on HealthBench-Hard stopped in trace generation | Appendix J, `\statRunMedQABioMistral<...>` | computed (CR) for the counts; the four runs from `cell_recipe.json` (`incomplete` in `numbers.json`) | |
| Second grader on seed 42 (r = 0.42, rho = 0.35, 43 shared prompts) | Appendix H | computed (`results/healthbench__medgemma-27b-text__bodhi__2026-04/analysis/qwen_grader_scores.json`, scores copied from the pilot's ignored eval files) | Settled: `make derived` rebuilds the copy from the source files |
| Plan specified in the repository on 2026-04-29 (not an outside registry), grader switch 2026-05-01 | Abstract, section 3.4, Appendix B | design | commits aa7eec4 (`RESULTS.md` 7.B) and 1376aec; the paper omits the hashes because they find the public repo (withdrawals W19) |
