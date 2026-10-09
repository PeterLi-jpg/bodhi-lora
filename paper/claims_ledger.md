# Claims ledger for paper/main.tex

Every number and claim in the abstract, the introduction and the results of `paper/main.tex`,
with the macro that prints it, the function in `scripts/paper_stats.py` that writes the macro,
and the input file. `make numbers` rebuilds every macro; nothing in this table is typed into
the paper by hand unless the Kind column says so.

The paper-level claim ledger, with the status of each claim made in the submission and the
replies, is `docs/CLAIMS.md`. This file is the number-level map for the current text.

Kinds:

- **computed**: the macro comes from committed data in this repository. For the added cells
  that data is `data/rebuttal_cells/generated_values.json` and `rebuttal/results/*.json`,
  which `make cells` computes from the raw outputs synced from the GPU host on 2026-10-08
  (the per-seed eval JSON and the host logs are gitignored; `make numbers` needs only the
  tracked files).
- **computed, untracked input**: the macro comes from a file that git ignores. The only
  such input left is `data/raw/healthbench/hard.jsonl`, a public download (`python -m bodhi_lora fetch healthbench`).
  `make numbers` stops with a message when it is absent.
- **posted**: the macro comes from `data/rebuttal_cells/posted_values.json`, a transcription
  of the OpenReview replies, and the paper quotes it as what was posted. Since the sync no
  added-cell result is of this kind.
- **design**: a setting of the experiment (seeds, prompts per seed, context length, tau,
  LoRA rank). Its source is the config or launcher named in the Source column.
- **unverified**: typed in the paper; the output that produced it is not in the repository.

Inputs: `RM` is `results_modal/seed_*/` (the four `<condition>.json` files and
`epistemic_scores.json`), `PV` is `data/rebuttal_cells/posted_values.json`, `GV` is
`data/rebuttal_cells/generated_values.json`, `CR` is `rebuttal/results/cell_recipe.json`.

## Abstract

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| 5 seeds, 667 unique prompts | `\statNSeeds`, `\statUniquePrompts` | `counts_and_failures` | RM | computed |
| Active inquiry 17.5% to 45.6%, +28.0 points, 95% CI [+22.9, +33.2] | `\statActiveInquiryBase`, `\statActiveInquiryLoRA`, `\statActiveInquiryLoRAvsBaseDiff`, `...CI` | `describe_conditions`, `compare_conditions` | RM | computed |
| Context-seeking 1.40 to 1.75 | `\statContextSeekingBase`, `\statContextSeekingLoRA` | `describe_conditions` | RM | computed |
| No detected change in aggregate score, +0.004 [-0.011, +0.020] | `\statHealthBenchLoRAvsBaseDiff`, `...CI` | `compare_conditions` | RM | computed |
| In 4 added cells outside the evaluator's family, active inquiry rises in 3 | `\statCellsAddedCompared`, `\statCellsAddedAIRose` | `generality_section` | GV | computed |
| Aggregate falls in all 4, by 0.014 to 0.086, largest paired t p = 0.016 | `\statCellsAddedWithAggregate`, `\statCellsAddedDropMin`, `\statCellsAddedDropMax`, `\statCellsAddedDropPMax` | `generality_section` | GV | computed |
| Adding CoT to the adapter changes red-flag identification by -0.161 [-0.262, -0.059] on the 0-2 score, Holm-adjusted p = 0.058 (the rate, -20.6 points, is a secondary measure stated in section 4.3) | `\statRedFlagLoRACoTvsLoRADiff`, `...CI`, `...PHolm` | `compare_conditions` | RM | computed |
| Format leakage in 54.6% of LoRA+BODHI answers | `\statLeakagePctLoRACoT` | `rebuttal_section` | `rebuttal/results/interference_mechanism.json` | computed |
| In a fifth such cell, MedQA, the aggregate rose, +0.018 [+0.003, +0.033], but the prompt told the model to ask | `\statCellMedQAMistralAggregateDiff`, `...DiffCI` | `generality_section` | GV | computed (the run identity: withdrawals W12; commitments C24 done) |
| Physician agreement kappa 0.35, target 0.6 set in advance for the filter grader, on rubric criteria only; no physician graded the seven dimensions | `\statKappaAll`; 0.6 is the target in the plan of 2026-04-29 | `kappa_section` | `results_modal/irr/kappa_results.json`, computed by `make kappa` from the returned rater file, which carries a name and is kept outside the repo; row design in `scripts/make_irr_excel.py` | computed from an input outside the repo (audit NUM-2): the committed `IRR-clinicians/` sheets are blank, so the value cannot be recomputed from the repo |
| BioMistral-7B is a null result | none in the abstract | | GV (see section 4.2 rows) | computed |

## Introduction

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| The two passes overran the context on 4.3% of prompts | `\statCompletionDropPctBaseCoT` | `counts_and_failures` | RM | computed |
| 40 of 1,000 Hard prompts tagged as written by a health professional; 63 carry the tag (lower bound) | `\statHardHealthProfessional`, `\statHardSpeakerTagged`, `\statHardPrompts` | `speaker_tags`, `difficulty_range` | `data/raw/healthbench/hard.jsonl` (`physician_agreed_category` tags) | computed; HealthBench is a public download |
| Posted 14 of 1,000 (Appendix H, as corrected); the committed keyword rule finds 4 of 667 evaluated prompts | `\statPostedClinicianPrompts`, `\statClinicianKeywordPrompts`, `\statMimicryEvalPrompts` | `generality_section`, `rebuttal_section` | PV; `rebuttal/results/mimicry_split.json` | posted (not reproduced; withdrawals W14); computed |
| Four LoRA - Base effects with Holm-adjusted p <= 0.0065 | `\statScopeBoundingLoRAvsBasePHolm` (the largest of the four) | `compare_conditions` | RM | computed |
| Active inquiry rises in 3 of 4 added cells; aggregate falls in all 4 | `\statCellsAddedAIRose`, `\statCellsAddedCompared`, `\statCellsAddedWithAggregate` | `generality_section` | GV | computed |
| BODHI expansion "Balanced, Open-minded, Diagnostic, Humble, Inquisitive" | none | | the cited BMJ paper (docs/DECISIONS.md, D15 note) | design |
| Pass 1 has seven fields: task type, audience, best assessment, key uncertainties, questions to ask, red flags, safe recommendations | none | | `bodhi/prompts.py` in bodhi-llm 0.1.4 (pinned in `requirements.txt`) | design |

## Method and evaluation

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| 4,000 HealthBench prompts outside the Hard set | `\statFilterPool` | `tau_calibration` | constant `FILTER_POOL` (5,000 HealthBench minus 1,000 Hard; legacy launcher) | design |
| Seed-7 training split: 2,816 traces, 0 in the Hard set | `\statFilterTrainNSeedSeven`, `\statTrainHardOverlapSeedSeven` | `tau_calibration` | `data/derived/seed_7_filter_scores.json` (scores copied from the ignored `data/sft_qwen_graded/seed_7_train.jsonl`), `data/raw/healthbench/hard.jsonl` | computed; HealthBench is a public download |
| tau = 0.4, r = 8, alpha = 16, 300 steps, 64 examples per step, 2,048 tokens | none | | `legacy/configs/lora_medgemma27b_tunix.yaml`, `legacy/tpu/launch_5seeds_tunix.sh` (Table hparams comments) | design; not checked against the saved adapters, because the GCS bucket returned 404 on 2026-10-08 (D1; BLOCKED.md item 3) |
| Added cells: r = 16, alpha = 32 (section 3.3, Table F.2, Appendix J) | `\statCellsLoRARank`, `\statCellsLoRAAlpha` | `recipe_macros` | CR (the synced `adapter_config.json` of all 35 adapters) | computed |
| 1,000 draws per condition cover 667 unique prompts; 271 repeat | `\statPromptEvaluations`, `\statUniquePrompts`, `\statRepeatedPrompts` | `counts_and_failures` | RM | computed |
| Epistemic parse failures 0.20% to 1.49% per condition | `\statEpistemicParseFailPctBase`, `\statEpistemicParseFailPctMax` | `counts_and_failures` | RM | computed |
| HealthBench criterion failures at most 0.087% per run | `\statHealthBenchParseFailPctMaxRun` | `counts_and_failures` | RM | computed |
| 1 of 3 physicians; kappa 0.35 [0.28, 0.43] on 509 rows from 10 prompts | `\statKappaRaters`, `\statKappaAll`, `\statKappaAllCI`, `\statKappaAllN`, `\statKappaPrompts` | `kappa_section` | `results_modal/irr/kappa_results.json` | computed from an input outside the repo (see the abstract row) |
| 4,096-token evaluation context, greedy decoding, A100-80GB | none | | `legacy/modal_eval.py` (max_model_len patch, GPU type) | design |
| Context-seeking 2 without a direct question: 74.4% of Base 2s, 47.2% of LoRA 2s | `\statContextSeekingTwoNoQuestionPctBase`, `\statContextSeekingTwoNoQuestionPctLoRA` | `context_seeking_anchor` | RM | computed |
| Mapping of the seven dimensions to the BODHI components (Table bodhi-map) | none | | names in the cited BMJ paper; Pass 1 fields in bodhi-llm 0.1.4 | design (our reading) |
| General graders as in HealthBench (GPT-4.1 grader) | none | | `healthbench2025` | design |
| Hedging 2 given to 4 of the 5 flagged blanket disclaimers (Appendix D) | `\statBlanketDisclaimerHedgingTwo`, `\statBlanketDisclaimerTotal` | `blanket_disclaimers` | RM | computed |

## Results: the MedGemma-27B cell (section 4.1, Tables 1 and 2; Figure E.1 is in Appendix E)

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| Table 1 (score, t interval, pooled n, n per seed) | `tab_main_healthbench.tex` | `build_tables` | RM | computed |
| Table 2 (seven dimensions, rates, scored n) | `tab_epistemic.tex` | `build_tables` | RM | computed |
| Figure E.1 (two panels) | `paper/figures/epistemic_bar.pdf` | `scripts/plot_epistemic_figures.py` | `results_modal/cross_seed_aggregate.json` | computed |
| Active inquiry, context-seeking, red flag, scope: diff, CI, Holm p | `\stat<Dim>LoRAvsBaseDiff`, `...CI`, `...PHolm` | `compare_conditions` | RM | computed |
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

## Results: added cells (section 4.2, Table 3)

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| Table 3, first row (MedGemma-27B cell) | `tab_generality.tex` | `generality_table` | RM | computed |
| Table 3, other rows | `tab_generality.tex` | `generality_table`, `resolve_cells` | GV | computed |
| Interval for each added cell (paired t(4) over seeds) | `tab_generality.tex` | `_diff_with_ci`, `resolve_cells` | GV (per-seed aggregate) | computed |
| BioMistral-7B: 11.7% to 10.1%, teacher 14.5%, filter kept 20.4% | `\statCellHBBioMistral<...>` | `generality_section` | GV; the filter share from CR | computed |
| Adapter at or above its teacher in 3 of 5 cells (submitted included) | `\statCellsAtOrAboveTeacher`, `\statCellsCompared` | `generality_section` | RM, GV | computed |
| Aggregate falls in all 4 added cells outside the evaluator's family, 0.014 to 0.086 | `\statCellsAddedWithAggregate`, `\statCellsAddedDropMin`, `\statCellsAddedDropMax` | `generality_section` | GV | computed |
| Largest p of the four drops 0.016; each survives Holm over the 6 added cells not set aside (largest p_Holm 0.016), as posted | `\statCellsAddedDropPMax`, `\statCellsAddedDropPHolmMax`, `\statCellsHolmFamily` | `generality_section` | GV | computed (`rebuttal/results/quality_drop.json` gives the same tests with Holm over 7 cells) |
| MedQA: Base asked in 99.0% to 100.0% of responses per graded seed; aggregate +0.018 [+0.003, +0.033]; the posted rise | `\statCellMedQAMistralBaseActiveInquirySeedMin`, `...SeedMax`, `\statCellMedQAMistralAggregateDiff`, `...DiffCI` | `generality_section` | GV | computed |
| MedQA dimensions graded in 4 of 5 seeds | `\statCellMedQAMistralEpistemicSeeds`, `\statCellMedQAMistralSeeds` | `generality_section` | GV (`per_seed`) | computed |
| Recipe and evaluation differences of the added cells | none | | the train and eval blocks of `configs/cells/*__2026-07.yaml` (from `61dada9:rebuttal/configs/` and `61dada9:rebuttal/launch/run_cell.sh`), comments above Table hparams-cells | design |

## Results: CoT on the adapted model (section 4.3)

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| Active inquiry 80.0% under LoRA+BODHI | `\statActiveInquiryLoRACoT` | `describe_conditions` | RM | computed |
| Red flag 1.67 to 1.51, -0.161 [-0.262, -0.059], p = 0.012, Holm 0.058 | `\statRedFlagLoRA`, `\statRedFlagLoRACoT`, `\statRedFlagLoRACoTvsLoRADiff`, `...CI`, `...P`, `...PHolm` | `describe_conditions`, `compare_conditions` | RM | computed |
| Red-flag rate -20.6 points [-29.8, -11.4], p = 0.0034 | `\statRedFlagRateLoRACoTvsLoRADiff`, `...CI`, `...P` | `compare_conditions` | RM | computed |
| Against Base: no detected difference on the score, lower on the rate | `\statRedFlagLoRACoTvsBaseDiff`, `...CI`, `\statRedFlagRateLoRACoTvsBaseDiff`, `...CI` | `compare_conditions` | RM | computed |
| Aggregate LoRA+BODHI minus LoRA -0.008 [-0.020, +0.004] | `\statHealthBenchLoRACoTvsLoRADiff`, `...CI` | `compare_conditions` | RM | computed |
| Leakage 54.6% against 0.2% to 3.0%: at least 2 of 9 analysis-format markers (the 7 Pass 1 headers, "differential", "scope of") | `\statLeakagePct<Cond>` | `rebuttal_section` | `rebuttal/results/interference_mechanism.json` | computed |
| Seven Pass 1 headers only (Appendix H): 53.2% against 0.0% to 0.8%; red flag 1.32 (n = 494) and 1.73 (n = 434) | `\statLeakagePassOnePct<Cond>`, `\statRedFlagLeakedPassOne`, `\statRedFlagCleanPassOne`, `\statLeakagePassOneN...` | `rebuttal_section` | `rebuttal/results/interference_mechanism.json` | computed |
| Red flag 1.33 (n = 507) leaked, 1.73 (n = 421) not leaked | `\statRedFlagLeaked`, `\statRedFlagClean`, `\statLeakageNLoRACoT`, `\statLeakageNCleanLoRACoT` | `rebuttal_section` | `rebuttal/results/interference_mechanism.json` | computed |

## Discussion

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| Aggregate falls by 0.014 to 0.086 in the added cells outside the family | `\statCellsAddedDropMin`, `\statCellsAddedDropMax` | `generality_section` | GV | computed |
| Baseline code exists and has not run | none | | `bodhi_lora/protocols/`, the `clarify` and `verbalized-confidence` records in `configs/protocols/` | design |
| Antidepressant example: Base asks nothing, LoRA asks about duration, intensity, sleep, appetite, energy, concentration and self-harm, and gives a crisis line | none (qualitative) | | `results_modal/seed_99/{base,lora}_no_wrapper.json` and `epistemic_scores.json`, prompt `2c915b71` | computed (read by hand; quoted in Appendix K) |
| Table 4: prior approaches compared on what they change, inference cost, what they need and their target | none | | the cited papers | design (positioning, not a measurement) |
| Second protocol implemented, not run | none | | `bodhi_lora/protocols/ask_first.py` | design |
| The adapter also asks more on self-contained prompts (15.9% to 37.5%; 17.0% to 43.2%); its gain in discrimination over Base includes zero (+8.4 [-1.2, +17.9]; +10.0 [-0.5, +20.1]); posted selectivity claim withdrawn | `\statAskSelfContained<Labeling><Cond>`, `\statDiscrimination<Labeling>LoRAvsBase`, `...CI`, `\statDiscrimination<Labeling><Cond>`, `...CI` | `rebuttal_section` | `rebuttal/results/mimicry_split.json` | computed (post hoc labelings; withdrawals W15) |
| Prior work compared: MediQ, CRAFT-MD, AMIE, R-Tuning, linguistic calibration (section 2, Table 4) | none | | the cited papers; venues checked against Crossref and arXiv on 2026-10-08 | design (positioning) |

## Appendix numbers that are not computed here

| Claim | Where | Kind | What would settle it |
|---|---|---|---|
| Contamination probe: 200 sampled, 105 long enough, 0 exact matches, 0 overlap | Appendix A | unverified | Re-run `bodhi_lora/pipeline/probe.py` on the base model (GPU) and commit the output |
| Filter kept 3,129 of 4,000 traces (tau table pool column) | Table tau-calibration | unverified (constant `FILTER_KEPT`) | The filter logs of the submitted run (PROV-11) |
| Figure 1 labels: 4,000 prompts, 2,816 train in seed 7, r = 8, 250 to 300 steps, 200 of 1,000 prompts | Figure 1 | computed for the counts (`paper/figures/pipeline_standalone.tex` inputs `generated/numbers.tex`); design for r and steps | `make figures` rebuilds the PDF |
| Every added cell on every dimension (Table J.3), MedQuAD specificity 1.71, 1.47, 1.33 | Appendix J, `tab_cell_dimensions.tex` | computed (GV; `\statCellMedQuADMistral<Cond><Dim>`; withdrawals W20, W21) | Settled by the sync |
| Per-cell tau, graded, kept, train, val, steps, evaluated epoch (Table J.1); tau = 0.4 in every cell, no re-filter | Appendix J, `tab_cell_filter.tex`, `\statCellsTau`, `\statCellsRefilterLines` | computed (CR, from `logs/host/filter_*.log` and the synced trainer states) | Settled by the sync (PROV-3) |
| Table F.2: rank, alpha, dropout, target modules, epochs, batch per device and accumulation of the added cells | `\statCellsLoRA*`, `\statCellsTrainEpochs`, `\statCellsEffectiveBatch`, `\statCellsBatchPerDevice*`, `\statCellsGradAccum*` | computed (CR, from the synced adapter and trainer files) | Settled by the sync; learning rate, schedule, sequence length and precision stay design values from the train blocks of `configs/cells/*__2026-07.yaml` |
| Askers minus non-askers: positive in 3 of 4 cells, -0.002 in BioMistral-7B; Med42-8B -0.036 and -0.025 | Appendix J, `\statCellsAddedAskersGapPositive`, `\statCell<Cell>AskersGap` | computed (GV, `lora_askers_minus_non_askers`) | Settled by the sync |
| Incomplete host runs: BioMistral-7B on MedQA, 1 seed, 40 traces, 3 steps, 10 prompts; MedGemma-27B on MedQA and MedQuAD and Gemma-2-9B on HealthBench-Hard stopped in trace generation | Appendix J, `\statRunMedQABioMistral<...>` | computed (CR) for the counts; the four runs from `cell_recipe.json` (`incomplete` in `numbers.json`) | |
| Second grader on seed 42 (r = 0.42, rho = 0.35, 43 shared prompts) | Appendix H | computed (`data/derived/seed_42_qwen_grader_scores.json`, scores copied from the ignored `results/seed_42/eval/`) | Settled: `make derived` rebuilds the copy from the source files |
| Plan specified in the repository on 2026-04-29 (not an outside registry), grader switch 2026-05-01 | Abstract, section 3.4, Appendix B | design | commits aa7eec4 (`RESULTS.md` 7.B) and 1376aec; the paper omits the hashes because they find the public repo (withdrawals W19) |
