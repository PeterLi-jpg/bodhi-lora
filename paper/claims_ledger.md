# Claims ledger for paper/main.tex

Every number and claim in the abstract, the introduction and the results of `paper/main.tex`,
with the macro that prints it, the function in `scripts/paper_stats.py` that writes the macro,
and the input file. `make numbers` rebuilds every macro; nothing in this table is typed into
the paper by hand unless the Kind column says so.

The paper-level claim ledger, with the status of each claim made in the submission and the
replies, is `docs/CLAIMS.md`. This file is the number-level map for the current text.

Kinds:

- **computed**: the macro comes from committed data in this repository.
- **computed, untracked input**: the macro comes from a file that git ignores. `make numbers`
  skips it with a message when the file is absent.
- **posted**: the macro comes from `data/rebuttal_cells/posted_values.json`, a transcription
  of the OpenReview replies. The raw outputs are on the shared GPU host (audit PROV-1). Once
  `make cells` writes `data/rebuttal_cells/generated_values.json`, the same macros come from
  the synced outputs instead, and the cross-cell table drops the asterisk.
- **design**: a setting of the experiment (seeds, prompts per seed, context length, tau,
  LoRA rank). Its source is the config or launcher named in the Source column.
- **unverified**: typed in the paper; the output that produced it is not in the repository.

Inputs: `RM` is `results_modal/seed_*/` (the four `<condition>.json` files and
`epistemic_scores.json`), `PV` is `data/rebuttal_cells/posted_values.json`.

## Abstract

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| 5 seeds, 667 unique prompts | `\statNSeeds`, `\statUniquePrompts` | `counts_and_failures` | RM | computed |
| Active inquiry 17.5% to 45.6%, +28.0 points, 95% CI [+22.9, +33.2] | `\statActiveInquiryBase`, `\statActiveInquiryLoRA`, `\statActiveInquiryLoRAvsBaseDiff`, `...CI` | `describe_conditions`, `compare_conditions` | RM | computed |
| Context-seeking 1.40 to 1.75 | `\statContextSeekingBase`, `\statContextSeekingLoRA` | `describe_conditions` | RM | computed |
| No detected change in aggregate score, +0.004 [-0.011, +0.020] | `\statHealthBenchLoRAvsBaseDiff`, `...CI` | `compare_conditions` | RM | computed |
| In 4 added cells outside the evaluator's family, active inquiry rises in 3 | `\statCellsAddedCompared`, `\statCellsAddedAIRose` | `generality_section` | PV | posted |
| Aggregate falls in all 4, by 0.014 to 0.086 | `\statCellsAddedWithAggregate`, `\statCellsAddedDropMin`, `\statCellsAddedDropMax` | `generality_section` | PV | posted |
| Adding CoT to the adapter changes the red-flag rate by -20.6 points [-29.8, -11.4] | `\statRedFlagRateLoRACoTvsLoRADiff`, `...CI` | `compare_conditions` | RM | computed |
| Format leakage in 54.6% of LoRA+BODHI answers | `\statLeakagePctLoRACoT` | `rebuttal_section` | `rebuttal/results/interference_mechanism.json` | computed |
| Physician agreement kappa 0.35, target 0.6 | `\statKappaAll`; 0.6 is the pre-registered target | `kappa_section` | `results_modal/irr/kappa_results.json` | computed (target: design) |

## Introduction

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| The two passes overran the context on 4.3% of prompts | `\statCompletionDropPctBaseCoT` | `counts_and_failures` | RM | computed |
| 14 of 1,000 Hard prompts name a clinician speaker | `\statPostedClinicianPrompts`, `\statHardPrompts` | `generality_section`, `difficulty_range` | PV; `data/raw/healthbench_hard.jsonl` | posted; computed, untracked input |
| Four LoRA - Base effects with Holm-adjusted p <= 0.0065 | `\statScopeBoundingLoRAvsBasePHolm` (the largest of the four) | `compare_conditions` | RM | computed |
| Active inquiry rises in 3 of 4 added cells; aggregate falls in all 4 | `\statCellsAddedAIRose`, `\statCellsAddedCompared`, `\statCellsAddedWithAggregate` | `generality_section` | PV | posted |
| BODHI expansion "Balanced, Open-minded, Diagnostic, Humble, Inquisitive" | none | | the cited BMJ paper (docs/DECISIONS.md, D15 note) | design |

## Method and evaluation

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| 4,000 HealthBench prompts outside the Hard set | `\statFilterPool` | `tau_calibration` | constant `FILTER_POOL` (5,000 HealthBench minus 1,000 Hard; legacy launcher) | design |
| Seed-7 training split: 2,816 traces, 0 in the Hard set | `\statFilterTrainNSeedSeven`, `\statTrainHardOverlapSeedSeven` | `tau_calibration` | `data/sft_qwen_graded/seed_7_train.jsonl`, `data/raw/healthbench_hard.jsonl` | computed, untracked input |
| tau = 0.4, r = 8, alpha = 16, 300 steps, 64 examples per step, 2,048 tokens | none | | `legacy/configs/lora_medgemma27b_tunix.yaml`, `legacy/tpu/launch_5seeds_tunix.sh` (Table hparams comments) | design, pending the GCS adapter_config.json check (D1) |
| 1,000 draws per condition cover 667 unique prompts; 271 repeat | `\statPromptEvaluations`, `\statUniquePrompts`, `\statRepeatedPrompts` | `counts_and_failures` | RM | computed |
| Epistemic parse failures 0.20% to 1.49% per condition | `\statEpistemicParseFailPctBase`, `\statEpistemicParseFailPctMax` | `counts_and_failures` | RM | computed |
| HealthBench criterion failures at most 0.087% per run | `\statHealthBenchParseFailPctMaxRun` | `counts_and_failures` | RM | computed |
| 1 of 3 physicians; kappa 0.35 [0.28, 0.43] on 509 rows from 10 prompts | `\statKappaRaters`, `\statKappaAll`, `\statKappaAllCI`, `\statKappaAllN`, `\statKappaPrompts` | `kappa_section` | `results_modal/irr/kappa_results.json` | computed |
| 4,096-token evaluation context, greedy decoding, A100-80GB | none | | `legacy/modal_eval.py` (max_model_len patch, GPU type) | design |

## Results: the submitted cell (section 4.1, Tables 1 and 2, Figure 2)

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| Table 1 (score, t interval, pooled n, n per seed) | `tab_main_healthbench.tex` | `build_tables` | RM | computed |
| Table 2 (seven dimensions, rates, scored n) | `tab_epistemic.tex` | `build_tables` | RM | computed |
| Figure 2 (two panels) | `paper/figures/epistemic_bar.pdf` | `scripts/plot_epistemic_figures.py` | `results_modal/cross_seed_aggregate.json` | computed |
| Active inquiry, context-seeking, red flag, scope: diff, CI, Holm p | `\stat<Dim>LoRAvsBaseDiff`, `...CI`, `...PHolm` | `compare_conditions` | RM | computed |
| Questions per response 0.45 to 1.16 | `\statQuestionsBase`, `\statQuestionsLoRA` | `describe_conditions` | RM | computed |
| Hedging and specificity significant before but not after Holm | `\statHedgingQualityLoRAvsBaseDiff`, `\statSpecificityLoRAvsBaseDiff`, `...PHolm` | `compare_conditions` | RM | computed |
| Uncertainty acknowledgment unchanged | `\statUncertaintyAckLoRAvsBaseDiff`, `...CI` | `compare_conditions` | RM | computed |
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
| Table 3, first row (submitted cell) | `tab_generality.tex` | `generality_table` | RM | computed |
| Table 3, other rows | `tab_generality.tex` | `generality_table`, `resolve_cells` | PV (or `generated_values.json` after the sync) | posted |
| Interval for the added cells "pending" | `tab_generality.tex` | `_diff_with_ci` | needs per-seed values from the sync | posted |
| BioMistral-7B: 11.7% to 10.1%, teacher 14.5%, filter kept 20.4% | `\statCellHBBioMistral<...>` | `generality_section` | PV | posted |
| Adapter at or above its teacher in 3 of 5 cells (submitted included) | `\statCellsAtOrAboveTeacher`, `\statCellsCompared` | `generality_section` | RM, PV | posted |
| Aggregate falls in all 4 added cells outside the evaluator's family, 0.014 to 0.086 | `\statCellsAddedWithAggregate`, `\statCellsAddedDropMin`, `\statCellsAddedDropMax` | `generality_section` | PV | posted |
| Posted paired t: each drop p < 0.02 after Holm over six cells | `\statPostedAggDropPBelow` | `generality_section` | PV | posted; recompute with `make cells` (`rebuttal/analyses/quality_drop.py`) |
| MedQA preamble run: base active inquiry 99% | `\statPostedMedQAPreambleAI` | `generality_section` | PV | posted |
| MedQA aggregate "rises" | none (no value was posted) | | `rebuttal/responses/posted/HomM_2_followup_part1.md:11` | posted, no number |
| Recipe and evaluation differences of the added cells | none | | `rebuttal/configs/*.yaml`, `rebuttal/launch/run_cell.sh`, `scripts/_vllm_engine.py` (comments above Table hparams-cells) | design |

## Results: CoT on the adapted model (section 4.3)

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| Active inquiry 80.0% under LoRA+BODHI | `\statActiveInquiryLoRACoT` | `describe_conditions` | RM | computed |
| Red flag 1.67 to 1.51, -0.161 [-0.262, -0.059], p = 0.012, Holm 0.058 | `\statRedFlagLoRA`, `\statRedFlagLoRACoT`, `\statRedFlagLoRACoTvsLoRADiff`, `...CI`, `...P`, `...PHolm` | `describe_conditions`, `compare_conditions` | RM | computed |
| Red-flag rate -20.6 points [-29.8, -11.4], p = 0.0034 | `\statRedFlagRateLoRACoTvsLoRADiff`, `...CI`, `...P` | `compare_conditions` | RM | computed |
| Against Base: level on the score, lower on the rate | `\statRedFlagLoRACoTvsBaseDiff`, `...CI`, `\statRedFlagRateLoRACoTvsBaseDiff`, `...CI` | `compare_conditions` | RM | computed |
| Aggregate LoRA+BODHI minus LoRA -0.008 [-0.020, +0.004] | `\statHealthBenchLoRACoTvsLoRADiff`, `...CI` | `compare_conditions` | RM | computed |
| Leakage 54.6% against 0.2% to 3.0% | `\statLeakagePct<Cond>` | `rebuttal_section` | `rebuttal/results/interference_mechanism.json` | computed |
| Red flag 1.33 (n = 507) leaked, 1.73 (n = 421) not leaked | `\statRedFlagLeaked`, `\statRedFlagClean`, `\statLeakageNLoRACoT`, `\statLeakageNCleanLoRACoT` | `rebuttal_section` | `rebuttal/results/interference_mechanism.json` | computed |

## Discussion

| Claim | Macro(s) | Function | Source | Kind |
|---|---|---|---|---|
| Aggregate falls by 0.014 to 0.086 in the added cells outside the family | `\statCellsAddedDropMin`, `\statCellsAddedDropMax` | `generality_section` | PV | posted |
| Baseline code exists and has not run | none | | `scripts/baselines/`, `rebuttal/launch/run_baselines.sh` | design |

## Appendix numbers that are not computed here

| Claim | Where | Kind | What would settle it |
|---|---|---|---|
| Contamination probe: 200 sampled, 105 long enough, 0 exact matches, 0 overlap | Appendix A | unverified | Re-run `scripts/contamination_probe.py` on the base model (GPU) and commit the output |
| Filter kept 3,129 of 4,000 traces (tau table pool column) | Table tau-calibration | unverified (constant `FILTER_KEPT`) | The filter logs of the submitted run (PROV-11) |
| Second grader on seed 42 (r = 0.42, rho = 0.35, 43 shared prompts) | Appendix H | computed, untracked input (`results/seed_42/eval/`) | Commit the Qwen-graded files, or keep them as an untracked input |
| Pre-registration date 2026-04-29, grader switch 2026-05-01 | Appendix B | design | commits aa7eec4 and 1376aec |
