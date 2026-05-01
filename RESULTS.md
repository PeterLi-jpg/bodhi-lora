# RESULTS

**Eval protocol (applies to all sections):** test set = 5 independent 200-sample draws from HealthBench Hard 1K (seeds 0–4); scores = mean ± std across those 5 draws. Training uses Hard 1K + Full ~5K minus the per-seed eval draw. Run `scripts/check_dataset_overlap.py --seeds 5 --tag-overlap --train-jsonl data/sft/train.jsonl` before every cluster job. See [#3](https://github.com/PeterLi-jpg/bohdi-lora/issues/3) for the full protocol. Do not fill any cell from a single draw.

- **Clinical involvement:** Sections 7.B–7.D require physician judgment ([#73](https://github.com/PeterLi-jpg/bohdi-lora/issues/73), [#74](https://github.com/PeterLi-jpg/bohdi-lora/issues/74), [#75](https://github.com/PeterLi-jpg/bohdi-lora/issues/75)). Peter/Sahil should defer these to Zineb, Ash Doulla, Hillary, and Felipe (felipeocampoos) and add their GitHub handles to the relevant issues. Manual annotation tasks (Sections 6, 7.A) involve Max (maximinl), Ashley, Martha, and Sohyeon.

---

## TLDR

- **Claim 1 — BOHDI wrapper improves calibration:** The paper claims the BOHDI prompt wrapper reduces overconfidence in model outputs. Requires Main Evaluation calibration columns (Brier, ECE) and Calibration Validity Spearman rho. NOT RUN.
- **Claim 2 — LoRA fine-tuning on BOHDI traces improves HB-Hard score:** Requires Main Evaluation rows comparing Base vs LoRA+BOHDI with statistical confidence. NOT RUN.
- **Claim 3 — Gains are not explained by verbosity:** Requires Ablation D (truncated-length control). NOT RUN.
- **Claim 4 — Gains hold across model scales:** Requires Ablation C (E2B, E4B, MedGemma-27B). NOT RUN.
- **Claim 5 — Gains generalize beyond training distribution:** Requires Generalization tables — MedQA accuracy, cross-grader consistency, adversarial robustness, knowledge retention. NOT RUN.
- **Claim 6 — Hard-tier examples benefit most (U-shape):** Requires U-shape Analysis with bootstrap CIs per difficulty tier. NOT RUN.
- **Claim 7 — Training data quality is not uniformly-hedged noise:** Requires Pre-filter Data Quality hedge-rate-by-quartile analysis. NOT RUN.
- **Claim 8 — The work is publication-ready for a medical-AI venue:** Requires Section 7 (clinical safety & threats to validity, [#72](https://github.com/PeterLi-jpg/bohdi-lora/issues/72)–[#75](https://github.com/PeterLi-jpg/bohdi-lora/issues/75)) — pretraining contamination probe, physician-validated grader, bias audit, harm-mode analysis. NOT RUN. Without these, the paper is not submittable to NEJM AI, npj Digital Medicine, or JAMA AI.

---

## 1. Main Evaluation ([#25](https://github.com/PeterLi-jpg/bohdi-lora/issues/25) gold baseline, [#3](https://github.com/PeterLi-jpg/bohdi-lora/issues/3) grader independence, [#1](https://github.com/PeterLi-jpg/bohdi-lora/issues/1) calibration metrics)

Tests whether BOHDI wrapper and LoRA fine-tuning improve clinical answer quality and calibration on HealthBench-Hard.

**Data split (enforced — see [#3](https://github.com/PeterLi-jpg/bohdi-lora/issues/3)):**
- **Training pool:** HealthBench Hard (1000 examples) + HealthBench Full (5000 examples), deduplicated by `prompt_id` → ~5000 unique prompts. Run through the BOHDI wrapper, graded, and filtered to produce `data/sft/train.jsonl`.
- **Test set:** 5 independent 200-sample draws from HealthBench Hard (1000 examples), one per seed (seeds 0–4). Each draw is a different random subset of the 1K Hard examples. Scores are reported as mean ± std across the 5 draws. This gives a variance estimate on the evaluation without requiring 5 full training runs.
- **Leakage prevention:** for each seed, the 200-sample draw must be excluded from training data generation via `--exclude-ids`. Run the pre-training leakage check before every cluster job:
  ```
  python scripts/check_dataset_overlap.py \
      --train-jsonl data/sft/train.jsonl \
      --seeds 5 \
      --tag-overlap
  ```
  This checks (1) no eval IDs leaked into the training file per seed, and (2) tag/theme distribution is balanced. Do not report any number from a single-seed draw.

| Config | Status | HB-Hard score (mean±std, 5 seeds) | Brier model | ECE model | Brier grader* | ECE grader* | Mean tokens | Parse fail % |
|---|---|---|---|---|---|---|---|---|
| Base | NOT RUN | -- | -- | -- | -- | -- | -- | -- |
| Base+BOHDI | NOT RUN | -- | -- | -- | -- | -- | -- | -- |
| LoRA no wrapper | NOT RUN | -- | -- | -- | -- | -- | -- | -- |
| LoRA+BOHDI | NOT RUN | -- | -- | -- | -- | -- | -- | -- |
| Gold-SFT baseline ([#25](https://github.com/PeterLi-jpg/bohdi-lora/issues/25)) | NOT RUN | -- | -- | -- | -- | -- | -- | -- |

\* Grader Brier/ECE are legacy grader-consistency proxies, not model calibration.

---

## 2. Calibration Validity ([#1](https://github.com/PeterLi-jpg/bohdi-lora/issues/1) open; [#12](https://github.com/PeterLi-jpg/bohdi-lora/issues/12) closed — token-logprob extraction implemented)

Tests whether the token-logprob proxy correlates with rubric outcomes. [#12](https://github.com/PeterLi-jpg/bohdi-lora/issues/12) is resolved (logprobs extracted during eval); [#1](https://github.com/PeterLi-jpg/bohdi-lora/issues/1) remains open (proxy validity unconfirmed).

| Config | Status | geomean token prob vs rubric Spearman rho | verbalized confidence (planned) | token-logprob proxy valid? |
|---|---|---|---|---|
| Base | NOT RUN | -- | -- | -- |
| Base+BOHDI | NOT RUN | -- | -- | -- |
| LoRA no wrapper | NOT RUN | -- | -- | -- |
| LoRA+BOHDI | NOT RUN | -- | -- | -- |

Note: grader Brier/ECE are legacy grader-consistency proxies, not model calibration.

---

## 3. Ablations

### A — Trace quantity ([#30](https://github.com/PeterLi-jpg/bohdi-lora/issues/30))

Effect of training set size on HB-Hard score.

| n traces | Status | HB-Hard score | hard-tier fail rate | 95% CI |
|---|---|---|---|---|
| 250 | NOT RUN | -- | -- | -- |
| 500 | NOT RUN | -- | -- | -- |
| 1000 | NOT RUN | -- | -- | -- |
| 2000 | NOT RUN | -- | -- | -- |
| full | NOT RUN | -- | -- | -- |

### B — LoRA rank ([#17](https://github.com/PeterLi-jpg/bohdi-lora/issues/17))

Effect of adapter rank on score and compute cost.

| rank | Status | HB-Hard score | trainable params (M) | GPU mem (GB) |
|---|---|---|---|---|
| r=4 | NOT RUN | -- | -- | -- |
| r=8 | NOT RUN | -- | -- | -- |
| r=16 | NOT RUN | -- | -- | -- |
| r=32 | NOT RUN | -- | -- | -- |
| r=64 | NOT RUN | -- | -- | -- |

### C — Model scale ([#26](https://github.com/PeterLi-jpg/bohdi-lora/issues/26))

Whether gains transfer across base model sizes.

| Model | Status | base score | LoRA score | delta |
|---|---|---|---|---|
| E2B | NOT RUN | -- | -- | -- |
| E4B | NOT RUN | -- | -- | -- |
| MedGemma-27B | NOT RUN | -- | -- | -- |

### D — Verbosity baseline ([#61](https://github.com/PeterLi-jpg/bohdi-lora/issues/61))

Controls for score inflation from longer outputs by truncating LoRA responses to base-model length.

| Config | Status | HB-Hard score | mean tokens | score/100 tokens |
|---|---|---|---|---|
| Base | NOT RUN | -- | -- | -- |
| LoRA full | NOT RUN | -- | -- | -- |
| LoRA truncated to base length | NOT RUN | -- | -- | -- |

### E — Per-theme training ([#18](https://github.com/PeterLi-jpg/bohdi-lora/issues/18))

Whether training on safety-heavy vs non-safety traces drives theme-specific vs general gains.

| Training pool | Status | HB-Hard overall | emergency_referrals score | hedging score |
|---|---|---|---|---|
| Full pool | NOT RUN | -- | -- | -- |
| Safety-only | NOT RUN | -- | -- | -- |
| Non-safety | NOT RUN | -- | -- | -- |

### F — BOHDI wrapper component ablation ([#69](https://github.com/PeterLi-jpg/bohdi-lora/issues/69))

Which component of the BOHDI wrapper drives the behavioral change in the LoRA.

| BOHDI component removed | Status | HB-Hard score | delta vs full BOHDI LoRA |
|---|---|---|---|
| None (full BOHDI) | NOT RUN | -- | -- |
| Domain framing | NOT RUN | -- | -- |
| Calibration prompt | NOT RUN | -- | -- |
| Abstention instruction | NOT RUN | -- | -- |
| Multi-turn clarification | NOT RUN | -- | -- |

### G — Multi-seed training variance ([#70](https://github.com/PeterLi-jpg/bohdi-lora/issues/70))

Whether gains are robust to LoRA weight initialization seed (independent of eval-draw seeds).

| Init seed | Status | HB-Hard score (mean across 5 eval draws) | std across 5 eval draws |
|---|---|---|---|
| seed 0 | NOT RUN | -- | -- |
| seed 1 | NOT RUN | -- | -- |
| seed 2 | NOT RUN | -- | -- |

### H — Score filter sensitivity ([#71](https://github.com/PeterLi-jpg/bohdi-lora/issues/71), [#4](https://github.com/PeterLi-jpg/bohdi-lora/issues/4))

Whether the choice of `--score-field` in `filter_traces.py` materially changes the trained LoRA.

| Score field | Status | Training set size | HB-Hard score | delta vs overall_score |
|---|---|---|---|---|
| overall_score (default) | NOT RUN | -- | -- | -- |
| positive_criteria_rate | NOT RUN | -- | -- | -- |
| absolute_point_score | NOT RUN | -- | -- | -- |

---

## 4. Generalization & Robustness

### A — MedQA ([#19](https://github.com/PeterLi-jpg/bohdi-lora/issues/19))

Out-of-distribution accuracy on a standard medical QA benchmark.

| Config | Status | accuracy % | abstention rate % | refusal rate % |
|---|---|---|---|---|
| Base | NOT RUN | -- | -- | -- |
| Base+BOHDI | NOT RUN | -- | -- | -- |
| LoRA no wrapper | NOT RUN | -- | -- | -- |
| LoRA+BOHDI | NOT RUN | -- | -- | -- |

### B — Cross-grader consistency ([#3](https://github.com/PeterLi-jpg/bohdi-lora/issues/3), [#11](https://github.com/PeterLi-jpg/bohdi-lora/issues/11))

Whether scores are stable across different grader models.

| Config | Status | Qwen-14B score | Llama-70B score | delta | Spearman rho |
|---|---|---|---|---|---|
| Base | NOT RUN | -- | -- | -- | -- |
| Base+BOHDI | NOT RUN | -- | -- | -- | -- |
| LoRA no wrapper | NOT RUN | -- | -- | -- | -- |
| LoRA+BOHDI | NOT RUN | -- | -- | -- | -- |

### C — Adversarial robustness ([#28](https://github.com/PeterLi-jpg/bohdi-lora/issues/28))

Whether the model capitulates to credential pressure or authority cues.

| Config | Status | credential-pressure capitulation % | authority capitulation % | false-refusal on controls % |
|---|---|---|---|---|
| Base | NOT RUN | -- | -- | -- |
| Base+BOHDI | NOT RUN | -- | -- | -- |
| LoRA no wrapper | NOT RUN | -- | -- | -- |
| LoRA+BOHDI | NOT RUN | -- | -- | -- |

### D — Knowledge retention ([#27](https://github.com/PeterLi-jpg/bohdi-lora/issues/27))

Whether LoRA fine-tuning degrades general medical knowledge.

| Config | Status | MedMCQA acc % | MMLU-Medical acc % | hedge rate on clear-cut % |
|---|---|---|---|---|
| Base | NOT RUN | -- | -- | -- |
| LoRA | NOT RUN | -- | -- | -- |

---

## 5. U-shape Analysis (eval_ushape.py, [#63](https://github.com/PeterLi-jpg/bohdi-lora/issues/63))

Tests whether gains are concentrated in hard examples, with statistical confidence per tier.

| Tier | Status | n examples | base fail rate % | LoRA fail rate % | delta (pp) | bootstrap 95% CI |
|---|---|---|---|---|---|---|
| easy | NOT RUN | -- | -- | -- | -- | -- |
| medium | NOT RUN | -- | -- | -- | -- | -- |
| hard | NOT RUN | -- | -- | -- | -- | -- |

CI overlap at any tier means that tier's claim is not statistically distinguishable at current sample size.

---

## 6. Pre-filter Data Quality ([#64](https://github.com/PeterLi-jpg/bohdi-lora/issues/64))

Tests whether the BOHDI training traces contain meaningful calibration signal or just uniform hedging.

| Subset | Status | n traces | BOHDI hedge rate % | mean rubric score |
|---|---|---|---|---|
| overall | NOT RUN | -- | -- | -- |
| Q1 (bottom quartile) | NOT RUN | -- | -- | -- |
| Q2 | NOT RUN | -- | -- | -- |
| Q3 | NOT RUN | -- | -- | -- |
| Q4 (top quartile) | NOT RUN | -- | -- | -- |

If hedge rate is uniformly high regardless of quartile, the training signal is unconditional hedging.

---

## 7. Clinical Safety & Threats to Validity ([#72](https://github.com/PeterLi-jpg/bohdi-lora/issues/72), [#73](https://github.com/PeterLi-jpg/bohdi-lora/issues/73), [#74](https://github.com/PeterLi-jpg/bohdi-lora/issues/74), [#75](https://github.com/PeterLi-jpg/bohdi-lora/issues/75))

### A — Pretraining contamination probe ([#72](https://github.com/PeterLi-jpg/bohdi-lora/issues/72))

If positive, every other number in this file is meaningless. Run this first. Probe protocol: feed base model the first 30 tokens of 100 random HealthBench Hard prompts; flag if exact-completion rate is significantly elevated vs random medical text baseline (p < 0.05).

| Probe | Status | model | exact-completion rate (%) | n probes | p-value vs baseline |
|---|---|---|---|---|---|
| completion-rate on first-30-tokens probe (HealthBench Hard prompts) | NOT RUN | -- | -- | -- | -- |
| random medical text baseline | NOT RUN | -- | -- | -- | -- |

### B — Human expert grader validation ([#73](https://github.com/PeterLi-jpg/bohdi-lora/issues/73))

Cohen's kappa between Qwen-14B and physician consensus on a sample of 50 responses per configuration (200 total), graded blind by ≥2 board-certified physicians. **Success threshold: kappa ≥ 0.6 across both physicians; below this the grader is unreliable for any calibration claim in this file.** Defer to Zineb, Ash Doulla, Hillary, Felipe.

| Config | Status | n responses graded | physician 1 vs Qwen-14B kappa | physician 2 vs Qwen-14B kappa | inter-physician kappa | overall agreement % |
|---|---|---|---|---|---|---|
| Base | NOT RUN | -- | -- | -- | -- | -- |
| Base+BOHDI | NOT RUN | -- | -- | -- | -- | -- |
| LoRA no wrapper | NOT RUN | -- | -- | -- | -- | -- |
| LoRA+BOHDI | NOT RUN | -- | -- | -- | -- | -- |

### C — Demographic / bias audit ([#74](https://github.com/PeterLi-jpg/bohdi-lora/issues/74))

Stratified per-tag scores. Flag tags where LoRA shift differs from overall by >2σ. Clinical interpretation: Zineb, Ash Doulla, Hillary, Felipe.

| example_tag | Status | n examples | base score | LoRA score | shift (pp) | overall shift (pp) | flagged? |
|---|---|---|---|---|---|---|---|
| emergency_referrals | NOT RUN | -- | -- | -- | -- | -- | -- |
| hedging | NOT RUN | -- | -- | -- | -- | -- | -- |
| context_seeking | NOT RUN | -- | -- | -- | -- | -- | -- |
| demographic-flagged | NOT RUN | -- | -- | -- | -- | -- | -- |
| severity-flagged | NOT RUN | -- | -- | -- | -- | -- | -- |

### D — Harm-potential failure-mode review ([#75](https://github.com/PeterLi-jpg/bohdi-lora/issues/75))

High-stakes prompts where appropriate response is direct and confident. Clinician-led review: Zineb, Ash Doulla, Hillary, Felipe.

| clinical category | Status | n prompts | base+wrapper care-delay rate % | LoRA care-delay rate % | hedging-on-clear-cut % | clinician verdict |
|---|---|---|---|---|---|---|
| chest pain triage | NOT RUN | -- | -- | -- | -- | -- |
| anaphylaxis | NOT RUN | -- | -- | -- | -- | -- |
| suicide risk | NOT RUN | -- | -- | -- | -- | -- |
| drug-allergy contraindications | NOT RUN | -- | -- | -- | -- | -- |
| total | NOT RUN | -- | -- | -- | -- | -- |
