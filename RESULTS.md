# RESULTS

## TLDR

- **Claim 1 — BOHDI wrapper improves calibration:** The paper claims the BOHDI prompt wrapper reduces overconfidence in model outputs. Requires Main Evaluation calibration columns (Brier, ECE) and Calibration Validity Spearman rho. NOT RUN.
- **Claim 2 — LoRA fine-tuning on BOHDI traces improves HB-Hard score:** Requires Main Evaluation rows comparing Base vs LoRA+BOHDI with statistical confidence. NOT RUN.
- **Claim 3 — Gains are not explained by verbosity:** Requires Ablation D (truncated-length control). NOT RUN.
- **Claim 4 — Gains hold across model scales:** Requires Ablation C (E2B, E4B, MedGemma-27B). NOT RUN.
- **Claim 5 — Gains generalize beyond training distribution:** Requires Generalization tables — MedQA accuracy, cross-grader consistency, adversarial robustness, knowledge retention. NOT RUN.
- **Claim 6 — Hard-tier examples benefit most (U-shape):** Requires U-shape Analysis with bootstrap CIs per difficulty tier. NOT RUN.
- **Claim 7 — Training data quality is not uniformly-hedged noise:** Requires Pre-filter Data Quality hedge-rate-by-quartile analysis. NOT RUN.

---

## 1. Main Evaluation (#25 gold baseline, #3 grader independence, #1 calibration metrics)

Tests whether BOHDI wrapper and LoRA fine-tuning improve clinical answer quality and calibration on HealthBench-Hard.

**Data split (enforced — see #3):**
- **Training pool:** HealthBench Hard (1000 examples) + HealthBench Full (5000 examples), deduplicated by `prompt_id` → ~5000 unique prompts, minus the 200 holdout IDs. These are run through the BOHDI wrapper, graded, and filtered to produce `data/sft/train.jsonl`.
- **Test set:** 200-sample holdout drawn exclusively from HealthBench Hard (`data/raw/hard_200_sample_ids.json`). Hard-isolated — zero overlap with training data. Verify with `scripts/check_dataset_overlap.py` before every cluster run.
- **Seeds:** all scores are mean ± std across **5 different training seeds** (seeds 0–4), each producing an independent LoRA checkpoint evaluated on the same 200-sample holdout. Do not fill any cell from a single-seed run.

| Config | Status | HB-Hard score (mean±std, 5 seeds) | Brier model | ECE model | Brier grader* | ECE grader* | Mean tokens | Parse fail % |
|---|---|---|---|---|---|---|---|---|
| Base | NOT RUN | -- | -- | -- | -- | -- | -- | -- |
| Base+BOHDI | NOT RUN | -- | -- | -- | -- | -- | -- | -- |
| LoRA no wrapper | NOT RUN | -- | -- | -- | -- | -- | -- | -- |
| LoRA+BOHDI | NOT RUN | -- | -- | -- | -- | -- | -- | -- |
| Gold-SFT baseline (#25) | NOT RUN | -- | -- | -- | -- | -- | -- | -- |

\* Grader Brier/ECE are legacy grader-consistency proxies, not model calibration.

---

## 2. Calibration Validity (#1, #12)

Tests whether the token-logprob proxy is a valid stand-in for verbalized confidence, and whether it correlates with rubric scores.

| Config | Status | geomean token prob vs rubric Spearman rho | verbalized confidence (planned) | token-logprob proxy valid? |
|---|---|---|---|---|
| Base | NOT RUN | -- | -- | -- |
| LoRA no wrapper | NOT RUN | -- | -- | -- |

Note: grader Brier/ECE are legacy grader-consistency proxies, not model calibration.

---

## 3. Ablations

### A — Trace quantity (#30)

Effect of training set size on HB-Hard score.

| n traces | Status | HB-Hard score | hard-tier fail rate | 95% CI |
|---|---|---|---|---|
| 250 | NOT RUN | -- | -- | -- |
| 500 | NOT RUN | -- | -- | -- |
| 1000 | NOT RUN | -- | -- | -- |
| 2000 | NOT RUN | -- | -- | -- |
| full | NOT RUN | -- | -- | -- |

### B — LoRA rank (#17)

Effect of adapter rank on score and compute cost.

| rank | Status | HB-Hard score | trainable params (M) | GPU mem (GB) |
|---|---|---|---|---|
| r=4 | NOT RUN | -- | -- | -- |
| r=8 | NOT RUN | -- | -- | -- |
| r=16 | NOT RUN | -- | -- | -- |
| r=32 | NOT RUN | -- | -- | -- |
| r=64 | NOT RUN | -- | -- | -- |

### C — Model scale (#26)

Whether gains transfer across base model sizes.

| Model | Status | base score | LoRA score | delta |
|---|---|---|---|---|
| E2B | NOT RUN | -- | -- | -- |
| E4B | NOT RUN | -- | -- | -- |
| MedGemma-27B | NOT RUN | -- | -- | -- |

### D — Verbosity baseline (#61)

Controls for score inflation from longer outputs by truncating LoRA responses to base-model length.

| Config | Status | HB-Hard score | mean tokens | score/100 tokens |
|---|---|---|---|---|
| Base | NOT RUN | -- | -- | -- |
| LoRA full | NOT RUN | -- | -- | -- |
| LoRA truncated to base length | NOT RUN | -- | -- | -- |

### E — Per-theme training (#18)

Whether training on safety-heavy vs non-safety traces drives theme-specific vs general gains.

| Training pool | Status | HB-Hard overall | emergency_referrals score | hedging score |
|---|---|---|---|---|
| Full pool | NOT RUN | -- | -- | -- |
| Safety-only | NOT RUN | -- | -- | -- |
| Non-safety | NOT RUN | -- | -- | -- |

---

## 4. Generalization & Robustness

### A — MedQA (#19)

Out-of-distribution accuracy on a standard medical QA benchmark.

| Config | Status | accuracy % | abstention rate % | refusal rate % |
|---|---|---|---|---|
| Base | NOT RUN | -- | -- | -- |
| Base+BOHDI | NOT RUN | -- | -- | -- |
| LoRA no wrapper | NOT RUN | -- | -- | -- |
| LoRA+BOHDI | NOT RUN | -- | -- | -- |

### B — Cross-grader consistency (#3, #11)

Whether scores are stable across different grader models.

| Config | Status | Qwen-14B score | Llama-70B score | delta | Spearman rho |
|---|---|---|---|---|---|
| Base | NOT RUN | -- | -- | -- | -- |
| Base+BOHDI | NOT RUN | -- | -- | -- | -- |
| LoRA no wrapper | NOT RUN | -- | -- | -- | -- |
| LoRA+BOHDI | NOT RUN | -- | -- | -- | -- |

### C — Adversarial robustness (#28)

Whether the model capitulates to credential pressure or authority cues.

| Config | Status | credential-pressure capitulation % | authority capitulation % | false-refusal on controls % |
|---|---|---|---|---|
| Base | NOT RUN | -- | -- | -- |
| Base+BOHDI | NOT RUN | -- | -- | -- |
| LoRA no wrapper | NOT RUN | -- | -- | -- |
| LoRA+BOHDI | NOT RUN | -- | -- | -- |

### D — Knowledge retention (#27)

Whether LoRA fine-tuning degrades general medical knowledge.

| Config | Status | MedMCQA acc % | MMLU-Medical acc % | hedge rate on clear-cut % |
|---|---|---|---|---|
| Base | NOT RUN | -- | -- | -- |
| LoRA | NOT RUN | -- | -- | -- |

---

## 5. U-shape Analysis (eval_ushape.py, #63)

Tests whether gains are concentrated in hard examples, with statistical confidence per tier.

| Tier | Status | n examples | base fail rate % | LoRA fail rate % | delta (pp) | bootstrap 95% CI |
|---|---|---|---|---|---|---|
| easy | NOT RUN | -- | -- | -- | -- | -- |
| medium | NOT RUN | -- | -- | -- | -- | -- |
| hard | NOT RUN | -- | -- | -- | -- | -- |

CI overlap at any tier means that tier's claim is not statistically distinguishable at current sample size.

---

## 6. Pre-filter Data Quality (#64)

Tests whether the BOHDI training traces contain meaningful calibration signal or just uniform hedging.

| Subset | Status | n traces | BOHDI hedge rate % | mean rubric score |
|---|---|---|---|---|
| overall | NOT RUN | -- | -- | -- |
| Q1 (bottom quartile) | NOT RUN | -- | -- | -- |
| Q2 | NOT RUN | -- | -- | -- |
| Q3 | NOT RUN | -- | -- | -- |
| Q4 (top quartile) | NOT RUN | -- | -- | -- |

If hedge rate is uniformly high regardless of quartile, the training signal is unconditional hedging.
