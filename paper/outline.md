# Paper outline — plain prose version for quick review

Reading time: ~5 min. Pairs with `main.tex` — edit prose here first, port to LaTeX once the shape is right. Page budget is 4 pages main body; rough per-section split below.

---

## Title
**Training for Humility: Distilling Epistemic Wrappers into Clinical LLM Weights via LoRA**

Working alternatives if the primary sounds too clever:
- *From Wrapper to Weights: LoRA-Distilling BOHDI into MedGemma*
- *Internalising Epistemic Humility in Clinical LLMs*

## Abstract (~150 words, 1 paragraph)

**Hook:** Medical LLMs fail hardest at the extremes — easy cases and emergencies — an inverted-U of failure that flat averages hide.

**Gap:** Inference-time epistemic wrappers (BOHDI et al.) fix this but cost 2–5× compute and don't transfer.

**Contribution:** Distill the wrapper's behaviour into a LoRA adapter so the model is humble without the wrapper at serve time.

**Method in one line:** generate wrapper traces → grade → filter → LoRA SFT.

**Eval in one line:** 4-config sweep (base/LoRA × with/without wrapper) on HealthBench Hard, stratified by difficulty tertile and theme.

**Result placeholder:** LoRA without wrapper matches base+wrapper overall and flattens the U on emergency and hedging cases.

---

## 1. Introduction (~0.8 page)

Three paragraphs.

**¶1 — The failure shape.** Cite *Nature Medicine* 2026 triage study showing ChatGPT undertriages 52% of gold-standard emergencies while handling classical ones fine. Generalise: when failure rate is plotted against case difficulty / acuity, medical LLMs trace an inverted U. Averages obscure this.

**¶2 — Why it persists.** RLHF reinforces confident answers over uncertainty or clarification because benchmarks mostly reward confident answers. Inference-time wrappers (BOHDI for humility/calibration/abstention; Constitutional AI; self-consistency) help but add latency, cost, and prompt-engineering brittleness — and don't survive when the wrapper is stripped.

**¶3 — Our question + contributions.**
- Can we internalise wrapper behaviour into model *weights*?
- Contributions:
  1. Trace-distillation pipeline for LoRA SFT on wrapper-conditioned outputs, on MedGemma-27B + Gemma-3n-E4B ablation.
  2. Stratified evaluation protocol (U-shape inspired by the *Nature Medicine* paper).
  3. Preliminary evidence that the U *can* flatten in the weights-only condition.

## 2. Related Work (~0.4 page)

Four micro-paragraphs. One sentence each, focused on what differs from us.

- **Clinical LLMs** (MedGemma, Med-PaLM): strong averages, no tail-stratification.
- **Calibration** (Brier, ECE, Kadavath et al.): measure-definition gap — "confidence" is usually the evaluator's own grade, not an independent model signal. We inherit this problem honestly.
- **Epistemic wrappers** (Constitutional AI, self-consistency, BOHDI): inference-time, we're weight-time.
- **LoRA for behaviour distillation** (DPO and friends): standard tool; we apply to wrapper traces specifically.

## 3. Method (~1.2 pages)

A **pipeline figure** (Fig 1) does most of the work.

### 3.1 Trace generation
HealthBench prompts (~4,800 after excluding the 200-prompt eval holdout) → run through `BODHI(π_base)` → save the *content* of each response (not the internal wrapper turns).

### 3.2 Rubric grading
Each HealthBench prompt has a per-criterion rubric with positive/negative points and tags. An independent grader model scores each trace per-criterion. Aggregate score:
$$\text{score}(x,y) = \frac{\sum_{i \in \text{met}} p_i}{\sum_{j: p_j > 0} p_j}$$

Paper run: Qwen2.5-14B-Instruct-AWQ. Cheap iteration: Qwen2.5-0.5B-Instruct.

### 3.3 Filtering
Keep traces with score ≥ 0.4. 10% held out as validation.

### 3.4 LoRA SFT
r=16, α=32, target all attn/MLP projections. 3 epochs, cosine decay, lr 1e-4, bf16, seed 42. Loss on assistant turn only.

### 3.5 Stratified evaluation (the methodological contribution)
- **Difficulty tertiles** — easy/medium/hard by rubric positive-point sum.
- **Theme breakdown** — HealthBench `theme:*` tags (emergency_referrals, hedging, context_seeking, …).
- Report: mean, fail rate (score < 0.4), Brier and ECE labelled clearly as *grader-consistency* not model calibration (Limitation §5).

## 4. Experiments (~0.7 page)

**Setup table or inline.** 4 configs × 200 prompts, greedy decoding, same grader.

### Main table (Table 1)
| Config | Mean ↑ | Fail (easy) ↓ | Fail (hard) ↓ |
|---|---|---|---|
| Base | [TBD] | [TBD] | [TBD] |
| Base + Wrapper | [TBD] | [TBD] | [TBD] |
| LoRA | [TBD] | [TBD] | [TBD] |
| LoRA + Wrapper | [TBD] | [TBD] | [TBD] |

### Two figures
- **Fig 2** `u_fail_smooth.png` — failure rate vs continuous difficulty, with 10-bin means and quadratic fits per config. The headline money figure.
- **Fig 3** `theme_fail.png` — per-theme bar chart, with `emergency_referrals` and `hedging` highlighted.

### Ablation
Same pipeline on Gemma-3n-E4B. Either:
- Confirms the effect is robust to base model (good)
- Shows it needs scale (still interesting, qualifies the claim)

Whichever, report honestly.

## 5. Limitations and Open Choices (~0.5 page)

Three specific open issues, explicit:

1. **Grader coupling** — same model grades the training set and the eval. Cross-check with a second grader family is future work.
2. **Calibration metric semantics** — Brier/ECE here measure grader-internal consistency, not model-internal confidence. Fields renamed `*_grader_consistency` in released outputs.
3. **Score normalisation** — `earned / Σ positive-points` can go negative; alternatives (`earned / Σ|points|`, per-prompt z-score) are on the table.

Plus: single seed, single test split, text-only HealthBench. Variance bars absent.

## 6. Conclusion (~0.2 page)

One paragraph. The weight-level internalisation hypothesis is testable and — [TBD: the preliminary evidence does or does not support it]. The pipeline and stratified eval generalise beyond BOHDI.

## Impact Statement (ICML-required, ~0.15 page)

Clinical LLMs' dangerous failures concentrate on the cases that matter most. A method that makes models quieter about what they don't know plausibly reduces harm. The risk we're alert to is over-generalisation: a too-humble model that refuses legitimate questions. The per-theme breakdown in §3.5 is what we use to detect this failure mode.

---

## What's blocking a real draft

Everything marked `[TBD]` above. All of them dissolve into the same gating step: **one real pilot run** (even on Gemma-3n-E4B with 500 prompts) produces:
- the table rows
- the `u_fail_smooth.png`
- the `theme_fail.png`
- one honest sentence for the abstract, intro, and conclusion

Without that run: the skeleton is credible, the results section is a placeholder, reviewers see through it.

## Planning questions for Sebastian and Felipe

- **Venue confirmation:** Is SD4H actually the target, or are we debating workshops?
- **Data timing:** Is a Felipe-run pilot (E4B, 500 prompts, ~8h H100) doable this weekend?
- **Scope:** Full 27B run or only the 4B ablation by deadline? If only 4B, does the paper still work? (I think yes — reframe as "pipeline demonstration on a small model" with 27B relegated to the arXiv non-workshop version.)
- **Authorship:** assume blind for SD4H; post-acceptance version for ICLR/ICML/NEJM-AI needs a real author list.
