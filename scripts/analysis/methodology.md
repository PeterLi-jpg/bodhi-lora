# BODHI workshop methodology — internalization vs. surface-pattern matching

This note documents the analyses we run on top of the headline 5-seed
multi-config eval, framing what each one rules out and what it doesn't.
The audience is a workshop reviewer asking the natural question: *"are
the LoRA's epistemic-virtue gains genuine internalization, or did the
model learn to recite hedging phrases on medical-domain triggers?"*

## Primary discriminator: 4-config ablation per seed

Every seed produces four eval files in `eval/seed_<N>/`:

| Config | Inference-time setup |
|---|---|
| `base_no_wrapper.json` | Off-the-shelf gemma3-4b, plain decoding |
| `base_bodhi.json` | Same model + BODHI inference-time wrapper |
| `lora_no_wrapper.json` | LoRA-tuned model, plain decoding |
| `lora_bodhi.json` | LoRA + wrapper |

The decisive contrast is **`lora_no_wrapper` − `base_no_wrapper`**: gains
that survive without the BODHI scaffolding at inference time are evidence
that training shifted the model's *default* behavior. If the gain only
materializes in `lora_bodhi`, the LoRA absorbed nothing the wrapper isn't
already providing at runtime — i.e. scaffolding-dependent, not
internalized.

A secondary contrast — `lora_no_wrapper` vs. `base_bodhi` — tests whether
LoRA *replaced* the wrapper's effect (LoRA-only matches or beats
wrapper-only).

These contrasts are reported on:

- HealthBench rubric correctness (Stage 4 grader scores).
- BODHI epistemic virtues — uncertainty acknowledgment, context seeking,
  red-flag identification, scope bounding, specificity, hedging
  (Stage 5 epistemic grader, `eval_epistemic.py`).

## Limitations the 4-config alone doesn't address

1. **Surface lexical mimicry on the rubric grader.** The grader could be
   keying on hedging phrases ("I'm not sure", "you should consult a
   professional") rather than calibrated uncertainty. The grader prompt
   includes explicit instructions to penalize hollow hedging that doesn't
   track question difficulty (see `eval_epistemic.py:58-150`), but this
   is best-effort; cross-grader (#3) and calibration (#2) below are the
   complementary defences.

2. **Calibration check (`scripts/analysis/calibration.py`).** Computes
   Spearman rho per config between the model's per-prompt
   `geomean_token_prob` and the grader's overall_score. Genuine humility
   should produce *positive* rho — more confident on questions it gets
   right. We report
   `internalization_calibration_delta = rho(lora_no_wrapper) − rho(base_no_wrapper)`;
   positive = LoRA training improved fluency-level calibration without
   inference-time scaffolding. **Caveat already documented in
   `eval_healthbench.py`:** `geomean_token_prob` is token fluency, not
   clinical-correctness probability. We report the delta only and do not
   over-claim absolute calibration numbers.

3. **Cross-grader bias control
   (`scripts/analysis/cross_grader_post_hoc.py`).** Re-grades the same
   generated responses with a second grader (default
   `Qwen/Qwen2.5-14B-Instruct`, different family from the primary
   Llama-3.1-8B). Reports per-config Spearman rho between the two
   graders' scores. Low cross-grader correlation indicates the rubric
   isn't reliable; the workshop appendix reports the *worst-case* rho
   across the 4 configs as the bound. Wired via
   `eval_healthbench.py`'s `--secondary-grader-model` so the primary
   numbers stay comparable across seeds.

4. **Theme-stratified scores
   (`scripts/analysis/theme_stratified.py`).** From issue #60: rubric
   themes (`emergency_referrals`, `hedging`, `context_seeking`, etc.)
   that appear disproportionately in training would let surface
   pattern-matching look like global gain. The launcher already runs
   `check_dataset_overlap.py --tag-overlap` per seed and writes the
   tag-frequency table to `pipeline.log`; the analysis script compares
   per-tag eval scores across the 4 configs and surfaces tags where
   `lora_no_wrapper − base_no_wrapper` is much larger than the global
   delta. Such tags are flagged for narrower interpretation.

5. **Out-of-distribution probe
   (`scripts/analysis/ood_probe.py`).** ~30 prompts across legal,
   ethical, and predictive categories where epistemic humility is also
   appropriate but the surface domain differs from medical. Run the same
   epistemic grader on `base_no_wrapper` vs `lora_no_wrapper` on these
   prompts. Generalized internalization → LoRA gains carry over.
   Domain-specific surface mimicry → LoRA looks like base on OOD. Set
   size is intentionally appendix-grade; larger / category-balanced OOD
   sets are future work.

## Contamination protocol (pre-existing in the codebase)

- Stage 1 trace generation excludes **all 1000 HealthBench Hard prompts**
  via `--exclude-ids data/raw/healthbench_hard.jsonl
  data/raw/hard_200_sample_ids.json` (issue #60, Path 1 from PR #124's
  thread). Training pool = HealthBench Full \ Hard = 4000 prompts.
- Each seed's Stage 4 evaluation draws a per-seed random 200-prompt
  subset of the 1000 Hard via `make_bootstrap_eval_ids.py --seed ${SEED}`
  with deterministic `Random(seed)` sampling.
- Per-seed `check_dataset_overlap.py` preflight gate (PR #124) aborts
  the run if any of the 1000 Hard prompts leaked into a seed's
  `train.jsonl`.
- Each seed runs its own Stage 2 filter pass (`filter_traces.py --seed
  ${SEED}`) against the deterministically-shared `raw_traces.jsonl`,
  producing a different filtered SFT subset per seed. Stage 3 LoRA
  training is per-seed (own init, own optimizer noise, own shuffle).
  Per-seed checkpoints land at `checkpoints/seed_<N>/best/`.

## What the multi-seed CIs estimate (and don't)

Bootstrap CIs across the 5 seeds estimate variance from:

- Stage 2 filter shuffle (different SFT subsets per seed)
- LoRA init RNG, optimizer noise, training-shuffle order
- Per-seed bootstrap draw of the 200/1000 Hard eval subset

They do **not** estimate variance from the Stage 1 BODHI rollouts
themselves (greedy decoding → deterministic). Running Stage 1 five times
would produce five identical files; the leader/follower share is a
compute optimization with zero methodological cost.

## Reproducibility

All analysis scripts are deterministic given the eval JSON inputs and
their own `--seed` arguments. The OOD probe set is committed at
`data/raw/ood_humility_probe.jsonl` (materialised by
`scripts/analysis/ood_probe.py`) so reviewers can reproduce the OOD eval.

Headline tables / figures the workshop submission cites:

- Stage 4 mean ± seed-bootstrap-std per config × 5 seeds.
- Stage 5 per-virtue mean per config × 5 seeds.
- Calibration delta per seed (and across-seeds mean ± std).
- Cross-grader Spearman rho per config (worst-case bound across 4 configs).
- Theme-stratified delta table (with imbalance flags).
- OOD probe per-virtue base-vs-lora deltas.
