We thank the reviewer for detailed comments and suggestions. We are glad the reviewer finds the use
case meaningful for high-stakes clinical AI, considers a system for evaluating epistemic virtues
valuable, and judges the significance and originality of the work highly. Two of the reviewer's
concerns specified concrete tests, and we ran both on the submitted evaluation data.

**Is the model learning calibration, or surface template behavior?**

This is the right test to ask for, and we implemented it exactly as the reviewer proposed. To avoid
circularity we labelled prompts using signals independent of the grader's own context-seeking score:
HealthBench's `context_seeking` theme tag, and separately whether a prompt's expert rubric rewards
asking for clarification. For each condition we computed a discrimination index, defined as
P(ask | information missing) minus P(ask | self-contained):

| Condition | Discrimination | Bootstrap 95% CI |
|---|---|---|
| Base | +2.1pp | contains zero |
| Wrapper (CoT) | +11.0pp | excludes zero |
| LoRA | +10.5pp | excludes zero |
| LoRA+CoT | −3.5pp | contains zero |

On the theme-only labelling the adapter reaches +13.2pp, CI [+4.1, +21.7]. The base model does not
reliably discriminate; the wrapper and the adapter both do. The adapter therefore does not ask more
indiscriminately, it acquires targeting the base model lacks, which is the opposite of what surface
mimicry predicts (either the base model's non-discrimination carrying forward, or both groups rising
uniformly). The collapse under LoRA+CoT matches the interference finding below.

A second line of evidence comes from benchmarks that withhold different amounts of information: on
ChatDoctor, where real patient messages routinely omit age, duration and medications, active inquiry
rises to 86.4% (Med42-8B); on the better-specified MedQuAD questions the same recipe yields only
6.5% to 26.5%. The behavior tracks whether asking is warranted.

**The CoT/LoRA competition claim is speculative**

Admittedly our wording asserted a mechanism we had not demonstrated. We tested the reviewer's
alternative (truncation or attention dilution from long outputs) directly against ours, on the
submitted evaluation outputs. Under LoRA+CoT, 54.6% of responses leak the protocol's internal Pass-1
analysis format into the patient-facing answer, versus under 3% in all other conditions. Red-flag
identification collapses only within that leaked subset (1.33); non-leaked responses hold at 1.73,
which is above LoRA alone (1.67). Crucially, dilution predicts red-flag falling with length, but
within non-leaked responses it *rises* across length quartiles, from 1.44 to 1.87. Leaked responses
are indeed longer, but their low score is attributable to emitting "RED FLAGS: None" in analysis
format rather than to length.

We therefore retain the competition interpretation but state it precisely: the two conditioning
sources compete for control of the output format, and the failure is format leakage rather than
capacity exhaustion. We are grateful for the push, since the original claim exceeded our evidence.
We also adopt the reviewer's suggestion to surface red flags early in the protocol, before the
extended reasoning.

**Why the sample sizes differ (200 vs 191 vs 192)**

The two-pass CoT generation exceeds the 4,096-token context window on ~4-5% of prompts, so those
conditions produce fewer completions; LoRA uses a single pass and has a 100% response rate. All
comparisons use available completions (n ≈ 985-998 pooled across 5 seeds). We add this to the Table 1
caption rather than leaving it in the body text.

**Why Llama-3.1 rather than a medical judge**

We agree this is a limitation and have not yet run the alternative. The original choice was
structural: the evaluator must sit in a different model family from the Qwen-14B filter, since that
separation is what prevents filter-grader circularity in a self-distillation pipeline. Meditron-3,
Med42-v2 and the Aloe family are all Llama-derived, so promoting one to evaluator would place filter
and grader in adjacent families and weaken exactly that property. We therefore plan to add a
clinical judge as an additional robustness panel reported alongside the primary grader rather than
replacing it, so both the cross-family guarantee and the clinical-specificity check are available.

**The dual-axis figure is misleading**

We agree. Figure 2 is split into Panel A (0-2 dimensions) and Panel B (percentage rates), each with
its own labelled axis.

**Additional runs, and one result bearing on the mimicry question**

Because generality was raised across reviews, we also re-ran the pipeline unchanged on two further
model families and two further benchmarks (5 seeds each). One result speaks directly to the reviewer's
concern: on MedQuAD the CoT wrapper *degrades* scope bounding (1.85 to 1.61) and hedging (1.45 to
1.40) while the distilled adapter *improves* both (1.89 and 1.78). The student does not copy the
teacher indiscriminately; the quality-filtering step removes the teacher's failures before they
reach the weights. We also found a scope condition: on BioMistral-7B the recipe produces no effect,
and the diagnostic is that the teacher fails there too (wrapper 14.5%), so there is nothing to
distill.

We will incorporate all feedback into the paper.
