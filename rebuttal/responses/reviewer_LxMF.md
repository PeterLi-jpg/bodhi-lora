Thank you for your thorough review and helpful suggestions. We are glad you find the use case
meaningful for high-stakes clinical AI, consider a system for evaluating epistemic virtues valuable,
and judge the significance and originality of the work highly. Two of your concerns specified
concrete tests; we ran both on the submitted evaluation data and report them below.

**Is the model learning calibration, or surface template behavior?**

Short answer: we ran exactly the test you proposed, and the adapter discriminates while the base
model does not.

To avoid circularity we labelled prompts using signals independent of the grader's own
context-seeking score: HealthBench's `context_seeking` theme tag, and separately whether a prompt's
expert rubric rewards asking for clarification. For each condition we computed a discrimination
index, defined as $P(\text{ask} \mid \text{info missing}) - P(\text{ask} \mid \text{self-contained})$.

**Table R1.** Discrimination index by condition, pooled over 5 seeds, bootstrap 95% CIs (5,000
resamples).

| Condition | Discrimination | 95% CI |
|---|---|---|
| Base | +2.1pp | contains zero |
| Wrapper (CoT) | +11.0pp | excludes zero |
| LoRA | +10.5pp | excludes zero |
| LoRA+CoT | −3.5pp | contains zero |

On the theme-only labelling the adapter reaches +13.2pp, CI [+4.1, +21.7]. The base model does not
reliably discriminate; the wrapper and the adapter both do. The adapter therefore does not ask more
indiscriminately, it acquires targeting the base model lacks, which is the opposite of what surface
mimicry predicts (either the base model's non-discrimination carrying forward, or both groups rising
uniformly). The collapse under LoRA+CoT is consistent with the interference finding below.

**Corroborating evidence across benchmarks:** the effect scales with how much a benchmark withholds.
Holding the base model fixed at Mistral-Small-24B, on ChatDoctor, where real patient messages
routinely omit age, duration and medications, active inquiry rises from 7.8% to 58.6%; on the
better-specified MedQuAD questions the same recipe yields only 6.5% to 26.5%. The gap is between
benchmarks rather than between models, and the behavior tracks whether asking is warranted.

**The CoT/LoRA competition claim is speculative**

Admittedly our wording asserted a mechanism we had not demonstrated. We tested your alternative
(truncation or attention dilution from long outputs) directly against ours, on the submitted
evaluation outputs.

- **Format leakage is pervasive under LoRA+CoT:** 54.6% of responses leak the protocol's internal
  Pass-1 analysis format into the patient-facing answer, versus under 3% in all other conditions.
- **The drop is confined to leaked responses:** red-flag identification is 1.33 in the leaked subset
  but 1.73 in non-leaked responses, which is *above* LoRA alone (1.67).
- **Dilution predicts the opposite of what we observe:** within non-leaked responses, red-flag
  identification *rises* across length quartiles, from 1.44 (shortest) to 1.87 (longest).

Leaked responses are indeed longer, but their low score is attributable to emitting "RED FLAGS: None"
in analysis format rather than to length. We therefore retain the competition interpretation but state
it precisely: the two conditioning sources compete for control of the output *format*, and the failure
is format leakage rather than capacity exhaustion or attention dilution. We are grateful for the push,
since the original claim exceeded our evidence. We also adopt your suggestion to surface red flags
early in the protocol, before the extended reasoning, so safety-critical content cannot be displaced.

**Why the sample sizes differ (200 vs 191 vs 192)**

The two-pass CoT generation exceeds the 4,096-token context window on ~4–5% of prompts, so those
conditions produce fewer completions; LoRA uses a single forward pass and has a 100% response rate.
All comparisons use available completions ($n \approx 985$–$998$ pooled across 5 seeds). We add this to the
Table 1 caption rather than leaving it in the body text where it is easy to miss.

**Why Llama-3.1 rather than a medical judge**

We agree this is a limitation and have not yet run the alternative. The original choice was
structural: the evaluator must sit in a different model family from the Qwen-14B filter, since that
separation is what prevents filter-grader circularity in a self-distillation pipeline. Meditron-3,
Med42-v2 and the Aloe family are all Llama-derived, so promoting one to evaluator would place filter
and grader in adjacent families and weaken exactly the property the asymmetric design guarantees. We
therefore plan to add a clinical judge as an additional robustness panel reported alongside the
primary grader rather than replacing it, so both the cross-family guarantee and the
clinical-specificity check are available to the reader.

**The dual-axis figure is misleading**

We agree, and will split Figure 2 into Panel A (0–2 dimensions: uncertainty, context-seeking,
red-flag, scope, hedging, specificity) and Panel B (percentage rates: active inquiry, red-flag,
scope-bounded, blanket disclaimer), each with its own labelled axis. This requires regenerating the
figure, so it will appear in the camera-ready rather than in the current revision.

**Additional runs, and one result bearing directly on mimicry**

Because generality was raised across reviews, we also re-ran the pipeline unchanged on two further
model families and two further benchmarks, 5 seeds each. One result speaks to your concern
specifically: on MedQuAD the CoT wrapper *degrades* scope bounding (1.85 → 1.61) and hedging quality
(1.45 → 1.40) while the distilled adapter *improves* both (1.89 and 1.78). The student does not copy
the teacher indiscriminately; the quality-filtering step removes the teacher's failures before they
reach the weights. We also identified a scope condition: on BioMistral-7B the recipe produces no
effect, and the diagnostic is that the teacher fails there too (wrapper reaches only 14.5% active
inquiry), so there is nothing to distill. Only 20% of its traces cleared the quality filter, against
62% for Mistral-Small-24B, so teacher incapacity and the smaller surviving training set are not fully
separable.

We will incorporate all feedback into the paper.
