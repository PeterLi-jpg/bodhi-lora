# Response to Reviewer LxMF

We thank the reviewer for two methodological critiques that were the most useful in the review set,
because each specified a concrete test. We ran both on the submitted evaluation data, and one
required us to state our original claim more precisely.

### Q2: Is the model learning calibration, or surface template behavior?

The reviewer proposed isolating samples that genuinely lack information from those answerable as
posed, then checking whether inquiry rises more in the former. We implemented exactly this on the
submitted runs. To avoid circularity we labelled prompts using signals independent of the grader's
own context-seeking score: HealthBench's `context_seeking` theme tag, and separately whether a
prompt's expert rubric rewards asking for clarification. For each condition we computed a
discrimination index, defined as P(ask | information missing) minus P(ask | self-contained).

| Condition | Discrimination index | Bootstrap 95% CI |
|---|---|---|
| Base | +2.1pp | contains zero |
| CoT wrapper | +11.0pp | excludes zero |
| **LoRA adapter** | **+10.5pp** | **excludes zero** |
| LoRA + CoT | negative 3.5pp | contains zero |

On the theme-only labelling the adapter reaches +13.2pp, CI [+4.1, +21.7]. The adapter does not ask
more indiscriminately; it acquires targeting the base model lacks. Surface mimicry predicts the
opposite: either the base model's non-discrimination carrying forward, or both groups rising
uniformly. The collapse under LoRA+CoT matches the interference result below.

### Q4: The CoT/LoRA competition claim is speculative

The reviewer is right that our wording asserted a mechanism we had not shown, and proposed a
specific alternative: truncation or attention dilution from long outputs. We tested that alternative
directly against ours, on the submitted evaluation outputs.

- **Format leakage:** under LoRA+CoT, 54.6% of responses leak the protocol's internal Pass-1
  analysis format into the patient-facing answer, versus under 3% in every other condition.
- **The drop is confined to leaked responses:** red-flag identification is 1.33 in the leaked
  subset, but 1.73 in non-leaked responses, which is above LoRA alone (1.67).
- **Dilution predicts the opposite of what we see:** within non-leaked responses, red-flag *rises*
  across length quartiles, from 1.44 in the shortest to 1.87 in the longest.

Leaked responses are longer, but their low score is attributable to emitting "RED FLAGS: None" in
analysis format rather than to length. We therefore keep the competition interpretation but state it
precisely: the two conditioning sources compete for control of the output format, and the failure is
format leakage rather than capacity exhaustion. We are grateful for the push, since the original
claim was broader than our evidence. We also adopt the reviewer's suggestion to surface red flags
early in the protocol, before the extended reasoning.

### Q1: Why do sample sizes differ (200 vs 191 vs 192)?

The CoT protocol's two-pass generation exceeds the 4,096-token context window on roughly 4 to 5% of
prompts, so those conditions produce fewer completions. The LoRA condition uses a single pass and has
a 100% response rate. All comparisons use available completions (n approximately 985 to 998 pooled
across five seeds). We add this as a table footnote rather than leaving it in the body text.

### Q3: Why Llama-3.1 rather than a medical judge?

We agree this is a limitation and have not yet run the alternative. The original choice was
structural rather than incidental:

- The evaluator must sit in a different model family from the Qwen-14B filter, since that separation
  is what prevents filter-grader circularity in a self-distillation pipeline.
- Meditron-3, Med42-v2 and the Aloe family are all Llama-derived, so promoting one to evaluator
  would place filter and grader in adjacent families and weaken exactly that property.

Our plan is to add a clinical judge as an additional robustness panel reported alongside the primary
grader rather than replacing it, so both the cross-family guarantee and the clinical-specificity
check are available.

### Clarity: the dual-axis figure

Agreed. Figure 2 is split into Panel A (0 to 2 dimensions) and Panel B (percentage rates), each with
its own labelled axis.

### Supporting evidence from the discussion period

Because generality was raised across reviews, we also ran the pipeline unchanged on two further
model families and two further benchmarks, five seeds each. Two results bear on the mimicry question
specifically:

- **The student does not copy the teacher's failures.** On MedQuAD the CoT wrapper *degrades* scope
  bounding (1.85 to 1.61) and hedging (1.45 to 1.40), while the adapter *improves* both (1.89 and
  1.78). The filtering step removes the teacher's failures before they reach the weights.
- **A boundary condition.** On BioMistral-7B the recipe produces no effect, and the diagnostic is
  that the teacher fails there too (wrapper 14.5%). Where the protocol cannot elicit the behavior,
  there is nothing to distill.
