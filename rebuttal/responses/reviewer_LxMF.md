# Response to Reviewer LxMF

We thank the reviewer for two methodological critiques that were the most useful in the review set,
because each specified a concrete test. We ran both on the submitted evaluation data, and one of
them required us to state our original claim more precisely.

**W2: Is the model learning calibration or surface template behavior?** The reviewer proposed
isolating samples that genuinely lack information from those answerable as posed, then checking
whether inquiry rises more in the former. We implemented this on the submitted runs. To avoid
circularity we labelled prompts using signals independent of the grader's own context-seeking score:
HealthBench's `context_seeking` theme tag, and separately whether a prompt's expert rubric rewards
asking for clarification. For each condition we computed a discrimination index, defined as
P(ask | information missing) minus P(ask | self-contained).

The base model does not reliably discriminate: +2.1pp, with a bootstrap 95% CI containing zero. The
CoT wrapper does: +11.0pp, CI excluding zero. The LoRA adapter does: +10.5pp, CI excluding zero, and
+13.2pp with CI [+4.1, +21.7] on the theme-only labelling. The adapter therefore does not ask more
indiscriminately; it acquires targeting that the base model lacks. Surface mimicry would predict
either the base model's non-discrimination carrying forward, or both groups rising uniformly.
Consistent with this, under LoRA+CoT the discrimination index collapses to negative 3.5pp, matching
the interference result below.

**W4: The CoT/LoRA competition claim is speculative.** The reviewer is right that our wording
asserted a mechanism we had not shown, and proposed a specific alternative: truncation or attention
dilution from long outputs. We tested that alternative directly on the submitted evaluation outputs.

Under LoRA+CoT, 54.6% of responses leak the protocol's internal Pass-1 analysis format into the
patient-facing answer, versus under 3% in every other condition. Red-flag identification collapses
specifically within that leaked subset (1.33), while non-leaked responses hold at 1.73, above LoRA
alone (1.67). If attention dilution were responsible, red-flag should fall as length grows; within
non-leaked responses it does the opposite, rising across length quartiles from 1.44 to 1.87. Leaked
responses are longer, but their low score is attributable to emitting "RED FLAGS: None" in analysis
format rather than to length.

We therefore keep the competition interpretation but state it precisely: the two conditioning
sources compete for control of the output format, and the failure is format leakage rather than
capacity exhaustion. We are grateful for the push, because the original claim was broader than our
evidence. We also adopt the reviewer's suggestion to surface red flags early in the protocol, before
the extended reasoning.

**Q1: Why do sample sizes differ (200 vs 191 vs 192)?** The CoT protocol's two-pass generation
exceeds the 4,096-token context window on roughly 4 to 5% of prompts, so those conditions produce
fewer completions. The LoRA condition uses a single pass and has a 100% response rate. All
comparisons use available completions (n approximately 985 to 998 pooled across five seeds). We add
this as a table footnote.

**Q3: Why Llama-3.1 rather than a medical judge?** We agree this is a limitation and we have not yet
run the alternative. The original choice was structural: the evaluator must sit in a different model
family from the Qwen-14B filter, since that separation is what prevents filter-grader circularity.
The clinically tuned models named (Meditron-3, Med42-v2, Aloe) are Llama-derived, so promoting one
to evaluator would place filter and grader in adjacent families and weaken the property the
asymmetric design exists to guarantee. Our plan is to add a clinical judge as an additional
robustness panel alongside the primary grader rather than replacing it.

**Clarity: the dual-axis figure.** Agreed. Figure 2 is split into Panel A for the 0 to 2 dimensions
and Panel B for percentage rates, each with its own labelled axis.

**Supporting evidence from the discussion period.** Because generality was raised across reviews, we
also ran the same pipeline unchanged on two further model families and two further benchmarks, five
seeds each. One result bears on the mimicry question specifically. On MedQuAD the CoT wrapper
degrades two dimensions relative to base (scope bounding 1.85 to 1.61, hedging 1.45 to 1.40) while
the distilled adapter improves both (1.89 and 1.78). The student is not copying the teacher
indiscriminately; the filtering step removes the teacher's failures before they reach the weights.
We also observed a boundary condition: on BioMistral-7B the recipe produces no effect, and the
diagnostic is that the teacher fails there too (wrapper 14.5%), so there is nothing to distill.
