# Response to Reviewer LxMF

We thank the reviewer for two methodological critiques that were, in our view, the sharpest in the
review set: whether the model learns calibration or surface template behavior, and whether our
interference explanation is supported or merely asserted. Both specified a concrete test. We ran
both, and we report the results below — including one that required us to state our original claim
more precisely.

## W2: Is the model learning calibration, or surface template behavior?

The reviewer proposed exactly the right experiment: isolate samples that genuinely have missing
information from those that can be answered as posed, and check whether inquiry rises more in the
former group than the latter. We implemented it.

To avoid circularity we labelled prompts using signals independent of the grader's own
context-seeking score: HealthBench's own `context_seeking` theme tag, and separately whether a
prompt's expert rubric rewards asking for clarification. For each condition we then computed a
discrimination index, defined as P(ask | information genuinely missing) − P(ask | self-contained).

The base model does **not** reliably discriminate: +2.1pp, with a bootstrap 95% confidence interval
that contains zero. The CoT wrapper does: +11.0pp, CI excluding zero. The LoRA adapter does:
+10.5pp, CI excluding zero, and on the theme-only labelling it reaches +13.2pp with CI
[+4.1, +21.7]. The adapter therefore does not ask more indiscriminately — it acquires the targeting
that the teacher has and the base model lacks. Template mimicry predicts the opposite: either the
base model's non-discrimination carried forward, or both groups rising uniformly.

Two further pieces of evidence point the same way. First, under LoRA+CoT the discrimination index
collapses to −3.5pp, consistent with the interference result below: when the format breaks down, the
targeting goes with it. Second, we extended the evaluation to new benchmarks that differ in how much
information they withhold. On ChatDoctor, which consists of unedited questions real patients asked
physicians online and which routinely omit age, duration and medications, active inquiry rises from
2.9% to 86.4% with Med42-8B and from 7.8% to 58.6% with Mistral-Small-24B. On MedQuAD, whose
questions are better specified, the same recipe produces a much smaller rise (6.5% to 26.5%). The
behavior tracks whether asking is actually warranted, across benchmarks as well as within one.

## W4: The CoT/LoRA competition claim is speculative

The reviewer is right that our original wording asserted a mechanism we had not demonstrated, and
proposed a specific alternative: context-window truncation or attention dilution from long outputs.
We tested that alternative directly against our proposed mechanism, using the existing evaluation
outputs.

Under LoRA+CoT, 54.6% of responses leak the CoT protocol's internal Pass-1 analysis format into the
patient-facing answer, compared with under 3% in every other condition. Red-flag identification
collapses specifically within that leaked subset (1.33), while the non-leaked responses hold at
1.73 — which is above LoRA alone (1.67). If attention dilution from long outputs were responsible,
red-flag should fall as length grows. Within the non-leaked responses it does the opposite, rising
monotonically across length quartiles from 1.44 in the shortest to 1.87 in the longest. The leaked
responses are indeed longer, but their low red-flag score is attributable to their emitting "RED
FLAGS: None" in analysis format rather than to their length.

We therefore retain the competition interpretation but state it more precisely in the revision: the
two conditioning sources compete for control of the output format, and the observed failure is
format leakage rather than capacity exhaustion or attention dilution. We are grateful for the push,
because the original claim was broader than the evidence supported. We also adopt the reviewer's
constructive suggestion and will modify the protocol so that red flags are surfaced early, before
the extended reasoning, so safety-critical content cannot be displaced by a long analysis pass.

## Q1: Why do the sample sizes differ (200 vs 191 vs 192)?

The differences arise entirely from the CoT protocol's two-pass generation exceeding the
4,096-token context window on roughly 4–5% of prompts, so those two conditions produce fewer
completions. The LoRA condition requires a single forward pass and has a 100% response rate. All
statistical comparisons use the available completions (n ≈ 985–998 after pooling five seeds). We
will state this in a table footnote rather than leaving it in the body text where it is easy to
miss.

## Q3: Why Llama-3.1 as the judge rather than a medical model?

We agree this is a real limitation, and we have not yet run the alternative, so we will not imply
otherwise. Our reason for the original choice was structural: the evaluator must sit in a different
model family from the Qwen-14B quality filter, because that separation is what prevents
filter-grader circularity in a self-distillation pipeline. We note that the clinically tuned models
the reviewer names — Meditron-3, Med42-v2, and the Aloe family — are Llama-derived, so substituting
one for the evaluator would place the filter and grader in adjacent families and weaken the property
the asymmetric design exists to guarantee.

Our plan is therefore to add a clinical judge as an additional robustness panel reported alongside
the primary grader, rather than replacing it, so that both the cross-family guarantee and the
clinical-specificity check are available to the reader. We list this as outstanding work.

## Clarity: the dual-axis figure is misleading

Agreed. Figure 2 is split into two panels: Panel A for the 0–2 scale dimensions (uncertainty
acknowledgment, context-seeking, red-flag identification, scope bounding, hedging quality,
specificity) and Panel B for the percentage rates (active inquiry, red-flag rate, scope-bounded
rate, blanket-disclaimer rate). Each panel has its own labelled axis.

## Additional evidence added during the discussion period

Because the generality of the finding was raised across reviews, we also ran the full pipeline —
unchanged, with the paper's hyperparameters fixed — on three model families and three benchmarks
with five seeds per cell. Beyond the numbers cited above, one result bears directly on the mimicry
question. On MedQuAD, the CoT wrapper *degrades* two dimensions relative to the base model: scope
bounding falls from 1.85 to 1.61 and hedging quality from 1.45 to 1.40. The distilled adapter
*improves* both (1.89 and 1.78). The student is therefore not reproducing the teacher
indiscriminately; the quality-filtering step removes the teacher's failures before they reach the
weights. We think this is the clearest single piece of evidence against the surface-mimicry reading.

We also report a boundary condition: on BioMistral-7B the recipe produces no effect (active inquiry
11.7% to 10.1%), and the diagnostic is that the teacher fails there as well (wrapper 14.5%). Where
the protocol cannot elicit the behavior, there is nothing to distill.
