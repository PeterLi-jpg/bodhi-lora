Thank you for this thorough review and for the specific suggestions. We are glad you find the use case
meaningful for high-stakes clinical AI, consider a system for evaluating epistemic virtues valuable,
and judge the significance and originality highly. Two of your concerns named concrete tests, and we
ran both on the submitted evaluation data rather than argue about them; the results are below, and one
of them changed our interpretation.

In brief: the discrimination test you proposed separates the adapter from the base model (+10.5pp
against +2.1pp), and the interference test refutes our own original wording, since the red-flag drop
turns out to be confined to responses that leak the protocol's internal format rather than to long
responses.

**1. Why the sample sizes differ (200 vs 191 vs 192)**

The two-pass CoT generation exceeds the 4,096-token context window on ~4–5% of prompts, so those
conditions return fewer completions. LoRA uses a single forward pass and has a 100% response rate. All
comparisons use the available completions, $n \approx 985$–$998$ per condition pooled across 5 seeds. If
accepted we will state this in the Table 1 caption rather than leave it in the body text, where it is
easy to miss.

**2. Is the model learning calibration, or surface template behavior?**

We ran the test you proposed: isolate prompts that genuinely withhold information from those answerable
as posed, and check whether inquiry rises more in the former.

To keep the split independent of the grader we labelled prompts from signals it never receives: the
epistemic grader is shown only the prompt and the response, so we labelled from HealthBench's
`context_seeking` theme tag and, separately, from whether a prompt's expert rubric rewards asking for
clarification. Neither signal reaches the grader, so the label and the measured outcome come from
different sources. For each condition we computed a discrimination index, defined as
$P(\text{ask} \mid \text{info missing}) - P(\text{ask} \mid \text{answerable})$, over 667 evaluated
prompts (508 information-withholding, 158 self-contained), with bootstrap 95% CIs from 5,000 resamples.

**Table R1.** Discrimination index by condition.

| Condition | Discrimination | 95% CI |
|---|---|---|
| Base | +2.1pp | [−5.0, +8.7] |
| Wrapper (CoT) | +11.0pp | [+2.1, +19.9] |
| LoRA | +10.5pp | [+1.7, +18.9] |
| LoRA+CoT | −3.5pp | [−10.1, +3.2] |

The base model does not reliably discriminate; the wrapper and the adapter both do. On the theme-only
labelling the adapter reaches +13.2pp, CI [+4.1, +21.7]. So the adapter does not simply ask more often,
it acquires targeting the base model lacks, which is the opposite of what surface mimicry predicts:
mimicry would either carry the base model's non-discrimination forward or raise both groups uniformly.
We should be straight about one limit: the LoRA${-}$Base difference in discrimination is +8.4pp with CI
[−1.2, +17.9], which includes zero, so the *difference between conditions* is suggestive rather than
established. The within-condition result is what carries weight.

Under LoRA+CoT the discrimination is no longer detectable, which is consistent with the interference
finding below.

**3. Why Llama-3.1 rather than a medical judge**

We agree this is a limitation and have not yet run the alternative. The original choice was structural
rather than incidental: the evaluator must sit in a different model family from the Qwen-14B filter,
because that separation is what prevents filter-grader circularity in a self-distillation pipeline.
Meditron-3, Med42-v2 and the Aloe family are all Llama-derived, so promoting one to evaluator would put
filter and grader in adjacent families and weaken exactly the property the asymmetric design buys. If
accepted we will add a clinical judge as an additional robustness panel reported alongside the primary
grader rather than replacing it, so both the cross-family guarantee and the clinical-specificity check
are visible to the reader.

Your suggestion did shape the new runs in a related way: Med42-8B, one of the models you named, is now
one of the base models we adapt (Table R2), which at least puts a clinically tuned Llama-family model
inside the study.

We should also report where our human check currently stands, since it bears on the same concern. The
physician validation is partial: one of three raters has returned grades, giving $\kappa = 0.35$ against
the Llama-3.1-8B grader, below our pre-registered target of 0.6. We report that as a limitation on the
aggregate-quality claims rather than presenting LLM grading as settled.

**4. The CoT/LoRA competition claim is speculative**

You are right, and our wording asserted a mechanism we had not demonstrated. We tested your
alternative, truncation or attention dilution from long outputs, directly against ours on the submitted
evaluation outputs.

- **Format leakage is pervasive under LoRA+CoT:** 54.6% of responses leak the protocol's internal
  Pass-1 analysis format into the patient-facing answer, against 0.2–3.0% in every other condition.
- **The drop is confined to leaked responses:** red-flag identification is 1.33 in the leaked subset
  ($n{=}507$) but 1.73 in the non-leaked subset ($n{=}421$), which is *above* LoRA alone (1.67).
- **Dilution predicts the opposite of what we see:** within non-leaked responses, red-flag
  identification *rises* across length quartiles, from 1.44 (median 1,508 chars) to 1.87 (median
  5,395 chars).

Truncation does not explain it either: prompts whose two-pass generation exceeds the window return no
response at all and are therefore absent from the sample (928 completions against 998 for the base),
rather than being present and scored low. Leaked responses are indeed longer (median 8,564 versus 3,457
characters), but their low score is attributable to emitting "RED FLAGS: None" in analysis format
rather than to length, since length correlates positively with the score once leakage is excluded. We therefore keep the competition
reading but state it precisely: the two conditioning sources compete for control of the output
*format*, and the failure is format leakage, not capacity exhaustion or attention dilution. Put plainly,
where leakage does not occur, stacking the protocol on the adapter is not harmful at all. We are
grateful for the push, because the original claim exceeded our evidence. If accepted we will also adopt
your suggestion to surface red flags early in the protocol, before the extended reasoning, so
safety-critical content cannot be displaced. The leakage diagnosis predicts that this should recover
most of the gap, which makes your suggestion a test of the mechanism as well as a fix.

**Clarity. The dual-axis figure is misleading**

Agreed. If accepted we will split Figure 2 into Panel A (0–2 dimensions: uncertainty, context-seeking,
red-flag, scope, hedging, specificity) and Panel B (percentage rates: active inquiry, red-flag,
scope-bounded, blanket disclaimer), each with its own labelled axis.

**Additional runs, and one result bearing on mimicry**

Because generality was raised across reviews, we re-ran the pipeline unchanged on two further model
families and two further benchmarks, 5 seeds per cell.

**Table R2.** Active inquiry, Base → LoRA. Row 1 is the submitted result.

| Benchmark | Base model | Active inquiry |
|---|---|---|
| HealthBench | MedGemma-27B (submitted) | 17.5 → 45.6% |
| HealthBench | Mistral-Small-24B | 25.6 → 56.6% |
| HealthBench | Med42-8B | 11.8 → 52.5% |
| ChatDoctor | Mistral-Small-24B | 7.8 → 58.6% |
| ChatDoctor | Med42-8B | 2.9 → 86.4% |
| MedQuAD | Mistral-Small-24B | 6.5 → 26.5% |
| HealthBench | BioMistral-7B | 11.7 → 10.1% (null) |

Two results speak to your concern rather than merely repeating the original one.

First, the effect scales with how much a benchmark withholds. Holding the base model at
Mistral-Small-24B, active inquiry rises 7.8% to 58.6% on ChatDoctor, where real patient messages
routinely omit age, duration and medications, but only 6.5% to 26.5% on the better-specified MedQuAD.
The gap is between benchmarks, not between models.

Second, the student does not copy the teacher indiscriminately. On MedQuAD the CoT wrapper *degrades*
scope bounding (1.85 → 1.61) and hedging quality (1.45 → 1.40) while the distilled adapter *improves*
both (1.89 and 1.78). The quality-filtering step removes the teacher's failures before they reach the
weights, which is hard to reconcile with pure imitation.

We also found a precondition: on BioMistral-7B there is no effect and the teacher fails there too
(wrapper 14.5%; only 20% of traces cleared the filter against 62% for Mistral-24B), so teacher
incapacity and the smaller surviving training set are not fully separable.

Reporting the unfavourable side too: on the new benchmarks aggregate rubric quality decreases modestly,
largest on MedQuAD (0.672 → 0.586). We read that partly as a rubric property, since it rewards
agreement with a reference answer and a response that asks a question instead of answering scores lower
by construction. Consistent with that, the inference-time wrapper drops further than the adapter (0.542
against 0.586) despite changing no weights, and the two stacked drop furthest (0.505). That is the
measurement problem the paper is about, and why we report the decomposition alongside the aggregate.

Still outstanding, and we do not claim otherwise: a head-to-head against inference-time calibration
methods, the completed three-physician validation, and a second CoT protocol.

Thank you again for proposing two concrete tests rather than only raising the concerns; one of them
made us restate a claim we had overreached on. Please do let us know if any questions remain; we would
be glad to run further analyses while the discussion period is open.
