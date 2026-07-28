Thank you for this thorough review and for the specific suggestions. We are glad you find the use case
meaningful for high-stakes clinical AI, consider a system for evaluating epistemic virtues valuable,
and judge the significance and originality highly. Two of your concerns named concrete tests, and we ran
both on the submitted evaluation data.

In brief: on the discrimination test you proposed, the adapter distinguishes information-withholding
prompts from self-contained ones (+10.5pp, CI excluding zero) while the base model does not (+2.1pp, CI
including zero); and the interference test refutes our own original wording, since the red-flag drop is
confined to responses that leak the protocol's internal format rather than to long responses.

**1. Why the sample sizes differ (200 vs 191 vs 192)**

The two-pass CoT generation exceeds the 4,096-token context window on ~4–5% of prompts, so those
conditions return fewer completions. LoRA uses a single forward pass and has a 100% response rate. All
comparisons use the available completions, $n \approx 985$–$998$ per condition pooled across 5 seeds. If
accepted we will state this in the Table 1 caption rather than leave it in the body text, where it is
easy to miss.

**2. Is the model learning calibration, or surface template behavior?**

We ran the test you proposed: separate prompts that genuinely withhold information from those
answerable as posed, and check whether inquiry rises more in the former.

The labels come from signals the grader never receives. The epistemic grader is shown only the prompt
and the response, and the asking outcome is its active-inquiry judgment; the split instead comes from
HealthBench's `context_seeking` theme tag and, separately, from whether a prompt's expert rubric
rewards asking for clarification. Label and outcome therefore have different sources. The index is
$P(\text{ask} \mid \text{withholding}) - P(\text{ask} \mid \text{self-contained})$ over 667 distinct
prompts (508 withholding, 158 self-contained), bootstrap 95% CIs, 5,000 resamples.

**Table R1.** Discrimination index by condition.

| Condition | Discrimination | 95% CI |
|---|---|---|
| Base | +2.1pp | [−5.0, +8.7] |
| Wrapper (CoT) | +11.0pp | [+2.1, +19.9] |
| LoRA | +10.5pp | [+1.7, +18.9] |
| LoRA+CoT | −3.5pp | [−10.1, +3.2] |

The base model does not reliably discriminate; the wrapper and the adapter both do. So the adapter does
not simply ask more often, it acquires targeting the base model lacks, which is the opposite of what
surface mimicry predicts: mimicry would either carry the base model's non-discrimination forward or
raise both groups uniformly.

Two qualifications are in order. The rubric-based and theme-only labellings disagree substantially about
which prompts withhold information (508 versus 127 prompts in that group), yet both give the same
ordering, and the adapter reaches +13.2pp, CI [+4.1, +21.7], under the theme-only split; that agreement
across near-disjoint labellings is the strongest robustness check available at this sample size. Separately, the
LoRA${-}$Base difference is +8.4pp with CI [−1.2, +17.9], which includes zero, so the *difference
between conditions* is suggestive rather than established. The within-condition results carry the
weight.

Under LoRA+CoT the discrimination is no longer detectable, which is consistent with the interference
finding below.

**3. Why Llama-3.1 rather than a medical judge**

We agree this is a limitation and have not yet run the alternative. The original choice was structural
rather than incidental: the evaluator must sit in a different model family from the Qwen-14B filter,
because that separation is what prevents filter-grader circularity in a self-distillation pipeline.
Meditron-3, Med42-v2 and the Aloe family are all Llama-derived, so promoting one to evaluator would put
filter and grader in adjacent families and weaken exactly the property the asymmetric design buys. If
accepted we will add a clinical judge as an additional robustness panel reported alongside the primary
grader rather than replacing it.

Your suggestion did shape the new runs: Med42-8B, from the Med42-v2 family you named, is now one of the
base models we adapt (Table R2), which puts a clinically tuned Llama-family model inside the study.

We should also report the current state of the physician validation, since it bears on the same
concern. It remains partial: one of three raters has returned grades, giving $\kappa = 0.35$ against the
Llama-3.1-8B grader, below our pre-registered target of 0.6. We report that as a limitation on the
aggregate-quality claims rather than presenting LLM grading as settled.

**4. The CoT/LoRA competition claim is speculative**

You are right, and our wording asserted a mechanism we had not demonstrated. We tested your
alternative, truncation or attention dilution from long outputs, directly against ours.

- **Format leakage is pervasive under LoRA+CoT:** 54.6% of responses leak the protocol's internal
  Pass-1 analysis format into the patient-facing answer, against 0.2–3.0% in every other condition.
- **The drop is confined to leaked responses:** red-flag identification is 1.33 in the leaked subset
  ($n{=}507$) but 1.73 in the non-leaked subset ($n{=}421$), which is *above* LoRA alone (1.67).
- **Dilution predicts the opposite of what we see:** within non-leaked responses, red-flag
  identification *rises* across length quartiles, from 1.44 (median 1,508 chars) to 1.87 (median
  5,395 chars).

Truncation does not explain it either: prompts whose two-pass generation exceeds the window return no
response at all, so they are absent from the sample rather than present and scored low, which is the
~5% attrition noted above. Leaked responses are longer (median 8,564 versus 3,457 characters), but their
low score is attributable to emitting "RED FLAGS: None" in analysis format rather than to length, since
length correlates positively with the score once leakage is excluded. We note that the non-leaked subset is
identified after the fact by a format property rather than by its score, so we cannot fully exclude
leakage correlating with prompt difficulty.

We therefore keep the competition reading but state it precisely: the two conditioning sources compete
for control of the output *format*, and the failure is format leakage, not capacity exhaustion or
attention dilution. Where leakage does not occur, stacking the protocol on the adapter is not harmful.
If accepted we will also adopt your suggestion to surface red flags early in the protocol, before the
extended reasoning. The leakage diagnosis predicts that this should recover most of the gap, which makes
your suggestion a test of the mechanism as well as a fix.

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

Two of these results bear on your concern directly rather than restating the original finding.

First, the effect scales with how much a benchmark withholds. Holding the base model at
Mistral-Small-24B, active inquiry rises 7.8% to 58.6% on ChatDoctor, where real patient messages
routinely omit age, duration and medications, but only 6.5% to 26.5% on the better-specified MedQuAD.
The gap is between benchmarks, not between models.

Second, the student does not copy the teacher indiscriminately. On MedQuAD the CoT wrapper *degrades*
scope bounding (1.85 → 1.61) and hedging quality (1.45 → 1.40) while the distilled adapter *improves*
both (1.89 and 1.78). The quality-filtering step removes the teacher's failures before they reach the
weights, which is hard to reconcile with pure imitation.

We also identified a precondition: on BioMistral-7B there is no effect, and the teacher fails there too
(wrapper 14.5%; only 20% of traces cleared the filter against 62% for Mistral-24B), so teacher
incapacity and the smaller surviving training set are not fully separable.

For completeness, including the results that do not favour the method: aggregate rubric quality
decreases modestly on the new benchmarks, most on MedQuAD (0.672 → 0.586). We read that partly as a rubric property, since it rewards
agreement with a reference answer and a response that asks a question instead of answering scores lower
by construction. Consistent with that, the inference-time wrapper drops further than the adapter (0.542
against 0.586) despite changing no weights, and the two stacked drop furthest (0.505). That is the
measurement problem the paper is about, and why we report the decomposition alongside the aggregate.

Remaining open, and we do not claim otherwise: a head-to-head comparison against inference-time
calibration methods, and completion of the three-physician validation.

Thank you again for proposing two concrete tests; one of them made us restate a claim we had
overreached on. Please do let us know if any questions remain; we would be
glad to run further analyses while the discussion period is open.
