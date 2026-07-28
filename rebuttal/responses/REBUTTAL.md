# Rebuttal — Submission 32818
## Ask Before You Answer: Distilling Clinical Epistemic Calibration into Model Weights

---

## Common response to all reviewers and the Area Chair

We thank the reviewers for a set of critiques that converged on one central issue: the paper
claimed a general framework but validated a single model on a single benchmark. Rather than
answer this with promises of future work, we ran the experiments during the discussion period.

We have now applied the identical pipeline — unchanged in its four steps of generate, grade,
filter, distill — to **three model families and three benchmarks, with five independent seeds per
cell**. The new base models are Mistral-Small-24B-Instruct (general-purpose, Mistral family) and
Med42-8B (clinically tuned, Llama-3 family), joining the original MedGemma-27B (Gemma family).
The new benchmarks are ChatDoctor/HealthCareMagic (unedited questions real patients asked
physicians online) and MedQuAD (NIH consumer-health question answering), joining HealthBench. A
fourth benchmark, MedQA-USMLE reframed open-ended, is still running and will be reported in the
camera-ready. Every cell holds the paper's hyperparameters fixed: LoRA r=16, effective batch 16,
three epochs, the asymmetric Qwen-14B filter and Llama-3.1-8B evaluator.

The central result replicates in every capable configuration. Active inquiry, the paper's largest
effect, rises from base to adapter by +31pp on HealthBench with Mistral-24B (25.6% to 56.6%),
+41pp on HealthBench with Med42-8B (11.8% to 52.5%), +51pp on ChatDoctor with Mistral-24B (7.8% to
58.6%), +84pp on ChatDoctor with Med42-8B (2.9% to 86.4%), and +20pp on MedQuAD with Mistral-24B
(6.5% to 26.5%). Context-seeking moves the same way in all five (for example 0.32 to 1.23 on
MedQuAD, 0.71 to 1.81 on ChatDoctor). In four of the five, the adapter **matches or exceeds its own
teacher** — the inference-time CoT protocol — while requiring no CoT at deployment. This is the
internalization claim, now demonstrated across two model families beyond the original and two
benchmark families beyond the original.

We also report a negative result that we think is more useful than a clean sweep. BioMistral-7B
shows no effect (11.7% to 10.1% active inquiry). The diagnostic detail is that the *teacher* also
fails on this model: the CoT wrapper only reaches 14.5%. If the protocol cannot elicit the behavior
in the first place, there is nothing to distill. The recipe therefore has a precondition — a base
model capable of following the structured CoT protocol — and we now state that as a scope
condition rather than leaving a reader to discover it.

Two further reviewer concerns turned out to be answerable from the data we already had, without new
runs, and we describe those analyses under Reviewer LxMF below: a direct test of whether the model
learned calibration or template mimicry, and a direct test of the mechanism behind the
CoT-adapter interference.

We want to be equally clear about what these runs do **not** settle. We have not yet run the
clinical-judge re-grade or a head-to-head comparison against prior calibration methods; those
remain future work and we no longer describe them otherwise. Physician validation is still partial.
And the new benchmarks lack native expert rubrics, so their filter signal is synthesized from
ground truth, which we detail below.

---

## Reviewer FzpA

**On the "framework" overclaim (W1).** We accept this criticism as originally written and we have
changed both the claim and the evidence. The paper's two statements were genuinely in tension: one
asserted that any CoT protocol, any PEFT method and any domain would work, while the next admitted
to one model, one protocol, one domain and one benchmark. Our revision resolves the tension in both
directions. We have narrowed the language — the contribution is a recipe plus an evaluation
decomposition, demonstrated on open-ended, information-seeking clinical Q&A, and we say explicitly
that we do not claim generality to task types where there is no missing information to seek, such
as closed-form multiple choice or summarization. At the same time, the empirical basis is no longer
a single point: three model families and three benchmarks now support the transfer claim, and the
BioMistral null defines its boundary. We would rather claim less and show more.

**On MedGemma not matching HealthBench's use case (W2).** The reviewer is right that a clinically
tuned model answering consumer-style questions is a mismatch, and our original defence — that
commercial models cannot be LoRA-adapted — answered a different question than the one asked. The
substantive answer is the experiment: Mistral-Small-24B-Instruct is a general-purpose,
non-clinical, widely used open-weight model, and it produces the paper's effect on HealthBench
(active inquiry 25.6% to 56.6%, context-seeking 1.05 to 1.76, red-flag 1.03 to 1.67). The effect is
therefore not an artifact of using a clinical model on consumer questions.

**On clinician-posed questions and MIMIC/eICU (Q3).** We agree that the original evaluation only
covered patient-style conversations. We checked HealthBench-Hard directly and found that only about
0.6% of the evaluated prompts explicitly identify the speaker as a clinician, which confirms the
reviewer's characterization rather than disputing it. We have added MedQA-USMLE reframed as
open-ended clinician-facing questions, which is running now, and ChatDoctor and MedQuAD to widen
the patient-facing side. We were not able to use MIMIC or eICU: the derived text corpora require
credentialed PhysioNet access under a data use agreement that we could not complete inside the
discussion period. We name this as a concrete next step rather than claiming coverage we do not
have.

**On the seven dimensions and their provenance (Q1).** The decomposition is not BODHI's five
letters renamed. BODHI supplies the generation protocol; the seven evaluation dimensions were
derived by grouping the behaviors that a calibrated clinical response must exhibit into three
functions — self-assessment (uncertainty acknowledgment, hedging quality, specificity),
information-seeking (active inquiry, context-seeking) and risk management (red-flag identification,
scope bounding) — and were shaped by the practising physician co-authors on this paper, who work in
the US, Uganda, Colombia and the UK. The camera-ready will state this origin explicitly and give a
verbatim text example of each dimension so a reader can see what is being scored, which also
answers Q2.

**On clarity of results and discussion (W3).** We have removed the "not statements". The claim now
reads positively: the adapter shifts specific epistemic behaviors while preserving aggregate
clinical quality, and we give the per-dimension numbers rather than gesturing at them. Figure 2 is
split into two panels so that 0–2 scores and percentage rates no longer share an axis. Table 2 now
carries explicit direction arrows, and red-flag identification is annotated as higher-is-better
with a footnote stating plainly that the metric does not capture false positives. We have added a
short paragraph on what each behavior means clinically — that active inquiry is history-taking
before recommending, that red-flag identification is the recognition of presentations warranting
escalation, that scope bounding is the refusal to advise past the available evidence.

**On the compute comparison (Strength 4).** We will report training and inference cost per
configuration explicitly, including the measured wall-clock of the new runs: on a single H100, a
24B cell takes roughly 128 minutes per seed and an 8B cell roughly 39 minutes, of which training is
about 93 and 30 minutes respectively and the remainder is the four-configuration evaluation.

---

## Reviewer LxMF

**On calibration versus surface mimicry (W2).** This was the sharpest methodological critique in
the set, and the reviewer specified exactly the right test: isolate prompts that genuinely lack
information from those that do not, and check whether inquiry rises more in the former. We ran it.
Using an independent label — HealthBench's own `context_seeking` theme tag, and separately whether
a prompt's rubric rewards asking for clarification, neither derived from the grader's own
context-seeking score — we computed a discrimination index for each condition, defined as
P(ask | information missing) − P(ask | self-contained).

The base model does **not** reliably discriminate: +2.1pp, with a bootstrap 95% CI that contains
zero. Both the wrapper and the adapter do: +11.0pp and +10.5pp respectively, with CIs excluding
zero, and on the theme-only cut the adapter reaches +13.2pp with CI [+4.1, +21.7]. In other words,
the adapter does not ask more indiscriminately; it acquires the *targeting* that the teacher has
and the base model lacks. Pure template mimicry predicts the opposite — either the base model's
non-discrimination carried forward, or both groups rising uniformly. We also note that under
LoRA+CoT the discrimination collapses to −3.5pp, which is consistent with the interference result
below. The complementary evidence comes from the new benchmarks: on ChatDoctor, where real patients
routinely omit age, duration and medications, inquiry rises to 86.4%, whereas on MedQA, whose
vignettes are deliberately self-contained, it stays far lower. The behavior tracks whether asking is
warranted.

**On the interference claim being speculative (W4).** The reviewer proposed a specific alternative:
context-window truncation or attention dilution from long outputs. We tested it directly against
the mechanism we proposed, using the existing evaluation outputs. Under LoRA+CoT, 54.6% of responses
leak the CoT protocol's internal Pass-1 analysis format into the patient-facing answer, compared
with under 3% in every other condition. Red-flag identification collapses specifically in that
leaked subset (1.33) while the non-leaked responses hold at 1.73, which is *above* LoRA alone
(1.67). If attention dilution were responsible we would expect red-flag to fall with output length;
within the non-leaked responses it does the opposite, rising monotonically from 1.44 in the
shortest quartile to 1.87 in the longest. The leaked responses are indeed longer, but their low
red-flag score is attributable to their emitting "RED FLAGS: None" in analysis format rather than to
their length. We therefore retain the competition interpretation but state it more precisely: the
two conditioning sources compete for the output format, and the failure is format leakage rather
than capacity exhaustion. We are grateful for the push, because the original wording claimed more
than we had shown, and we adopt the reviewer's suggestion of surfacing red flags early in the
protocol as a mitigation.

**On differing sample sizes in Table 1 (Q1).** The differences are entirely attributable to the CoT
protocol's two-pass generation exceeding the 4,096-token context window on roughly 4–5% of prompts,
so those conditions produce fewer completions. The LoRA condition needs a single pass and has a
100% response rate. We will add this as a table footnote rather than leaving it to the body text.

**On using Llama-3.1 rather than a clinical judge (Q3).** We agree and have not yet run it, so we
will not imply otherwise. Our reason for the original choice was to keep the evaluator in a
different model family from the Qwen-14B filter, which is what makes the asymmetric grading design
work. We note that the clinically tuned models the reviewer names are Llama-derived, so substituting
one for the evaluator would place the filter and grader in adjacent families and weaken that
property. Our plan is to add a clinical judge as a robustness panel reported alongside the primary
grader rather than replacing it.

**On the dual-axis figure (Clarity).** Agreed and fixed; Figure 2 is now two panels.

---

## Reviewer HomM

**On the evaluation being the weakest part (W1, Q1, Q2).** The reviewer suspected we were already
pursuing the obvious extensions across more models and benchmarks. We were, and they are now done.
The grid described in the common response above — Mistral-Small-24B and Med42-8B added to
MedGemma-27B, ChatDoctor and MedQuAD added to HealthBench, five seeds per cell — is a direct
response to Q2's request for "at least one or two additional base models, or one additional
clinical evaluation set." We have both, and across families rather than within one.

We would highlight one result that we did not anticipate and that speaks to the reviewer's concern
about whether the recipe is merely copying a teacher. On MedQuAD, the CoT wrapper *degrades* two
dimensions relative to the base model — scope bounding falls from 1.85 to 1.61 and hedging quality
from 1.45 to 1.40 — while the distilled adapter *improves* both (1.89 and 1.78). The student is not
reproducing the teacher indiscriminately; quality filtering removes the teacher's failures before
they reach the weights. This is the clearest evidence we have that the filter step does real work.

**On aggregate quality (a check the reviewer will want).** Behavior gains are not free everywhere,
and we report this plainly. On HealthBench the adapter's aggregate rubric score is essentially
unchanged (Mistral-24B 0.425 to 0.411; Med42-8B 0.405 to 0.392), consistent with the paper's
original non-inferiority result. On the new benchmarks there are modest decreases, largest on
MedQuAD (0.672 to 0.586). We think the MedQuAD case is informative rather than merely negative: its
synthesized rubric rewards agreement with a reference answer, so a response that asks a clarifying
question instead of answering immediately scores lower by construction. Consistent with that
reading, the CoT wrapper — which asks most — drops furthest (0.672 to 0.542), further than the
adapter. This is precisely the measurement problem the paper is about: aggregate rubric scores can
penalize the epistemic behavior that makes a response safer, which is why we argue for the
decomposition alongside the aggregate rather than instead of it. We state the decreases in the
camera-ready rather than reporting only the dimensions that improved.

**On dependence on automated grading (W3, Q4).** We share this concern and have not resolved it. The
physician validation is still partial: one of three physicians has returned grades, giving
κ=0.35 against the LLM grader, below our pre-registered target of 0.6. We report this as a
limitation on the aggregate-quality claims and will not present LLM grading as settled clinical
validation. Adding a clinically tuned judge as a robustness panel is planned, with the family-overlap
caveat noted under LxMF Q3.

**On the absence of comparison to prior work (W4, Q3).** This is a fair criticism that our new runs
do not address, and we will not pretend otherwise. We will add a Discussion subsection positioning
the method against uncertainty prompting (Lin et al., Tian et al.), which achieves calibration at
inference time whereas we internalize it into weights; STaR-style self-improvement (Zelikman et
al.), whose filter-then-finetune logic we apply to behavioral rather than factual demonstrations;
and Constitutional AI (Bai et al.), which uses self-critique where we use a structured CoT protocol
as a behavioral teacher. A direct empirical head-to-head against an inference-time calibration
baseline is the single most valuable experiment we have not run, and we name it as the next step
rather than folding it into a claim.

**On workshop versus conference scope.** We understand the assessment as originally given. Our
position is that the specific gap identified — one model, one benchmark — has now been closed with
five-seed evidence across three model families and three benchmark families, together with two
mechanistic analyses that address the mimicry and interference concerns directly. We would ask the
reviewer to reconsider in light of the added evidence, while acknowledging that the prior-work
comparison and the completed physician validation remain outstanding.

---

## Summary of new evidence

| Cell | Model family | Active inquiry (Base → LoRA) | Context-seeking |
|---|---|---|---|
| HealthBench × Mistral-Small-24B | Mistral, general | 25.6% → 56.6% | 1.05 → 1.76 |
| HealthBench × Med42-8B | Llama-3, clinical | 11.8% → 52.5% | 0.52 → 1.40 |
| ChatDoctor × Mistral-Small-24B | Mistral, general | 7.8% → 58.6% | 0.83 → 1.49 |
| ChatDoctor × Med42-8B | Llama-3, clinical | 2.9% → 86.4% | 0.71 → 1.81 |
| MedQuAD × Mistral-Small-24B | Mistral, general | 6.5% → 26.5% | 0.32 → 1.23 |
| HealthBench × BioMistral-7B | scope condition | 11.7% → 10.1% (null) | 0.27 → 0.28 |

Five seeds per cell; approximately 200 prompts per seed per condition; identical pipeline and
hyperparameters throughout. MedQA-USMLE (clinician-facing, open-ended) is in progress.

**Outstanding and not claimed:** head-to-head comparison against prior calibration methods;
clinical-judge robustness panel; completion of the three-physician validation; MIMIC/eICU-derived
clinician questions.
