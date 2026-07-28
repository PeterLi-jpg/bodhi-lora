Thank you for your thorough and constructive review. We are pleased
you find epistemic calibration an important area of study, consider our asymmetric cross-family
grading protocol to address concerns about using a single model across pipeline steps, and find the
experiment carefully constructed. Your criticisms of our claim language and
presentation are correct, and we have acted on all of them.

**W1. The "framework" claim is overly broad**

Admittedly our two statements were in tension, as you point out, and the fault is ours. We have
narrowed the claim: the contribution is a recipe plus an evaluation decomposition, demonstrated on
open-ended, information-seeking clinical Q&A. We now state what is *not* covered, namely task types
with no missing information to seek, such as closed-form multiple choice, extraction, or
summarization. The sentence beginning "Any structured CoT protocol..." is replaced with a scoped
version separating what we demonstrated from what we conjecture.

On how the decomposition shifts with context: the dimensions are not equally meaningful everywhere. Active inquiry and context-seeking presuppose that information is
missing and recoverable from the user, so they are informative on patient-facing Q&A and close to
vacuous on self-contained exam items. Scope bounding and red-flag identification travel further,
since they concern what a response commits to rather than what it asks for. Our new runs bear this
out: holding the base model fixed at Mistral-Small-24B, active inquiry rises 7.8% to 58.6% on
ChatDoctor, whose patient messages routinely omit age and medications, but only 6.5% to 26.5% on the
better-specified MedQuAD. The revision says which dimensions are load-bearing in which settings
rather than presenting all seven as uniformly applicable.

**W2. MedGemma is not aligned with HealthBench's use case**

You are right, and our original reasoning about closed weights answered a different question. As you
suggested a more widely used model would be more appropriate, we re-ran the identical pipeline with
all hyperparameters fixed, on **Mistral-Small-24B-Instruct**, a general-purpose non-clinical
open-weight model, and on **Med42-8B**, clinically tuned from a third family (Table R1).

The pattern across the three is the substantive answer, not just the reproduction. Base active
inquiry varies by more than a factor of two across the three models (11.8%, 17.5%, 25.6%), which is
what one would expect if the behavior were an idiosyncrasy of a particular model's alignment. After
adaptation all three converge into a narrow band (45.6%, 52.5%, 56.6%). The endpoint is a property of
the training signal rather than of the starting model, which is the claim your critique put in doubt.

**W3 / Q3. One benchmark is not sufficient; consider clinician-posed questions**

Short answer: we added two benchmarks, attempted a third and excluded it for the reason below,
and could not use MIMIC or eICU.

We first checked your premise, and it holds: auditing HealthBench-Hard for explicit clinician
self-identification ("my patient", "as a physician"), 14 of 1,000 prompts (1.4%) identify the speaker
as a clinician. The keyword audit is a lower bound, but the benchmark is overwhelmingly patient-facing,
as you suspected. Per your suggestion we added
**ChatDoctor** (unedited questions real patients asked physicians online) and **MedQuAD** (NIH
consumer-health QA); the effect reproduces on both. MIMIC and eICU require credentialed PhysioNet
access under a data use agreement we could not complete in the discussion period, so we list this
as a next step rather than claiming coverage.

We also attempted the clinician-facing set you asked for, by reframing MedQA-USMLE as open-ended
questions, and we exclude it for a reason worth stating: our reframing preamble instructed the model
to ask for information it needed, which drove base-model active inquiry to 99% and left the cell
measuring instruction-following rather than calibration. We report it rather than omit it. The prompt
must establish the clinician setting without naming the behavior scored, and we will rebuild it for the
camera-ready.

**Table R1.** Transfer across model families and benchmarks. 5 seeds per cell, all hyperparameters
fixed; every cell reads Base → LoRA. Mistral-Small-24B and BioMistral-7B are Mistral-family,
Med42-8B is Llama-3; the submitted result used MedGemma-27B (Gemma).

| Benchmark | Base model | Active inquiry | Context-seek |
|---|---|---|---|
| HealthBench | **MedGemma-27B (submitted)** | **17.5 → 45.6%** | **1.40 → 1.75** |
| HealthBench | Mistral-Small-24B | 25.6 → 56.6% | 1.05 → 1.76 |
| HealthBench | Med42-8B | 11.8 → 52.5% | 0.52 → 1.40 |
| ChatDoctor | Mistral-Small-24B | 7.8 → 58.6% | 0.83 → 1.49 |
| ChatDoctor | Med42-8B | 2.9 → 86.4% | 0.71 → 1.81 |
| MedQuAD | Mistral-Small-24B | 6.5 → 26.5% | 0.32 → 1.23 |
| HealthBench | BioMistral-7B | 11.7 → 10.1% (null) | 0.27 → 0.28 |

We report a precondition rather than only successes: on BioMistral-7B there is no effect, and the
diagnostic is that the teacher fails there too (wrapper 14.5%; only 20% of its traces cleared the
filter, against 62% for Mistral-24B, leaving 733 rows). The recipe requires a base model already
capable of following the protocol.

**Q1. Where the seven dimensions came from, and whether clinicians were involved**

They are deliberately not BODHI's five letters renamed, and yes.

BODHI (**B**ridging, **O**pen, **D**iscerning, **H**umble, **I**nquiring) is the *generation* protocol;
the seven dimensions are the *evaluation* decomposition. We kept them distinct on purpose: scoring the
student on the teacher's own five categories would grade the pipeline against the rubric it was built
to satisfy. The correspondence is therefore partial. Inquiring maps onto
active inquiry and context-seeking; Humble onto hedging quality and uncertainty acknowledgment;
Discerning onto red-flag identification and specificity; Open and Bridging bear on scope bounding but
have no single dimension of their own.

The seven instead cover the three epistemic functions a calibrated clinical response must serve:
*self-assessment* (uncertainty acknowledgment, hedging quality, specificity), *information-seeking*
(active inquiry, context-seeking), and *risk management* (red-flag identification, scope bounding).
The selection was shaped by the practising physician co-authors on this paper, who work in the US,
Uganda, Colombia and the UK. We excluded candidates such as empathy and patient-education depth
because they conflate what is communicated with how accurately uncertainty is represented. The
revision states the origin and the exclusions explicitly.

**Q2. How each value is operationalized in text**

We add a verbatim example per dimension so the reader can see what is scored: *uncertainty
acknowledgment* ("without a chest X-ray I cannot confirm..."); *active inquiry* ("when did the
symptoms start?"); *context-seeking* ("I would need the patient's age and medication history");
*red-flag identification* ("shortness of breath with chest pain warrants immediate evaluation");
*scope bounding* ("I can give general information, but diagnosis requires an examination"); *hedging
quality* (a specific qualified statement rather than a generic disclaimer); *specificity* (a concrete
dose or timeframe).

**W3. Results and discussion lack clarity**

We agree. Three of the four changes are made for the revision; the fourth needs the figure
regenerated:

- **Vague "not statements" removed** (done), replaced with positive claims stating the per-dimension
  numbers, including the accuracy (0.117 → 0.121) and completeness (0.168 → 0.169) figures that
  support the communication-not-knowledge reading.
- **Table 2 now carries a direction arrow on every row** (done), with the caption note below.
- **Clinical implications paragraph added** (done): active inquiry is history-taking before
  recommending, red-flag identification is recognizing presentations warranting escalation, scope
  bounding is declining to advise beyond the available evidence.
- **Figure 2 will be split into two panels** so 0–2 scores and percentage rates no longer share an
  axis. This requires regenerating the figure and will appear in the camera-ready.

**Is red-flag rate missing a down arrow?**

Higher is better, and the ambiguity is our fault: the dimension measures sensitivity to warning signs
that warrant escalation. In Table 2, blanket disclaimer rate is the only row where lower is better, and
it is now the only one carrying a down arrow. We add a caption note that red-flag identification does
not capture false positives, so a model flagging indiscriminately would also score highly; measuring
over-flagging needs a labelled set of prompts containing no genuine red flag, which we leave to future
work.

**A more explicit compute comparison**

We will report training and inference cost per configuration. From these runs, one seed of a 24B
configuration end to end (generation, filtering, training, evaluation) takes about 2 hours on a
single H100; the 8B configurations are cheaper, but our timings came off a shared node, so we will
quote per-stage figures measured in isolation rather than numbers confounded by queueing. This sits
alongside the existing one-pass-versus-two comparison, where the adapter delivers wrapper-equivalent
behavior at ~58% lower inference output cost.

Remaining open, and we do not claim otherwise: a head-to-head against prior calibration methods
[1,2], a clinical-judge robustness panel, completion of the three-physician validation (1 of 3 raters
returned, kappa = 0.35), and a second CoT protocol. We will incorporate all feedback into the paper.

[1] Lin, Hilton, Evans. "Teaching Models to Express Their Uncertainty in Words." TMLR 2022.
[2] Tian et al. "Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from
Fine-Tuned Language Models with Human Feedback." EMNLP 2023.
