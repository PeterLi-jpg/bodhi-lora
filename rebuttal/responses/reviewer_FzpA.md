Thank you for your thorough and constructive review, and for your helpful suggestions. We are pleased
you find epistemic calibration an important area of study, consider our asymmetric cross-family
grading protocol to address concerns about using a single model across pipeline steps, and find the
experiment carefully constructed and well described. Your criticisms of our claim language and
presentation are correct, and we have acted on all of them.

**The "framework" claim is overly broad**

Admittedly our two statements were in tension, as you point out, and the fault is ours. We have
narrowed the claim: the contribution is a recipe plus an evaluation decomposition, demonstrated on
open-ended, information-seeking clinical Q&A. We now state what is *not* covered, namely task types
with no missing information to seek, such as closed-form multiple choice, extraction, or
summarization. The sentence beginning "Any structured CoT protocol..." is replaced with a scoped
version separating what we demonstrated from what we conjecture.

**MedGemma is not aligned with HealthBench's use case**

You are right, and our original reasoning about closed weights answered a different question. As you
suggested a more widely used model would be more appropriate, we re-ran the identical pipeline with
all hyperparameters fixed on **Mistral-Small-24B-Instruct**, a general-purpose non-clinical
open-weight model (Table R1). The effect reproduces on HealthBench: active inquiry 25.6% to 56.6%,
context-seeking 1.05 to 1.76 (5 seeds). We also ran **Med42-8B**, clinically tuned from a third model
family. The submitted finding therefore does not appear to be an artifact of pairing a clinical model
with consumer-style questions.

**One benchmark is not sufficient; consider clinician-posed questions**

Short answer: we added two benchmarks, attempted a third and excluded it for a reason we give below,
and could not use MIMIC or eICU.

We first checked your premise and it holds: in HealthBench-Hard only ~0.6% of evaluated prompts
explicitly identify the speaker as a clinician. Per your suggestion we added **ChatDoctor** (unedited
questions real patients asked physicians online) and **MedQuAD** (NIH consumer-health QA); the effect
reproduces on both (Table R1). MIMIC and eICU require credentialed PhysioNet access under a data use
agreement we could not complete within the discussion period, so we list this as a next step rather
than claiming coverage.

We also attempted the clinician-facing set you asked for, by reframing MedQA-USMLE as open-ended
questions, and we exclude it for a reason worth stating: our reframing preamble instructed the model
to ask for information it needed, which drove base-model active inquiry to 99% and left the cell
measuring instruction-following rather than calibration. We report this rather than omit it. The
prompt needs to establish the clinician setting without cueing the behavior being scored, and we
will rebuild it that way for the camera-ready.

**Table R1.** Transfer across model families and benchmarks. 5 seeds per cell, all hyperparameters
fixed; every cell reads Base → LoRA. Families: Mistral-Small-24B and BioMistral-7B (Mistral),
Med42-8B (Llama-3); the submitted result used MedGemma-27B (Gemma).

| Benchmark | Base model | Active inquiry | Context-seek |
|---|---|---|---|
| HealthBench | **MedGemma-27B (submitted)** | **17.5 → 45.6%** | **1.40 → 1.75** |
| HealthBench | Mistral-Small-24B | 25.6 → 56.6% | 1.05 → 1.76 |
| HealthBench | Med42-8B | 11.8 → 52.5% | 0.52 → 1.40 |
| ChatDoctor | Mistral-Small-24B | 7.8 → 58.6% | 0.83 → 1.49 |
| ChatDoctor | Med42-8B | 2.9 → 86.4% | 0.71 → 1.81 |
| MedQuAD | Mistral-Small-24B | 6.5 → 26.5% | 0.32 → 1.23 |
| HealthBench | BioMistral-7B | 11.7 → 10.1% (null) | 0.27 → 0.28 |

**Where the seven dimensions came from, and whether clinicians were involved**

Short answer: they are not BODHI's five letters renamed, and yes.

BODHI supplies the generation protocol. The seven evaluation dimensions group the behaviors a
calibrated clinical response must exhibit into three functions: *self-assessment* (uncertainty
acknowledgment, hedging quality, specificity), *information-seeking* (active inquiry,
context-seeking), and *risk management* (red-flag identification, scope bounding). The selection was
shaped by the practising physician co-authors on this paper, who work in the US, Uganda, Colombia and
the UK, and who identified these as the safety-critical behaviors for clinical decision support. The
revision states this origin explicitly in a new subsection.

**How each value is operationalized in text**

We add a verbatim example per dimension so the reader can see what is scored: *uncertainty
acknowledgment* ("without a chest X-ray I cannot confirm..."); *active inquiry* ("when did the
symptoms start?"); *context-seeking* ("I would need the patient's age and medication history");
*red-flag identification* ("shortness of breath with chest pain warrants immediate evaluation");
*scope bounding* ("diagnosis requires an examination"); *hedging quality* (a specific qualified
statement rather than a generic disclaimer); *specificity* (a concrete dose or timeframe).

**Results and discussion lack clarity**

We agree. We have made three of the four changes for the revision; the fourth is a figure regeneration:

- **Vague "not statements" removed** (done). They are replaced with positive claims stating the
  per-dimension numbers, including the accuracy (0.117 → 0.121) and completeness (0.168 → 0.169)
  figures that support the communication-not-knowledge reading.
- **Table 2 now carries a direction arrow on every row** (done), with the caption note on false
  positives described below.
- **Clinical implications paragraph added** (done): active inquiry is history-taking before
  recommending, red-flag identification is recognizing presentations warranting escalation, scope
  bounding is declining to advise beyond the available evidence.
- **Figure 2 will be split into two panels** so that 0–2 scores and percentage rates no longer share
  an axis; this requires regenerating the figure and will appear in the camera-ready.

**Is red-flag rate missing a down arrow?**

Higher is better, and the ambiguity is our fault for not labelling it: the dimension measures
sensitivity to warning signs that warrant escalation. We add a caption note that it does not capture
false positives, so a model flagging indiscriminately would also score highly, and that measuring
over-flagging requires a labelled set of prompts containing no genuine red flag, which we leave to
future work.

**A more explicit compute comparison**

We will report training and inference cost per configuration. From these runs, one seed of a 24B
configuration end to end (generation, filtering, LoRA training, evaluation) takes about 2 hours on a
single H100; the 8B configurations are cheaper, but our timings came off a shared node and we will
quote per-stage figures measured in isolation rather than publish numbers confounded by queueing.
This sits alongside the existing inference comparison of one forward pass versus two.

Remaining open, and we do not claim otherwise: a head-to-head against prior calibration methods
[1,2], a clinical-judge robustness panel, and completion of the three-physician validation. We will
incorporate all feedback into the paper.

[1] Lin, Hilton, Evans. "Teaching Models to Express Their Uncertainty in Words." TMLR 2022.
[2] Tian et al. "Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from
Fine-Tuned Language Models with Human Feedback." EMNLP 2023.
