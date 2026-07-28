We thank the reviewer for detailed comments and suggestions. We are pleased the reviewer finds
epistemic calibration an important area of study, considers our asymmetric cross-family grading
protocol to address concerns about using a single model across pipeline steps, and finds the
experiment carefully constructed and well described. The criticisms of our claim language and
presentation are correct, and we have acted on all of them.

**The "framework" claim is overly broad**

Admittedly our two statements were in tension, as the reviewer points out, and the fault is ours. We
have narrowed the claim: the contribution is a recipe plus an evaluation decomposition, demonstrated
on open-ended, information-seeking clinical Q&A. We now state what is *not* covered, namely task
types with no missing information to seek, such as closed-form multiple choice, extraction, or
summarization. The sentence beginning "Any structured CoT protocol..." is replaced with a scoped
version separating what we demonstrated from what we conjecture.

**MedGemma is not aligned with HealthBench's use case**

The reviewer is right, and our original reasoning about closed weights answered a different
question. As the reviewer suggested a more widely used model would be more appropriate, we re-ran
the identical pipeline with all hyperparameters fixed on **Mistral-Small-24B-Instruct**, a
general-purpose non-clinical open-weight model. The effect reproduces on HealthBench: active inquiry
25.6% to 56.6%, context-seeking 1.05 to 1.76 (5 seeds). We also ran **Med42-8B**, clinically tuned
from a third family (Llama-3). The submitted finding therefore does not appear to be an artifact of
pairing a clinical model with consumer-style questions.

**One benchmark is not sufficient; consider clinician-posed questions**

We checked the premise and it holds: in HealthBench-Hard only ~0.6% of evaluated prompts explicitly
identify the speaker as a clinician. Per the reviewer's suggestion we added two benchmarks:
**ChatDoctor** (unedited questions real patients asked physicians online) and **MedQuAD** (NIH
consumer-health QA). The effect reproduces on both (ChatDoctor 7.8% to 58.6% with Mistral-24B, 2.9%
to 86.4% with Med42-8B; MedQuAD 6.5% to 26.5%). A clinician-facing set, MedQA-USMLE reframed
open-ended, is in progress. We could not use MIMIC or eICU: the derived text corpora require
credentialed PhysioNet access we could not complete within the discussion period, and we list this
as a next step rather than claiming coverage.

**Where the seven dimensions came from**

They are not BODHI's five letters renamed. BODHI supplies the generation protocol; the seven
evaluation dimensions group the behaviors a calibrated clinical response must exhibit into three
functions: self-assessment (uncertainty acknowledgment, hedging quality, specificity),
information-seeking (active inquiry, context-seeking), and risk management (red-flag identification,
scope bounding). The selection was shaped by the practising physician co-authors on this paper, who
work in the US, Uganda, Colombia and the UK, and who identified these as the safety-critical
behaviors for clinical decision support. The revision states this origin explicitly.

**How each value is operationalized in text**

We add a verbatim example per dimension: uncertainty acknowledgment ("without a chest X-ray I cannot
confirm..."), active inquiry ("when did the symptoms start?"), context-seeking ("I would need the
patient's age and medication history"), red-flag identification ("shortness of breath with chest
pain warrants immediate evaluation"), scope bounding ("diagnosis requires an examination"), hedging
quality (a specific qualified statement rather than a generic disclaimer), specificity (a concrete
dose or timeframe).

**Results and discussion lack clarity**

We agree, and all four changes are made. The "not statements" are replaced with positive claims
stating the per-dimension numbers, including the accuracy (0.117 to 0.121) and completeness (0.168
to 0.169) figures that support the communication-not-knowledge reading. Figure 2 is split into two
panels so 0-2 scores and percentage rates no longer share an axis. Table 2 carries direction arrows
on every row. A new paragraph states what each behavior means clinically: active inquiry is
history-taking before recommending, red-flag identification is recognizing presentations warranting
escalation, scope bounding is declining to advise beyond the available evidence.

**Is red-flag rate missing a down arrow?**

Higher is better, and the ambiguity is our fault for not labelling it: red-flag identification
measures sensitivity to warning signs that warrant escalation. We add a caption note that the metric
does not capture false positives, so a model flagging indiscriminately would also score highly, and
that measuring over-flagging requires a labelled set of prompts with no genuine red flag, which we
leave to future work.

**A more explicit compute comparison**

We will report training and inference cost per configuration, including measured wall-clock: on a
single H100, ~128 min/seed for a 24B configuration and ~39 min/seed for an 8B one, alongside the
existing inference comparison of one forward pass versus two.

Remaining open, and we do not claim otherwise: a head-to-head against prior calibration methods, a
clinical-judge robustness panel, and completion of the three-physician validation. We will
incorporate all feedback into the paper.
