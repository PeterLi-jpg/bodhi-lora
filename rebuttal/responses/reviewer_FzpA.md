# Response to Reviewer FzpA

We thank the reviewer for a careful reading. The criticisms about our claim language and about
presentation are correct, and we have acted on all of them. Below we address each point, and we note
where confirmatory runs completed during the discussion period support the submitted results.

**W1: The "framework" claim is overstated.** We accept this. The two statements the reviewer quotes
were in tension, and the fault was ours. In the revision we narrow the claim: the contribution is a
recipe plus an evaluation decomposition, demonstrated on open-ended, information-seeking clinical
question answering. We now say explicitly what is not covered, namely task types with no missing
information to seek, such as closed-form multiple choice or summarization. The sentence beginning
"Any structured CoT protocol..." is replaced with a scoped version separating what we demonstrated
from what we conjecture.

**W2: MedGemma is not aligned with HealthBench's use case.** The reviewer is right, and our original
reasoning about closed weights answered a different question. To check whether the result depends on
using a clinical model for consumer-style questions, we ran the same pipeline on
Mistral-Small-24B-Instruct, a general-purpose open-weight model, holding all hyperparameters fixed.
The effect reproduces: active inquiry rises from 25.6% to 56.6% and context-seeking from 1.05 to
1.76 (five seeds). We also ran Med42-8B, a clinically tuned model from a third family. The submitted
MedGemma result therefore does not appear to be an artifact of model choice.

**Q3: One benchmark, and only patient-style conversations.** We checked the premise before
responding, and it holds: in HealthBench-Hard only about 0.6% of evaluated prompts explicitly
identify the speaker as a clinician. We have added two further benchmarks, ChatDoctor
(unedited questions real patients asked physicians online) and MedQuAD (NIH consumer-health QA),
where the effect also reproduces. A clinician-facing set, MedQA-USMLE reframed open-ended, is in
progress. We could not use MIMIC or eICU because the text corpora require credentialed PhysioNet
access we could not complete in the discussion window, and we list this as a next step rather than
claiming coverage we do not have.

**Q1: Where do the seven dimensions come from?** They are not BODHI's five letters renamed. BODHI
supplies the generation protocol; the seven evaluation dimensions group the behaviors a calibrated
clinical response must exhibit into three functions: self-assessment (uncertainty acknowledgment,
hedging quality, specificity), information-seeking (active inquiry, context-seeking), and risk
management (red-flag identification, scope bounding). The selection was shaped by the practising
physician co-authors on this paper, who work in the US, Uganda, Colombia and the UK. The revision
states this origin explicitly.

**Q2: How is each value operationalized in text?** We add a verbatim example per dimension so the
reader can see what is scored: uncertainty acknowledgment ("without a chest X-ray I cannot
confirm..."), active inquiry ("when did the symptoms start?"), context-seeking ("I would need the
patient's age and medication history"), red-flag identification ("shortness of breath with chest
pain warrants immediate evaluation"), scope bounding ("diagnosis requires an examination"), hedging
quality (a specific qualified statement rather than a generic disclaimer), and specificity (a
concrete dose or timeframe).

**W3: Clarity of results and discussion.** All four changes are made. The "not statements" are
replaced with positive claims stating the per-dimension numbers. Figure 2 is split into two panels
so that 0 to 2 scores and percentage rates no longer share an axis. Table 2 carries direction
arrows. We add a paragraph on what each behavior means clinically: active inquiry is history-taking
before recommending, red-flag identification is recognizing presentations that warrant escalation,
scope bounding is declining to advise beyond the available evidence.

**Table 2, red-flag direction.** Higher is better, and the ambiguity is our fault for not labelling
it. Red-flag identification measures sensitivity to warning signs that warrant escalation. We add a
footnote stating that the metric does not capture false positives, and that measuring over-flagging
is necessary future work.

**Strength 4: compute comparison.** We will report training and inference cost per configuration,
including measured wall-clock: on a single H100, roughly 128 minutes per seed for a 24B
configuration and 39 minutes for an 8B one.

**What remains open.** We have not completed a head-to-head comparison against prior calibration
methods, the clinical-judge robustness panel, or the full three-physician validation. We state these
as outstanding rather than folding them into a claim.
