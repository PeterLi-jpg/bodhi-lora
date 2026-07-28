# Response to Reviewer FzpA

We thank the reviewer for a careful reading. The criticisms of our claim language and presentation
are correct, and we have acted on all of them.

### W1: The "framework" claim is overstated

We accept this. The two statements quoted were in tension, and the fault was ours. In the revision:

- The contribution is described as a recipe plus an evaluation decomposition, demonstrated on
  open-ended, information-seeking clinical question answering.
- We state what is not covered: task types with no missing information to seek, such as closed-form
  multiple choice, extraction, or summarization.
- "Any structured CoT protocol..." is replaced with a scoped version separating what we demonstrated
  from what we conjecture.

### W2: MedGemma is not aligned with HealthBench's use case

The reviewer is right, and our original reasoning about closed weights answered a different
question. To test whether the result depends on using a clinical model for consumer-style questions,
we ran the same pipeline, hyperparameters fixed, on a general-purpose open-weight model:

- **Mistral-Small-24B-Instruct** (general-purpose): active inquiry 25.6% to 56.6%, context-seeking
  1.05 to 1.76 (five seeds).
- **Med42-8B** (clinical, third model family) also reproduces the effect.

The submitted MedGemma result therefore does not appear to be an artifact of model choice.

### Q3: One benchmark, and only patient-style conversations

We checked the premise before responding, and it holds: in HealthBench-Hard only about 0.6% of
evaluated prompts explicitly identify the speaker as a clinician.

- Added **ChatDoctor** (unedited questions real patients asked physicians online) and **MedQuAD**
  (NIH consumer-health QA). The effect reproduces on both.
- A clinician-facing set, **MedQA-USMLE reframed open-ended**, is in progress.
- MIMIC and eICU were not usable: the text corpora require credentialed PhysioNet access we could
  not complete in the discussion window. We list this as a next step rather than claiming coverage.

### Q1: Where do the seven dimensions come from?

They are not BODHI's five letters renamed. BODHI supplies the generation protocol; the seven
evaluation dimensions group the behaviors a calibrated clinical response must exhibit into three
functions:

- **Self-assessment:** uncertainty acknowledgment, hedging quality, specificity
- **Information-seeking:** active inquiry, context-seeking
- **Risk management:** red-flag identification, scope bounding

The selection was shaped by the practising physician co-authors on this paper, who work in the US,
Uganda, Colombia and the UK. The revision states this origin explicitly.

### Q2: How is each value operationalized in text?

We add a verbatim example per dimension so the reader can see what is scored:

- *Uncertainty acknowledgment:* "without a chest X-ray I cannot confirm..."
- *Active inquiry:* "when did the symptoms start?"
- *Context-seeking:* "I would need the patient's age and medication history"
- *Red-flag identification:* "shortness of breath with chest pain warrants immediate evaluation"
- *Scope bounding:* "diagnosis requires an examination"
- *Hedging quality:* a specific qualified statement rather than a generic disclaimer
- *Specificity:* a concrete dose or timeframe rather than "consider medication"

### W3: Clarity of results and discussion

All four changes are made:

- Vague "not statements" replaced with positive claims that state the per-dimension numbers.
- **Figure 2** split into two panels so 0 to 2 scores and percentage rates no longer share an axis.
- **Table 2** carries explicit direction arrows.
- New paragraph on clinical meaning: active inquiry is history-taking before recommending, red-flag
  identification is recognizing presentations warranting escalation, scope bounding is declining to
  advise beyond the available evidence.

### Table 2: red-flag direction

Higher is better, and the ambiguity is our fault for not labelling it. Red-flag identification
measures sensitivity to warning signs that warrant escalation. We add a footnote noting the metric
does not capture false positives, and that measuring over-flagging is necessary future work.

### Strength 4: compute comparison

We will report training and inference cost per configuration, including measured wall-clock: on a
single H100, approximately 128 minutes per seed for a 24B configuration and 39 minutes for an 8B
one, alongside the existing inference comparison (one forward pass versus two).

### What remains open

- Head-to-head comparison against prior calibration methods
- Clinical-judge robustness panel
- Completion of the three-physician validation

We state these as outstanding rather than folding them into a claim.
