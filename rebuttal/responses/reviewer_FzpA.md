# Response to Reviewer FzpA

We thank the reviewer for a careful reading that identified a real inconsistency in our claims and
several concrete presentation failures. Rather than answer with promises, we used the discussion
period to run the experiments the review implies. We have now applied the identical pipeline —
unchanged in its four steps of generate, grade, filter, distill, and with the paper's
hyperparameters held fixed — to three model families and three benchmarks, five independent seeds
per cell. We address each point below.

## W1: The "framework" claim is overstated

We accept this criticism as written. The two statements the reviewer quotes were genuinely in
tension, and the fault was ours. Our revision resolves it in both directions.

First, we narrow the language. The contribution is a recipe plus an evaluation decomposition,
demonstrated on open-ended, information-seeking clinical question answering. We now state
explicitly what the approach does not claim to cover: task types where there is no missing
information to seek, such as closed-form multiple choice, extraction, or summarization. The
sentence beginning "Any structured CoT protocol..." is replaced with a scoped version that
distinguishes what we demonstrated from what we conjecture.

Second, the empirical basis is no longer a single point. Active inquiry rises from base to adapter
by +31pp on HealthBench with Mistral-Small-24B (25.6% to 56.6%), +41pp on HealthBench with Med42-8B
(11.8% to 52.5%), +51pp on ChatDoctor with Mistral-Small-24B (7.8% to 58.6%), +84pp on ChatDoctor
with Med42-8B (2.9% to 86.4%), and +20pp on MedQuAD with Mistral-Small-24B (6.5% to 26.5%).
Context-seeking moves the same way in every cell. In four of the five the adapter matches or exceeds
its own teacher while requiring no chain-of-thought at inference.

We also report a boundary condition rather than only successes. BioMistral-7B shows no effect
(11.7% to 10.1%). The diagnostic detail is that the teacher fails there too — the CoT wrapper
reaches only 14.5% on that model — so there is nothing to distill. The recipe requires a base model
capable of following the structured protocol, and we now state that as a precondition.

We would rather claim less and show more, and we think the revised framing does that.

## W2: MedGemma is not aligned with HealthBench's use case

The reviewer is right, and our original defence — that commercial models cannot be LoRA-adapted —
answered a different question than the one asked. The substantive answer is an experiment.
Mistral-Small-24B-Instruct is a general-purpose, non-clinical, widely used open-weight model, and it
reproduces the paper's effect on HealthBench: active inquiry 25.6% to 56.6%, context-seeking 1.05 to
1.76, red-flag identification 1.03 to 1.67, scope bounding 1.77 to 1.97. The result is therefore not
an artifact of applying a clinically tuned model to consumer-style questions. We have also added
Med42-8B, a clinically tuned model from a third family (Llama-3), so the comparison now spans
clinical and general models across three families rather than one.

## Q3: One benchmark is not enough; consider clinician-posed questions from records

We agree, and we checked the specific premise before responding. In HealthBench-Hard, only about
0.6% of the evaluated prompts explicitly identify the speaker as a clinician, which confirms the
reviewer's characterization rather than disputing it: the original evaluation is essentially all
patient-style conversation.

We have widened the evaluation on both sides. On the patient-facing side we added ChatDoctor
(HealthCareMagic), which consists of unedited questions real people asked physicians online and
which routinely omit age, duration, severity and concurrent medications; this is the benchmark where
the effect is largest (+84pp with Med42-8B), which is what one would predict if the behavior being
learned is genuinely context-seeking. We also added MedQuAD, NIH consumer-health question answering
from a different source than HealthBench. On the clinician-facing side we added MedQA-USMLE reframed
as open-ended questions with the answer options stripped; that cell is still running and will appear
in the camera-ready.

We were not able to use MIMIC or eICU. The derived text corpora require credentialed PhysioNet
access under a data use agreement we could not complete within the discussion period. We name this
as a concrete next step rather than claiming coverage we do not have.

## Q1: Where do the seven dimensions come from, and did clinicians provide input?

The decomposition is not BODHI's five letters renamed. BODHI supplies the generation protocol; the
seven evaluation dimensions were derived by grouping the behaviors a calibrated clinical response
must exhibit into three functions: self-assessment (uncertainty acknowledgment, hedging quality,
specificity), information-seeking (active inquiry, context-seeking), and risk management (red-flag
identification, scope bounding). The selection was shaped by the practising physician co-authors on
this paper, who work in the United States, Uganda, Colombia and the United Kingdom, and who
identified these as the safety-critical behaviors for clinical decision support. The camera-ready
adds a subsection stating this origin explicitly.

## Q2: How is each value operationalized in the text?

We will give a verbatim example of each dimension so the reader can see exactly what is scored:
uncertainty acknowledgment ("without a chest X-ray I cannot confirm..."), active inquiry ("when did
the symptoms start?"), context-seeking ("I would need the patient's age and medication history"),
red-flag identification ("shortness of breath with chest pain warrants immediate evaluation"), scope
bounding ("I can give general information, but diagnosis requires an examination"), hedging quality
(a specific qualified statement rather than a generic "consult a doctor" disclaimer), and
specificity (a concrete dose or timeframe rather than "consider medication").

## W3: Results and discussion lack clarity

We have made all four changes. The "not statements" are gone; the claim now reads positively —
the adapter shifts specific epistemic behaviors while preserving aggregate clinical quality — with
the per-dimension numbers stated rather than gestured at. Figure 2 is split into two panels so that
0–2 scores and percentage rates no longer share an axis. Table 2 carries explicit direction arrows.
We have added a paragraph on what each behavior means clinically: that active inquiry is
history-taking before recommending, that red-flag identification is recognizing presentations that
warrant escalation, that scope bounding is declining to advise beyond the available evidence.

## W3 (Table 2): Is red-flag rate missing a down arrow?

No — higher is better, and the ambiguity is our fault for not labelling it. Red-flag identification
measures sensitivity to genuine clinical warning signs that warrant escalation, so a higher score is
the desirable direction. We add a footnote stating plainly that the metric does not capture
false-positive rates, and that measuring over-flagging is necessary future work.

## Strength 4: A more explicit compute comparison

We will report training and inference cost per configuration explicitly, including measured
wall-clock from the new runs. On a single H100, a 24B cell takes approximately 128 minutes per seed
and an 8B cell approximately 39 minutes, of which training is about 93 and 30 minutes respectively
and the remainder is the four-condition evaluation. This sits alongside the existing inference-cost
comparison, where the adapter delivers wrapper-equivalent behavior in one forward pass instead of
two.

## What we do not claim

The new runs do not resolve everything the review raises. We have not yet completed a head-to-head
comparison against prior calibration methods, nor the clinical-judge robustness panel, nor the full
three-physician validation. We state these as outstanding rather than folding them into a claim.
