Thank you for this careful and constructive review. We are glad you find epistemic calibration
important, our asymmetric cross-family grading a safeguard against using one model across pipeline
steps, and the experiment carefully constructed.

**W1. The "framework" claim is overly broad**

You are right that our two statements are in tension, and the fault is ours. If accepted we will
narrow the claim to a recipe plus an evaluation decomposition, demonstrated on open-ended,
information-seeking clinical Q&A, and state what is *not* covered: task types with no missing
information to seek, such as closed-form multiple choice, extraction, or summarization. The sentence
beginning "Any structured CoT protocol..." will be replaced with a scoped version separating what we
demonstrated from what we conjecture, and positioning the recipe against inference-time calibration
methods, which act at prompt time where we act on the weights.

We will also address how the decomposition changes across contexts, which you asked for. The
dimensions are not equally meaningful everywhere. Active inquiry and context-seeking presuppose
information that is missing and recoverable from the user, so they are informative on patient-facing
Q&A and near-vacuous on self-contained exam items, whereas scope bounding and red-flag identification
travel further, since they concern what a response commits to rather than what it asks for. The new runs
bear this out: with the base fixed at Mistral-Small-24B, active inquiry rises 7.8% to 58.6% on
ChatDoctor but only 6.5% to 26.5% on the better-specified MedQuAD.

**W2. MedGemma is not aligned with HealthBench's use case**

Your characterization of HealthBench is right, and the mismatch is real. One part of our reasoning holds:
ChatGPT, Claude and Gemini cannot be LoRA-adapted, and weight-level adaptation is the object of study,
so a closed model could not be the base. But that never required a *clinical* base model, and that is where our choice was weak.

You suggested a more widely used model or a smaller variant. **Mistral-Small-24B-Instruct** is exactly
that: general-purpose, non-clinical, widely used and open-weight. We re-ran the pipeline unchanged on
it, and on **Med42-8B**, clinically tuned from a third family (Table R1 below).

The pattern across the three is the substantive answer, not the reproduction. Base active inquiry
varies by more than a factor of two (11.8%, 17.5%, 25.6%), as one would expect if it were an
idiosyncrasy of one model's alignment. After adaptation all three converge into a narrow band (45.6%,
52.5%, 56.6%): the endpoint is a property of the training signal, not of the starting model.

**W3. Results and discussion lack clarity**

We agree on all four points, and Clarity was your lowest score. If accepted, we will:

- **Remove the vague "not statements"** you quoted, replacing them with positive claims that state the
  per-dimension numbers, including accuracy 0.117 → 0.121 and completeness 0.168 → 0.169, which support the
  communication-not-knowledge reading.
- **Add a direction arrow to every row of Table 2,** with the caption note below.
- **Add a clinical implications paragraph:** active inquiry is history-taking before recommending,
  red-flag identification is recognizing presentations that warrant escalation, and scope bounding is
  declining to advise beyond the available evidence.
- **Split Figure 2 into two panels,** the 0–2 dimensions in one and the percentage rates in the other,
  each with its own labelled axis. This is the one change that requires regenerating the figure.

*On your question of whether red-flag rate is missing a down arrow:*

Higher is better, and the ambiguity is our fault for not labelling it: it measures sensitivity to
warning signs that warrant escalation. In Table 2, blanket disclaimer rate is the only row where lower
is better, and will be the only one with a down arrow. We will add a caption note that it does
not capture false positives, so a model flagging indiscriminately would also score highly; measuring
over-flagging requires a labelled set of prompts with no genuine red flag, which we will build if
accepted.

**Q1. Where the seven dimensions came from, and whether clinicians were involved**

They are deliberately not BODHI's five letters renamed, and yes, clinicians shaped them.

BODHI (**B**ridging, **O**pen, **D**iscerning, **H**umble, **I**nquiring) is the *generation*
protocol; the seven dimensions are the *evaluation* decomposition. We kept them separate since
scoring the student on the teacher's own five categories would grade the pipeline against the
rubric it was built to satisfy. The correspondence is therefore partial: Inquiring maps onto active
inquiry and context-seeking; Humble onto hedging quality and uncertainty acknowledgment; Discerning
onto red-flag identification and specificity; Open and Bridging bear on scope bounding without owning a
dimension.

The seven cover the three epistemic functions a calibrated clinical response must serve:
*self-assessment* (uncertainty acknowledgment, hedging quality, specificity), *information-seeking*
(active inquiry, context-seeking), and *risk management* (red-flag identification, scope bounding).
The selection was shaped by the international practising physician co-authors on this paper, who identified these as
the behaviors that most affect safety in clinical decision support. We excluded candidates such as
empathy and patient-education depth because they conflate what is communicated with how accurately
uncertainty is represented. We will state the origin and exclusions explicitly.

**Q2. How each value is operationalized in text**

We will give a verbatim example per dimension so the reader sees what is scored: *uncertainty
acknowledgment* ("without a chest X-ray I cannot confirm pneumonia"); *active inquiry* ("when did the
symptoms start?"); *context-seeking* ("I would need the patient's age and medication history");
*red-flag identification* ("shortness of breath with chest pain warrants immediate evaluation");
*scope bounding* ("diagnosis requires an examination"); *hedging quality* (a qualified statement, not
a generic "I am not a doctor"); *specificity* (a concrete dose rather than "consider medication").
These anchors are scored by the Llama-3.1-8B evaluator, with a three-physician validation underway;
one of three raters has returned, giving kappa = 0.35 against the grader, below our pre-registered
target of 0.6.

**Q3. One benchmark is not sufficient; consider clinician-posed questions**

We have added two benchmarks, though they do not fully answer your ask.

Your premise holds. Auditing HealthBench-Hard for explicit clinician self-identification ("my
patient", "as a physician"), 14 of 1,000 prompts (1.4%) identify the speaker as a clinician. That
audit is a lower bound, but the benchmark is overwhelmingly patient-facing.

We added **ChatDoctor** (unedited questions real patients asked physicians online) and **MedQuAD** (NIH
consumer-health QA); the effect reproduces on both, broadening source and underspecification. **Both remain patient-facing, so they do not deliver the
clinician-perspective comparison you proposed.** The cell that would have was MedQA-USMLE reframed as
open-ended clinician questions, which we excluded: our preamble told the model to ask for information
it needed, which drove base-model active inquiry to 99% and left the cell measuring
instruction-following rather than calibration. We report this rather than omit it. It also shows that
instruction raises the asking rate without the targeting: base discrimination between prompts that
withhold information and those that do not is +2.1pp (CI contains zero) against +10.5pp for the adapter
(CI excludes zero). A rate obtained by instruction is indiscriminate; the adapter's is selective. MIMIC and eICU require
credentialed PhysioNet access we could not obtain in time. We
have therefore added benchmark diversity but not yet the clinician-posed setting, which we will add if
accepted.

**Table R1.** New runs, 5 seeds per cell, hyperparameters fixed; each cell reads Base → LoRA.
Mistral-Small-24B and BioMistral-7B are Mistral-family, Med42-8B is Llama-3; row 1 is the submitted result.

| Benchmark | Base model | Active inquiry | Context-seek |
|---|---|---|---|
| HealthBench | **MedGemma-27B (submitted)** | **17.5 → 45.6%** | **1.40 → 1.75** |
| HealthBench | Mistral-Small-24B | 25.6 → 56.6% | 1.05 → 1.76 |
| HealthBench | Med42-8B | 11.8 → 52.5% | 0.52 → 1.40 |
| ChatDoctor | Mistral-Small-24B | 7.8 → 58.6% | 0.83 → 1.49 |
| ChatDoctor | Med42-8B | 2.9 → 86.4% | 0.71 → 1.81 |
| MedQuAD | Mistral-Small-24B | 6.5 → 26.5% | 0.32 → 1.23 |
| HealthBench | BioMistral-7B | 11.7 → 10.1% (null) | 0.27 → 0.28 |

**A more explicit compute comparison**

We will report training and inference cost per configuration. One seed of a 24B
configuration end to end takes about two hours on a single H100, and the 8B ones less;
our timings came off a shared node, so we will quote per-stage figures measured in isolation rather
than numbers confounded by queueing. This sits alongside the comparison already in the paper: the CoT
protocol costs roughly 2x inference and drops ~5% of responses, whereas the adapter uses one forward
pass and reaches the same behavioral effect at under a quarter of the length overhead.

**Significance** was your other low score, and your objection was that we described one instantiation
as a framework. You were right. We have narrowed the claim to what we demonstrated, and it is
now 5 seeds across three model families and three benchmarks rather than one of each, failures reported
alongside successes.

Thank you again for your thoughtful comments, which have improved the paper. We hope the additional evidence
and changes above address your concerns. Please do let us know if any questions remain; we would be glad
to run further analyses while the discussion period is open.
