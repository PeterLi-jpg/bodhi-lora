Thank you for your thorough review. We are pleased you find epistemic calibration important, our asymmetric cross-family grading a real safeguard against using
one model across pipeline steps, and the experiment carefully built.

**W1. The "framework" claim is overly broad**

You are right that our two statements are in tension, and the fault is ours. If accepted we will
narrow the claim
to a recipe plus an evaluation decomposition, demonstrated on open-ended, information-seeking clinical
Q&A, and will state what is *not* covered: task types with no missing information to seek, such as
closed-form multiple choice, extraction, or summarization. The sentence beginning "Any structured CoT
protocol..." will be replaced with a scoped version separating what we demonstrated from what we
conjecture, and positioned against inference-time calibration methods, which act at prompt time where we act on weights.

We will also address how the decomposition changes across contexts, which you asked for. The
dimensions are not equally meaningful everywhere: active inquiry and context-seeking presuppose
information that is missing and recoverable from the user, so they are informative on patient-facing
Q&A and near-vacuous on self-contained exam items, whereas scope bounding and red-flag identification
travel further, since they concern what a response commits to rather than what it asks for. The new
runs bear this out: with the base model fixed at Mistral-Small-24B, active inquiry rises 7.8% to 58.6% on
ChatDoctor but only 6.5% to 26.5% on the better-specified MedQuAD. It tracks whether asking is
warranted rather than rising uniformly, which separates calibration from a template.

**W2. MedGemma is not aligned with HealthBench's use case**

Your characterization of HealthBench is right and the mismatch is real. One part of our reasoning holds: ChatGPT, Claude and Gemini cannot be LoRA-adapted, and weight-level adaptation is the
object of study, so a closed model could not be the base. But that never required a *clinical* base model, which is where our choice was weak.

You suggested a more widely used model or a smaller variant. **Mistral-Small-24B-Instruct** is exactly
that: general-purpose, non-clinical, widely used, open-weight. We re-ran the pipeline unchanged on it,
and on **Med42-8B**, clinically tuned from a third family (Table R1).

The pattern across the three is the substantive answer, not just the reproduction. Base active inquiry
varies by more than a factor of two (11.8%, 17.5%, 25.6%), as one would expect if the
behavior were an idiosyncrasy of one model's alignment. After adaptation all three converge into a
narrow band (45.6%, 52.5%, 56.6%). The endpoint is a property of the training signal rather than of the
starting model, which is what your critique put in doubt.

**Q3. One benchmark is not sufficient; consider clinician-posed questions**

We added two benchmarks, and want to be straight that they do not fully answer it.

Your premise holds: auditing HealthBench-Hard for explicit clinician self-identification ("my patient",
"as a physician"), 14 of 1,000 prompts (1.4%) identify the speaker as a clinician. The keyword audit is a
lower bound, but the benchmark is overwhelmingly patient-facing, as you suspected.

We added **ChatDoctor** (unedited questions real patients asked physicians online) and **MedQuAD** (NIH
consumer-health QA); the effect reproduces on both, broadening source and underspecification.
**Both are still patient-facing, so they do not deliver the clinician-perspective comparison you
proposed.** The cell that would have was MedQA-USMLE reframed as open-ended clinician questions, which we
excluded: our reframing preamble told the model to ask for information it needed, which drove
base-model active inquiry to 99% and left the cell measuring instruction-following rather than
calibration. We report it rather than quietly drop it. MIMIC and eICU need credentialed PhysioNet
access under a data use agreement we could not complete in time. So we added benchmark diversity but not yet
the clinician-posed setting.

**Table R1.** New runs, 5 seeds per cell, hyperparameters fixed; each cell reads Base → LoRA.
Mistral-Small-24B and BioMistral-7B are Mistral-family, Med42-8B Llama-3; row 1 is the submitted result.

| Benchmark | Base model | Active inquiry | Context-seek |
|---|---|---|---|
| HealthBench | **MedGemma-27B (submitted)** | **17.5 → 45.6%** | **1.40 → 1.75** |
| HealthBench | Mistral-Small-24B | 25.6 → 56.6% | 1.05 → 1.76 |
| HealthBench | Med42-8B | 11.8 → 52.5% | 0.52 → 1.40 |
| ChatDoctor | Mistral-Small-24B | 7.8 → 58.6% | 0.83 → 1.49 |
| ChatDoctor | Med42-8B | 2.9 → 86.4% | 0.71 → 1.81 |
| MedQuAD | Mistral-Small-24B | 6.5 → 26.5% | 0.32 → 1.23 |
| HealthBench | BioMistral-7B | 11.7 → 10.1% (null) | 0.27 → 0.28 |

The BioMistral-7B row is a scope condition we report rather than hide: the teacher fails there too
(wrapper 14.5%), so the recipe needs a base capable of following the protocol.

**Q1. Where the seven dimensions came from, and whether clinicians were involved**

They are deliberately not BODHI's five letters renamed, and yes, clinicians shaped them.

BODHI (**B**ridging, **O**pen, **D**iscerning, **H**umble, **I**nquiring) is the *generation* protocol;
the seven dimensions are the *evaluation* decomposition. We kept them separate deliberately:
scoring the student on the teacher's own five categories would grade the pipeline against the rubric it
was built to satisfy. The correspondence is partial: Inquiring maps onto active inquiry and
context-seeking; Humble onto hedging and uncertainty acknowledgment; Discerning onto red-flag
identification and specificity; Open and Bridging bear on scope bounding without owning a dimension.

The seven instead cover the three epistemic functions a calibrated clinical response must serve:
*self-assessment* (uncertainty acknowledgment, hedging quality, specificity), *information-seeking*
(active inquiry, context-seeking), and *risk management* (red-flag identification, scope bounding). The
selection was shaped by the practising physician co-authors on this paper, who work in the US, Uganda,
Colombia and the UK, and who named these as the safety-critical behaviors. We excluded candidates such as empathy and patient-education depth because they conflate what
is communicated with how accurately uncertainty is represented. We will state the origin and the exclusions explicitly.

**Q2. How each value is operationalized in text**

We will give a verbatim example per dimension so the reader sees what is scored: *uncertainty
acknowledgment* ("without a chest X-ray I cannot confirm pneumonia"); *active inquiry* ("when did the
symptoms start?"); *context-seeking* ("I would need the patient's age and medication history");
*red-flag identification* ("shortness of breath with chest pain warrants immediate evaluation"); *scope
bounding* ("diagnosis requires an examination"); *hedging quality* (a specific qualified statement, not
a generic "I am not a doctor"); *specificity* (a concrete dose rather than "consider medication").
These anchors are scored by the Llama-3.1-8B evaluator, with a three-physician validation underway.

**W3. Results and discussion lack clarity**

We agree on all four points, and Clarity was your lowest score, so we treat these as required rather
than optional. If accepted we will:

- **Remove the vague "not statements"** you quoted, replacing them with positive claims stating the
  per-dimension numbers, including accuracy 0.117 → 0.121 and completeness 0.168 → 0.169, which actually support the communication-not-knowledge reading.
- **Add a direction arrow to every row of Table 2,** with the caption note below.
- **Add clinical implications:** active inquiry is history-taking before recommending, red-flag
  identification is recognizing presentations that warrant escalation, scope bounding is declining to
  advise beyond the available evidence.
- **Split Figure 2 into two panels,** 0–2 dimensions in one and percentage rates in the other, each
  with its own labelled axis. This is the one that requires regenerating the figure.

**Is red-flag rate missing a down arrow?**

Higher is better, and the ambiguity is our fault for not labelling it: the dimension measures
sensitivity to warning signs that warrant escalation. In Table 2, blanket disclaimer rate is the only
row where lower is better, and it will be the only one with a down arrow. We will add a caption note
that it does not capture false positives, so a model that flagged indiscriminately
would also score highly; measuring over-flagging needs a labelled set of prompts containing no genuine
red flag, which we leave to future work.

**A more explicit compute comparison**

We will report training and inference cost per configuration, as you suggested. From these runs one
seed of a 24B configuration end to end takes about 2 hours on a single H100, the 8B ones less; our timings came off a shared node, so we will quote per-stage figures measured in isolation
rather than numbers confounded by queueing. This
sits alongside the existing one-pass-versus-two comparison, where the adapter is wrapper-equivalent at
~58% lower output cost.

Nothing above is claimed as done that is not. Of what you asked for, three things are not yet in hand,
and if accepted we commit to all three: the clinician-posed benchmark rebuilt with a neutral preamble,
the training and inference cost table, and the regenerated two-panel Figure 2.

On your two lowest scores. **Clarity**: the four fixes above are the ones you specified, and we treat
them as required. **Significance**: your objection was that we called one
instantiation a framework, and you were right. We have narrowed the claim to what we actually
demonstrated, and it is now 5 seeds across three model families and three benchmarks rather
than one of each, failures reported alongside successes. We hope you will reconsider the rating
in that light.
