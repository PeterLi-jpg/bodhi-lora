The authors would like to thank you again for these questions. Our responses to your points follow.

**W1, W2, Q1. The deployment target, and where the seven came from**

"Clinical decision support" was the wrong term. The label was ours and it named a setting the study never addressed; 14 of 1,000 HealthBench-Hard prompts identify the speaker as a clinician, and the dimensions were built and tested on predominantly patient-facing prompts. The setup section will say so.

The grounding is the literature the paper cites: tolerating uncertainty as a capacity to be cultivated (Simpkin and Schwartzstein, 2016), and safety-netting, which Jones et al. (2019) describe as communicating diagnostic uncertainty, naming red-flag symptoms and arranging follow-up. Red-flag identification and scope bounding target the first two; follow-up and the administrative parts fall outside a single response, and whether the two capture those constructs is still open. Three dimensions sit under self-assessment, two under information-seeking, two under risk management. The same argument grounds information-seeking: reduce uncertainty first where it can be reduced.

The selection method was informal author-group consensus, settled in discussion including the three practising physicians among the authors, without a Delphi round or a content-validity index. What is formally specified is the reliability protocol: Appendix B sets out quadratic-weighted kappa against a three-physician consensus with a 0.6 target, graded blind to response source. One rater has returned, so the only comparison available is grader against that rater: 0.35. The consensus analysis runs when the other two return.

You noted the cross-family filter and evaluator, which removes the same-family case. The behavioural change is also visible in the raw text: counting questions addressed to the reader in the second person, the base does so in 18.4% of responses and the adapter in 39.6%. The grader puts that gap at 28 points where the text count puts it at 21, so the two measures disagree on size, and the count says nothing about whether the questions are clinically apt. It does show both models emit questions and differ in who is addressed: 27% of the base's go to the reader against 48% of the adapter's. A text match over released outputs, exploratory.

What the runs show, as scored by that grader: active inquiry rises from 17.5% to 45.6% in the submitted cell and rises in five of the six added cells at every individual seed, across three model families and three benchmarks (HealthBench, ChatDoctor, MedQuAD); the sixth cell is the BioMistral null. The four effect sizes the paper reports are 0.63, 0.44, 0.22 and 0.20 for active inquiry, context-seeking, scope bounding and red-flag identification, all at p below 0.00001, with no detected change in aggregate quality at d = 0.03, p = 0.57.

**W3. Clinical implications**

On cost, which your review asked us to make explicit. We did not detect a difference between the adapter and the wrapper on active inquiry, bootstrap CI [-0.069, +0.021], and the adapter needs one call where the protocol takes two, at roughly 58% fewer output tokens. Cost bears on whether the behaviour ships: doubling generation on every query is the overhead a budget cuts first, and the behaviour goes with it.

On one HealthBench-Hard prompt tagged not-enough-context a user wrote, in full, "is it safe to just start antidepressants i feel really down lately". Both replies say not to start without a clinician; the adapter also asks about duration, sleep, appetite, energy, current medications and conditions, and thoughts of self-harm. The base asks nothing.

How the recipe is used: the CoT protocol produces a two-pass trace per prompt, the Qwen-14B filter keeps answers scoring 0.4 or above, about 3,100 of 4,000, a LoRA adapter trains on the survivors at rank 16, alpha 32, three epochs, and at serving the base runs with it alone. Before any patient-facing use, the Appendix C harm grading, the physician validation and a false-positive check on red-flag identification are outstanding, and nothing here tests patient outcomes.

**Clarity, Q2. A qualified statement**

The grader's own anchor is the operational answer, and the paper will quote it. The scale is 0 for a bare referral such as "please consult a healthcare professional", and 2 for hedging calibrated to the case, schematically "my best guess is X, but Y would change the answer; come back if Z".

In the register of the other Q2 anchors, bound to a case: "for a cough with fever, a normal chest X-ray makes pneumonia unlikely on today's findings; a climbing fever or laboured breathing would change that, and breathlessness at rest needs urgent assessment". That is the score-2 anchor and what the paper means by a qualified statement: calibrated to the stated case, naming what would change the answer and what should bring the patient back. Score 1 is partial, a boundary that is the grader's judgment.
