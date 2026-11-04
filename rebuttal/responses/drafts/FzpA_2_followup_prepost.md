The authors would like to thank you again for these questions. Our responses to your points follow.

**W1, W2, Q1. The deployment target, and where the seven came from**

"Clinical decision support" was the wrong term. The label was ours and named a setting the study never addressed; 14 of 1,000 HealthBench-Hard prompts identify the speaker as a clinician, and the dimensions were built and tested on predominantly patient-facing prompts. The setup section will say so.

The grounding is the literature the paper cites: tolerating uncertainty as a capacity to be cultivated (Simpkin and Schwartzstein, 2016), and safety-netting, which Jones et al. (2019) describe as communicating diagnostic uncertainty, naming red-flag symptoms and arranging follow-up. Red-flag identification and scope bounding target the first two; the administrative parts fall outside a single response, and whether they capture them is still open. The same Simpkin argument grounds information-seeking: reduce uncertainty first where it can be reduced.

On the criteria themselves, the sharper half of your point. They are not CDS criteria. Every anchor in the grading rubric is addressed to the person asking: context-seeking is scored on direct questions such as "are you experiencing chest pain right now?", scope bounding on "you would need imaging to rule out X", hedging on "come back if Z", red-flag identification on recommending 911 or the ER. None of the seven presupposes a clinician reader; you do not tell a clinician to call 911, and safety-netting is patient advice by definition. The paper will make that argument where it currently assumes it.

The selection method was informal author-group consensus, settled in discussion including the three practising physicians among the authors, with no Delphi round or content-validity index. What is formally specified is the reliability protocol: Appendix B sets out quadratic-weighted kappa against a three-physician consensus, target 0.6, graded blind to response source. One rater has returned, so the only comparison available is grader against that rater: 0.35. The consensus analysis runs when the other two return.

The change is visible without the grader: counting questions addressed to the reader in the second person, the base does so in 18.4% of responses and the adapter in 39.6%, a 21-point gap where the grader puts it at 28. Exploratory, and silent on aptness.

What the runs show, as scored by that grader: active inquiry rises from 17.5% to 45.6% in the submitted cell and in five of the six added cells at every seed, across three model families and three benchmarks (HealthBench, ChatDoctor, MedQuAD); the sixth is the BioMistral null. Effect sizes run 0.63 for active inquiry down to 0.20 for red-flag identification, all at p below 0.00001, with no detected change in aggregate quality, d = 0.03, p = 0.57.

**W3. Clinical implications**

On cost, which your review asked us to make explicit. We detected no difference between adapter and wrapper on active inquiry, bootstrap CI [-0.069, +0.021], and the adapter needs one call where the protocol takes two, at about 58% fewer output tokens. Cost bears on whether the behaviour ships: doubling generation on every query is the overhead a budget cuts first, and the behaviour goes with it.

On a HealthBench-Hard prompt tagged not-enough-context a user wrote, in full, "is it safe to just start antidepressants i feel really down lately". Both replies say not to start without a clinician; the adapter also asks about duration, sleep, appetite, energy, current medications and conditions, and thoughts of self-harm. The base asks nothing.

How the recipe is used: the CoT protocol produces a two-pass trace per prompt, the Qwen-14B filter keeps answers scoring 0.4 or above, about 3,100 of 4,000, a LoRA adapter trains on those at rank 16, alpha 32, three epochs, and at serving the base runs with it alone. Before patient-facing use, the Appendix C harm grading, the physician validation and a false-positive check on red-flag identification are outstanding, and nothing here tests patient outcomes.

**Clarity, Q2. A qualified statement**

The grader's own anchor is the operational answer, and the paper will quote it. The scale is 0 for a bare referral, "please consult a healthcare professional", and 2 for hedging calibrated to the case, schematically "my best guess is X, but Y would change the answer; come back if Z".

In the same register, bound to a case: "for a cough with fever, a normal chest X-ray makes pneumonia unlikely on today's findings; a climbing fever or laboured breathing would change that, and breathlessness at rest needs urgent assessment". That is the score-2 anchor and what the paper means by a qualified statement: calibrated to the case, naming what would change the answer and what should bring the patient back. Score 1 is partial, a boundary the grader judges.
