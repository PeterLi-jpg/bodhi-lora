The authors would like to once again extend their sincerest gratitude to all the reviewers for their incredibly rigorous insights, as well as the AC, as the Area Chair for facilitating these thoughtful discussions. We are grateful too for the acknowledgment that the problem is meaningful and well motivated, that using models of varying capability across generation and evaluation is a strength, and that the experimentation is well outlined. The following points address the six weaknesses in the meta review, over two comments.

**W1. MedGemma does not align with HealthBench use cases.**

The scope is corrected. The setting is patient-facing clinical Q&A, and the clinical-decision-support wording is withdrawn. Weight-level adaptation is the object of study, so the base had to be open-weight; the clinically tuned choice was ours. We re-ran the pipeline unchanged on three further bases. On HealthBench, active inquiry moves 17.5 to 45.6% (MedGemma-27B, the submitted cell), 25.6 to 56.6% (Mistral-Small-24B), 11.8 to 52.5% (Med42-8B) and 11.7 to 10.1% (BioMistral-7B). Mistral-Small-24B is general-purpose and widely used, and the active-inquiry effect reproduces on it.

**W2. One pipeline, and the contribution stated as a general framework.**

The evidence now spans three model families (Gemma, Mistral, Llama-3) and three benchmarks (HealthBench, ChatDoctor, MedQuAD), five seeds per cell, every hyperparameter fixed. Active inquiry rises in five of the six added cells. The sixth is the BioMistral-7B null, which within HealthBench carries the weakest teacher, 14.5% active inquiry against 47.9 to 57.1, and the lowest filter clearance, 20.4% against 50.8 to 78. A seventh cell, MedQA on Mistral-24B, was set aside here as a single-best-answer exam task; it is the one added cell where aggregate score rose. The aggregate cost against these gains is under W5.

You and Reviewer FzpA identified language that outran the evidence. The contribution will be described as a recipe plus an evaluation decomposition for open-ended information-seeking clinical Q&A, and the sentences claiming that any CoT protocol or any PEFT method would serve are removed.

We varied the base model and the benchmark. One CoT protocol was tested, so protocol-independence remains open, and we commit to a second protocol.

**W3. Automated grading is limited.**

We share this concern. Filter and evaluator sit in different model families, which removes the most direct route to circularity. Correlated grading preferences remain possible, and clinical validity is a separate property; the paper will say both. If accepted a clinical judge joins as a robustness panel alongside the primary grader.

The validation is pre-registered in Appendix B against a three-physician consensus with a 0.6 quadratic-weighted kappa target, graded blind to response source. One of three raters has returned, and grader against that rater is 0.35. The consensus analysis runs when the other two return, whatever the outcome here.

One measurement in the record avoids the grader entirely and one uses labels it never sees, both bearing on inquiry behaviour, described under W4. On active inquiry the grader-free count agrees with the grader in direction, 21 points against 28. The remaining dimensions rest on the grader until the physician panel is complete.

**W4. Does SFT distill calibration, or mimic favourable responses?**

Reviewer LxMF proposed the test and we ran it on the submitted data, using labels the grader never sees: HealthBench theme tags and whether a prompt's rubric rewards asking. The index separates prompts that withhold information from those answerable as posed, over 667 prompts, bootstrap intervals at 5,000 resamples. The base index is +2.1pp, CI [-5.0, +8.7], an interval containing zero. The wrapper reaches +11.0pp, CI [+2.1, +19.9], and the adapter +10.5pp, CI [+1.7, +18.9], both intervals clear of zero. The adapter-minus-base contrast is +8.4pp with CI [-1.2, +17.9], which contains zero, so the size of the gain between conditions is unestablished and the within-condition results carry the weight.

A second measurement also avoids the grader: counting second-person questions, the base asks in 18.4% of responses and the adapter in 39.6%. It is a text match over released outputs, exploratory, and silent on clinical aptness.

A third result, this one scored by the grader, bears on imitation. On MedQuAD the unfiltered teacher degrades scope bounding, 1.85 to 1.61, and hedging quality, 1.46 to 1.40, while the adapter improves both, to 1.89 and 1.78. Specificity runs base 1.71, teacher 1.47, adapter 1.33, and red-flag identification base 0.57, teacher 0.74, adapter 0.68. The direction is dimension-specific.

Together these support a behavioural change carried in the weights. Latent internal uncertainty sits outside this evidence.

Part 2 follows.
