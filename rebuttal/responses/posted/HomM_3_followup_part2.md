Continued from the previous comment.

Table 3. Aggregate rubric score, all four conditions, for the six added cells. Source for the last paragraph of our earlier W1 answer, the one on aggregate quality.

| Benchmark | Base model | Base | Base+CoT | LoRA | LoRA+CoT |
|---|---|---|---|---|---|
| HealthBench | Mistral-Small-24B | 0.425 ± 0.004 | 0.416 ± 0.001 | 0.411 ± 0.006 | 0.404 ± 0.011 |
| HealthBench | Med42-8B | 0.405 ± 0.003 | 0.391 ± 0.006 | 0.392 ± 0.004 | 0.382 ± 0.007 |
| HealthBench | BioMistral-7B | 0.378 ± 0.003 | 0.363 ± 0.001 | 0.363 ± 0.007 | 0.362 ± 0.003 |
| ChatDoctor | Mistral-Small-24B | 0.557 ± 0.004 | 0.550 ± 0.006 | 0.541 ± 0.006 | 0.448 ± 0.015 |
| ChatDoctor | Med42-8B | 0.498 ± 0.003 | 0.497 ± 0.004 | 0.457 ± 0.007 | 0.461 ± 0.005 |
| MedQuAD | Mistral-Small-24B | 0.672 ± 0.003 | 0.542 ± 0.005 | 0.586 ± 0.022 | 0.505 ± 0.017 |

Base to LoRA falls in all six, by 0.013 to 0.015 on HealthBench and 0.016 to 0.086 on ChatDoctor and MedQuAD. Cross-seed SDs are now in the table. A paired t-test over the five seeds puts all six drops at two-sided p below 0.02, and all six survive Holm correction across the six tests; the tightest is BioMistral at p = 0.016. So even the small HealthBench losses are consistent rather than seed noise. That differs from the submitted cell, where the adapter came out slightly above base, 0.459 to 0.463, so we should not call these in line with it. The LoRA+CoT column shows the stacking interference the paper reports, here in aggregate score and in five of the six cells, worst on ChatDoctor with Mistral-24B and on MedQuAD; the adapter is meant to be used without the protocol, not with it.

We had read part of that as a rubric artefact, on the grounds that scoring agreement with a reference answer should penalise asking instead of answering. We tested it and it does not hold. Within the adapter's own outputs, the prompts it asks on do not score consistently lower than the ones it answers directly: that gap is positive in three cells and negative in three, and in one of the three it is within 0.002 of zero. Those prompts are self-selected, so on its own that is suggestive rather than decisive. The clearest case is BioMistral, where active inquiry barely moves at all, 11.7% to 10.1%, and aggregate still falls by 0.015. A penalty on asking cannot explain a drop in a cell where the asking did not change. So we withdraw that reading as well, and report the cost as it stands: consistent, small on HealthBench at 0.013 to 0.015, and largest on MedQuAD.

The trade, then. Active inquiry rises from 17.5% to 45.6% in the submitted cell, and by more than that in four of the six added ones. The aggregate cost against it is 0.013 to 0.015 on HealthBench and 0.016 to 0.086 on the other two, largest on MedQuAD. Those costs are measured against base. Against the wrapper, which is the alternative you would actually run, the adapter is lower on aggregate in three of the six added cells, level in two and higher in one, MedQuAD by 0.044. The wrapper is itself below base in all six, so neither route is free. What that buys is the behaviour surviving in the weights: one forward pass instead of the protocol's two, and no CoT scaffold at inference. In the submitted cell that scaffold also returned nothing at all on about 5% of prompts. We are not claiming the adapter uniformly beats the wrapper. We are claiming that where it transfers, it does the job without one.

**W3 and Q4. Automatic grading, and why the graders are not clinically fine-tuned**

Agreed on all three points. Whether a clinically tuned judge would score these dimensions differently is untested. Why a non-clinical one might do as well isn't understood. And whether the extra cost buys anything isn't something this study can answer. The cross-family split removes the most direct route to circularity without ruling out correlated grading preferences, and it doesn't buy clinical validity. Both are limitations of the design rather than settled choices, and the paper will say so.

**W4 and Q3. Comparison with prior work**

Understood, and thank you for treating it as a justified scope limitation. You meant an empirical comparison against one or two of the closest published methods, and we read it too narrowly. If the paper is accepted we will state the limitation plainly and name that comparison as the next experiment instead of folding it into a claim.

**On workshop versus conference scope**

Your questions made us withdraw the convergence sentence, put a scope on the four-of-five count, and correct how we described the aggregate scores. The tables are the rest of the answer, and we hope they give you what you needed to place the work.

If the code and per-cell outputs would be useful, the rules allow an anonymized link to the AC on request.
