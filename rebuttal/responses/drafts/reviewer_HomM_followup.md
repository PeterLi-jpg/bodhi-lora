Thanks for coming back with specifics. Answering under your headings, with all three tables.

**W1, Q1 and Q2. The evaluation is too narrow**

On "scope condition": poor wording. We ran that cell to find where the recipe stops working, and it stopped: active inquiry did not improve, falling from 11.7% to 10.1%. What we should have written is a hypothesis rather than a condition we had established. The recipe distills a teacher, so it plausibly needs the base model under the CoT protocol to produce filterable demonstrations at a usable rate, and BioMistral is where that broke down. Its teacher barely moves on active inquiry, 11.7% to 14.5%, where Mistral-24B goes 25.6% to 54.2%, and only 20.4% of its traces clear the filter against 50.8 to 78 for the other three. There is proportionally much less there to distill from. One null can't separate teacher strength from model size, instruction tuning, or a configuration that just doesn't suit a 7B. One useful next test would be a strong teacher on a small base, and we haven't run it.

On convergence: it shouldn't be expected, and **we withdraw the sentence**. It claimed a prediction we never derived, and it doesn't hold on our own numbers either. We computed that spread over the three cells where transfer worked and left the null out. Put all four HealthBench bases in and adapted active inquiry runs 10.1% to 56.6%, against a baseline spread of 11.7% to 25.6%. It widens. We withdraw it wherever we made it.

One thing does speak to your point about size. If parameter count set the endpoint, the adapted rates should order by it, and **they don't**. Med42-8B lands at 52.5%, Mistral-24B at 56.6%, MedGemma-27B at 45.6%, so the largest of the three HealthBench bases that transfer is the lowest, and BioMistral-7B at 10.1% is the smallest and also the null. Four bases, no causal claim, but it is the part of your objection we can answer with data.

On the missing tables: our fault for describing those results in prose. Both are below, with a third for the aggregate paragraph.

Table 1. MedQuAD, Mistral-Small-24B, dimensions on a 0 to 2 scale. Source for the "student does not copy the teacher indiscriminately" paragraph. Means over five seeds, 200 prompts per seed per condition, with cross-seed standard deviations.

| Dimension | Base | Base+CoT (teacher) | LoRA (adapter) |
|---|---|---|---|
| Scope bounding | 1.85 ± 0.02 | 1.61 ± 0.03 | 1.89 ± 0.02 |
| Hedging quality | 1.46 ± 0.01 | 1.40 ± 0.03 | 1.78 ± 0.03 |
| Context-seeking | 0.32 ± 0.02 | 0.87 ± 0.02 | 1.23 ± 0.05 |
| Red-flag identification | 0.57 ± 0.02 | 0.74 ± 0.02 | 0.68 ± 0.03 |
| Uncertainty ack. | 1.87 ± 0.01 | 1.78 ± 0.02 | 1.98 ± 0.00 |
| Specificity | 1.71 ± 0.01 | 1.47 ± 0.03 | 1.33 ± 0.03 |
| Active inquiry (%) | 6.5 ± 1.6 | 23.1 ± 1.2 | 26.5 ± 2.3 |

The first two rows are the ones we quoted: the teacher makes scope bounding and hedging worse, the adapter makes both better. Small fix, we wrote the hedging base as 1.45 and the unrounded value is 1.455, so 1.46. The rest of the table shows something we didn't quote, which is that specificity goes the other way, 1.71 to 1.33, which belonged next to them. So the table shows the change is selective, but it doesn't show which part of the pipeline produces it. An unfiltered-SFT control would isolate the filter's part of it, and that is the next ablation to run.

Table 2. Teacher behaviour and filter yield across all cells. Source for the BioMistral hypothesis. Row 1 is the submitted cell; the clearance column is a pooled retention fraction, since trace generation is shared across seeds.

| Benchmark | Base model | Teacher active inquiry (%) | Traces clearing filter (%) | Base to LoRA active inquiry (%) |
|---|---|---|---|---|
| HealthBench | MedGemma-27B (submitted) | 47.9 | ~78 | 17.5 to 45.6 |
| HealthBench | Mistral-Small-24B | 54.2 | 62.1 | 25.6 to 56.6 |
| HealthBench | Med42-8B | 57.1 | 50.8 | 11.8 to 52.5 |
| HealthBench | BioMistral-7B | 14.5 | 20.4 | 11.7 to 10.1 |
| ChatDoctor | Mistral-Small-24B | 54.5 | 97.6 | 7.8 to 58.6 |
| ChatDoctor | Med42-8B | 83.4 | 96.0 | 2.9 to 86.4 |
| MedQuAD | Mistral-Small-24B | 23.1 | 78.4 | 6.5 to 26.5 |

Within HealthBench, which is the clean comparison, BioMistral has the weakest teacher at 14.5% against 47.9 to 57.1, and the lowest clearance at 20.4% against 50.8 to 78. It is also the only cell where active inquiry did not rise. Across benchmarks the teacher rates aren't directly comparable. The submitted cell's clearance is the paper's roughly 3,100 traces kept of 4,000.

One correction to the paragraph above that table, since Table 2 makes it checkable. We wrote that the adapter matches or exceeds its teacher in four of five cells and never said what that counts or which cells. It counts active inquiry over the five added cells that transfer, so excluding the BioMistral null. Of those five the adapter is ahead in four, and the exception is Med42-8B on HealthBench. Table 2 above has seven rows, and across all of them it is four of seven: the submitted MedGemma cell (47.9 to 45.6) and BioMistral (14.5 to 10.1) are also behind their teachers. On context-seeking, the other column of Table R1 in our earlier reply, the adapter is ahead in one of those same five, on MedQuAD by 0.36, and behind in the other four by 0.04, 0.07, 0.10 and 0.01.

The trade is what we should have led with. On context-seeking, in the four cells where it trails, the adapter gives up at most 0.10 on a 0 to 2 scale, and the largest adapter-to-teacher shortfall in Table 1 is specificity, 1.47 against 1.33. What it drops is a protocol that, in the submitted MedGemma cell where we measured it, costs roughly twice the inference and returns nothing at all on about 5% of prompts, mostly because two-pass generation runs past the context window. The adapter reduces that rather than removing it: pooled over five seeds the wrapper returns 947 responses out of 1,000 and the adapter returns 985, against 998 for the unadapted base. Trading those losses for half the inference and under a third of the missing answers looks like a good trade to us, with both savings measured in the submitted cell and the losses in the added ones. We are not claiming the adapter uniformly beats the wrapper. We are claiming it does the job without one.

Table 3. Aggregate rubric score, all four conditions, for the six added cells. Source for the last paragraph.

| Benchmark | Base model | Base | Base+CoT | LoRA | LoRA+CoT |
|---|---|---|---|---|---|
| HealthBench | Mistral-Small-24B | 0.425 | 0.416 | 0.411 | 0.404 |
| HealthBench | Med42-8B | 0.405 | 0.391 | 0.392 | 0.382 |
| HealthBench | BioMistral-7B | 0.378 | 0.363 | 0.363 | 0.362 |
| ChatDoctor | Mistral-Small-24B | 0.557 | 0.550 | 0.541 | 0.448 |
| ChatDoctor | Med42-8B | 0.498 | 0.497 | 0.457 | 0.461 |
| MedQuAD | Mistral-Small-24B | 0.672 | 0.542 | 0.586 | 0.505 |

Base to LoRA falls in all six of these cells, by 0.013 to 0.015 on HealthBench and by 0.016 to 0.086 on ChatDoctor and MedQuAD. That is a change from the submitted cell, where the adapter came out slightly above base, 0.459 to 0.463, so we should not describe these as in line with it. On MedQuAD the teacher falls further than the adapter, 0.542 against 0.586, without touching any weights, which fits reading part of the drop as a rubric artefact, if scoring agreement with a reference answer penalises asking. We can't separate that from a real quality cost with what we have, so both the numbers and the ambiguity are there.

One cell is missing on purpose. We also ran MedQA on Mistral-24B and left it out because it is multiple choice: nothing is withheld, so asking is not the right behaviour there. For the record its aggregate rises, 0.665 to 0.683, and its inquiry rate moves the other way.

**W3 and Q4. Automatic grading, and why the graders are not clinically fine-tuned**

Agreed, and your framing is the right one. Whether a clinically tuned judge would score these dimensions differently is untested. Why a non-clinical one might do just as well isn't understood. And your practical question, whether the extra cost of a clinical judge buys anything, isn't something this study can answer. The cross-family split keeps the filter and the grader in separate model families, which removes the most direct route to circularity without ruling out correlated grading preferences. It doesn't buy clinical validity either. If the paper is accepted, both go in as limitations rather than as settled choices.

**W4 and Q3. Comparison with prior work**

Understood, and thank you for treating it as a justified scope limitation. You meant an empirical comparison against one or two of the closest published methods, not whether related work appears in the manuscript, and we had read it too narrowly. If the paper is accepted we will state the limitation plainly and name that comparison as the next experiment instead of folding it into a claim.

**On workshop versus conference scope**

You were explicit that the rating tracked evaluation breadth, so here is what has changed since it was written: seven model-benchmark cells spanning four base models, three families and three benchmarks, five seeds each, a documented null with the teacher and filter numbers that bound it, and a further analysis that corrected a mechanism claim of ours. Your questions made us withdraw the convergence sentence, put a scope on the four-of-five count, and correct how we described the aggregate scores. We would be grateful if you would reconsider the rating in that light.

If the code and per-cell outputs behind these tables would help, the rules let us send an anonymized link to the AC where a reviewer has asked for code, and we are glad to do that. The same goes for any other cut of the data, per seed or on a cell not shown here, while the discussion period is open.
