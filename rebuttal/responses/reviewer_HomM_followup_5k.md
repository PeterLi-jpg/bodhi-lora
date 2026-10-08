Thanks for coming back with specifics. Answering under your headings, with all three tables.

**W1, Q1 and Q2. The evaluation is too narrow**

On "scope condition": poor wording. That cell is where the recipe stopped working, and active inquiry did not improve, 11.7% to 10.1%. What we should have written is a hypothesis, not a condition we had established. The recipe distills a teacher, so it plausibly needs the base under CoT to produce filterable demonstrations at a usable rate, and BioMistral is where that broke down (Table 2). One null can't separate teacher strength from model size, instruction tuning, or a configuration that doesn't suit a 7B.

On convergence: it shouldn't be expected, and **we withdraw the sentence**. It claimed a prediction we never derived, and it doesn't hold on our own numbers. We computed that spread over the three cells where transfer worked and left the null out. With all four HealthBench bases, adapted active inquiry runs 10.1% to 56.6% against a baseline spread of 11.7% to 25.6%. It widens. We withdraw it wherever we made it.

On size: if parameter count set the endpoint, the adapted rates should order by it, and **they don't**. Med42-8B 52.5%, Mistral-24B 56.6%, MedGemma-27B 45.6%, so the largest of the three HealthBench bases that transfer is the lowest. No causal claim on four bases, but it is the part of your objection we can answer with data.

On the missing tables: our fault for describing those results in prose.

Table 1. MedQuAD, Mistral-Small-24B, 0 to 2 scale. Means over five seeds, 200 prompts per seed per condition, cross-seed SDs.

| Dimension | Base | Base+CoT (teacher) | LoRA (adapter) |
|---|---|---|---|
| Scope bounding | 1.85 ± 0.02 | 1.61 ± 0.03 | 1.89 ± 0.02 |
| Hedging quality | 1.46 ± 0.01 | 1.40 ± 0.03 | 1.78 ± 0.03 |
| Context-seeking | 0.32 ± 0.02 | 0.87 ± 0.02 | 1.23 ± 0.05 |
| Red-flag identification | 0.57 ± 0.02 | 0.74 ± 0.02 | 0.68 ± 0.03 |
| Uncertainty ack. | 1.87 ± 0.01 | 1.78 ± 0.02 | 1.98 ± 0.00 |
| Specificity | 1.71 ± 0.01 | 1.47 ± 0.03 | 1.33 ± 0.03 |
| Active inquiry (%) | 6.5 ± 1.6 | 23.1 ± 1.2 | 26.5 ± 2.3 |

We quoted the first two rows: the teacher makes scope bounding and hedging worse, the adapter makes both better. We did not quote specificity, which goes the other way, 1.71 to 1.33, and belonged next to them. The table shows the change is selective but not which part of the pipeline produces it. An unfiltered-SFT control would isolate the filter, and that is the next ablation to run.

Table 2. Teacher behaviour and filter yield, all cells. Row 1 is the submitted cell; clearance is pooled, since trace generation is shared across seeds.

| Benchmark | Base model | Teacher active inquiry (%) | Traces clearing filter (%) | Base to LoRA active inquiry (%) |
|---|---|---|---|---|
| HealthBench | MedGemma-27B (submitted) | 47.9 | ~78 | 17.5 to 45.6 |
| HealthBench | Mistral-Small-24B | 54.2 | 62.1 | 25.6 to 56.6 |
| HealthBench | Med42-8B | 57.1 | 50.8 | 11.8 to 52.5 |
| HealthBench | BioMistral-7B | 14.5 | 20.4 | 11.7 to 10.1 |
| ChatDoctor | Mistral-Small-24B | 54.5 | 97.6 | 7.8 to 58.6 |
| ChatDoctor | Med42-8B | 83.4 | 96.0 | 2.9 to 86.4 |
| MedQuAD | Mistral-Small-24B | 23.1 | 78.4 | 6.5 to 26.5 |

Within HealthBench, BioMistral has the weakest teacher, 14.5% against 47.9 to 57.1, the lowest clearance, 20.4% against 50.8 to 78, and is the only cell where active inquiry did not rise.

One correction to the paragraph above that table. We wrote that the adapter matches or exceeds its teacher in four of five cells without saying what that counted. It counts active inquiry over the five added cells that transfer, excluding the BioMistral null. Of those the adapter is ahead in four, the exception being Med42-8B on HealthBench. Across all seven rows above it is four of seven.

Table 3. Aggregate rubric score, all four conditions, six added cells.

| Benchmark | Base model | Base | Base+CoT | LoRA | LoRA+CoT |
|---|---|---|---|---|---|
| HealthBench | Mistral-Small-24B | 0.425 | 0.416 | 0.411 | 0.404 |
| HealthBench | Med42-8B | 0.405 | 0.391 | 0.392 | 0.382 |
| HealthBench | BioMistral-7B | 0.378 | 0.363 | 0.363 | 0.362 |
| ChatDoctor | Mistral-Small-24B | 0.557 | 0.550 | 0.541 | 0.448 |
| ChatDoctor | Med42-8B | 0.498 | 0.497 | 0.457 | 0.461 |
| MedQuAD | Mistral-Small-24B | 0.672 | 0.542 | 0.586 | 0.505 |

Base to LoRA falls in all six, by 0.013 to 0.015 on HealthBench and 0.016 to 0.086 on ChatDoctor and MedQuAD. That differs from the submitted cell, where the adapter came out slightly above base, 0.459 to 0.463, so we should not call these in line with it. On MedQuAD the teacher falls further than the adapter, 0.542 against 0.586, without touching any weights, which fits part of the drop being a rubric artefact if scoring against a reference answer penalises asking. We can't separate that from a real quality cost with what we have.

One cell is missing on purpose: MedQA on Mistral-24B, left out because it is multiple choice, so nothing is withheld and asking is not the right behaviour there. Its aggregate rises, 0.665 to 0.683, and its inquiry rate moves the other way.

**W3 and Q4. Automatic grading, and why the graders are not clinically fine-tuned**

Agreed, and your framing is the right one. Whether a clinically tuned judge would score these dimensions differently is untested, why a non-clinical one might do as well isn't understood, and whether the extra cost buys anything isn't something this study can answer. The cross-family split removes the most direct route to circularity without ruling out correlated grading preferences, and it doesn't buy clinical validity. If the paper is accepted, both go in as limitations rather than as settled choices.

**W4 and Q3. Comparison with prior work**

Understood, and thank you for treating it as a justified scope limitation. You meant an empirical comparison against one or two of the closest published methods, and we read it too narrowly. If the paper is accepted we will state the limitation plainly and name that comparison as the next experiment instead of folding it into a claim.

**On workshop versus conference scope**

Since the rating was written: seven model-benchmark cells over four base models, three families and three benchmarks, five seeds each, and a documented null with the teacher and filter numbers that bound it. Your questions made us withdraw the convergence sentence, put a scope on the four-of-five count, and correct how we described the aggregate scores. We would be grateful if you would reconsider the rating in that light.

If the code and per-cell outputs behind these tables would help, the rules let us send an anonymized link to the AC where a reviewer has asked for code, and we are glad to do that.
