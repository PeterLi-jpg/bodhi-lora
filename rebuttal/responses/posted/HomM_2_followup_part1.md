Thanks for coming back with specifics. Answering under your headings, with all three tables. The comment box caps at 5,000 characters, so this is in two parts.

**W1, Q1 and Q2. The evaluation is too narrow**

On "scope condition": poor wording. That cell is where the recipe stopped working, and active inquiry did not improve, 11.7% to 10.1%. What we should have written is a hypothesis, not a condition we had established. The recipe distills a teacher, so it plausibly needs the base under CoT to produce filterable demonstrations at a usable rate. BioMistral is where that broke down, and Table 2 shows it. One null can't separate teacher strength from model size, instruction tuning, or a configuration that doesn't suit a 7B.

On convergence: it shouldn't be expected, and **we withdraw the sentence**. It claimed a prediction we never derived, and it doesn't hold on our own numbers. We computed that spread over the three cells where transfer worked and left the null out. With all four HealthBench bases, adapted active inquiry runs 10.1% to 56.6% against a baseline spread of 11.7% to 25.6%. It widens. On the three that transfer it does narrow, 13.8 points of base spread to 11.0, and that is what we originally reported. The mistake was stating it as a general property with the null left out. We withdraw it wherever we made it.

On size: if parameter count set the endpoint, the adapted rates should order by it, and **they don't**. Med42-8B 52.5%, Mistral-24B 56.6%, MedGemma-27B 45.6%, so the largest of the three that transfer is the lowest. Including BioMistral the sequence by size is 10.1, 52.5, 56.6, 45.6, which fits a floor below 8B but not size setting the endpoint above it. No causal claim on four bases.

On the missing tables: our fault for describing those results in prose. We ran seven added cells and report six. The seventh is MedQA on Mistral-24B, a single-best-answer exam benchmark whose rubric grades agreement with a keyed answer, which is a different task from the open-ended setting the recipe targets. We set it aside for that reason. It is also the one added cell where aggregate score rises, so setting it aside does not flatter us.

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

We quoted the first two rows: the teacher makes scope bounding and hedging worse, the adapter makes both better. We did not quote specificity, which goes the other way, 1.71 to 1.33, and belonged next to them. The table shows the change is selective: the adapter moves some dimensions in the opposite direction to its teacher, which is not what indiscriminate copying would produce.

Table 2. Teacher behaviour and filter yield for the six added cells plus the submitted one. Row 1 is the submitted cell. Clearance comes from a single trace-generation run per cell, so there is no seed spread to report, and row 1's is approximate because the paper reports it as roughly 3,100 traces kept of 4,000.

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

One correction, now that Table 2 makes it checkable. In the first paragraph after Table R1 we wrote that the adapter matches or exceeds its teacher in four of five cells, without saying what that counted. It counts active inquiry over the five added cells that transfer, excluding the BioMistral null. Of those the adapter is ahead in four, the exception being Med42-8B on HealthBench. Across all seven rows of Table 2 it is four of seven; the other two behind their teachers are the submitted cell and BioMistral.

Table 3 and the rest of your questions follow in the next comment.
