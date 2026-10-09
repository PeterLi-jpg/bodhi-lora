# Why aggregate rubric score falls Base -> LoRA

Internal. NOT in any reviewer response. Run 2026-08-01 against
`/data/PROJECTS/bohdi/results_rebuttal/` on mit-critical-data-1, 5 seeds per cell.

## 1. The drop is real, not seed noise

Paired over 5 seeds, base_no_wrapper vs lora_no_wrapper:

| Cell | base | lora | delta | paired t (df=4) |
|---|---|---|---|---|
| HealthBench / Mistral-24B | 0.4248 | 0.4111 | -0.0137 | -5.47 |
| HealthBench / Med42-8B | 0.4050 | 0.3920 | -0.0130 | -6.20 |
| HealthBench / BioMistral-7B | 0.3781 | 0.3629 | -0.0152 | -4.01 |
| ChatDoctor / Mistral-24B | 0.5573 | 0.5414 | -0.0159 | -5.34 |
| ChatDoctor / Med42-8B | 0.4984 | 0.4574 | -0.0410 | -10.51 |
| MedQuAD / Mistral-24B | 0.6716 | 0.5860 | -0.0856 | -9.02 |

All six exceed 2 SE.

## 2. It is NOT the rubric penalising the model for asking

Restricting to prompts where neither base nor adapter asks anything, the drop
persists in all six cells, and is *larger* than the overall drop in four:

| Cell | overall | both-silent | n silent |
|---|---|---|---|
| HB / Mistral | -0.012 | -0.019 | 139 |
| HB / Med42 | -0.013 | -0.008 | 450 |
| HB / BioMistral | -0.015 | -0.015 | 824 |
| CD / Mistral | -0.018 | -0.089 | 70 |
| CD / Med42 | -0.041 | -0.022 | 134 |
| MedQuAD / Mistral | -0.108 | -0.126 | 143 |

Within the adapter's own outputs, asking is not consistently penalised: askers
minus non-askers is +0.015, -0.036, -0.002, +0.081, -0.025, +0.069. No sign.

Decisive case: BioMistral. Active inquiry does not move (11.7 -> 10.1) yet
quality still falls 0.015.

**The rubric-artefact explanation in the submitted reply is wrong.** It was
withdrawn in the HomM follow-up.

## 3. Specificity is not the mechanism either

Specificity base -> LoRA: +0.235, -0.082, -0.032, +0.343, -0.110, -0.388.
Rises in two cells while quality still falls. Do not use this as the story.

## 4. Where the points actually go (point-weighted, by rubric axis)

Mean contribution to the total score change across the six cells:

| Axis | mean contribution | negative in |
|---|---|---|
| instruction_following | -0.0160 | 6/6 |
| accuracy | -0.0094 | 5/6 |
| context_awareness | -0.0093 | 6/6 |
| communication_quality | -0.0070 | 5/6 |
| completeness | -0.0038 | 4/6 |

**Instruction following is the largest and the only universally negative
contributor.** Consistent with the adapter having learned a behavioural prior
that partly overrides prompt-specific instructions. This is the most defensible
mechanism statement available.

## 5. OPEN PROBLEM: context_awareness disagrees with our own grader

HealthBench's native `axis:context_awareness` falls in all six cells. Splitting
those criteria by sign on the three HealthBench cells, the adapter meets FEWER
of the reward-asking criteria than the base:

| Cell | reward-asking criteria met, base -> lora |
|---|---|
| HB / Mistral | 0.090 -> 0.064 (-0.026) |
| HB / Med42 | 0.060 -> 0.039 (-0.022) |
| HB / BioMistral | 0.032 -> 0.012 (-0.020) |

Our seven-dimension grader reports context-seeking rising (1.40 -> 1.75 in the
submitted cell). HealthBench's expert-written rubric reports the adapter earning
less on the axis that rewards exactly that.

**Working hypothesis, untested:** the two measure different things. HealthBench
criteria are prompt-specific ("asks whether they would like a refined version
suitable for recording in clinical notes"); our grader scores generic clarifying
behaviour. An adapter that asks more but asks *generically* moves one up and the
other down.

If that holds it is a publishable nuance and belongs in the limitations. If it
does not, it is a threat to the central behavioural claim. Either way this is
about fifty lines of analysis against the released code, so someone will run it.
Resolve before camera-ready.

## Reproduce

Section 1 (paired t over seeds, Holm over the six cells) is now
`rebuttal/analyses/quality_drop.py`. It read the host copy, then in the layout
`61dada9:rebuttal/launch/run_cell.sh` writes; since the move it reads the cell folders of
`results/` and writes `results/cross_cell/quality_drop.json`.
On 2026-10-08 it ran against the cells synced from the host, and its six t values match
the table above. Its Holm family also holds the MedQA cell, so its adjusted p values differ
from the paper's, which runs Holm over the six cells that are not set aside.

The askers-minus-non-askers gap of section 2 is now recomputed by
`rebuttal/analyses/cell_values.py` (`lora_askers_minus_non_askers` in
`results/cross_cell/generated_values.json`), on LoRA responses with a parsed grade, pooled
over seeds. It gives the same signs in all six cells. The values differ in the three
Mistral-Small-24B cells (HealthBench +0.004, ChatDoctor +0.083, MedQuAD +0.044), so the ad
hoc code above used another definition or other grade files; the paper uses the script.

Sections 2 to 5 were computed with ad hoc code that is not in the repo. The pattern is: join
`seed_*/eval/<config>.json` (`results[].score`, `criteria_results[]` with
`points` / `criteria_met` / `tags`) against
`seed_*/eval/epistemic_scores.json` (`configs[].examples[].scores`) on
`prompt_id`.
