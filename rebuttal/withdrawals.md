# Withdrawn and corrected claims

This file lists every claim that the authors corrected or withdrew during the NeurIPS 2026
discussion of submission 32818. It also lists posted claims that the discussion never
corrected but that the audit or a later check found wrong (W10, W11, and W14 onward; W21 came
from the raw outputs synced on 2026-10-08, W22 from the stored outputs on 2026-10-09). W23 to
W25 withdraw posted numbers that came from keyword rules, under the first author's rule of
2026-10-09 (docs/CONVENTIONS.md rule 15): a keyword rule misses negation, hypotheticals and
quotes, so it cannot decide what a text means. Each
entry gives the claim as made, the correction with its source, and the statement the paper
must carry.

The paper must carry the corrected statement in every place where the claim appears. A
correction that exists only in a reply does not count.

## Where the locations point

- **Submitted:** `paper/submitted/neurips2026/main.tex`. This is the frozen source of record,
  checked against the official PDF (commit 416113b on `cleanup/resubmission`). The earlier
  `paper/neurips_submission/main.tex` is a later working copy and is archived.
- **Posted:** `rebuttal/responses/posted/<file>.md:<line>`.
- **Live paper:** `paper/main.tex` at commit 61dada9. Later paper edits will move these lines.

Each "must say" text is plain wording for the paper to adapt. Numbers in it come from
the posted replies (see `rebuttal/posted_values.json`) or from files in this repo.

---

## W1. "Framework"

**Claim as made.**
- Submitted, abstract, `main.tex:36`: "We contribute a framework for studying these epistemic
  behaviors as first-class objects of evaluation and design".
- Submitted `:52`: "This paper contributes a framework for studying epistemic behaviors".
- Submitted `:56`: "An \textbf{epistemic virtue evaluation framework}".
- Submitted `:58`: "The LoRA recipe is one concrete instantiation; the evaluation framework is
  what generalizes."
- Submitted `:251`: "The framework does not depend on any of these choices. Any structured CoT
  protocol that elicits scorable behavioral demonstrations can serve as the generation step.
  Any parameter-efficient fine-tuning method can serve as the distillation step."
- Submitted `:260`, `:262`: "We contribute a framework ..."; "The recipe is not tied to
  \bodhi{}, MedGemma, or HealthBench."
- Live paper: `paper/main.tex:78`, `:97`, `:102`, `:103` still say "framework".
  `:416-419` already has the narrowed sentence.

**Correction posted.**
- `FzpA_1_rebuttal.md:5`: "If accepted we will narrow the claim to a recipe plus an evaluation
  decomposition, demonstrated on open-ended, information-seeking clinical Q&A, and state what
  is *not* covered". The same line says the "Any structured CoT protocol..." sentence will be
  replaced.
- `FzpA_1_rebuttal.md:66`: "You were right. We have narrowed the claim to what we
  demonstrated".
- `HomM_1_rebuttal.md:7`: "not as a validated general framework".
- `HomM_1_rebuttal.md:27`: the protocol-agnostic claim "remains untested and we do not assert
  it".

**The paper must say.** We contribute a recipe and an evaluation decomposition for open-ended,
information-seeking, patient-facing clinical Q&A. We tested one CoT protocol (BODHI) and one
PEFT method (LoRA). We do not claim that other protocols, other PEFT methods, or tasks with no
missing information to seek (closed-form multiple choice, extraction, summarization) give the
same result.

---

## W2. "Clinical decision support" as the setting

**Claim as made.**
- Submitted: the phrase does not appear.
- Posted `FzpA_1_rebuttal.md:36`: "who identified these as the behaviors that most affect
  safety in clinical decision support".
- Live paper `paper/main.tex:187`: "identified these as the behaviors that most affect safety
  in clinical decision support".

**Correction posted.**
- `FzpA_2_followup.md:5`: "\"Clinical decision support\" was the wrong term. The label was
  ours and it named a setting the study never addressed".
- The draft of that reply also said "None of the seven presupposes a clinician reader"
  (`responses/drafts/FzpA_2_followup_prepost.md:9`). That paragraph was not posted.
- `HomM_4_consolidation.md:3`: "The scope is named as patient-facing clinical Q&A."

**The paper must say.** The setting is patient-facing clinical Q&A. In HealthBench-Hard, 14 of
1,000 prompts identify the speaker as a clinician. This audit counts explicit
self-identification only, so it is a lower bound. Every anchor in the grading rubric addresses
the person who asks.

---

## W3. Origin of the seven dimensions

**Claim as made.**
- Submitted: no origin is stated. `main.tex:140` lists the dimensions.
- Posted `FzpA_1_rebuttal.md:32`: "yes, clinicians shaped them."
- Posted `FzpA_1_rebuttal.md:36`: "The selection was shaped by the international practising
  physician co-authors on this paper".
- Live paper `paper/main.tex:182-187`: "was shaped by the practising physician co-authors of
  this paper, who work across several health systems".

**Correction posted.**
- `FzpA_2_followup.md:9`: "The selection method was informal author-group consensus, settled
  in discussion including the three practising physicians among the authors, without a Delphi
  round or a content-validity index."
- `FzpA_2_followup.md:7`: the grounding is the cited literature (Simpkin and Schwartzstein,
  2016; Jones et al., 2019).

**The paper must say.** We chose the seven dimensions by informal consensus in the author
group, which includes three practising physicians. We used no formal method (no Delphi round,
no content-validity index). The grounding is the literature on tolerating uncertainty (Simpkin
and Schwartzstein, 2016) and on safety-netting (Jones et al., 2019).

---

## W4. Convergence of adapted active inquiry

**Claim as made.**
- Submitted: not present. It was first made in the rebuttal.
- Posted `FzpA_1_rebuttal.md:15`: "After adaptation all three converge into a narrow band
  (45.6%, 52.5%, 56.6%): the endpoint is a property of the training signal, not of the
  starting model."
- Posted `HomM_1_rebuttal.md:23`: "converge into a narrow band after adaptation ... which is
  what one would expect if the endpoint is a property of the training signal".
- Live paper: not present.

**Correction posted.**
- `HomM_2_followup_part1.md:7`: "**we withdraw the sentence** ... With all four HealthBench
  bases, adapted active inquiry runs 10.1% to 56.6% against a baseline spread of 11.7% to
  25.6%. It widens. ... We withdraw it wherever we made it."
- `HomM_3_followup_part2.md:30`.

**The paper must say.** Nothing about convergence. If the spread is reported: across the four
HealthBench bases, base active inquiry spans 11.7% to 25.6% and adapted active inquiry spans
10.1% to 56.6%. We draw no conclusion about what sets the endpoint.

---

## W5. "Scope condition" for the BioMistral cell

**Claim as made.**
- Submitted: not present.
- Posted `HomM_1_rebuttal.md:11`: "BioMistral-7B is a scope condition".
- Posted `HomM_1_rebuttal.md:53`: "with a scope condition showing where the recipe fails and
  why".
- Posted `HomM_1_rebuttal.md:25` and `LxMF_1_rebuttal.md:78`: "there is a precondition"; "We
  also identified a precondition".
- Live paper `paper/main.tex:756`: "BioMistral-7B (scope cond.)".
- Live paper `:768-772`: "A precondition, not a universal recipe ... We therefore state a
  scope condition: the recipe requires a base model capable of following the structured CoT
  protocol."
- Live paper `:411-413`: "identifies a precondition".

**Correction posted.**
- `HomM_2_followup_part1.md:5`: "On \"scope condition\": poor wording. ... What we should have
  written is a hypothesis, not a condition we had established. ... One null can't separate
  teacher strength from model size, instruction tuning, or a configuration that doesn't suit
  a 7B."

**The paper must say.** BioMistral-7B on HealthBench is a null result: active inquiry went from
11.7% to 10.1%. Its teacher was the weakest (14.5% against 47.9% to 57.1% for the other
HealthBench bases), and its filter clearance was the lowest (20.4% against 50.8% to about
78%). Our hypothesis is that the recipe needs a teacher that produces demonstrations the filter
keeps at a usable rate. We have not tested this hypothesis. One null cannot separate teacher
strength from model size, instruction tuning, or a configuration that does not suit a 7B
model. The table label is "null", not "scope cond.".

---

## W6. "Four of five" cells where the adapter matches its teacher

**Claim as made.**
- Submitted: not present.
- Posted `HomM_1_rebuttal.md:23`: "In four of the five working cells the adapter matches or
  exceeds its own teacher while requiring no CoT at inference; the exception is Med42-8B on
  HealthBench".
- Live paper `paper/main.tex:762-763`: "in four of the five the adapter matches or exceeds its
  own teacher".

**Correction posted.**
- `HomM_2_followup_part1.md:41`: "It counts active inquiry over the five added cells that
  transfer, excluding the BioMistral null. ... Across all seven rows of Table 2 it is four of
  seven".
- `HomM_3_followup_part2.md:30`: "put a scope on the four-of-five count".

**The paper must say.** On active inquiry, the adapter is at or above its teacher in 4 of the
5 added cells where the effect transfers. The exception is Med42-8B on HealthBench (52.5%
against 57.1%). Over all 7 cells the count is 4 of 7. In the submitted cell (45.6% against
47.9%) and in BioMistral (10.1% against 14.5%) the adapter is behind its teacher.

---

## W7. The rubric explains the aggregate drop

**Claim as made.**
- Submitted: not present.
- Posted `LxMF_1_rebuttal.md:80`: "We read that partly as a rubric property, since it rewards
  agreement with a reference answer, so asking instead of answering scores lower by
  construction."
- Posted `HomM_1_rebuttal.md:29`: the same reading.
- Live paper `paper/main.tex:783-789`: "We read this as partly a property of the rubric rather
  than of the model".
- Live paper `:786` also says "the wrapper, which asks most, drops furthest of all
  ($0.672\to0.542$)". Both parts of that sentence are false against the posted tables. On
  MedQuAD the wrapper asks less than the adapter (23.1% against 26.5%,
  `HomM_2_followup_part1.md:23`). LoRA+CoT drops further, to 0.505
  (`HomM_3_followup_part2.md:12`).

**Correction posted.**
- `HomM_3_followup_part2.md:16`: "We had read part of that as a rubric artefact ... We tested
  it and it does not hold. ... A penalty on asking cannot explain a drop in a cell where the
  asking did not change. So we withdraw that reading as well".

**The paper must say.** We tested whether the rubric penalises asking, and the drop is not
explained that way. Within the adapter's outputs, prompts where it asks do not score
consistently lower than prompts where it answers directly. The gap is positive in three cells
and negative in three. In BioMistral, active inquiry did not change, yet the aggregate score
still fell by 0.015. Delete the "property of the rubric" paragraph, or restate it as tested
and rejected. The internal analysis in `rebuttal/responses/drafts/FINDINGS_quality_drop.md` (section
2) reaches the same result.

---

## W8. Aggregate quality "preserved", "unchanged", "non-inferior"

**Claim as made.**
- Submitted, abstract, `main.tex:36`: "with aggregate clinical quality preserved ($p{=}0.57$)".
- Submitted `:52`: "without degrading clinical quality".
- Submitted `:58`: "with no degradation in aggregate clinical quality ($d=0.03$, $p=0.57$)".
- Submitted `:169`: "it establishes a non-inferiority constraint: the adapter does not degrade
  clinical quality".
- Submitted `:171`: "Clinical quality is preserved."
- Submitted `:260`: "while leaving aggregate clinical quality unchanged".
- Posted `HomM_1_rebuttal.md:29`: "On HealthBench the aggregate rubric score is essentially
  unchanged (Mistral-Small-24B 0.425 → 0.411; Med42-8B 0.405 → 0.392), consistent with the
  submitted non-inferiority result."
- Posted `HomM_1_rebuttal.md:41`: "rather than to the non-inferiority one".
- Live paper `paper/main.tex:78`, `:97`, `:103`, `:236`, `:241`, `:780-782`.

**Correction posted.**
- `HomM_3_followup_part2.md:14`: "Base to LoRA falls in all six ... A paired t-test over the
  five seeds puts all six drops at two-sided p below 0.02, and all six survive Holm correction
  ... we should not call these in line with it."
- `HomM_3_followup_part2.md:18` states the trade.
- `HomM_4_consolidation.md:3`: "the trade-off goes into the abstract alongside the gain, where
  the submitted version claimed quality was preserved."

**Left over in a later post.** `FzpA_2_followup.md:13` (dated after the correction) still
reports "no detected change in aggregate quality at d = 0.03, p = 0.57". It does not mention the
six drops. The sentence is true for the submitted cell only.

**The paper must say.** In the submitted cell we detected no change in aggregate score (0.459
to 0.463; d = 0.03, p = 0.57). We set no equivalence margin, so this result does not show
non-inferiority (audit EVAL-1). In all six added cells, aggregate score fell from Base to
LoRA. The drop was 0.013 to 0.015 on HealthBench and 0.016 to 0.086 on ChatDoctor and
MedQuAD. In a paired t-test over five seeds, all six had p < 0.02, and all six passed Holm
correction. The abstract states this cost next to the behavioural gain. Remove "preserved",
"non-inferiority" and "no degradation" everywhere.

---

## W9. CoT and LoRA "compete for the same behavioral capacity"

**Claim as made.**
- Submitted `main.tex:59`: "CoT-induced and weight-based epistemic behaviors compete rather
  than compose".
- Submitted `:228`: "The CoT protocol's analysis pass consumes context that the adapter's
  learned patterns would otherwise use for red-flag recognition. This is the clearest evidence
  of internalization: the two mechanisms compete for the same behavioral capacity."
- Submitted `:247`: "competes with the adapter's learned red-flag patterns for the model's
  attention budget".
- Submitted `:260`: "compete for the same behavioral capacity".
- Live paper `paper/main.tex:97`, `:101`, `:309`. `:326` already names the Pass 1 format
  leakage, but it still says "compete".

**Correction posted.**
- `LxMF_1_rebuttal.md:42`: "our wording asserted a mechanism we had not demonstrated".
- `LxMF_1_rebuttal.md:44-48` gives the evidence.
- `LxMF_1_rebuttal.md:50`: "the two conditioning sources compete for control of the output
  *format*, and the failure is format leakage, not capacity exhaustion or dilution."

**The paper must say.** Under LoRA+CoT, 54.6% of responses leak the protocol's Pass 1 analysis
format into the answer. In the other conditions the rate is 0.2% to 3.0%. The red-flag drop
is in the leaked responses (1.33, n = 507). The other responses score 1.73 (n = 421), which is
above LoRA alone (1.67). Within those other responses, red-flag score rises with length. We
call this format leakage. We do not claim competition for capacity or attention. We found the
non-leaked subset after the fact, by a format property, so leakage may correlate with prompt
difficulty.

**Changed 2026-10-10 (W25).** The leakage numbers above came from a rule with two free words.
With the seven Pass 1 headers only, 53.2% of graded LoRA+CoT responses leak (519 of all 959,
graded or not) against 0.0% to 0.8%, and the red-flag scores are 1.32 (n = 494) and 1.73
(n = 434). The paper carries these. Among the non-leaked responses, red-flag identification
rises across length quartiles, 1.45, 1.80, 1.82 and 1.84 (medians 1,529 to 5,477 characters,
about 108 responses each), and section 4.3 says so; this answers whether long outputs dilute
attention (LxMF). The check is post hoc, like the split. Prompts whose two passes overran the
context got no response and are in neither group.

---

## W10. "n ≈ 985–998 per condition" (not corrected in the discussion)

**Claim as made.**
- Submitted `main.tex:212`: "$n\approx985$--$998$ per condition, pooled across 5 seeds".
- Posted `LxMF_1_rebuttal.md:7`: "All comparisons use the available completions,
  $n \approx 985$–$998$ per condition pooled across 5 seeds."
- Live paper `paper/main.tex:221`, `:560`.

**Correction posted.** None. Audit finding PROV-4 found the range wrong for both CoT
conditions.

**Checked here.** I recounted from `results_modal/seed_*/`. Epistemic scores excluding parse
failures: Base 998, Base+CoT 947, LoRA 985, LoRA+CoT 928. HealthBench-scored responses: 1,000,
957, 1,000 and 959.

**The paper must say.** Give the n for each condition: 998, 947, 985 and 928 for the epistemic
dimensions, and 1,000, 957, 1,000 and 959 for the HealthBench score. Label the Table 1 values
191 and 192 as rounded per-seed means. The responses come from 667 unique prompts, and some
prompts repeat across seeds (audit CLAIM-10).

---

## W11. "Re-ran the pipeline unchanged, every hyperparameter fixed" (not corrected in the discussion)

**Claim as made.**
- Submitted, hyperparameter table, `main.tex:328-335`: r = 16, alpha = 32, seven target
  modules, 3 epochs, effective batch 16, max sequence 4096.
- Posted `HomM_1_rebuttal.md:9`: "We re-ran the pipeline unchanged, every hyperparameter fixed
  (LoRA $r=16$, $\alpha=32$, effective batch 16, 3 epochs, $\tau=0.4$, Qwen-14B filter,
  Llama-3.1-8B evaluator)".
- The same claim is posted at `FzpA_1_rebuttal.md:13` ("We re-ran the pipeline unchanged"),
  `FzpA_1_rebuttal.md:50` ("hyperparameters fixed"), `LxMF_1_rebuttal.md:58` and
  `FzpA_2_followup.md:21` ("rank 16, alpha 32, three epochs").
- Live paper `paper/main.tex:710-713`.

**Correction posted.** None. Audit findings HYG-4, PROV-3, REPRO-3 and REPRO-12 contradict it.

**The paper must say.** State what ran in each cell, not "unchanged":
- The added cells used `61dada9:rebuttal/configs/`: r = 16, alpha = 32, effective batch 16 and
  3 epochs. Mistral-Small-24B was trained with 4-bit QLoRA and served on a bf16 base.
- The submitted MedGemma adapters very likely used r = 8 and alpha = 16 (tunix path). The GCS
  check in HYG-4 cannot run: the bucket returned 404 on 2026-10-08. The paper says the values come
  from the code and could not be checked against the saved files (D1, D19).
- The eval settings differ from the submitted cell: context window 8,192 against 4,096 tokens,
  64 against 32 concurrent sequences, one fixed set of 200 HealthBench prompts for all seeds
  against a new draw per seed, grader weights from a mirror repo, and H100 against A100.
- tau was 0.4 in every added cell, MedQuAD included. Each filter log prints `threshold=0.4`,
  and no host log has the 22nd-percentile re-filter line (`results/cross_cell/cell_recipe.json`,
  commit 5d5890e; PROV-3). The earlier reading that MedQuAD used a percentile cut came from its
  clearance, 78.4%, which a 22nd-percentile cut would also give. Done in Appendix J (Table J.1)
  of `paper/main.tex`, commit 38b800e. The synced `adapter_config.json` files confirm r = 16,
  alpha = 32, dropout 0.05 and seven target modules in all 35 adapters, and the trainer states
  confirm 3 epochs and an effective batch of 16 (Table F.2 now reads them).

---

## W12. Why the MedQA cell was set aside

**Claim as made.** The posted replies give two different reasons.
- `FzpA_1_rebuttal.md:48`: "our preamble told the model to ask for information it needed,
  which drove base-model active inquiry to 99% and left the cell measuring
  instruction-following rather than calibration."
- `HomM_2_followup_part1.md:11`: "a single-best-answer exam benchmark whose rubric grades
  agreement with a keyed answer ... We set it aside for that reason. It is also the one added
  cell where aggregate score rises".
- Live paper `paper/main.tex:724-731` gives the preamble reason only.

**Correction posted.** `HomM_4_consolidation.md:3` commits to moving the MedQA cell into the
main evaluation.

**The paper must say.** One account. We ran MedQA-USMLE as open-ended questions. The first
prompt set had a preamble that told the model to ask for missing information. Base active
inquiry reached 99%, so that run measured instruction-following. Commit 035d85f removed the
preamble. The cell in the main evaluation must name which run it reports. It must also say
that MedQA is a single-best-answer exam task, scored against a keyed answer.

**Checked against the raw outputs (2026-10-08).** The host holds one MedQA run, and it is the
preamble run. Its traces were generated on 2026-07-27, and its eval files were written between
2026-07-27 and 2026-07-28 01:23 (Pacific); 035d85f, on another machine, removed the preamble at
01:04 that night. Base active inquiry is 99.0% to 100.0% in each of the four seeds with an
epistemic grade (seed 42's grading found none of the MedQA prompts), as posted for the preamble
run. Its aggregate score rises from 0.665 to 0.683
(+0.018, 95% t interval [+0.003, +0.033], paired over five seeds), the values in the unposted
drafts (`rebuttal/README.md`, "Not in these files"). So the posted "aggregate score
rises" (`HomM_2_followup_part1.md:11`) describes the preamble run, which measures instruction
following. No run without the preamble exists. Done in section 4.2, the abstract, the Limitations
and Appendix J of `paper/main.tex` (commit 38b800e), with every number from `make numbers`. Since
ab58ee5 this is section 4.1, where the cell sits last in Table 1, set aside.

---

## W13. Cross-family grading "prevents" filter-grader bias

**Claim as made.**
- Submitted `main.tex:36`, `:52`: "prevents filter-grader bias from inflating reported
  metrics".
- Submitted `:119`: "This asymmetry ensures that any Qwen-specific grading bias affects only
  which traces enter training".
- Posted `LxMF_1_rebuttal.md:34` and `HomM_1_rebuttal.md:35`: "that separation is what
  prevents filter-grader circularity".
- Live paper `paper/main.tex:78`, `:97`, `:102`.

**Correction posted.**
- `HomM_3_followup_part2.md:22`: "The cross-family split removes the most direct route to
  circularity without ruling out correlated grading preferences, and it doesn't buy clinical
  validity."

**The paper must say.** The filter (Qwen2.5-14B) and the evaluator (Llama-3.1-8B) come from
different model families. This removes the most direct route to circularity. It does not rule
out correlated grading preferences, and it gives no clinical validity. In the two Med42-8B
cells, the evaluator is from the same family as the base model (audit EVAL-7).

---

## W14. "14 of 1,000 HealthBench-Hard prompts identify a clinician" (corrected after the discussion)

**Claim as made.**
- Posted `FzpA_1_rebuttal.md:46`: "Auditing HealthBench-Hard for explicit clinician
  self-identification ("my patient", "as a physician"), 14 of 1,000 prompts (1.4%)
  identify the speaker as a clinician."
- Posted `FzpA_2_followup.md:5`: the same count.

**Correction.** Not corrected in the discussion. The keyword rule behind 14 was never
committed, and no script in the repo gives 14. The committed rule, `CLINICIAN_RE` in
`rebuttal/analyses/mimicry_split.py`, finds 4 of the 667 evaluated prompts
(`results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/withdrawn/mimicry_split.json`). HealthBench's own physician-agreed tag
(`physician_agreed_category:health-professional`) marks 40 of the 1,000 Hard prompts as
written by a health professional. Only 63 prompts, all in the communication theme, carry
the tag, so 40 is a lower bound.

**The paper must say.** HealthBench's physicians tagged 40 of the 1,000 Hard prompts as
written by a health professional, and only 63 carry the tag, so this is a lower bound.
The posted 14 came from a keyword rule we did not keep; the tag count replaces it. The
benchmark is still mostly patient-facing. Done in `paper/main.tex` (introduction and
Appendix H, macros `\statHardHealthProfessional`, `\statHardSpeakerTagged`).

**Changed 2026-10-10.** Both keyword counts are withdrawn as measures (docs/CONVENTIONS.md
rule 15): the posted 14 and the committed `CLINICIAN_RE` count of 4 of 667. A keyword rule
cannot tell a clinician who writes about a patient from a patient who quotes a doctor. The
paper no longer cites the 4 of 667, and `rebuttal/analyses/mimicry_split.py` is kept only as a
record. The tag count stays.

---

## W15. "The adapter asks selectively" (corrected after the discussion)

**Claim as made.**
- Posted `LxMF_1_rebuttal.md:3`: "the adapter distinguishes information-withholding prompts from
  self-contained ones (+10.5pp, CI excluding zero) while the base model does not (+2.1pp, CI
  including zero)". `LxMF_1_rebuttal.md:24`: "the opposite of what surface mimicry predicts".
- Posted `HomM_1_rebuttal.md:39,53` and `FzpA_1_rebuttal.md:48`: "the adapter's is selective".

**Correction.** Not corrected in the discussion. The posted numbers compare each condition's
discrimination index with zero. The test of a change is LoRA minus Base: +8.4 points
[-1.2, +17.9] with the rubric-based labeling and +10.0 [-0.5, +20.1] with the theme-only one
(`results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/withdrawn/mimicry_split.json`). Both intervals include zero. The adapter also asks more
on self-contained prompts (15.9% to 37.5%, and 17.0% to 43.2%).

**The paper must say.** We do not show that the adapter learned when to ask, and template
imitation remains possible. Done in `paper/main.tex`: the Discussion, the Limitations item
"Selectivity" and Appendix H, which names the posted claim as withdrawn.

**Changed 2026-10-10 (W24).** The rubric-based labelling is withdrawn. The paper keeps the
theme-tag labelling only: LoRA minus Base +10.0 [-0.5, +20.1], and asking on self-contained
prompts 17.0% to 43.2%, now from the uniform analysis (`results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json`,
`discrimination.active_inquiry.needs-info-theme`).

**Changed 2026-10-10 (paper).** The paper gives the index in every HealthBench cell
(section 4.4, Table H.6): in none is LoRA minus Base detectably above zero, and with
BioMistral-7B it falls. The abstract says that the data do not show that the adapter
learned when to ask.

---

## W16. The BODHI expansion and the dimension mapping (changed after the discussion)

**Claim as made.**
- Submitted `main.tex:50` and posted `FzpA_1_rebuttal.md:34`: BODHI is "Bridging, Open,
  Discerning, Humble, Inquiring", and scope bounding maps to Open and Bridging.

**Correction.** The paper cites the BMJ Health & Care Informatics paper (DOI
10.1136/bmjhci-2025-101877), which expands BODHI as "Balanced, Open-minded, Diagnostic, Humble,
Inquisitive" (`docs/DECISIONS.md`, D15 notes). The Bridging expansion comes from the bodhi-llm
0.1.4 package, which cites the PLOS Digital Health paper (DOI 10.1371/journal.pdig.0001013).

**The paper must say.** The submission and the replies used the package's expansion and mapping.
This version uses the cited BMJ expansion, so the mapping changed (scope bounding now maps to
Humble). Cite the PLOS Digital Health paper as the source of the bodhi-llm protocol. Done in
Appendix D of `paper/main.tex` (`bodhiplos2026` in `paper/references.bib`).

---

## W17. "All p < 1e-5" over pooled responses (changed after the discussion)

**Claim as made.**
- Submitted `main.tex:218`, `:222`: "$p < 10^{-5}$" for scope bounding and red-flag
  identification, from tests that treat each response as independent.
- Posted `FzpA_2_followup.md:13`: "all at p below 0.00001".

**Correction.** The seed is now the unit of inference (D7). At seed level the LoRA minus Base
p-values of the four dimensions that rise are 5.1e-5 to 1.6e-3 before Holm correction, and
hedging quality and specificity are not significant after it (`paper/generated/numbers.json`).

**The paper must say.** The submitted version treated responses as independent; this version
uses the seed as the unit. Done in Section 3.4 of `paper/main.tex`.

---

## W18. "Released code and adapters" (corrected after the discussion)

**Claim as made.** Posted `HomM_1_rebuttal.md:57`: "released code and adapters".

**Correction.** The code is public. No adapter is released. The five MedGemma-27B adapters were
in `gs://bohdi-runs-tokyo-micron/prod-v39-tunix-27b-neurips/`, and that bucket returned 404 on
2026-10-08. The only other copy that the code names is the Modal volume `bohdi-data-cache`
(`legacy/modal_eval.py:119,435-447`). The first author decided on 2026-10-08 not to use Modal
and to treat these adapters as not retained (DECISIONS.md D19), so nobody will check it. The
added-cell adapters are a separate case: the sync of 2026-10-08 copied their configs and trainer
states, not their weights, which stay on the GPU host.

**The paper must say.** No adapter is released, because the five MedGemma-27B adapters were not
retained. Done in Code availability (`paper/main.tex`, ab58ee5) and in the checklist
(`paper/checklist.tex`, faf8579).

---

## W19. How the physician check was specified (corrected after the discussion)

**Claim as made.**
- Paper and abstract: "pre-registered".
- Posted `FzpA_2_followup.md:9`: "quadratic-weighted kappa against a three-physician consensus".

**Correction.** The plan was written into the repository on 2026-04-29 (commit aa7eec4,
`aa7eec4:RESULTS.md` section 7.B), not registered with an outside registry. It specifies Cohen's kappa
between the Qwen2.5-14B filter grader and the consensus of at least two physicians, with a target
of 0.6.

**The paper must say.** "Specified in advance in the repository", the plan as written, and that
the 0.6 target was set for the filter grader. Done in the abstract, Section 3.4 and Appendix B of
`paper/main.tex`. The paper does not give the commit hash, because the hash would find the public
repository during double-blind review.

---

## W20. MedQuAD specificity left out (admitted in the discussion)

**Claim as made.** Posted `LxMF_1_rebuttal.md:76` and `HomM_1_rebuttal.md:25` quoted scope bounding
and hedging on MedQuAD, where the adapter moves away from its teacher, and concluded that the
adapter does not copy its teacher.

**Correction posted.** `HomM_2_followup_part1.md:25`: "We did not quote specificity, which goes the
other way, 1.71 to 1.33, and belonged next to them."

**The paper must say.** Give specificity next to the other two: Base 1.71, teacher 1.47, LoRA 1.33,
so the adapter lowers specificity on MedQuAD. Done in Appendix J of `paper/main.tex`, with every
dimension of every cell in the generated tables of Appendix E (`paper/generated/tab_cell_detail_1.tex`
and `tab_cell_detail_2.tex`; before ab58ee5, `tab_cell_dimensions.tex`).

---

## W21. MedQuAD base hedging quality posted as 1.46 (found in the raw outputs)

**Claim as made.** Posted Table 1, `HomM_2_followup_part1.md:18`: MedQuAD with Mistral-Small-24B,
Base hedging quality 1.46 ± 0.01. Two earlier posts (`LxMF_1_rebuttal.md:76`,
`HomM_1_rebuttal.md:25`) gave the same value as 1.45.

**Correction posted.** None.

**What the raw outputs give.** `503b975:rebuttal/analyses/cell_values.py`, then run by `make cells`, computes 1.4548
from the per-seed eval files, which rounds to 1.45. It is the only posted value of the added cells
that differs from the raw outputs at the posted precision
(`results/cross_cell/generated_values.json`, `mismatches`). So 1.45 was right and Table 1's 1.46
was not.

**The paper must say.** 1.45, from the generated values. Done: Appendix J and
`paper/generated/tab_cell_detail_2.tex` print the generated value, and
`rebuttal/posted_values.json` keeps 1.46 as the record of what was posted.

---

## W22. "Roughly 58% fewer output tokens" (found in the outputs)

**Claim as made.** Posted `FzpA_2_followup.md:17`: the adapter "needs one call where the protocol
takes two, at roughly 58% fewer output tokens".

**Correction posted.** None.

**What the outputs give.** The serving engine stored output token counts for the one-pass
conditions, Base and LoRA. The two-pass conditions stored only their final response, so no file
counts the protocol's tokens, and no script or output gives 58%
(`results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json`, `cost`).

**The paper must say.** The engine's counts for Base and LoRA, that the two-pass conditions cannot
be counted, and that the posted 58% is withdrawn. Done in section 4.2 and Appendix H of
`paper/main.tex` (ab58ee5), with the posted value from `rebuttal/posted_values.json`. Since
2026-10-10, section 4.4 and Table H.6 give the engine's counts for Base and LoRA in every
cell.

---

## W23. Second-person questions, 18.4% against 39.6% (withdrawn 2026-10-10, keyword measure)

**Claim as made.** Posted `FzpA_2_followup.md:11`: "counting questions addressed to the reader
in the second person, the base does so in 18.4% of responses and the adapter in 39.6%. The
grader puts that gap at 28 points where the text count puts it at 21". The same paragraph
adds that "27% of the base's go to the reader against 48% of the adapter's".

**Why withdrawn.** The count is a keyword rule: a sentence that ends in "?" and contains "you"
or "your". It counts quoted and rhetorical questions and misses indirect ones ("tell me how
long it has lasted"), so it cannot decide whether a reply asks. No script gave the posted
values (audit PROV-7). The committed rule (`rebuttal/analyses/second_person_questions.py`)
gave 17.9% and 36.7%, and the paper printed those until 2026-10-10; they are withdrawn with
the posted ones. The rule is gone from `configs/report.yaml`, so no cell has the column now.

**The paper must say.** The posted rate is withdrawn, and the paper gives no count of this
kind. The evaluator's active-inquiry grade is the measure of asking. A check by a judge from
another family, with a semantic rubric and a hand-label gate, is specified but has not run
(docs/ANALYSIS_PLAN.md section 12, 2026-10-10; docs/RUN_PLAN.md R17). Done in Appendix H of
`paper/main.tex`.

---

## W24. The discrimination index under the rubric-keyword labelling (withdrawn 2026-10-10, keyword measure)

**Claim as made.** Posted `LxMF_1_rebuttal.md:13-28` (Table R1): the discrimination index is
+2.1 points for Base, +11.0 for the wrapper, +10.5 for LoRA and -3.5 for LoRA+CoT, over 508
"withholding" and 158 self-contained prompts. On self-contained prompts the adapter asks 37.5%
of the time against Base's 15.9%. The odds ratios of asking are 1.16, 1.54 and 1.56. LoRA minus
Base is +8.4 [-1.2, +17.9]. `LxMF_1_rebuttal.md:3` and W15 carry the selectivity claim made
from these numbers.

**Why withdrawn.** The labelling counted a prompt as withholding information when its theme
tag was context_seeking or when any rubric criterion matched a keyword rule (`CLARIFY_RE` in
`rebuttal/analyses/mimicry_split.py`: "ask", "clarify", "follow up", "more information" and
others). A keyword match cannot tell a criterion that rewards asking from one that only
mentions it, or penalizes it. Every number above rests on that labelling. The theme-only
labelling of the same reply (+13.2 [+4.1, +21.7] for LoRA, 127 prompts) uses a structured tag
and stands. The rubric part is gone from `configs/benchmarks/healthbench.yaml`; the
`cluster:` tags cannot replace it, because they carry the physician labels, which only the
label analyses of the analysis plan may read (unseen until its adoption on 2026-10-11).

**The paper must say.** The theme-tag numbers only, from the uniform analysis: LoRA minus Base
+10.0 [-0.5, +20.1]; asking on self-contained prompts 17.0% under Base and 43.2% under LoRA;
the adapter's own index +13.2 [+4.1, +21.7] and Base's +3.2 [-4.2, +11.2]. The posted index is
named as withdrawn. Done in the Discussion and Appendix H of `paper/main.tex`.

---

## W25. Format leakage in 54.6% of LoRA+CoT responses (withdrawn 2026-10-10, keyword measure)

**Claim as made.** Posted `LxMF_1_rebuttal.md:44-46`: 54.6% of LoRA+CoT responses leak the
Pass 1 format, against 0.2% to 3.0% in the other conditions; red-flag identification is 1.33
in the leaked subset (n = 507) and 1.73 in the rest (n = 421); within the rest it rises
across length quartiles, from 1.44 (median 1,508 characters) to 1.87 (median 5,395). The
leaked responses have a median of 8,564 characters against 3,457.

**Why withdrawn.** The rule counted nine markers: the seven Pass 1 section headers of
bodhi-llm 0.1.4, and DIFFERENTIAL and SCOPE OF/BOUNDAR. These two are not headers of the
protocol's format. They are clinical words, so the rule read meaning from the text. In the
other cells they inflate the rate most: on MedQuAD the LoRA+CoT rate falls from 7.6% to
0.2% without them, on ChatDoctor with Med42-8B from 2.5% to 0.5%.

**What the outputs give.** With the seven headers only, which the protocol defines (the
`analysis-format` rule of `configs/protocols/bodhi.yaml`, from the uniform analysis,
`results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/stats.json`): 53.2% of LoRA+CoT responses leak, against 0.0% to 0.8% in the other
conditions; red-flag identification is 1.32 in the leaked subset (n = 494) and 1.73 in the
rest (n = 434), which is above LoRA alone (1.67). The diagnosis of W9 holds under this rule.

**The paper must say.** The seven-header numbers, and that the posted nine-marker rule is
withdrawn. Done in the abstract, section 4.3 and Appendix H of `paper/main.tex`. The posted
length quartiles and medians are withdrawn with the rule; under the seven-header rule the
quartiles rise from 1.45 to 1.84, and section 4.3 gives the two ends (W9). Since
2026-10-10, section 4.3 and Table H.6 give leakage in every cell under the seven-header
rule. Also since 2026-10-10, the rule counts a header only where a header starts (the start
of a line, after optional markdown, a bullet or a section number), so a sentence such as
"which is a key uncertainty" no longer counts. The MedGemma-27B values above do not change;
`docs/ANALYSIS_PLAN.md` section 12 lists the cells whose counts do.

---

## Posted claims that the discussion never corrected

The discussion did not withdraw the claims below, but the evidence does not support them. The
paper must decide on each one.

- **"the filter removes the teacher's failures before they reach the weights"**
  (`HomM_1_rebuttal.md:25`; `LxMF_1_rebuttal.md:76` says "the quality filter"), live paper `paper/main.tex:777-778`. The
  study has no unfiltered-SFT arm, so it cannot attribute the effect to the filter. A later
  post (`HomM_2_followup_part1.md:25`) leaves out the causal clause but does not withdraw it.
- **"blanket disclaimers eliminated"** Submitted `main.tex:36`, live `paper/main.tex:78`,
  `:103`. This rests on 5 of 998 against 0 of 985 responses (Fisher p = 0.062, audit CLAIM-8).
- **MedQuAD base hedging, 1.45 against 1.46.** Settled by the raw outputs; see W21.
- **"in five of the six added cells at every individual seed"** `FzpA_2_followup.md:13`. Checked against
  the raw outputs on 2026-10-08 (`results/cross_cell/generated_values.json`, per_seed): LoRA
  active inquiry is above Base in all five seeds of five added cells, and in one of five seeds of
  BioMistral-7B. The claim holds as posted. It counts the two Med42-8B cells, which the paper
  leaves out of cross-model statements (D6).
- **Second-person questions, 18.4% against 39.6%.** `FzpA_2_followup.md:11`. No script or
  output exists. A simple regex gives 17.9% and 37.2% (audit PROV-7). Withdrawn on
  2026-10-10 as a keyword measure (W23).
- **What the physician check covers.** `FzpA_1_rebuttal.md:40` reports kappa = 0.35 right after
  the dimension anchors, as if the physician check graded the seven dimensions. It grades
  HealthBench rubric criteria (`scripts/make_irr_excel.py`, `scripts/compute_irr_kappa.py`).
  The paper now says that no physician graded the seven dimensions (a59cb82).
- **The antidepressant example.** `FzpA_2_followup.md:19` says the adapter asks about "current
  medications and conditions". The answer in `results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_99/eval/lora.json`
  (prompt `2c915b71`) does not. The paper lists only the questions it asks (5c2e9b0).
- **The hedging scale.** `FzpA_2_followup.md:25` says the grader's scale is "0 for a bare
  referral". The grader prompt gives the two ends but ties no score to them, and the grader
  scored most flagged blanket disclaimers 2 on hedging. The paper says so (a59cb82).
- **Context-seeking needs a question for a 2.** The paper's earlier text, and the grader
  prompt, say a 2 needs a direct question. The grader gave 2 with no direct question to most
  Base answers it scored 2 (`\statContextSeekingTwoNoQuestionPctBase`). The paper reports
  the rate (a59cb82).
