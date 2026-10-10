# Analysis plan

Dated 2026-10-08. Status: **proposed**. The plans lane wrote it for the first author, who
adopts it or changes it (section 12). It is the dated plan of convention 3
([CONVENTIONS.md](CONVENTIONS.md)). It covers every run in [RUN_PLAN.md](RUN_PLAN.md),
and every statistic in [EXPERIMENTS.md](EXPERIMENTS.md) that nobody has computed on real
outputs yet.

**Registration.** The commit that adds this file registers the plan. Convention 3
compares the time of that commit with the first commit of each result the plan covers,
and every run manifest records the commit it ran at (`bodhi_lora/manifest.py`). The date
above is the working day; the commit time is what counts.

**Changes.** Nobody edits sections 1 to 11 after registration. A change is a dated entry
in section 12 that says what changed, why, and whether any output of a covered cell
existed at that time. A result that a late change touches is reported as exploratory.

**State at registration.** No planned cell has run: no folder of a planned cell exists
under `results/`. No file in the repo joins the HealthBench physician labels (section
2.4) to a model output: `rebuttal/analyses/mimicry_split.py` labels prompts by rubric
keywords and the theme tag, and `scripts/paper_stats.py` only counts the speaker tag. So
the label analyses of H3, H7 and H11 are unseen on every cell, the existing ones too.

## 1. Questions

Each question has an id. Sections 3 to 9 fix its test. "Claim" numbers refer to
[CLAIMS.md](CLAIMS.md); C-numbers to `rebuttal/commitments.md`.

| ID | Question | Experiments | Claims and promises it can change |
|---|---|---|---|
| H1 | Does the adapter raise asking without the protocol, under one set of settings? | E13, E16, E15, E21, E24, E25 | claims 1, 2 |
| H2 | What does the adapter cost on the rubric score? | E13 and every new training cell; E05 | claims 3, 4 |
| H3 | Does the adapter ask more where information is missing than where it is not? | E14 (primary), E02 (secondary) | MR-W4, LxMF-2; the Limitations sentence on selective asking |
| H4 | Does the adapter beat a prompt that asks for the same behavior? | E12 | claims 13, 16; C27 |
| H5 | Does the recipe work with a second teaching protocol? | E16 | claim 16; C12, C29 |
| H6 | Does stating red flags first recover the red-flag drop of LoRA plus the protocol? | E17 | claim 9; C31 |
| H7 | Does the red-flag gain come from flagging cases that are not emergent? | E18 | claim 1 (red flags); C18, C30 |
| H8 | Do the contrasts hold under a judge from another family? | E09, E10 | claim 14 and every LLM-graded claim; C26, D6 |
| H9 | Did the BioMistral-7B null come from the teacher or the student? | E21 | claim 6 |
| H10 | Does the filter change what the adapter learns? | E22 | the withdrawn filter claim (`rebuttal/withdrawals.md`) |
| H11 | Does the shift hold on prompts that clinicians wrote? | E19 | C32; FzpA-Q3 |

Section 10 lists the analyses that are descriptive or exploratory. Section 11 lists each
result that would change the paper.

## 2. Rules for every analysis

### 2.1 Unit and test

- **Unit.** The seed. The seeds are the cell record's list: 42, 7, 13, 99 and 101 in every
  planned cell.
- **Seed value.** For one seed and one condition, a dimension is the mean over the
  responses that the evaluator scored. The rubric score (the paper's aggregate score) is
  the mean over completions.
- **Contrast.** For "A minus B", take the difference d_s of the two seed values for each
  seed s. Report the mean of d_s, its 95% t interval with n - 1 degrees of freedom (t(4)
  with five seeds), and the two-sided one-sample t test of d_s against 0
  (`scipy.stats.ttest_1samp`, the same as a paired t test).
- **Level.** Two-sided alpha 0.05, after the correction that the section names.
- **Pooled numbers.** Response-level numbers are descriptive only: Cohen's d with a
  prompt-clustered bootstrap interval, 5,000 resamples, fixed seed 0 (D7).
- **Shared prompts.** Seeds that draw their own eval ids share some prompts, so seed
  values are not fully independent. D7 accepts this; the prompt-clustered bootstrap is
  the check.
- **Contrasts between cells** (H9, H10, E24) pair seed s of one cell with seed s of the
  other. This is valid only when both cells evaluated the same prompts for that seed. The
  analysis compares the two cells' eval id lists per seed first. If any list differs, it
  uses Welch's t test on the seed values (`scipy.stats.ttest_ind`, `equal_var=False`)
  and marks the result.
- **Seed-invariant conditions.** In a cell whose eval ids are the same for every seed
  (`fixed_pool`), a `base` condition gives the same outputs for every seed under greedy
  decoding. Its contrasts then measure the spread of the adapters only. The report says
  so for each such cell.

### 2.2 Outcomes

- **The seven dimensions** (the Holm family of each contrast): uncertainty acknowledgment
  (0 to 2), active inquiry (share of responses with at least one direct question),
  context-seeking (0 to 2), red-flag identification (0 to 2), scope bounding (0 to 2),
  hedging quality (0 to 2), specificity (0 to 2). Higher is better for all seven.
- **Secondary outcomes**, outside every Holm family: the rubric score (0 to 1, normalized),
  questions per response, red-flag rate (share with red-flag score 2), scope-bounded rate,
  specificity rate (concrete numbers), appropriate hedging, blanket disclaimer (lower is
  better), and the grader-free question rate.
- **Ask.** A response asks when the evaluator marks `active_inquiry` true.
- **Ask without a grader.** A response asks when it holds a second-person question, by
  the four steps in the docstring of `rebuttal/analyses/second_person_questions.py` at
  commit bf771eb. The analysis lane may move that code; the definition stays.
- **Harm proxy.** The share of responses that meet at least one rubric criterion with
  negative points, and the points lost to such criteria. It is LLM-graded and is not the
  physician harm review (E28).

### 2.3 Missing data and failed runs

- An evaluator parse failure drops the response from that dimension's mean (D13). A
  worst-case version scores every failure at the undesirable end. If the worst case
  changes a decision of this plan, the report says so next to the result.
- A completion lost to the context window is missing for the dimensions and drops out of
  the rubric mean. The report gives the share lost per condition (`finish_length`).
- A seed enters a contrast only when both conditions have outputs for it.
- A seed that fails for a technical reason (crash, out of memory, timeout) runs again
  with the same seed and settings, and its manifest records the rerun. Nobody drops or
  replaces a seed because of its result. If a seed still cannot run, the contrast uses
  the seeds that remain, its degrees of freedom fall with them, and the table shows the
  seed count.
- Nobody adds a seed after any output of the cell exists.

### 2.4 Labels

The physician labels of HealthBench are pre-specified in the benchmark record
(`configs/benchmarks/healthbench.yaml`, `timing: pre_specified`). The graders never see
them. A row with no tag of a label is in neither group.

| Label | Positive | Negative | Hard set (pos / neg) | Full set (pos / neg) |
|---|---|---|---|---|
| `needs-info-physician` | not enough context, not enough information to complete the task, context matters but is unclear | enough context, enough information, context matters and is clear, context does not matter | 204 / 113 | 580 / 857 |
| `emergent-physician` | emergent, conditionally emergent | non-emergent | 48 / 8 | 319 / 134 |
| `clinician-user` | health professional | not a health professional | 40 / 23 | 322 / 424 |

The counts were read from the downloaded files (sha256 pinned in the record) when this
plan was written. Per seed, the eval sets of the existing HealthBench cells hold 33 to 47
`needs-info-physician` positives and 19 to 27 negatives (2026-05 draws), or 46 and 19
(the fixed 2026-07 set). These are small groups; a null is "not detected".

The post hoc labellings (rubric keywords, the theme tag, `needs-info-theme`) are
exploratory only (D9).

### 2.5 No interim look

While a cell runs, progress checks read the manifests only: status, counts and times.
Nobody computes a contrast, or reads the scores that the stage logs print, before every
seed of the cell has finished its grade stage. The one-seed technical check of
RUN_PLAN.md follows the same rule. Enforced by: review step. The eval stage prints each
condition's mean rubric score to its log, so the operator follows progress through the
manifests, not the logs.

### 2.6 Wording

- A result "supports" a question only by the decision rule of its section.
- An interval that includes 0 is "not detected". "Preserved", "equivalent" and
  "non-inferior" need a margin that section 12 sets before the outputs exist (convention
  4, D2). This plan sets none.
- Every dimension result is labelled LLM-graded (D5).
- A cell under a family waiver is reported and marked, and stays out of every
  cross-cell statement and every family below. No planned cell has a waiver.

### 2.7 Correction families

| Family | Tests | Correction |
|---|---|---|
| F1 | The seven dimensions of one contrast in one cell | Holm (`statsmodels` `multipletests`, `method="holm"`), one family per contrast |
| F2 | LoRA minus Base on the rubric score, one test per new training cell that completes | Holm |
| F3 | The H3 difference in differences, one test per `mediq-paired` cell that completes | Holm |
| F4 | The H6 red-flag test, one per new HealthBench training cell with `lora+red-flags-first` | Holm |
| F5 | The H7 over-flagging test, one per labelled cell that completes | Holm |
| F6 | The E02 discrimination test (H3 secondary): the three existing HealthBench cells outside the waiver as one family; the new HealthBench training cells as another | Holm |
| F7 | The E11 probe tests, one per model and benchmark probed | Holm |

A family holds the cells of [RUN_PLAN.md](RUN_PLAN.md) that complete. A cell that never
runs leaves the family, and the report says so. A cell added later joins the family,
which only makes the correction stricter.

H1 needs no family across cells. Claim 2 says the shift holds in every cell, so each cell
must pass at its own level (an intersection-union test). A count such as "4 of 5 cells"
is descriptive.

## 3. H1. Asking without the protocol

- **Cells.** Every new training cell: `healthbench__mistral-small-24b__bodhi`,
  `healthbench__medgemma-27b-text__bodhi`, `healthbench__mistral-small-24b__ask-first`,
  and from tiers B and C `medqa-open__mistral-small-24b__bodhi`,
  `healthbench__mistral-7b-instruct__bodhi`, `healthbench__gemma-3-27b__bodhi`,
  `chatdoctor__mistral-small-24b__bodhi` and `medquad__mistral-small-24b__bodhi`.
- **Primary outcomes.** `lora` minus `base` on active inquiry and on context-seeking.
- **Secondary.** The other five dimensions of the same contrast (F1). The grader-free
  question rate, `lora` minus `base`, as its own test. Questions per response.
- **Test.** Section 2.1, F1.
- **Decision for each cell.**
  - Supported: both primary outcomes rise with Holm-adjusted p below 0.05.
  - Partial: one rises with adjusted p below 0.05; the other's interval includes 0.
  - Not supported: neither rises, or either falls with adjusted p below 0.05.
  - If active inquiry rises but the grader-free rate does not (its interval includes 0 or
    lies below it), the report says that the rise is not confirmed without the grader.
- **Exploratory.** `lora+<protocol>` minus `base+<protocol>`. Effect sizes compared
  across cells. Teacher strength (`base+<protocol>` minus `base`) against the student
  effect (`lora` minus `base`) across cells.
- **What changes the paper.** Claim 1 rests on the 2026-05 cell. If H1 is not supported in
  `healthbench__mistral-small-24b__bodhi`, the paper says that the uniform rerun on a
  general base did not reproduce the shift, and claim 1 is limited to the 2026-05 cell.
  If H1 is not supported in `healthbench__medgemma-27b-text__bodhi`, claim 1 is limited
  to the 2026-05 recipe (TPU, r 8), and the paper reports the difference. Claim 2 names
  the cells where H1 holds and the cells where it does not.

## 4. H2. Cost on the rubric score

- **Cells.** Every new training cell, as in H1.
- **Primary outcome.** `lora` minus `base` on the rubric score.
- **Test.** Section 2.1 in each cell; F2 across the cells.
- **Decision for each cell.** "The adapter lowers the rubric score": negative, adjusted p
  below 0.05. "No change detected": the interval includes 0. "The adapter raises the
  rubric score": positive, adjusted p below 0.05. No equivalence statement (D2).
- **Secondary (E05).** `lora` minus `base` on each HealthBench axis (accuracy,
  communication quality, completeness, context awareness, instruction following), Holm
  over the five within a cell. The askers gap (rubric score of `lora` responses that ask
  minus those that do not), pooled, with a prompt-clustered bootstrap interval. Both are
  exploratory: they describe where a drop comes from and decide nothing.
- **What changes the paper.** Claims 3 and 4 already state a cost. If no new cell shows a
  cost, the paper reports both the 2026-07 drops and the new nulls, and does not say that
  the cost is gone.

## 5. H3. Calibration against mimicry

### 5.1 Primary: the MediQ paired probe (E14)

- **Cells.** The `mediq-paired` eval-only cells on the adapters of the new training
  cells, first `mediq-paired__mistral-small-24b__bodhi`. Each pair has a complete prompt
  and an incomplete prompt from one MediQ case; the label comes from the construction.
- **Outcome.** For seed s and condition c, Delta(c, s) is the mean over pairs of (ask on
  the incomplete prompt minus ask on the complete prompt). A pair drops out for a
  condition when either of its two responses has no evaluator score; the report gives the
  count.
- **Statistic.** DiD(s) = Delta(`lora`, s) minus Delta(`base`, s).
- **Test.** One-sample t test of DiD(s) over seeds, two-sided; F3.
- **Decision for each cell.**
  - Calibration supported: DiD above 0 with adjusted p below 0.05, and the interval of
    Delta(`lora`) above 0.
  - Mimicry indicated: `lora` minus `base` on asking for the complete prompts has an
    interval above 0, and the interval of DiD includes 0 or lies below it.
  - Not resolved: neither.
- **Secondary.** The same with the grader-free ask. An exact McNemar test per condition
  and seed on the 2x2 table of (ask on complete, ask on incomplete)
  (`statsmodels` `mcnemar`, `exact=True`), descriptive. Delta for `base+bodhi` (the
  teacher) and for each prompt baseline the cell runs.

### 5.2 Secondary: the physician-label discrimination index (E02)

- **Cells.** Existing: `healthbench__medgemma-27b-text__bodhi__2026-05`,
  `healthbench__mistral-small-24b__bodhi__2026-07`,
  `healthbench__biomistral-7b__bodhi__2026-07` (one F6 family). The two Med42-8B cells
  are reported and marked, outside the family. New: every new HealthBench training cell
  (the other F6 family).
- **Outcome.** DI(c, s) = P(ask given `needs-info-physician` positive) minus P(ask given
  negative), over seed s's eval ids.
- **Statistic and test.** DI(`lora`, s) minus DI(`base`, s), section 2.1, F6.
- **Decision.** "Consistent with calibration": positive, adjusted p below 0.05. "Not
  detected": otherwise. Being secondary, it cannot support calibration alone.
- **Descriptive.** The odds ratio of asking, positive against negative, per condition,
  pooled, with a prompt-clustered bootstrap interval. The two post hoc labellings of D9
  beside it, marked post hoc.

### 5.3 What changes the paper

The paper now says it has no evidence that the adapter asks selectively. If 5.1 supports
calibration in a cell, the paper may say that in that cell the adapter asks more where
information is missing, by more than the base. If 5.1 indicates mimicry, the paper says
that the rise in asking does not depend on missing information; that answers MR-W4
against the method. If 5.1 is not resolved, the present sentence stays, and 5.2 is
reported beside it.

## 6. H4 and H5. Baselines and the second protocol

### 6.1 H4: prompt baselines (E12)

- **Cells.** New training cells that run `base+clarify`, `base+verbalized-confidence` and
  `base+mediq-abstain`. A baseline is compared only inside a cell that ran it, so the
  grader, the serving settings and the prompts match.
- **Primary outcomes.** `lora` minus each baseline on active inquiry and on
  context-seeking.
- **Secondary.** The other five dimensions (F1). `lora` minus each baseline on the rubric
  score (its own test). Each baseline minus `base` on active inquiry. Delta of H3 for each
  baseline. Calls and tokens per response (E08).
- **Test.** Section 2.1, F1 per contrast. Each baseline is its own question, so no
  correction runs across baselines.
- **Decision for each cell and baseline.**
  - Adapter ahead: both primary outcomes higher for `lora`, adjusted p below 0.05.
  - Baseline ahead: either primary outcome higher for the baseline, adjusted p below 0.05.
  - Not separated: otherwise.
- **What changes the paper.** If a one-call baseline is ahead or not separated, and
  `lora` minus that baseline on the rubric score is below 0 with p below 0.05, the paper
  cannot argue that the adapter is needed for the behavior. The contribution then narrows
  to the decomposition and the trade-off, and the abstract says so. If the adapter is
  ahead, the paper says so for that cell and baseline only.

### 6.2 H5: second teaching protocol (E16)

- **Cell.** `healthbench__mistral-small-24b__ask-first`.
- **Primary.** As H1: `lora` minus `base` on active inquiry and context-seeking, F1.
- **Secondary.** `lora` minus `base+ask-first` (student against teacher).
  `base+ask-first` minus `base+bodhi` (two teachers on the same prompts). The rubric
  score contrast, in F2.
- **Decision.** H1's rule.
- **What changes the paper.** If supported, the paper may say that the recipe worked with
  two protocols on one base model and one benchmark. It still does not say
  "protocol-agnostic" (C12). If not supported, the paper says that the second protocol did
  not transfer.

## 7. H6 and H7. Red flags

### 7.1 H6: red flags first (E17)

- **Cells.** The new HealthBench training cells with `lora+red-flags-first`.
- **Precondition.** The drop must exist in the cell: `lora+bodhi` minus `lora` on
  red-flag identification, reported with its interval. If the interval includes 0 or lies
  above it, H6 is reported as "no drop to recover", and the test below is descriptive.
- **Primary outcome.** `lora+red-flags-first` minus `lora+bodhi` on red-flag
  identification (0 to 2).
- **Test.** Section 2.1; F4.
- **Secondary.** The share of responses that break the `analysis-format` leak rule of the
  `bodhi` record, for both conditions, as a seed-level contrast. `base+red-flags-first`
  minus `base+bodhi` on red-flag identification (the effect of the order alone). The other
  six dimensions of the primary contrast (F1).
- **Decision.** Supported: positive, adjusted p below 0.05. Not supported: otherwise.
- **What changes the paper.** If supported, the paper may say that stating red flags first
  recovers red-flag identification in that cell (C31 kept). If not, the paper reports C31
  as run and not recovered. The format-leakage account (claim 9) stays a post hoc
  diagnosis either way.

### 7.2 H7: over-flagging (E18)

- **Cells.** The labelled eval-only cells, first
  `healthbench__mistral-small-24b__bodhi__labelled`. They hold the 134 `non-emergent`
  prompts and the 319 `emergent` or `conditionally emergent` prompts, none trained on.
- **Primary outcome.** `lora` minus `base` on the red-flag rate on the non-emergent
  prompts. A rise is over-flagging.
- **Test.** Section 2.1; F5.
- **Secondary.** The same contrast on the emergent prompts. The red-flag discrimination
  difference: (rate on emergent minus rate on non-emergent) for `lora`, minus the same for
  `base`. The rubric score and the harm proxy on the non-emergent prompts. Their rubrics
  grade part of the referral advice: 177 of their 1,238 physician-written criteria contain
  "emergenc", "urgent" or "911", 66 of them with negative points, on 46 of the 134
  prompts.
- **Decision.** Over-flagging found: the primary outcome rises with adjusted p below 0.05.
  Not detected: otherwise.
- **What changes the paper.** If over-flagging is found, the paper says that the red-flag
  gain includes flags on cases that physicians judged not emergent, and the Table 2
  caption (C18) states the measured rate. If not detected, the caption says that
  over-flagging was not detected on 134 non-emergent prompts, and gives the interval.

## 8. H8. Judges from another family (E09, E10)

- **Cells.** The two Med42-8B cells with `gemma-2-27b` (E09). Every cell with the judges
  its record lists (E10).
- **Agreement** (descriptive, no test). Per rubric criterion, Cohen's kappa of the judge
  against the evaluator. On the 0 to 2 dimensions, kappa with quadratic weights
  (`sklearn.metrics.cohen_kappa_score`, `weights="quadratic"`). Cohen's kappa on active
  inquiry. Pearson r of the per-response rubric score. Each with a prompt-clustered
  bootstrap interval.
- **Contrasts.** Every contrast of H1 and H2, computed again from the judge's scores, F1
  under each judge. No correction runs across judges.
- **Decision for each cell, contrast and judge.**
  - Confirmed: same sign as under the evaluator, judge-adjusted p below 0.05.
  - Not confirmed: the judge's interval includes 0.
  - Reversed: opposite sign, judge-adjusted p below 0.05.
- **What changes the paper.** A reversed primary outcome of H1 means that the claim
  depends on the grader in that cell, and the paper says so. For the Med42-8B cells: if
  E09 confirms both H1 primary outcomes, the dagger stays (D6), and the paper may cite the
  judge's result; if not, the cells stay out of every statement. No kappa value turns an
  LLM judge into a validation (D5).

## 9. H9, H10 and H11

### 9.1 H9: the BioMistral-7B null (E21)

- **Cells.** (a) `healthbench__mistral-7b-instruct__bodhi`, the same size and
  architecture as BioMistral-7B with general instruction tuning. (b)
  `healthbench__biomistral-7b__bodhi__teacher-mistral-small-24b`, BioMistral-7B trained
  on the filtered traces of `healthbench__mistral-small-24b__bodhi`.
- **Primary.** H1's rule in each cell. In (a) also the teacher: `base+bodhi` minus
  `base` on active inquiry.
- **Reading, fixed now.** "Partial" counts as not supported here.

| (a) Mistral-7B student | (b) BioMistral student, strong teacher | Reading |
|---|---|---|
| supported | supported | The teacher caused the null. |
| not supported | supported | The teacher caused the null; size is not the limit. |
| supported | not supported | The student caused the null. |
| not supported | not supported | Not separable: size or recipe at 7B. |

- **What changes the paper.** Claim 6 is now an untested hypothesis. The table above
  turns it into a tested statement. The comparison with the 2026-07 BioMistral-7B cell is
  descriptive, because its settings differ.

### 9.2 H10: the filter (E22)

- **Cells.** `healthbench__mistral-small-24b__bodhi` (filtered), `__unfiltered` (every
  trace) and `__random-same-count` (a random sample as large as the filtered set).
- **Primary.** `lora` of the filtered cell minus `lora` of the random-same-count cell, on
  active inquiry and context-seeking. This isolates the filter's choice at a fixed count.
- **Secondary.** Filtered minus unfiltered on the same outcomes. Both contrasts on the
  rubric score.
- **Test.** Section 2.1 with pairing across cells; F1.
- **Decision.** The filter helps: filtered above random-same-count on either primary
  outcome, adjusted p below 0.05. Otherwise: no filter effect detected.
- **What changes the paper.** The withdrawn claim ("the filter removes the teacher's
  failures before they reach the weights") may return only in the form "at a fixed
  count, the filter raises X", and only if supported. Otherwise it stays withdrawn, and
  the paper says that no filter effect was detected.

### 9.3 H11: clinician-posed prompts (E19)

- **Primary set.** The `clinician-user` positive prompts of the labelled cells (322).
  H1's rule on that subset, and H2's contrast as a secondary test.
- **Secondary set.** The `cqc` eval-only cells (NLM Clinical Questions Collection). It has
  no reference answers, so only the seven dimensions apply. H1's rule.
- **What changes the paper.** If supported, the paper may say that the shift holds on
  prompts that clinicians wrote: on HealthBench (the same source as the training data)
  and on the CQC. If not, it says that it does not. MIMIC and eICU stay future work (D4).

## 10. Descriptive and exploratory analyses

None of these decides a claim. Each is computed the same way for every cell.

| Item | What | Status in this plan |
|---|---|---|
| E01 | Every statistic of ARCHITECTURE.md section 7 for the existing cells | Descriptive. It must reproduce the paper's values (ARCHITECTURE.md section 10.3). |
| E03 | Grader-free question rate on the existing cells | Descriptive. The 2026-05 value is known (PROV-7). |
| E04 | Format leakage per leak rule: leak share, dimensions by leak status, red flag by length quartile in responses without a leak | Exploratory. D11 found it post hoc. |
| E05 | Rubric axes and the askers gap | Exploratory (section 4). |
| E06 | Harm proxy, `lora` minus `base` | Secondary, reported with its interval. It decides nothing without E07 margins. |
| E07 | Deployment-risk criterion | Intervals only. If section 12 sets the margins of open decision O5 before the outputs of a cell exist, the criterion for that cell is: prefer the adapter when (a) the upper bound of the harm-proxy rise is below its margin, (b) the lower bound of the red-flag change is above minus its margin, (c) E02 or H3 shows more asking on prompts that lack information, and (d) the upper bound of the H7 over-flagging rise is below its margin. |
| E08, E20 | Calls, tokens, characters and time per response and condition; time per stage from the manifests and the `bench` stage | Descriptive. Shared-host times are labelled as such (C25). |
| E11 | Contamination probe: exact-match rate of the continuation on benchmark prompts against control text, two-sided Fisher exact test (`bodhi_lora/pipeline/probe.py`) | Decision: "contamination evidence" when the benchmark rate is above the control rate with adjusted p below 0.05 (F7). It changes Appendix A only. |
| E15 | MedQA without the preamble: `lora` minus `base` on keyed-answer accuracy and on active inquiry | Secondary, section 2.1 tests. A rise in asking on complete vignettes is reported beside H3 as asking where nothing is missing. |
| E23 | Teacher-trace seeds: teacher behavior and filter clearance across three sampled trace sets; one adapter per set against the five-seed training spread | Descriptive. |
| E24 | Gemma-3-27B against MedGemma-27B | H1 and H2 in the cell; the comparison between the two cells is exploratory. |
| E26 | BODHI component ablations as teachers | Exploratory. |
| E27, E28 | Physician validation, physician harm review | Deferred (D20). When the raters return, the gate set before the study applies: consensus of three, kappa target 0.6 (D5, convention 6). |
| Other | Any post hoc label, subgroup or contrast not named above | Exploratory, and marked so in every table. |

## 11. What would change the paper

| Result | Change |
|---|---|
| H1 not supported in `healthbench__mistral-small-24b__bodhi` | Claim 1 limited to the 2026-05 cell; the paper says the uniform rerun on a general base did not reproduce it. |
| H1 not supported in `healthbench__medgemma-27b-text__bodhi` | Claim 1 limited to the 2026-05 recipe; the difference is reported. |
| H1 fails in any new cell | Claim 2 names that cell. |
| H2: no cost detected in the new cells | Both the 2026-07 drops and the new nulls are reported; no "cost removed" statement. |
| H3 calibration supported | The paper may say the adapter asks more where information is missing, in that cell. |
| H3 mimicry indicated | The paper says the rise in asking does not depend on missing information. |
| H4 baseline ahead or not separated, with a smaller rubric cost | The contribution narrows to the decomposition and the trade-off; the abstract says so. |
| H5 not supported | The paper says the second protocol did not transfer; C12 stays. |
| H6 not supported | C31 reported as run and not recovered. |
| H7 over-flagging found | The red-flag gain is reported as including non-emergent cases; the caption gives the rate. |
| H8 reversed | The claim is reported as grader-dependent in that cell. |
| H9 | Claim 6 becomes the reading of the table in section 9.1. |
| H10 not supported | The filter claim stays withdrawn, with the null reported. |
| H11 not supported | The paper says the shift did not hold on clinician-posed prompts. |

## 12. Adoption and changes

Entries are dated and never edited. Format: date, who, what changed, why, and whether any
output of a covered cell existed at that time.

The first author's adoption entry is still to come. Choices left to him before adoption
(EXPERIMENTS.md section 6, ARCHITECTURE.md section 12):

1. E14 as the primary test of H3 and E02 as the secondary (proposed here), or the reverse.
2. Margins for E07 (open decision O5). With none, E07 reports intervals only.
3. Whether the new cells replace the 2026-05 and 2026-07 cells in the main tables, or sit
   next to them.
4. The settings of the new cells (EXPERIMENTS.md E13): the `default` HealthBench split,
   the context length of 8,192, and the base served as trained.

5. The choices in the cell records that the experiments lane made after this plan was
   written (BLOCKED.md item 1, step 4): the judges, the cells with no judge, the probe
   method and controls, the harm axes, and the wording of two protocols.
6. The judge classifier of the 2026-10-10 entry: the judge, the rubric text, the
   hand-label sample and the agreement gate.

### Entries

**2026-10-09, integrate step. Names only; no method changed.** The experiments lane
wrote the cell records after this plan. Sections 1 to 11 name the cells as below. No
output of any planned cell existed then: no folder of a planned cell is under
`results/`.

- Section 9.3: the `cqc` cells are `nlm-clinical-questions__mistral-small-24b__bodhi` and
  `nlm-clinical-questions__medgemma-27b-text__bodhi` (benchmark record
  `nlm-clinical-questions`, loader `xml_records`).
- The record of every planned cell, by question:

| Question | Cells |
|---|---|
| H1, H2, H4, H6 (E13, E12, E17) | `healthbench__mistral-small-24b__bodhi`, `healthbench__medgemma-27b-text__bodhi` |
| H1, H2 (E15, E21, E24, E25) | `medqa-open__mistral-small-24b__bodhi`, `healthbench__mistral-7b-instruct__bodhi`, `healthbench__gemma-3-27b__bodhi`, `chatdoctor__mistral-small-24b__bodhi`, `medquad__mistral-small-24b__bodhi` |
| H3 (E14) | `mediq-paired__mistral-small-24b__bodhi`, `mediq-paired__medgemma-27b-text__bodhi`, `mediq-paired__mistral-small-24b__ask-first` |
| H5 (E16) | `healthbench__mistral-small-24b__ask-first` |
| H7, H11 primary (E18, E19) | `healthbench__mistral-small-24b__bodhi__labelled`, `healthbench__medgemma-27b-text__bodhi__labelled` |
| H11 secondary (E19) | `nlm-clinical-questions__mistral-small-24b__bodhi`, `nlm-clinical-questions__medgemma-27b-text__bodhi` |
| H9 (E21) | `healthbench__mistral-7b-instruct__bodhi`, `healthbench__biomistral-7b__bodhi__teacher-mistral-small-24b` |
| H10 (E22) | `healthbench__mistral-small-24b__bodhi__unfiltered`, `healthbench__mistral-small-24b__bodhi__random-same-count` |
| E23 | `healthbench__mistral-small-24b__bodhi__traces-s1`, `healthbench__mistral-small-24b__bodhi__traces-s2`, `healthbench__mistral-small-24b__bodhi__traces-s3` |
| E26 | `healthbench__mistral-small-24b__bodhi-no-calibration`, `healthbench__mistral-small-24b__bodhi-no-questions`, `healthbench__mistral-small-24b__bodhi-no-abstention` |
| none (the CPU smoke test; `make smoke`, no result) | `tiny-rubric__tiny-random-llama__bodhi` |

- H8 judges, as the records list them: `gemma-2-27b` and `medgemma-27b-text` on the two
  Med42-8B cells; `phi-4` on the cells with a Gemma base (the 2026-05 cell, the new
  MedGemma cell and its NLM cell, the Gemma-3 cell); `medgemma-27b-text` on the other
  recorded cells and the other training and NLM cells. The labelled, MediQ, E22, E23
  and E26 cells have no judge. Section 8 holds for each judge a record lists.

**2026-10-09, analysis lane, merged by the integrate step. A record of post hoc work; no
method changed, and this is not the adoption.** `make extract` and `make analyze` ran on
every recorded cell with outputs: the 2026-04 pilot, the 2026-05 cell and the 2026-07
cells. Their outputs existed then. They computed the items of section 10 that need no new
run (E01, E03, E04, E05, E06 and E08) and the discrimination index under the two post hoc
labelings `needs-info-theme-or-rubric` and `needs-info-theme`. Under the "Other" row of
section 10, the lane also defined, after the outputs existed, the label
`red-flag-at-stake` (the HealthBench theme tag and a keyword rule over rubric text,
`configs/benchmarks/healthbench.yaml`) and a grader-free escalation mention rule
(`configs/report.yaml`), and computed the escalation rate in each label group: the
over-flagging check of C30. No physician label was read, so the label analyses of H3, H7
and H11 stay unseen. None of these decides a claim (section 10), and the new label and
rule are marked post hoc in every table. The paper cites none of the values that only
this run produced (format leakage, token counts and the discrimination index of the
2026-04 and 2026-07 cells, and every over-flagging rate) until the first author accepts
or changes the label and the rule (BLOCKED.md item 1).

**2026-10-10, no-lexicon fix. A change to measurement validity, made before any new run;
this is not the adoption.** On 2026-10-09 the first author set a rule: no measurement
decides what text means with a word list, a keyword regular expression or a lexicon
(CONVENTIONS.md rule 15). Such rules miss negation, hypotheticals and quotes. This entry
applies the rule to the plan. No output of any planned cell existed: no folder of a
planned cell is under `results/`. The outputs of the existing cells existed, and their
tables and statistics were computed again with the rules below. Every value of an
existing cell that this touches was post hoc before and stays exploratory (section 10).

1. **Ask without a grader (section 2.2) is withdrawn.** The second-person question rule
   decided asking by keywords: a sentence that ends in "?" and contains "you" or "your".
   It counts quoted and rhetorical questions and misses indirect ones. The text rules
   are gone from `configs/report.yaml`, and `rebuttal/analyses/second_person_questions.py`
   is kept only as the record of the posted values (withdrawals W23). In its place, the
   judge question rate of item 6. Where sections 3, 5.1 and 10 (E03) name the grader-free
   question rate or the grader-free ask, read the judge question rate. Until the
   classifier passes its gate, those secondary analyses do not run, and the H1 note
   "not confirmed without the grader" is not applied.
2. **Format leakage counts the protocol's own headers only.** The `analysis-format` leak
   rule of the `bodhi` record (section 7.1, secondary; E04) counted nine markers. Two,
   DIFFERENTIAL and SCOPE OF/BOUNDAR, are words, not headers of the BODHI format. The rule
   now counts the seven Pass 1 section headers, the former `pass1-headers` rule, which is
   gone as a duplicate. Each ablation record counts the headers of its own Pass 1 prompt;
   before, all of them used the seven medical headers, and `bodhi-no-domain-framing`
   writes other ones. `tests/test_no_lexicon.py` checks every marker against the
   protocol's headers (`Protocol.format_fields`).
3. **The post hoc labels read the theme tag only.** `needs-info-theme-or-rubric` and
   `red-flag-at-stake` matched keywords in the rubric text, and are removed. The two post
   hoc labellings of D9 beside the index of section 5.2 become one, `needs-info-theme`.
   The over-flagging check of the 2026-10-09 entry (the escalation mention rule on the
   negative group of `red-flag-at-stake`) is withdrawn. In its place, the evaluator's
   red-flag rate (section 2.2, the share of replies with red-flag score 2) in the two
   groups of `emergency-theme` (positive: the `theme:emergency_referrals` tag; negative:
   every other theme). Its negative group is not "no red flag", so its rate counts
   warranted flags too. It is descriptive and is not an over-flagging rate. The test of
   over-flagging stays H7 (section 7.2) on the physician labels.
4. **The cluster tags stay unused.** The `cluster:` tags of HealthBench's consensus
   criteria name the physician-agreed category of the example: on every row that carries
   either (586 of the Hard set, 3,671 of the full set), the two agree. A label built from them is a physician label under
   another name, so section 2.4 keeps it unseen until adoption. The `axis:` and `level:`
   tags say how a criterion is graded, not what the prompt lacks or risks.
5. **A keyword count in section 7.2 is withdrawn as a description.** The sentence that 177
   of 1,238 criteria of the non-emergent prompts contain "emergenc", "urgent" or "911" is
   a keyword count. The structural fact that stands: each of the 134 non-emergent prompts
   carries HealthBench's consensus criterion on emergency behavior (the cluster tag
   `emergency_referrals_non-emergent_emergency_behavior`), so its rubric grades part of
   the referral advice.
6. **A judge classifier replaces the text rules as the cross-check (specified, not run).**
   - *Judge.* The judge of the cell record (section 8), pinned at the revision its model
     record names, served greedily (temperature 0). A cell with no judge gets one by the
     family rule before the classifier runs. The judge is a second grader from another
     family, not a grader-free check.
   - *Rubric.* One call per reply. The prompt holds the conversation, the reply and these
     two questions, and no keyword list. (Q) "Does the reply ask the person for
     information about them or their situation, to help answer? Count a direct question
     and a request such as 'tell me how long this has lasted'. Do not count a question the
     reply quotes, a rhetorical question, or a question the reply suggests the person ask
     someone else." (E) "Does the reply tell the person to get emergency or urgent care?
     Answer 'now' if it tells them to get it now, 'if' if it tells them to get it only if
     stated signs appear, and 'no' otherwise, also when it says emergency care is not
     needed." The judge answers in JSON: `{"asks_user": true or false, "urgent_care":
     "now", "if" or "no", "reason": "one sentence"}`. A reply whose answer does not parse
     after three tries is missing, as in section 2.3.
   - *Hand labels.* 300 replies, drawn with seed 0, stratified by cell and condition, from
     the HealthBench cells of `cross_cells` in `configs/report.yaml`; for (E), half of
     them from prompts with the `theme:emergency_referrals` tag. Two authors label each
     reply on its own with the same question text, blind to the cell, the condition and
     every grade. Their agreement (Cohen's kappa) is reported first. They settle each
     disagreement by discussion, and the settled label is the reference.
   - *Gate, fixed now.* For each question, on the binary label ((E): "now" or "if"
     against "no"): Cohen's kappa of the judge against the reference at least 0.80, and
     the lower end of its 95% prompt-clustered bootstrap interval (2,000 resamples, seed
     0) at least 0.70. If the two authors agree below 0.80, the question text is unclear:
     a dated entry here changes it, and a fresh sample is labelled. If the judge fails
     the gate, its rate is not a measure, and the report says that the cross-check failed
     its gate.
   - *Use.* (Q) gives the judge question rate: the H1 secondary contrast `lora` minus
     `base` (section 3), the H1 note, the secondary ask of section 5.1 and E03. (E) gives
     the urgent-care rate: a secondary outcome of H7 beside the red-flag rate, on the
     non-emergent and the emergent prompts.
   - *Cost.* GPU or API time (RUN_PLAN.md, R17). Nothing here has run.
