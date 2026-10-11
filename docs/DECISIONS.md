# Decision log

Decisions for the resubmission of NeurIPS 2026 submission 32818. They were made on
2026-10-07 by the first author's delegate, after the audit of the same date. Each
decision is quoted verbatim. The line under it gives the reason, drawn from the audit
finding named in brackets.

To change a decision, add a new dated entry below it that says what changed and why.
Do not edit the text of an old entry.

**Status (2026-10-11).** The first author delegated D1 to D16 on 2026-10-07 and asked
that they be applied. They are applied under that delegation, and the audit items closed
on them are closed. D17 to D21 and the quoted sentence of D23 are his; D22 applies his
requests. On 2026-10-11 he adopted the analysis plan (D23). Its choices 3 and 7, and the
open decisions O1, O2, O7 and O8 of ARCHITECTURE.md section 12, are proposals that wait
for his yes (D25 corrects D23 and D24; BLOCKED.md item 1). The first author may still
revise any decision with a dated entry below it.

## 2026-10-07

**D1** LoRA recipe: the paper reports what actually ran. The submitted MedGemma-27B cell used the tunix path; repo evidence (launch_5seeds_tunix.sh exports with --r 8 --alpha 16, the only committed adapter at 607440e is r=8, alpha=16) says r=8, alpha=16. Derive the full submitted recipe (rank, alpha, target modules, epochs/steps, effective batch, seq len, optimizer, lr schedule, hardware) from history at the commit that produced prod-v39-tunix-27b-neurips and correct the hparams table. Report the rebuttal cells' actual configs (rebuttal/configs/*.yaml, r=16 alpha=32, QLoRA 4-bit for Mistral-Small-24B) in a separate per-cell table. Never say "unchanged" or "every hyperparameter fixed". Mark the submitted adapter config as pending direct confirmation from the GCS adapter_config.json files.

Why: the paper's r=16, alpha=32 table matches neither the tunix launcher nor the only committed adapter, so "every hyperparameter fixed" was not supported. [HYG-4, PROV-3]

**D2** Aggregate quality: drop "preserved", "non-inferiority", "no degradation", "statistically indistinguishable". Report the seed-level paired difference LoRA minus Base with a t(4) 95% CI for every cell, and state the trade-off between behavior gain and aggregate cost plainly, including in the abstract. No post-hoc equivalence margin.

Why: "preserved" rested on a non-significant test with no stated margin, and the aggregate score fell significantly in all six added cells. [EVAL-1, PROV-8, CLAIM-2]

**D3** Title: remove the unmeasured "calibration" claim. New title: "Ask Before You Answer: Distilling Clinical Epistemic Behaviors into Model Weights". Use "epistemic behaviors" for the seven dimensions; "calibration" only as background motivation, never as a measured property.

Why: the title and abstract claim calibration, but the paper reports no calibration metric. [CLAIM-7]

**D4** Scope: patient-facing, open-ended clinical question answering. Remove "clinical decision support" as the target. Clinician-posed settings (MIMIC, eICU) are future work.

Why: reviewers found that the model and benchmarks do not fit clinical decision support, and every benchmark used is patient-facing. [CLAIM-3; MR-W1, FzpA-W2]

**D5** Physician validation: amend on the record. Report 1 of 3 raters returned, kappa 0.35 (n=509 if the recomputation confirms it) with a prompt-clustered bootstrap CI, below the pre-registered 0.6 target. Label every epistemic-dimension result as LLM-graded. Do not claim validation.

Why: the pre-registered design needed more than one rater and a kappa of at least 0.6; one rater returned and the gate failed without a dated amendment. [EVAL-5, PROV-5]

**D6** Med42-8B cells: keep them, mark them with a dagger in every table as "base model shares the evaluator's Llama-3 family, which breaks the paper's grader-separation rule", exclude them from any cross-model statement, and ship a runnable regrade with a judge outside Llama, Qwen and the graded base's family.

Why: Med42-8B is built on Llama-3, the evaluator's family, and the planned mitigation was never run. [EVAL-7]

**D7** Statistical unit: the seed is the unit of inference (paired t over 5 seeds, t intervals). Prompt-level pooled Cohen's d is descriptive only, with a prompt-clustered bootstrap CI. State 667 unique prompts and response counts per condition. Holm correction over the seven dimensions within each contrast.

Why: "n≈1,000 per condition" counted responses over 667 unique prompts, the main and added cells used different units, and the correction family was never stated. [CLAIM-10, REPRO-8, EVAL-13]

**D8** Canonical paper: paper/main.tex. The submitted version is frozen at paper/submitted/neurips2026/ (source only, anonymous). All other copies move out of the repo to the archive folder next to it, `../_archive/paper_copies_2026-10-07/`.

Why: the repo held seven copies of main.tex in five versions, and none was the committed submitted source. [HYG-9, CLAIM-4, CLAIM-5]

**D9** Discrimination analysis: report both labelings (rubric-based and theme-only) side by side, both labelled post hoc and exploratory. No primary.

Why: the labels and the leakage split were defined and committed together with their results, with no prior specification. [EVAL-10]

**D10** Blanket disclaimers: remove from the abstract; report counts with a two-sided Fisher exact p in results (5/998 vs 0/985, p=0.062 if recomputation agrees).

Why: "blanket disclaimers eliminated" rests on 5 events against 0 and is not significant. [CLAIM-8]

**D11** CoT plus LoRA interference: replace "compete for the same behavioral capacity" with the format-leakage account, labelled as a post hoc diagnosis.

Why: the red-flag drop comes mostly from Pass-1 analysis format leaking into the answer, an account found after the results. [CLAIM-2]

**D12** All rebuttal cells move into the main evaluation, including the BioMistral-7B null and the MedQA cell with the reason it does not measure the target. Numbers that exist only as posted rebuttal text are typeset from data/rebuttal_cells/posted_values.json with a visible "transcribed from posted values; raw outputs pending sync" footnote.

Why: the added-cell numbers exist only as posted prose, and the replies promised that all added cells would enter the main evaluation. [PROV-1; HomM consolidation]

**D13** Parse failures: report per-condition rates for both graders and a worst-case sensitivity (failures scored as not-met) instead of "<0.1% across all runs".

Why: epistemic-grader parse failures are dropped, differ by condition, and contradict the paper's "<0.1%". [EVAL-9]

**D14** Training-loss figure: remove it (it comes from an early 5-epoch run on 838 traces, not the reported adapters).

Why: the figure and its caption describe a different run from the one the paper reports. [PROV-6]

**D15** Name: BODHI everywhere in the paper and docs, with one expansion checked against the cited source. Repo slug rename is a remote action, deferred.

Why: the repo used BOHDI and BODHI and two acronym expansions, and the taxonomy drifted between documents. [CLAIM-13]

**D16** Public state: nothing remote changes in this run. History purge and remote branch deletion are prepared as scripts, not run.

Why: a history rewrite and remote branch deletion change public state, which is the first author's call. [Audit section 5, decisions 7 and 8; HYG-1, HYG-12, HYG-15]

## Notes on applying the decisions

- D15, expansion check (2026-10-07). The paper cites BODHI as Arslan et al., "Engineering
  Framework for Curiosity-Driven and Humble AI in Clinical Decision Support", BMJ Health
  & Care Informatics 33(1):e101877 (DOI 10.1136/bmjhci-2025-101877). Its abstract, as
  indexed by Europe PMC, expands BODHI as "Balanced, Open-minded, Diagnostic, Humble,
  and Inquisitive". The docs use that expansion. `paper/main.tex` uses "Bridging, Open,
  Discerning, Humble, Inquiring", which matches a different paper by the same group
  (PLOS Digital Health, 2026). The paper must either use the BMJ expansion or cite the
  paper whose expansion it uses.
- D15, mapping (2026-10-08). The cited BMJ paper defines two virtues, curiosity and humility,
  but not the five components one by one. The posted reply (`FzpA_1_rebuttal.md:34`) mapped
  the dimensions onto the other expansion (Bridging, Open, Discerning, Humble, Inquiring).
  The paper now gives one mapping onto the BMJ names, read from the names, the two virtues and
  the Pass 1 fields of bodhi-llm 0.1.4: Inquisitive covers active inquiry and context-seeking;
  Humble covers uncertainty acknowledgment, hedging quality and scope bounding; Diagnostic
  covers red-flag identification and specificity; Balanced and Open-minded have no dimension
  of their own (`paper/main.tex`, Table bodhi-map). The installed bodhi-llm 0.1.4 package
  still expands the acronym the other way in its docstrings and cites the PLOS Digital Health
  paper.
- D12, after the sync (2026-10-08). The added cells' raw outputs were copied from the GPU
  host (commit 3b19a4c). Their numbers now come from `results/cross_cell/generated_values.json`,
  which `make cells` computes from those outputs, with a paired t(4) interval for every cell.
  The "transcribed from posted values; raw outputs pending sync" footnote is gone from the
  paper (72a9565). `rebuttal/posted_values.json` stays as the record of what was
  posted; one posted value differs from the raw outputs (`rebuttal/withdrawals.md` W21). The
  MedQA cell is in Table 3 with its aggregate score and interval. Its only run is the
  preamble run, so it stays set aside with that reason, and its seven dimensions stay out of
  the table because seed 42 has no epistemic grades (W12).
- D1, the GCS check (2026-10-08). The bucket `gs://bohdi-runs-tokyo-micron` returns 404, and
  its project has no buckets, so the submitted adapter configs cannot be read there. The
  MedGemma-27B recipe stays as derived from the code in `legacy/`, and the paper now says it
  could not be checked against the saved files instead of "pending". The evidence left: the
  launcher, config and exporter in `legacy/`, the early committed adapter at 607440e (an
  HF-path run, not an evaluated adapter), and possibly the Modal volume `bohdi-data-cache`,
  into which `legacy/modal_eval.py` (lines 119 and 435-447) copied each seed's `checkpoints/best/` and
  `train.jsonl`/`val.jsonl` (old BLOCKED.md item 3, dropped by D19). The added cells'
  recipe is now read from their synced adapter and trainer files
  (`results/cross_cell/cell_recipe.json`, Table F.2).
- D6, after the sync (2026-10-08). The askers-minus-non-askers count is now made without
  the Med42-8B cells (Appendix J). The Med42-8B cells stay in the Holm family of the
  aggregate tests (the six added cells that are not set aside), as in the posted tests; that
  family is a correction, not a cross-model statement.

## 2026-10-08

Facts and decisions that the first author gave on 2026-10-08. They are recorded as given,
not quoted. Each one says which earlier entry or item it changes. D17 to D21 are his;
D22 records the design that answers his request of the same day.

**D17** The physician raters gave consent.

Changes: the step "ask the raters what they consent to" (old BLOCKED.md item 1) is done,
and consent no longer drives a history purge (D18). The cleanup still writes no rater
name into a tracked file: raters stay AnonClinician1 to 4 (convention 10).

**D18** The repository stays public. There is no privacy purge of history. A purge is
optional, and if it runs it removes only the AI-tool traces (the two trailers of
`def7eee` and `9a0dbd6`).

Changes: the steps to make the repository private and to purge rater names and the author
block (old BLOCKED.md items 1 and 10.2) are dropped. D16 still holds: nothing remote
changes without the first author's yes.

**D19** No Modal. The submitted MedGemma-27B adapters are treated as not retained.
History and docs may still say where the submitted evaluation ran.

Changes: the check of the Modal volume `bohdi-data-cache` (old BLOCKED.md item 3) is
dropped. The MedGemma-27B recipe of D1 stays as derived from the code in `legacy/`, now
for good: no saved adapter will confirm it. The paper's Code availability paragraph and
checklist items 4, 5, 11 and 13 must say that the adapters are not retained, instead of
"we have not yet checked a second copy". Every new experiment on MedGemma-27B needs a
rerun, which is a new cell (open decision O7 in ARCHITECTURE.md).

**D20** The two remaining physician raters are deferred.

Changes: D5 stands as written: one of three raters, kappa 0.35, below the 0.6 target, and
every dimension result is LLM-graded. C28 and the physician harm review (C33, E28) wait
for the first author.

**D21** No GPU use until the first author says so.

Changes: every run of EXPERIMENTS.md waits, listed with its command and cost in
[RUN_PLAN.md](RUN_PLAN.md). CPU dry runs with tiny public models stay allowed.

**D22** Architecture. On 2026-10-08 the first author asked that the code be made general
and its conventions fixed. The design of [ARCHITECTURE.md](ARCHITECTURE.md) does that:

- one registry: every model, benchmark, protocol and cell is one YAML record in
  `configs/`, checked by a schema;
- one runner: `python -m bodhi_lora run <cell id>` runs every stage of every cell, and the
  Makefile calls only the command line and the repository tools;
- one results root: `results/<cell id>/seed_<n>/<stage>/`, with a manifest in every stage
  folder;
- no names in code: the package names no model, benchmark, results path or seed list;
- uniform analysis: the same code computes every statistic for every cell;
- a dated analysis plan before any new run ([ANALYSIS_PLAN.md](ANALYSIS_PLAN.md)).

A record of a past run is frozen: the runner refuses to run it again, except to add judge
re-grades. `legacy/` keeps the launchers of the 2026-05 cell as the record of what ran.
CONVENTIONS.md rules 3, 7 and 11 to 14 say how each part is enforced. The open choices in
ARCHITECTURE.md section 12 (O1 to O8) stay the first author's.

Why: the code had two launch paths, six result locations, analysis that special-cased
one cell, condition names in code and settings in four places (ARCHITECTURE.md section
1). Each review phase added its own copy, which is how the evaluation became hard to
check. [HYG-5, REPRO-3, REPRO-6, REPRO-7]

## 2026-10-11

**D23** "the analysis plan is good" (the first author, in chat, 2026-10-11).

Changes: docs/ANALYSIS_PLAN.md is adopted as written. Its section 12 records his entry
and what it settles for each open choice: E14 primary and E02 secondary for H3; no E07
margins, so intervals only (O5); the new cells sit next to the 2026-05 and 2026-07 cells,
which stay the record; the E13 settings; the choices in the cell records as written; the
judge classifier as specified, used only after its gate; and the frozen evaluator template
for the new cells (O10). The post hoc values that the paper cites stay in the paper, marked
post hoc. The GPU go-ahead stays open (D21).

**D24** The open decisions of ARCHITECTURE.md section 12, settled on 2026-10-11 to apply
D23: O1, keep the package name `bodhi_lora`; O2, the default, one tracked rule for every
cell, with a tracked response table in each; O3, a versioned archive of all raw outputs
is deferred, because publishing it is a remote action; O7, the MedGemma-27B rerun is a new
QLoRA cell and the 2026-05 cell stays the record; O8, git tracks the bench output and the
probe's rates and test, and ignores the probed benchmark text and the continuations. O6
(GPU priorities and where the runs go) and O9 (a merged-weights condition in the timing
bench) stay open.

Changes: D22 said that O1 to O8 stay the first author's; O1 to O5, O7 and O8 are now
closed. O2 removed the 2026-05 exception from `.gitignore`: that cell's raw eval and grade
JSON left the tree and stay in history at 843c90e, and the three figures that read them
read the response tables instead (the PDFs are byte-identical). O8 split the probe output
into `probe.json` (tracked) and `samples.jsonl` (ignored).

Why: O1, O2 and O7 take the option that ARCHITECTURE.md section 12 writes down. O3 needs a
remote action, which D16 keeps his. O8 had two options, track both files or ignore the
probe output; this keeps benchmark text out of a public repository and keeps the probe's
counts and test checkable from git. He may revise any of them with a dated entry.

**D25** Correction of D23 and D24 (the plans lane, 2026-10-11). The first author wrote one
sentence, "the analysis plan is good". It adopts the plan's proposals: choices 1, 2, 4, 5
and 6 of ANALYSIS_PLAN.md section 12. It does not choose where the plan makes no
proposal, and it does not answer the open decisions of ARCHITECTURE.md section 12.

Changes: the parts of D23 that settle choice 3 (the new cells next to the recorded cells)
and choice 7 (the frozen evaluator template for the new cells, O10), and all of D24, are
proposed defaults, not his decisions. Each waits for his yes in BLOCKED.md item 1: O1,
keep the name `bodhi_lora` (nothing changed); O2, one tracked rule for every cell; O7, the
MedGemma-27B rerun is a new QLoRA cell; O8, git tracks the probe's `probe.json` and
ignores its `samples.jsonl`. O3 stays deferred, because publishing an archive is a remote
action (D16). The O2 and O8 changes are in the tree as the proposed defaults, and each can
be undone. For O2, put back the two `.gitignore` lines for the 2026-05 cell's
`seed_*/eval/*.json` and `seed_*/grade/*.json` that d224bf8 removed, and add the 25 files
again (history keeps them at 843c90e, and the files are still in this checkout). For O8,
add a `.gitignore` exception for the probe's `samples.jsonl`. Neither change touched a
number of the paper. The ANALYSIS_PLAN.md section 12 entry of the same date records the
same correction.

Why: a choice of method is the first author's, and D22 kept O1 to O8 his. The entry that
settled them was written by the delegate, under his heading.

### D26. The proposed defaults accepted (2026-10-11)

The first author answered "Accept all (Recommended)" to the proposed defaults that D25
left open: choice 3 (new cells sit next to the recorded ones), choice 7 / O10 (keep the
frozen evaluator template), O1 (`bodhi_lora`), O2 (one tracked rule), O7 (R7 is a new
QLoRA cell) and O8 (track `probe.json`, ignore `samples.jsonl`). Recorded in
ANALYSIS_PLAN.md section 12. Still open: the R7 4,096-token pair, O9 and the R17 strata.
