# Blocked on the first author

What is left that only Peter can do: a decision, a go-ahead for compute, a remote action,
a login or a fact. Rewritten on 2026-10-08 after his facts of that day (DECISIONS.md D17
to D22). Items are in priority order. Each one says what to do and what changes in the
repo after it. Run commands from the repo root.

D-numbers refer to [docs/DECISIONS.md](docs/DECISIONS.md), C-numbers to
[rebuttal/commitments.md](rebuttal/commitments.md), audit IDs (PROV-1, EVAL-7) to
`CONVENTIONS_AUDIT_2026-10-07.md` next to the repo. The last section maps the old item
numbers, which other files still cite, to the items below.

Nothing here was pushed, and no remote state changed.

## 1. Adopt the analysis plan and the open design choices

Why: convention 3 needs a dated plan before any new run, and before the physician-label
analyses run on the existing outputs. The plans step drafted it; the choices in it are
methodology, so they are yours.

Action:

1. Read [docs/ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md). Adopt it as it is, or change it
   with a dated entry in its section 12. Its section 12 lists seven open choices: E14 or
   E02 as the primary test of calibration, the E07 margins (O5), whether the new cells
   replace the old ones in the main tables, the settings of the new cells, the
   choices in the cell records (step 4 below), the judge classifier of the
   2026-10-10 entry (its judge, rubric text, hand-label sample and agreement gate), and
   the evaluator template of the new cells (the frozen one names BODHI, the protocol
   under test; KNOWN_ISSUES.md).
2. Decide the open choices of [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) section 12
   that are still open: O1 (package name), O2 (raw outputs in git: the default stops
   tracking the 80 MB of the 2026-05 cell), O3 (a versioned archive of all raw outputs,
   about 450 MB; publishing it is a remote action), O7 (the MedGemma-27B rerun is a
   new QLoRA cell) and O8 (whether git tracks the probe and bench outputs; the probe
   output holds benchmark text). Decide O8 before the first probe or bench run.
3. Read D17 to D22 in [docs/DECISIONS.md](docs/DECISIONS.md). To revise one, add a dated
   entry below it.
4. Decide the choices the experiments step made in the cell records (2026-10-09). Each
   is in the records now; change a record before its cell runs, because the runner
   freezes it then.
   - Judges (E10): `medgemma-27b-text` for every base outside Gemma, `phi-4` (general)
     for the Gemma bases. The earlier candidates (Med42-8B, Meditron3-8B, Aloe-Beta-8B)
     are in no record.
   - No judge on the labelled, MediQ, filter-ablation, trace-seed and component-ablation
     cells, to save cost.
   - The labelled cells evaluate five conditions and the paired cells seven.
     RUN_PLAN.md section 4 costs them as they stand, and also with three and two
     conditions, which the primary tests need.
   - The probe (E11) continues raw token ids with no chat template, unlike the chat-mode
     probe of the paper's Appendix A; its control is HealthBench for the older
     benchmarks and ChatDoctor for HealthBench, 100 to 200 prompts each.
   - The judges' harm axes (extent and likelihood from Singhal et al. 2023, plus delayed
     care) are an LLM proxy for C33, not physician review.
   - `mediq-abstain` adapts MediQ's multiple-choice wording to open questions. It reads
     the rating from the DECISION: field only, matched to a full label; MediQ's parser
     also searches the free text, so "not very confident" scored as "very confident".
     A rating that does not parse asks, as in MediQ, and the eval output counts it.
     `red-flags-first` moves the red-flag section first and skips its pass-2 line on
     TECHNICAL and HYBRID tasks.

After: record the adoption as a dated entry in ANALYSIS_PLAN.md section 12 and commit it.
The uniform analysis of the existing outputs already ran on 2026-10-09 (`make extract`,
`make analyze`; tables and statistics committed): the descriptive and exploratory items
of the plan's section 10, with no physician label read. For E02, add
`needs-info-physician` to the `active_inquiry` entry of `discrimination` in
`configs/report.yaml`, then run `make analyze` and `make numbers`. Fix choice 1 (E14 or
E02 as the primary test of H3) in the adoption entry; RUN_PLAN.md may not start without
it. Also accept or change the 2026-10-10 entries of ANALYSIS_PLAN.md section 12. Only your
entry lifts a gate that an entry sets for you (the 2026-10-10 entry on gates), so the
paper's citations below are a draft until you decide. The no-lexicon entry removed
the keyword rules, put the `emergency-theme` red-flag rates in place of the over-flagging
check (KNOWN_ISSUES.md) and specified the judge classifier with its hand-label gate. The
paper entry records that the paper now cites the post hoc checks of every cell (section
4.4 and Table H.6 of `paper/main.tex`): format leakage, the red-flag rate outside the
emergency theme (with the statement that it is not an over-flagging rate), the theme-tag
asking index, the askers gap and the engine's token counts. The abstract also says that
no cell shows selective asking. If you do not want these values in the paper, say so:
they come out of section 4.4, Table H.6, that abstract sentence and CLAIMS.md row 26.
Then RUN_PLAN.md may start.

## 2. Go-ahead for GPU runs

Why: no GPU job runs until you say so (D21). [docs/RUN_PLAN.md](docs/RUN_PLAN.md) lists
every run in order, with its command, cells, seeds, H100-hour estimate and the reviewer
points it answers.

Action: say which runs may start, where, and when.

- The minimal first batch (RUN_PLAN.md section 5) is about 32 to 43 H100-hours: the
  Med42-8B judge re-grade, the uniform Mistral-Small-24B cell with the baselines, and the
  MediQ paired probe on its adapters. With the paired record cut to `base` and `lora`, it
  is 24 to 31. Tier A is about 159 to 249; Tiers A and B about 234 to 350; all tiers
  about 320 to 456, of which the judges are about 78 (RUN_PLAN.md section 6). RUN_PLAN.md costs each cell record as it stands, and its section 4
  lists the smaller records that would cost less.
- Where: the shared H100 host or a paid cloud. Not Modal (D19).
- Before the first run: `bash setup.sh` on the GPU machine, an `HF_TOKEN` with access to
  MedGemma and Llama-3.1, and the Gemma 2 licence accepted for the Med42-8B judge (Gemma 3
  too for R14).
- The new cell records and the two protocols (`mediq-abstain`, `red-flags-first`) exist.
  Before R3 and R10, `python -m bodhi_lora fetch mediq-paired` and
  `python -m bodhi_lora fetch nlm-clinical-questions` download the MediQ file (2.6 MB) and
  the four NLM XML files and print their sha256 values; put them in the benchmark
  records and commit.
- The estimates scale the measured 2026-07 stage times (RUN_PLAN.md section 3). The
  speed of a 4-bit base in vLLM and of the judges is not measured, so each run has a
  range. The run manifests of the first batch replace the estimates.

After: each finished cell lands in `results/<cell_id>/`. RUN_PLAN.md section 7 says how to
copy it back, analyze it and commit it.

## 3. Hugging Face facts

Why: two model facts need a token with access to gated repositories. Neither blocks a
planned run.

1. `configs/models/meditron3-8b.yaml` is stale, and no cell uses it. On 2026-10-09 the
   HF API listed the model as `EPFLiGHT/Meditron3-8B` (sha
   783c241b18b84692689e0336170b345e5732e48e, gated, licence llama3.1), and the
   OpenMeditron organisation listed only Meditron-CHUV. Update the record if E10 should
   use it again; otherwise it can go.
2. Compare the LFS sha256 of the weight files of `NousResearch/Meta-Llama-3.1-8B-Instruct`
   (the evaluator of the 2026-07 cells) with `meta-llama/Llama-3.1-8B-Instruct` (the
   evaluator of the 2026-05 cell and the new cells). If they match, KNOWN_ISSUES.md can
   say the two evaluators are the same weights.

## 4. Remote cleanup (needs your yes)

Why: D16 keeps remote changes yours. None of this ran. The repository stays public, and
there is no privacy purge (D18).

1. **Push.** `git push origin cleanup/resubmission` runs CI on this work for the first
   time. The CI step that downloads HealthBench, runs `make numbers` and diffs
   `paper/generated/` has never run off this machine. Check it with `gh pr checks` or the
   Actions tab.
2. **Delete the scratch branches.** The remote has 44, under the two scratch prefixes
   that `scripts/maintenance/prune_branches.sh` names (40 and 4), counted from the
   remote-tracking refs of the last fetch. Fetch first, read the dry run, then apply:

   ```bash
   git fetch --prune origin
   scripts/maintenance/prune_branches.sh            # dry run: read the list
   scripts/maintenance/prune_branches.sh --apply    # asks before it deletes
   ```

   `--apply` also deletes the temp branches and the branches already merged into the
   mainline. One temp branch, `origin/results-temp`, holds the 2026-04 pilot adapter
   (seed 42, 838 traces), not a submitted adapter. Its weights have the same git blob as
   the local copy in
   `results/healthbench__medgemma-27b-text__bodhi__2026-04/seed_42/train/`, and its
   `adapter_config.json` differs only in the order of the target modules.
3. **Optional: remove the two AI-tool trailers** (`cf72766` and its twin `72a428b`),
   and nothing else (D18). Follow `scripts/maintenance/purge_history.md` with four
   changes: skip steps 2 and 3 (no name map); in step 4 write only the `co-authored-by`
   line into `messages.txt` (no path list, no callback); in step 6 pass only
   `--replace-message "$SECURE/messages.txt"` to `git filter-repo`; in step 7 run only the
   `co-authored-by` check. Every commit id from the first rewritten commit onward
   changes, and commit links in the OpenReview replies stop working. Skipping this is a
   valid choice.
4. **Optional: rename the repository** `bohdi-lora` to `bodhi-lora` (D15). Then update
   the URLs in `README.md`, `KNOWN_ISSUES.md`, `contributions/contributing.md`,
   `contributions/index.md`, `contributions/getting-started/installation.md` and
   `scripts/maintenance/purge_history.md`, and run
   `git remote set-url origin git@github.com:PeterLi-jpg/bodhi-lora.git`.
5. **Local-only refs.** Check, then delete, `refs/codex/*`, `refs/original/*` and the tags
   `backup-before-email-rewrite*`. Run `git worktree prune` for the stale
   `/private/tmp/bohdi-main` entry. The merged branches `cleanup/lane-*` can go with
   `git branch -d`.

## 5. Fix the hooks path in each clone

Why: `core.hooksPath` in this clone points at a folder that no longer exists (the old
location of the repository, before it moved under `Substrate Labs/`). So
`hooks/commit-msg` and `hooks/pre-push` never run. Every commit message of the cleanup
was checked with `sh hooks/commit-msg` by hand and passes. [G14]

Action, in each clone:

```bash
git config core.hooksPath hooks
git config --get core.hooksPath     # expect: hooks
```

## 6. The two remaining physician raters (deferred)

Deferred by you (D20). D5 stands: one of three raters, kappa 0.35, below the 0.6 target.
When you resume:

1. Check that the returned file is the one the kappa came from:

   ```bash
   make kappa RATER_FILES="<private dir>/<returned Grading file>.xlsx"
   git diff --exit-code results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/kappa_results.json   # expect no diff (0.354, n=509)
   ```

2. Send the two other raters their
   `results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/Grading-reviewer<K>.xlsx`.
   Keep returned files outside the repo.
3. When both are back:

   ```bash
   python3 scripts/make_tiebreaker_excel.py --rater-files <r1.xlsx> <r2.xlsx> <r3.xlsx>
   # send the tiebreaker sheet to the fourth physician, then
   make kappa RATER_FILES="<r1.xlsx> <r2.xlsx> <r3.xlsx>"
   ```

4. Decide whether to commit de-identified per-row grades (grading id to pass, fail or
   unsure, with the `physician_id` column removed).

After: Appendix B, Table B.1, the abstract sentence on kappa, checklist items 14 and 15,
and C28.

## 7. Facts and records only you can supply

- **Checklist items 14 and 15.** The checklist says the participating physicians are
  co-authors and that there is no IRB determination. Neither could be checked. The raters
  gave consent (D17); say in item 15 what they consented to, if the venue asks.
- **LLM-usage statement.** `paper/checklist.tex` (LLM-usage item) says the authors used AI
  assistants for editing and drafting. That disclosure is your decision.
- **Posted replies against OpenReview.** Compare the 7 files in `rebuttal/responses/posted/`
  with the threads of submission 32818, including the math notation in the LxMF and HomM
  rebuttals. Check whether `rebuttal/responses/drafts/meta_review_part1.md` and
  `meta_review_part2.md` went to the AC; if they did, `git mv` them to
  `rebuttal/responses/posted/` and add them to `rebuttal/README.md`.
- **Byte-exact submitted source.** In the Overleaf project, download the version from
  just before the NeurIPS deadline and diff it with `paper/submitted/neurips2026/main.tex`.
- **Host environments of the 2026-07 cells (REPRO-10).** With a login to the host:

  ```bash
  ssh mit-critical-data-1 'cd /data/PROJECTS/bohdi && .venv/bin/pip freeze' > requirements-train.lock
  ssh mit-critical-data-1 'cd /data/PROJECTS/bohdi && .venv-infer/bin/pip freeze' > requirements-infer.lock
  ```

  Commit both with the freeze date, and point `REPRODUCIBILITY.md` section 4 at them.
- **Reviewer copy at resubmission (convention 10).** The repository stays public, so a
  double-blind review still needs an anonymous export:
  `scripts/export_anonymous.sh ../_export_review --deny-file "$PRIVATE/deny.txt"`, with the
  rater and co-author names in a deny file outside every repo. Publish it as a new
  anonymous repository and set `\codeurl` in `paper/main.tex`. With co-author names, the
  gate flags `paper/references.bib`, because the cited BODHI papers list them; decide
  whether to keep that self-citation.
- **Clean-environment install.** The dependency pins were never installed in a fresh
  environment (it downloads packages):
  `python3 -m venv /tmp/v && /tmp/v/bin/pip install -r requirements-dev.txt && /tmp/v/bin/python -m pytest -q`.
- **Names file.** `../_wt/.clinician_names.txt` starts with a blank line. The grep on this
  machine ignores a blank pattern; GNU grep without `-w` would match every line. Delete
  the blank line.

## Old item numbers

Other files still cite the items of the 2026-10-08 version of this file. They map here:

| Old item | Now |
|---|---|
| 1. Public exposure of rater names | Dropped: consent given and no privacy purge (D17, D18). The reviewer copy is in item 7. |
| 2. Review D1 to D16 | Item 1, step 3. |
| 3. Another copy of the submitted adapter configs (Modal volume) | Dropped: the adapters are treated as not retained (D19). |
| 4. Sync the added cells; freeze the host environments | The sync is done. The freeze is in item 7. |
| 5. A dated analysis plan | Item 1; `docs/ANALYSIS_PLAN.md`. |
| 6. Prompt baselines | Item 2; RUN_PLAN.md R2. |
| 7. Med42-8B judge re-grade | Item 2; RUN_PLAN.md R1. |
| 8. Remaining physician raters | Item 6 (deferred, D20). |
| 9. Second teaching protocol | Item 2; RUN_PLAN.md R5. |
| 10. Remote cleanup | Item 4 (no privacy purge, D18); the hooks path is item 5. |
| 11. Other GPU runs | Item 2; RUN_PLAN.md R4, R8, R11. |
| 12. Facts and records | Items 3 and 7. The adapter release follows from D19. |
