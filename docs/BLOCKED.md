# Blocked on the first author

What is left that only Peter can do: a decision, a go-ahead for compute, a remote action,
a login or a fact. Rewritten on 2026-10-08 after his facts of that day (DECISIONS.md D17
to D22). Items are in priority order. Each one says what to do and what changes in the
repo after it. Run commands from the repo root.

D-numbers refer to [DECISIONS.md](DECISIONS.md), C-numbers to
[rebuttal/commitments.md](../rebuttal/commitments.md), audit IDs (PROV-1, EVAL-7) to
[AUDIT_IDS.md](AUDIT_IDS.md). The last section maps the old item
numbers, which other files still cite, to the items below.

Nothing here was pushed. On 2026-10-11 the first author adopted the analysis plan, deleted
the remote scratch branches and set the hooks path; items 1, 4.2 and 5 say what is
left of them.

## 1. The analysis plan (adopted 2026-10-11) and the open design choices

Done: on 2026-10-11 you wrote "the analysis plan is good". The adoption entry at the end
of [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) section 12 records it: the plan as written, with
E14 as the primary test of H3 and E02 as the secondary, no E07 margins, the choices in the
cell records as they stand (the judges, the cells with no judge, the probe method, the
harm axes, the protocol wording) and the 2026-10-10 entries. So the paper keeps the post
hoc values of section 4.4, Table H.7 (Table H.6 until 2026-10-11) and the abstract sentence on selective asking.
Check the entry; to change any of it, add a dated entry below it. RUN_PLAN.md may start
once item 2 says so.

Still yours:

1. Decide the open choices of [ARCHITECTURE.md](ARCHITECTURE.md) section 12
   that are still open: O1 (package name), O2 (raw outputs in git: the default stops
   tracking the 80 MB of the 2026-05 cell), O3 (a versioned archive of all raw outputs,
   about 450 MB; publishing it is a remote action), O7 (the MedGemma-27B rerun is a
   new QLoRA cell) and O8 (whether git tracks the probe and bench outputs; the probe
   output holds benchmark text). Decide O8 before the first probe or bench run.
2. Read D17 to D22 in [DECISIONS.md](DECISIONS.md). To revise one, add a dated
   entry below it.
3. Two choices of the plan that it makes no proposal for: choice 3 (whether the new
   cells replace the 2026-05 and 2026-07 cells in the main tables) before the first new
   cell is analyzed, and choice 7 (keep the frozen evaluator template, which names BODHI,
   or write a protocol-neutral one as a new grader version) before the first new cell is
   graded. The code has only the frozen template; a second one is a small change
   (a template version key in the cell record's grading block), made only if you choose it.
4. The timing bench of R2 serves the adapter unmerged (vLLM `--enable-lora`, eager mode,
   the 4-bit base for QLoRA), as the 2026-07 cells were evaluated. The paper's argument is
   one pass with merged weights, as the 2026-05 cell was evaluated. Decide before R2
   whether the bench also times a merged-weights condition; it needs code and about
   50 GB of disk for the merged 24B weights, and is not built.

Done on 2026-10-11, after your adoption: E02 ran on the existing outputs (the
`needs-info-physician` label in `configs/report.yaml`, `make extract`, `make analyze`,
`make numbers`). LoRA - Base on the physician-label index is not positive in any of the
three cells of family F6; with BioMistral-7B it falls (Holm-adjusted p below 0.05). The
paper cites it in section 4.4 and Table C.1. The deployment criterion (E07) is in Table
C.1 as intervals only, as the adoption says.

## 2. Go-ahead for GPU runs

Why: no GPU job runs until you say so (D21).

How many GPUs: one H100 80GB runs any planned stage (the 2026-07 cells trained and
evaluated on one card per seed), and the runner runs one seed per GPU. So five H100s run a
five-seed cell in about the time of its shared stages plus one seed; one H100 runs the
same work at about five times the wall-clock time. The H100-hours below do not change
with the number of cards. [RUN_PLAN.md](RUN_PLAN.md) lists
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
  records and commit. These downloads need your go-ahead too, and so does
  `python -m bodhi_lora fetch medqa-open` for R8 (it builds the MedQA rows from Hugging
  Face). A dry run never downloads (since 2026-10-11), so until then the dry runs of those
  cells stop at the fetch command: R3, the paired cells of R5 and R7, R8 and R10. Every
  other run's dry run passes (RUN_PLAN.md section 4.1).
- The judge classifier (R17) has its stage since 2026-10-11. Its gate needs 300 hand
  labels from two authors first (ANALYSIS_PLAN.md section 12, 2026-10-10). Before the
  draw, fix in a dated plan entry how the 300 replies split over the 16 strata (four
  HealthBench cells, four conditions); the plan leaves that open.
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
2. **Delete the scratch branches.** Done on 2026-10-11: the remote `claude/*` and
   `worktree-agent*` branches are deleted. `scripts/maintenance/prune_branches.sh --apply`
   also deletes the temp branches and the branches already merged into the mainline; this
   checkout has not fetched since, so whether those went too is not confirmed here. One
   temp branch, `origin/results-temp`, holds the 2026-04 pilot adapter (seed 42, 838
   traces), not a submitted adapter. Its weights have the same git blob as the local copy
   in `results/healthbench__medgemma-27b-text__bodhi__2026-04/seed_42/train/`.
3. **Remove the two AI-tool trailers** (`cf72766` and its twin `72a428b`), and nothing
   else (D18). On 2026-10-11 you asked for this purge. It is not done: it rewrites the
   history of `main` and of this branch (`cf72766` is an ancestor of both, and of 17
   other local or remote refs), and it needs a force-push, so it waits for your yes on
   each. The four other `Co-authored-by` lines in history name a human collaborator and
   stay. Follow `scripts/maintenance/purge_history.md` with four
   changes: skip steps 2 and 3 (no name map); in step 4 write only its `co-authored-by`
   rule into `messages.txt` (the regex line that matches Claude or Anthropic; no path list,
   no callback), so the human co-author lines stay; in step 6 pass only `--replace-message "$SECURE/messages.txt"` to
   `git filter-repo`; in step 7 check that `git log --all -i --grep=noreply@anthropic`
   lists nothing. Every commit id from the first rewritten commit onward changes, and
   commit links in the OpenReview replies stop working.
4. **Optional: rename the repository** `bohdi-lora` to `bodhi-lora` (D15). Then update
   the URLs in `README.md`, `docs/KNOWN_ISSUES.md`, `CONTRIBUTING.md`,
   `docs/INSTALL.md`, `pyproject.toml` and `scripts/maintenance/purge_history.md`, and run
   `git remote set-url origin git@github.com:PeterLi-jpg/bodhi-lora.git`.
5. **Local-only refs.** Check, then delete, `refs/codex/*`, `refs/original/*` and the tags
   `backup-before-email-rewrite*`. Run `git worktree prune` for the stale
   `/private/tmp/bohdi-main` entry. The merged branches `cleanup/lane-*` can go with
   `git branch -d`.

## 5. The hooks path (done 2026-10-11 in this clone)

Done: `git config --get core.hooksPath` returns `hooks` in this clone, and on 2026-10-11 a
test commit whose message named a coding assistant was rejected by `hooks/commit-msg`.
Every commit of the cleanup before that date was checked with `sh hooks/commit-msg` by
hand. In any other clone, run once:

```bash
git config core.hooksPath hooks
```

## 6. The two remaining physician raters (deferred)

Deferred by you (D20). D5 stands: one of three raters, kappa 0.35, below the 0.6 target.
When you resume:

1. Check that the returned file is the one the kappa came from. The answer key is no
   longer in the repo (KNOWN_ISSUES.md); its copy is in `../_archive/validation/`:

   ```bash
   make kappa RATER_FILES="<private dir>/<returned Grading file>.xlsx" ANSWER_KEY=../_archive/validation/answer_key.xlsx
   git diff --exit-code results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/kappa_results.json   # expect no diff (0.354, n=509)
   ```

2. Decide first whether to re-key the sheets: the answer key was public until 2026-10-11
   and history keeps it (KNOWN_ISSUES.md). Then send the two other raters their
   `results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/Grading-reviewer<K>.xlsx`
   (or the re-keyed sheets). Keep returned files outside the repo.
3. When both are back:

   ```bash
   python3 scripts/make_tiebreaker_excel.py --rater-files <r1.xlsx> <r2.xlsx> <r3.xlsx>
   # send the tiebreaker sheet to the fourth physician, then
   make kappa RATER_FILES="<r1.xlsx> <r2.xlsx> <r3.xlsx>" ANSWER_KEY=<answer_key.xlsx>
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
- **Posted replies against OpenReview.** Done on 2026-10-11 from the copy of the thread
  you supplied: four of the seven files held text that was not posted, and now hold the
  posted text; the meta-review replies were not posted (`rebuttal/README.md`, "Posted
  replies"). Optional: export the raw markdown of each reply from OpenReview, which would
  settle the table and math formatting that the rendered copy cannot show.
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
  `python3 -m venv /tmp/v && /tmp/v/bin/pip install -e '.[train,dev]' && /tmp/v/bin/python -m pytest -q`.
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
