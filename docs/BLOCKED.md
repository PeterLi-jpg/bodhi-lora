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

## 1. The analysis plan and the open design choices

Done: on 2026-10-11 you wrote "the analysis plan is good". The adoption entry in
[ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) section 12 records it, adopted as written, with the
choices the plan proposes: E14 as the primary test of H3 and E02 as the secondary (choice
1); no E07 margins, so intervals only (choice 2); the E13 settings (choice 4); the choices
in the cell records as written (choice 5); and the judge classifier as specified, used
only after its gate (choice 6). The paper keeps the post hoc values of section 4.4, Table
H.7 and CLAIMS.md row 26, each marked post hoc.

Your answer of 2026-10-11, "Accept all (Recommended)", settled choice 3, choice 7 (O10),
O1, O2, O7 and O8 ([DECISIONS.md](DECISIONS.md) D26 and ANALYSIS_PLAN.md section 12).

Still yours:

1. Decided 2026-10-19 (DECISIONS.md D30): R7 runs on NRP with the 4,096-token pair, as the
   cell `healthbench__medgemma-27b-text__bodhi__ctx4096`, and R10 runs with it. The question
   as it stood: R7 and C31: the red-flag drop that C31 promised to recover occurred only at r 8 and a
   4,096-token eval context, and R7 runs at r 16 and 8,192, so it may find no drop to
   recover. Should R7 add `lora+bodhi` and `lora+red-flags-first` at 4,096 tokens
   (RUN_PLAN.md, "R7 and C31")? Needed before R7 starts, and it blocks R7 only. The
   queue of R3 to R7 on GPU 3 (`configs/queues/tier_a.txt`, RUN_PLAN.md section 4.2)
   runs R3 to R6 and keeps R7's three lines behind `#`. R3 to R6 take about 2.8 to 4.1
   days after R2 ends (an estimate). An answer before then lets R7 follow with no gap
   ([GPU_QUEUE.md](GPU_QUEUE.md) section 6).
2. Done 2026-10-11: the GPU go-ahead (O6). The runs go on the shared host
   mit-critical-data-1, GPU 3 only, reserved in /home/nvidia/GPUS.txt, at most 3 GPUs.
   R1 is done and R2 runs. R3 to R7 follow R2 on GPU 3 through `scripts/gpu_queue.sh`,
   which claims the card in that file for the whole queue and releases it at the end
   (RUN_PLAN.md section 4.2).
3. The timing bench of R2 serves the adapter unmerged (vLLM `--enable-lora`, eager mode,
   the 4-bit base for QLoRA), as the 2026-07 cells were evaluated. The paper's argument is
   one pass with merged weights, as the 2026-05 cell was evaluated. R2 started before
   this was decided, so its bench times the adapter unmerged. Decide whether a later
   bench also times a merged-weights condition (O9); it needs code and about
   50 GB of disk for the merged 24B weights, and is not built.
4. Read D17 to D26 in [DECISIONS.md](DECISIONS.md). To revise one, add a dated entry
   below it.

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

### 2.1 The reruns after the EOS fix (2026-10-17)

Why: the adapters of R2 and R5 never trained on the end of a reply (KNOWN_ISSUES.md,
Training; DECISIONS.md D28). R5's adapters loop in most replies; R2's mostly stop. No claim
uses their adapter conditions until they are trained again. The code fix needs no GPU and
is in. R3 to R6 finished on 2026-10-15 and GPU 3 was released; the lease file says which
cards are free now.

Scope decided on 2026-10-17: "R2–R5 adapters" (the table below), not the 2026-07 cells
(ANALYSIS_PLAN.md section 12). Then, the same day: run them on NRP, on at most three
A100-80GB cards, with the base conditions made again there too (DECISIONS.md D29,
docs/NRP.md). Nothing is left for you here.

| Run | What runs again | H100-hours |
|---|---|---|
| R2 again | train the five seeds; eval the three adapter conditions; the evaluator on all nine conditions; the MedGemma-27B judge, which never ran on R2; the bench of seed 42 | 19 to 20 |
| R3 again | eval the three adapter conditions on the new R2 adapters; the evaluator | 2 to 5 |
| R4 again | eval the three adapter conditions at 775 prompts; the evaluator | 6 to 7 |
| R5 again, HealthBench | train the five seeds; eval the two adapter conditions; the evaluator; the judge; the bench | 7 to 12 |
| R5 again, MediQ paired | eval the two adapter conditions on the new R5 adapters; the evaluator | 2 to 3 |
| All | | 36 to 47 |

The hours come from the measured stage times of R2 to R5 (RUN_PLAN.md section 4.4). R3 and
R4 need R2's new adapters, and the MediQ cell needs R5's. On one card the set takes about
two days; on three, with R2 and R5 at once, about one.

How: before each rerun, the outputs that depend on the adapters (each seed's `train`,
`grade` and `bench` folders, the adapter-condition eval files and the cell manifest) move to
`/data/PROJECTS/bohdi/results_archive/eos-masked-2026-10/<cell>/` on the host. The runner
then trains and evaluates them again and skips what is complete (traces, filter, base
conditions). Nothing is deleted.

The 2026-07 cells stay the record as posted (choice 3); their adapters had the same
fault, and the paper says so. Training them again is about 45 H100-hours by their
2026-07 times, plus about 15 for the MedGemma-27B judge on the new replies. Say if you
want that too.

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
2. **Delete the scratch branches.** Done on 2026-10-11: the remote scratch branches (the
   name prefixes that `hooks/pre-push` rejects) are deleted.
   `scripts/maintenance/prune_branches.sh --apply` also deletes the temp branches and the
   branches already merged into the mainline; this checkout has not fetched since, so
   whether those went too is not confirmed here. One temp branch, `origin/results-temp`,
   holds the 2026-04 pilot adapter (seed 42, 838 traces), not a submitted adapter. Its
   weights have the same git blob as the local copy in
   `results/healthbench__medgemma-27b-text__bodhi__2026-04/seed_42/train/`. Keep
   `origin/results-scaffold`: it is the only branch that holds aa7eec4, the commit that
   dates the physician-check plan to 2026-04-29 (`paper/NUMBERS.md`, withdrawals W19). The
   prune script lists it as unmerged and does not delete it.
3. **Remove the two tool trailers** (`def7eee` and its twin `9a0dbd6`), and nothing else
   (D18). On 2026-10-11 you asked for this purge. It is not done: it rewrites the history
   of `main` and of this branch (`def7eee` is in 20 refs, local and remote, the stash
   among them; `9a0dbd6` is in `origin/results-scaffold`, after aa7eec4, which keeps its
   id), and it needs a force-push, so it waits for your yes on each. The four other
   `Co-authored-by` lines in history name a human collaborator and stay. Follow
   `scripts/maintenance/purge_history.md` with five changes: skip steps 2 and 3 (no name
   map); in step 4 write only its trailer rule into `messages.txt` (the `printf` line; no
   path list, no callback), so the human co-author lines stay; in step 6 pass only
   `--replace-message "$SECURE/messages.txt"` to `git filter-repo`; in step 7 run only the
   trailer check (the `--grep` line), which must list nothing; then step 7b, which
   rewrites the commit ids that tracked files cite and runs `make check` and `make test`
   before the push. Without step 7b, `scripts/check_doc_paths.py` fails on most
   `<commit>:<path>` references, and the pre-commit hook blocks every later commit. Every
   commit id from the first rewritten commit onward changes, and commit links in the
   OpenReview replies stop working.
4. **Optional: rename the repository** `bohdi-lora` to `bodhi-lora` (D15). Then update
   the URLs in `README.md`, `docs/KNOWN_ISSUES.md`, `CONTRIBUTING.md`,
   `docs/INSTALL.md`, `pyproject.toml` and `scripts/maintenance/purge_history.md`, and run
   `git remote set-url origin git@github.com:PeterLi-jpg/bodhi-lora.git`.
5. **Local-only refs.** Check, then delete, the backup refs and tags that
   `scripts/maintenance/purge_history.md` step 9 lists. Run `git worktree prune` for the
   stale `/private/tmp/bohdi-main` entry. The merged branches `cleanup/lane-*` can go with
   `git branch -d`. One stash is left, `stash@{0}`, "interrupted integrator WIP 2026-10-09
   (paper table fitting)", on 176475f: it changes `paper/main.tex` (462 lines),
   `paper/generated/numbers.tex`, `paper/generated/numbers.json` and the removed
   `b533908:scripts/paper_stats.py`. Its paper text is older than the current tables.
   Drop it (`git stash drop`) or recover what you need from it before the purge, which
   rewrites it too.

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
