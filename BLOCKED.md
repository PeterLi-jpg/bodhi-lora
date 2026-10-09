# Blocked on the first author

Work that the cleanup of 2026-10-07 and 2026-10-08 could not finish. Each item needs
Peter: a login, a remote action, GPU time, people, or a decision. Items are in priority
order. Each one gives the exact command or action, then what to change in the repo
after it. Run commands from the repo root unless the item says otherwise.

Audit IDs (PROV-1, EVAL-7, ...) refer to `CONVENTIONS_AUDIT_2026-10-07.md`, which sits
next to the repo. D-numbers refer to [docs/DECISIONS.md](docs/DECISIONS.md). C-numbers
refer to [rebuttal/commitments.md](rebuttal/commitments.md). G- and NUM-numbers are
findings of the final verification pass.

The working tree on `cleanup/resubmission` is complete and clean. `make test`,
`make numbers`, `make figures` and `make paper` pass. Nothing was pushed, and no
remote state changed.

## 1. Stop the public exposure of rater names and author identity

Why: the public GitHub repo still shows physician rater names. In the local copy of
`origin/main` (last fetched 2026-07-17), 2 file paths under `results_modal/irr/` and 14
files carry rater name tokens. Public history also holds the author block of `fcdcd9e`
and two tool trailers (`cf72766`, `72a428b`). The paper's code link
(`\codeurl`, `paper/main.tex:47`) points at the anonymous.4open.science mirror of that
repo. [EVAL-6, HYG-1, G3; audit section 5, decision 7]

Action:

1. Decide now: make the GitHub repo private (Settings, General, Danger Zone, Change
   visibility) until item 10 is done, or push the cleaned tree. A private repo also
   cuts off the old anonymous mirror, which is what you want.
2. Ask the raters what they consent to. Their answer decides how far the purge in
   item 10 must go.
3. Build the reviewer copy and publish it as a new repository with a new anonymous
   mirror. First put the rater names and the co-author names in one deny file in a
   folder outside every repo (`PRIVATE` below):

   ```bash
   scripts/export_anonymous.sh ../_export_review --deny-file "$PRIVATE/deny.txt"
   ```

   The gate passed on 2026-10-08 with the rater names (387 files at 624a1db). With co-author
   names it will flag `paper/references.bib`, because the cited BODHI papers
   (`bodhi2026`, `bodhiplos2026`) list them. Decide whether to keep that third-person
   self-citation (the submitted version had `bodhi2026`).
4. Change the one line `\newcommand{\codeurl}{...}` in `paper/main.tex:47` to the new
   mirror and run `make paper`.

## 2. Review the applied decisions D1 to D16

Why: you delegated D1 to D16 on 2026-10-07, and they are applied: the title, the scope,
the Med42-8B handling, the canonical paper and the statistical unit all follow them, and
the audit items closed on them are closed. Three got notes after the sync of 2026-10-08:
D12 (the added cells now come from the raw outputs), D1 (the GCS bucket is gone, item 3)
and D6 (the askers count without Med42-8B). [G6]

Action: read [docs/DECISIONS.md](docs/DECISIONS.md), the "Status (2026-10-08)" paragraph
and the notes at the end. To revise a decision, add a dated entry below it that says what
changes and why; do not edit the old entries. Nothing here blocks the paper.

## 3. Find another copy of the submitted adapter configs (D1)

Why: the paper reports r=8, alpha=16, dropout 0.05, q/k/v target modules and an exported
step of 250 or 300 for the MedGemma-27B cell. These come from the launcher and trainer
code in `legacy/`, not from the saved adapters. The adapters were in
`gs://bohdi-runs-tokyo-micron/prod-v39-tunix-27b-neurips/`. On 2026-10-08 that bucket
returned 404, and its project has no buckets left. So the GCS check cannot run, and the
paper now says that the values come from the code and could not be checked against the
saved files (Section 3.3, the Table F.1 caption, the Limitations and the checklist).
[D1, HYG-4, PROV-6, PROV-11]

What evidence remains:

- The code that trained and exported the adapters: `legacy/tpu/launch_5seeds_tunix.sh`
  (exports with `--r 8 --alpha 16`), `legacy/configs/lora_medgemma27b_tunix.yaml` and
  `legacy/scripts/export_tunix_lora_to_peft.py`. The comments above Table F.1 in
  `paper/main.tex` cite each value.
- The only MedGemma adapter in git, `607440e:checkpoints/seed_42/best/adapter_config.json`
  (r=8, alpha=16). It comes from an early HF-path run on 838 traces, not from the
  evaluated adapters, so it supports the rank only indirectly.
- Possibly the Modal volume `bohdi-data-cache`. `legacy/modal_eval.py` (lines 119 and
  435-447) pulled each seed's `checkpoints/best/` (with `adapter_config.json`) and its
  `train.jsonl` and `val.jsonl` from the bucket into that volume, at `/seed_<N>/`, before
  the evaluation. A Modal volume persists until someone deletes it. No other store is named
  in `legacy/` or `results/healthbench__medgemma-27b-text__bodhi__2026-05/`: the second volume, `bohdi-hf-cache`, holds base-model
  weights only.

Action, with your Modal login (reading a volume uses no GPU; `-` writes to stdout, per the
help of the installed modal 1.4.2):

```bash
python3 -m modal volume ls bohdi-data-cache
for s in 7 13 42 99 101; do
  echo "seed $s"
  python3 -m modal volume get bohdi-data-cache "seed_$s/checkpoints/best/adapter_config.json" - | python3 -c \
    'import json,sys; d=json.load(sys.stdin); print(d["r"], d["lora_alpha"], d.get("lora_dropout"), sorted(d["target_modules"]))'
  python3 -m modal volume ls bohdi-data-cache "seed_$s"
done
```

Expected from the code: `8 16 0.0 ['k_proj', 'q_proj', 'v_proj']` (the exporter writes
dropout 0.0; training used 0.05). The exported step is not in `adapter_config.json`; look
for a trainer or export log next to it. If `train.jsonl` and `val.jsonl` are there, count
their lines per seed (PROV-11). Any other copy (a TPU VM disk, a local download) works the
same way.

After, if the values match: replace "could not check" in Section 3.3, the Table F.1
caption and the Limitations with the confirmation, update the comments at the top of
Section 3.3 and above Table F.1, and drop the "not confirmed" lines in
`REPRODUCIBILITY.md` and `KNOWN_ISSUES.md`. If a filter log turns up, replace the hand
constant `FILTER_KEPT, FILTER_POOL = 3129, 4000` in `scripts/paper_stats.py` with a value
read from it. If they do not match, correct Table F.1 and Section 3.3 instead. If no copy
exists, the configuration text stays: the paper already says what it cannot check.

The same copy decides the adapter release (item 12). If `seed_<N>/checkpoints/best/` also
holds the adapter weights, the five adapters can be released. If no copy exists, change
the Code availability paragraph and checklist items 4, 5, 11 and 13 to say that the five
adapters are lost.

## 4. Sync the added cells (done) and freeze the host environments

Done on 2026-10-08. The raw outputs and logs were copied read-only from the GPU host
(60c1a54). `make cells` computes every added-cell value, the paired tests and the per-cell
tau, trace counts and adapter recipe (5d5890e, d5d5637, b7bbdfb). The paper takes them
from there, with no "transcribed" footnote, names the MedQA run (the preamble run) and the
incomplete host runs (38b800e). The records followed (bd40a05, 5f3f858, 624a1db): one
posted value was wrong (W21), C24 is done, and the host copy left the `PENDING` list
of `scripts/check_doc_paths.py`; on 2026-10-08 it moved into the `results/*__2026-07/` cells. [PROV-1, PROV-2, PROV-3, CR-1, C24, W11, W12]

One step remains. It needs a login to the host (REPRO-10):

```bash
ssh mit-critical-data-1 'cd /data/PROJECTS/bohdi && .venv/bin/pip freeze' > requirements-train.lock
ssh mit-critical-data-1 'cd /data/PROJECTS/bohdi && .venv-infer/bin/pip freeze' > requirements-infer.lock
```

Commit both, and point `REPRODUCIBILITY.md` section 4 at them. The environment may have
changed since July, so record the freeze date with them.

## 5. Commit a dated analysis plan before any new GPU run

Why: convention 3 and EVAL-10 require a plan that fixes metrics, unit, test, correction
family and contrasts before the results exist. Items 6, 7 and 9 have no plan yet. The
plan is a methodology choice, so it is yours. [G7, EVAL-10]

Action: write a dated plan, `ANALYSIS_PLAN_<date>.md` in `docs/`, and commit it before
item 6 starts. The
cleanup proposed these defaults; none is adopted:

- Unit: the seed (42, 7, 13, 99, 101). Paired per-seed difference, 95% t(4) interval,
  two-sided paired t. Holm over the seven dimensions within each contrast. Aggregate
  score and rates are secondary, outside the Holm family. No equivalence margin; a null
  is "not detected". Pooled d with a prompt-clustered bootstrap is descriptive only.
- Baselines: LoRA minus `base_clarify` and LoRA minus `base_verbalized_confidence` on
  the seven dimensions and the aggregate score; `base_clarify` minus Base on active
  inquiry.
- Med42-8B re-grade: LoRA minus Base on active inquiry and the aggregate under the new
  judge; agreement with the original evaluator by kappa per criterion and Pearson r of
  the aggregate. The dagger stays.
- Second protocol: LoRA minus Base on the seven dimensions; leakage under both the
  nine-marker and the seven-header rule.
- Decide whether the MedGemma-27B baselines pair with the existing Base rows (run on
  Modal A100s) or with a Base rerun on the same H100.

## 6. Run the prompt baselines (C27, EVAL-8)

Why: reviewers asked for a head-to-head against an inference-time baseline. The prompt
protocols `clarify` and `verbalized-confidence` are records in `configs/protocols/`
(`bodhi_lora/protocols/instruction.py`), tested with the stub engine only.

Action, on the GPU host after `bash setup.sh` has built `.venv` and `.venv-infer`:

1. Write the cell records with the baseline conditions (`base+clarify`,
   `base+verbalized-confidence`), as EXPERIMENTS.md E12 describes. A recorded cell is
   never run again, so a baseline for an existing cell is a new eval-only cell with the
   same benchmark, model and eval settings.
2. Check, then run:

   ```bash
   python -m bodhi_lora run <cell id> --dry-run       # CPU: prints the GPU commands
   python -m bodhi_lora run <cell id> --gpus 0,1,2,3,4
   ```

Outputs land in the cell's `seed_<n>/eval/` as `base+clarify.json` and
`base+verbalized-confidence.json`, and the grade stage adds them to the evaluator's file.

Compute cost (rough estimate, not measured): each seed runs two single-pass conditions
of 200 prompts plus grading, about a third of a cell's evaluation. The posted replies
give about two H100-hours per 24B seed end to end, training included. That suggests 2
to 3 H100-hours per 24-27B cell and 1 to 2 per 7-8B cell. The two cells above: about 4
to 6 H100-hours. All seven cells: about 12 to 18 H100-hours. On the shared host this has
no meter but holds a card; at an on-demand cloud rate of $2 to $4 per H100-hour it is
about $10 to $25 for two cells and $25 to $75 for all. Measure the real per-stage times
while it runs (C25).

## 7. Re-grade the Med42-8B cells with a third-family judge (EVAL-7, D6, C26)

Why: Med42-8B is built on Llama-3, the evaluator's family. The cells carry a dagger and
stay out of cross-model statements until a judge from another family checks them.

Action, on the GPU host, where the outputs already are (accept the Gemma 2 license on
Hugging Face first). The two cell records list `gemma-2-27b` as their judge, and the
runner adds judge re-grades to a recorded cell without touching its outputs:

```bash
python -m bodhi_lora run healthbench__med42-8b__bodhi__2026-07 --stages grade --gpus 0
python -m bodhi_lora run chatdoctor__med42-8b__bodhi__2026-07 --stages grade --gpus 0
```

Each seed gets `results/<cell>/seed_<n>/grade/judge__gemma-2-27b.json` with the original and the judge score per
example, on the rubric and on the seven dimensions. Sync these files, then compute the
agreement and the re-graded contrasts as the plan in item 5 says.

Compute cost (rough estimate, not measured): per cell, 5 seeds x 4 conditions x 200
responses. That is about 48,000 rubric calls on HealthBench (12 criteria per prompt) or
32,000 on ChatDoctor (8 criteria), plus 4,000 epistemic calls, on a 27B judge. Expect 2
to 5 H100-hours per cell, so 4 to 10 for both cells.

## 8. Collect the two remaining physician raters (C28, EVAL-5, PROV-5, NUM-2)

Why: the pre-registered check needs three raters and a consensus kappa against a 0.6
target. One of three returned grades. The paper reports kappa 0.35 from that one file,
which is kept outside the repo, so nobody else can recompute it.

Action:

1. Confirm that the returned file in your Downloads folder is the one the kappa came
   from. Move it to a private folder outside the repo. Then check that it still gives
   the committed value:

   ```bash
   make kappa RATER_FILES="<private dir>/<returned Grading file>.xlsx"
   git diff --exit-code results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/kappa_results.json   # expect no diff (0.354, n=509)
   ```

2. Send the two raters who have not returned their sheet their
   `results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/Grading-reviewer<K>.xlsx`. Keep the returned files outside the
   repo.
3. When both are back:

   ```bash
   python3 scripts/make_tiebreaker_excel.py --rater-files <r1.xlsx> <r2.xlsx> <r3.xlsx>
   # send the tiebreaker sheet to the fourth physician, then
   make kappa RATER_FILES="<r1.xlsx> <r2.xlsx> <r3.xlsx>"
   make numbers && make paper
   ```

4. Decide whether to commit de-identified per-row grades (grading_id to pass, fail or
   unsure, with the `physician_id` column removed).

After: update Appendix B, Table B.1, the abstract sentence on kappa, checklist items
14 and 15, and C28.

## 9. Run the second teaching protocol (C29, HomM-W1)

Why: the posted replies promised a second CoT protocol. `bodhi_lora/protocols/ask_first.py`
implements a one-pass ask-first protocol (the `ask-first` record). It has run only with
the stub engine.

Action, on the GPU host, after the plan in item 5: write the cell record
`healthbench__mistral-small-24b__ask-first` (EXPERIMENTS.md E16), then

```bash
python -m bodhi_lora run healthbench__mistral-small-24b__ask-first --dry-run
python -m bodhi_lora run healthbench__mistral-small-24b__ask-first --gpus 0,1,2,3,4
```

The cell lands in `results/healthbench__mistral-small-24b__ask-first/`.

Compute cost (rough estimate): a full 24B cell with trace generation, filtering, five
trainings and evaluation. `rebuttal/PLAN.md` estimated 8 to 13 H100-hours for a 24-27B
cell on one card.

## 10. Remote cleanup (D16)

Why: none of this ran, because D16 forbids remote changes in the cleanup.

1. Branches. 44 scratch branches (the prefixes that `prune_branches.sh` names) and
   `origin/results-temp` (MedGemma adapter weights and optimizer state) are on the
   remote; 24 scratch and 3 temp branches are local.

   ```bash
   git fetch --prune origin
   scripts/maintenance/prune_branches.sh            # dry run: read the list
   scripts/maintenance/prune_branches.sh --apply    # type: delete
   ```

2. History purge of rater names, the author block and the two tool trailers. Follow
   `scripts/maintenance/purge_history.md`, steps 0 to 9. First fix one gap: the
   author-block rule in `scripts/maintenance/purge_callback.py` rewrites only an
   uncommented `\author{...}`. A selftest on 2026-10-08 with an empty name map left 32
   affiliation hits: 31 blob versions of `paper/main.tex` and one of
   `paper/neurips_submission/main.tex` hold the commented-out author block. Add a rule
   for it and rerun until the selftest prints `residual hits after rules: 0`.
3. Repo slug. Rename `bohdi-lora` to `bodhi-lora` in the GitHub settings (D15). Then
   update the URLs in `README.md`, `KNOWN_ISSUES.md`, `contributions/contributing.md`,
   `contributions/index.md`, `contributions/getting-started/installation.md` and
   `scripts/maintenance/purge_history.md`, and run
   `git remote set-url origin git@github.com:PeterLi-jpg/bodhi-lora.git`.
4. Hooks. `core.hooksPath` points at a folder that no longer exists, so
   `hooks/commit-msg` and `hooks/pre-push` never ran on this branch. Every commit message
   since `61dada9` was checked with `hooks/commit-msg` by hand and passes. Run
   `git config core.hooksPath hooks` in each clone. [G14]
5. Local-only refs. Check, then delete, `refs/codex/*`, `refs/original/*` and the tags
   `backup-before-email-rewrite*` (purge step 9). Run `git worktree prune` for the stale
   `/private/tmp/bohdi-main` entry. The four merged branches `cleanup/lane-*` can go
   with `git branch -d`.
6. Push. `git push origin cleanup/resubmission` runs CI for the first time on this work.
   The CI step that downloads HealthBench, runs `make numbers` and diffs
   `paper/generated/` has never run outside this machine. Check it with
   `gh pr checks` or the Actions tab.

## 11. Other runs that need GPU time or people

- **MedQA without the preamble (optional).** The only MedQA run used the prompt that told
  the model to ask, so the paper sets the cell aside. A run with the fixed prompt (035d85f)
  would test the recipe there:
  a cell record on the `medqa-open` benchmark (the template without the preamble, EXPERIMENTS.md
  E15), run with `python -m bodhi_lora run` (about one 24B cell, 8 to 13 H100-hours by
  `rebuttal/PLAN.md`). It needs a dated plan (item 5).
- **Contamination probe (NUM-7).** Appendix A types its numbers by hand and says the
  output was not kept. Run it on a GPU, commit the JSON, and add an emitter to
  `scripts/paper_stats.py`. The probe is a stage now: it reads a `probe` block in a cell
  record (n_prompts 200, prefix_tokens 15, continuation_tokens 25 for the Appendix A
  setting) and writes probe.json in the shared probe folder of that cell. No record has a `probe`
  block yet, and the runner refuses the recorded 2026-05 cell, so this needs a new cell
  record on the MedGemma-27B base first; then
  `python -m bodhi_lora run <cell id> --stages probe --gpus 0`.
  About one H100-hour (rough estimate).
- **Red flags early in the protocol (C31).** Promised to LxMF. No code exists yet. It
  needs a protocol variant that writes red flags before the extended reasoning, a cell
  run as in item 9, and a test of LoRA+variant minus LoRA+BODHI on red flag.
- **Measured cost per stage (C25).** Time each stage in isolation on one H100 while
  items 6 and 9 run. The paper reports no hours now.
- **Over-flagging set (C30).** Two physicians label prompts that have no genuine red
  flag, before any grading, with its own dated plan.
- **Harm grading (C33).** Appendix "Harm-Potential Analysis" says the clinician harm
  review is not done.
- **Clinician-posed benchmark (C32).** D4 makes MIMIC and eICU future work. Running them
  needs PhysioNet credentialed access.

## 12. Facts and records only you can supply

- **Checklist items 14 and 15.** The checklist says the participating physicians are
  co-authors and that there is no IRB determination. Neither could be checked. Edit both
  if either is wrong.
- **LLM-usage statement.** `paper/checklist.tex` (LLM-usage item) says the authors used
  AI assistants for editing and drafting. That disclosure is your decision.
- **Adapter release.** The paper (Code availability) and checklist items 4, 5, 11 and 13
  say that no adapter is released, that the cloud storage of the five MedGemma-27B
  adapters no longer exists, and that their release depends on a second copy (item 3).
  If you release any adapter, or if item 3 finds no copy, update that paragraph and
  those items.
- **Posted replies against OpenReview.** Compare the 7 files in
  `rebuttal/responses/posted/` with the threads of submission 32818, including the
  MathJax and plain-text math in the LxMF and HomM rebuttals. Check whether
  `rebuttal/responses/drafts/meta_review_part1.md` and `meta_review_part2.md` were
  posted to the AC; if they were, `git mv` them to `rebuttal/responses/posted/` and add them to
  `rebuttal/README.md`.
- **Byte-exact submitted source.** In the Overleaf project, download the version from
  just before the NeurIPS deadline and diff it with
  `paper/submitted/neurips2026/main.tex`.
- **Hugging Face pins.** With your token, read the revision of
  `OpenMeditron/Meditron3-8B` and add it to `configs/models/meditron3-8b.yaml`; compare the LFS
  sha256 of `NousResearch/Meta-Llama-3.1-8B-Instruct` with `meta-llama/Llama-3.1-8B-Instruct`.
- **Full responses of the added cells.** The per-seed eval JSON (about 370 MB) is in the
  `results/*__2026-07/` cells (gitignored), in a local archive outside the repo and on the host. The
  repo tracks only what `make numbers` needs. Decide whether to archive it with its
  sha256, as for the training traces below; reviewers who ask for per-cell outputs need it.
- **Training traces of the submitted cell.** `results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_7/filter/train.jsonl`
  (49 MB) is ignored. `make numbers` reads its scores from the tracked
  `results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_7/filter/filter_scores.json`; `make kappa` and `make derived` still need
  the file. Decide whether to archive it with its sha256.
- **Equivalence margin (EVAL-1, D2).** D2 says no margin. If you want a non-inferiority
  statement, state the margin first; the 90% interval is in
  `\statHealthBenchLoRAvsBaseCINinety`. Any margin set now is post hoc.
- **Clean-environment install.** The new dependency pins were never installed in a
  fresh environment (it needs a download):
  `python3 -m venv /tmp/v && /tmp/v/bin/pip install -r requirements-dev.txt && /tmp/v/bin/python -m pytest -q`.
- **Names file.** `../_wt/.clinician_names.txt` (next to the repo) starts with a blank line.
  A blank pattern in `grep -f` matches every line without `-w`. Delete the blank line.
