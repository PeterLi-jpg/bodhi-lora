# Known issues

Open problems first, then the fixed ones, then a short history. IDs such as PROV-1 refer
to the audit of 2026-10-07 ([AUDIT_IDS.md](AUDIT_IDS.md) says what each one found);
D-numbers refer to [DECISIONS.md](DECISIONS.md). Issue numbers such as #5 are GitHub issues on `PeterLi-jpg/bohdi-lora`.

## Open

### Blocked on data outside the repo

- **The added cells' full responses are not versioned (PROV-1).** The per-seed eval JSON
  (about 370 MB) is in the local copy, in the `results/*__2026-07/` cells, and on the
  shared H100 host, and it is gitignored. The repo tracks what the paper needs from it
  (`results/cross_cell/generated_values.json`) and the adapter configs, trainer states and
  eval ids. Rebuilding `generated_values.json` needs the local copy (`make cells`).
- **The submitted adapter config cannot be confirmed (HYG-4, D1, D19).** Repo evidence
  says r=8, alpha=16. The GCS bucket that held the five submitted adapters returned 404
  on 2026-10-08, and the adapters are treated as not retained (D19), so the paper says
  the values come from the code and could not be checked. A new experiment on
  MedGemma-27B needs a rerun, proposed as a new QLoRA cell, with the 2026-05 cell as the
  record (O7, which waits for the first author's yes; BLOCKED.md item 1).
- **The paper still says a second copy of the adapters may exist.** The Code
  availability paragraph and checklist items 4, 5, 11 and 13 say "we have not yet checked
  a second copy", and two comments in `paper/main.tex` point at the Modal volume. Under
  D19 they must say that the adapters are not retained.
- **Returned physician grades are not in the repo (PROV-5, EVAL-5).** One of three
  raters returned. `scripts/compute_irr_kappa.py` read that file in place and wrote
  `results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/kappa_results.json`, which names no rater. Nobody else can
  recompute it until the de-identified grades (grading_id to pass, fail or unsure) are
  committed. Strip the `physician_id` column first. Rater identities stay out of git.

### Measurement and design

- **The answer key of the physician sheets was public (C28, D20).** From 2026-05-04 to
  2026-10-11 the repo tracked
  `results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/answer_key.xlsx`.
  It maps each grading id of `Grading-reviewer1.xlsx` to `Grading-reviewer3.xlsx` to the
  condition and seed that made the response and the LLM's score. Git no longer tracks it
  (the validation README gives its sha256), but history keeps it, because there is no
  purge (D18). The one rater who returned grades graded while the key was public. The
  two deferred raters would grade the same sheets, and anyone who reads the repo can
  unblind them. The
  choice is the first author's: re-key the sheets before they go out (new shuffled
  grading ids from `scripts/make_irr_excel.py`, with a new answer key kept outside the
  repo), or send them as they are and say in the paper that the key was public.
- **Physician validation is incomplete (D5).** The check grades HealthBench rubric
  criteria only, and agreement is below the pre-registered 0.6 target. No physician
  graded the seven behaviors, so every epistemic-behavior result is labeled LLM-graded.
- **The grader departs from its own anchors.** It gives context-seeking 2 to answers
  that ask no direct question, and hedging 2 to most answers it flags as blanket
  disclaimers (`context_seeking_anchor` and `blanket_disclaimers` in
  `paper/generated/numbers.json`).
- **The evaluator's prompt names the protocol under test.** The template that graded
  every cell (`bodhi_lora/grading/epistemic.py`, unchanged since 7df83b4) says it judges
  "the epistemic virtues that BODHI-style prompting aims to instill", and its
  context-seeking definition adds "BODHI v0.1.3 specifically trains active inquiry".
  The repo pins `bodhi-llm` 0.1.4, so the version is also stale. The same words graded
  every condition, the base and the baselines too, but a grader told what the method
  aims at may credit its behaviors more readily. The recorded cells keep the template so
  their grades stay comparable; the paper's appendix on the dimensions says so. Whether
  the new cells keep it too (proposed, so their grades compare with the recorded ones) or
  use a protocol-neutral template (a new grader version, with its own hash in each grade
  file) is choice 7 of the analysis plan. The adoption did not settle it, so it waits for
  the first author (docs/BLOCKED.md item 1).
  `tests/test_no_hardcode.py` fails on any other grader text that names a protocol.
- **Med42-8B shares the evaluator's family (EVAL-7, D6).** The cells are marked † and
  kept out of cross-model statements. The judge re-grade is built
  (`bodhi_lora/pipeline/grade.py`) and waits for the GPU go-ahead (docs/RUN_PLAN.md R1).
- **No prior-work baseline has run (EVAL-8).** The three prompt baselines (`clarify`,
  `verbalized-confidence`, `mediq-abstain`) are protocol records in `configs/protocols/`,
  run by `bodhi_lora/protocols/`, and pass their stub tests. They run inside the new
  cells, so they share the grader and the prompts of the conditions they are compared
  with (docs/RUN_PLAN.md R2, docs/ANALYSIS_PLAN.md H4).
- **The MedQA cell has only the preamble run.** Its one run used the old prompt that told
  the model to ask (Base asks in nearly every response), so it measures instruction
  following. The fixed prompt (035d85f) was never run. The paper reports the cell's
  aggregate score, sets it aside and says why (withdrawals W12).
- **MedQA seed 42 has no epistemic grades.** Its grading looked the prompts up in
  HealthBench-Hard only and skipped all 200 per condition
  (`results/medqa-open-preamble__mistral-small-24b__bodhi__2026-07/logs/eval_medqa_mistralai_Mistral-Small-24B-Instruct-2501_s42.log`); later seeds
  loaded the MedQA prompts. The cell's seven dimensions have four seeds.
- **The two pipelines differ (REPRO-3).** Context length, server batch size, grader
  mirror, eval-prompt policy, trainer and hardware all differ between the submitted
  and added cells. REPRODUCIBILITY.md section 3 lists each one.
- **Mistral-Small-24B trains on 4-bit and serves on bf16 (REPRO-12).**
- **Seeds do not vary the teacher traces (REPRO-8).** Each cell generates its traces
  once. What a seed changes also differs between the two pipelines. The submitted
  cell has 667 unique eval prompts across its five draws (CLAIM-10).
- **Grader parse failures differ by condition (EVAL-9, D13).** The paper reports the
  rate per condition for both graders and a worst-case sensitivity.
- **HealthBench's context-awareness axis disagrees with our context-seeking score.**
  On the three added HealthBench cells the adapter meets fewer of the rubric items
  that reward asking, while our grader scores context-seeking up. The working
  hypothesis is that rubric items are prompt-specific and our grader scores generic
  asking. It is untested (`rebuttal/responses/drafts/FINDINGS_quality_drop.md`, section 5).
- **The discrimination label was defined after the results (EVAL-10, D9).** It reads
  HealthBench's theme tag only and is marked post hoc and exploratory. The second
  labelling of the discussion matched keywords in rubric text and was removed on
  2026-10-10 (CONVENTIONS.md rule 15, withdrawals W24).
- **Calibration is not measured (#1, D3).** The eval JSON has Brier and ECE fields from
  a token-probability proxy. The paper does not report them.
- **No over-flagging rate on the existing cells (C30).** The post hoc check of 2026-10-09
  (a keyword rule for emergency mentions on the `red-flag-at-stake` label, itself a
  keyword rule over rubric text) was removed on 2026-10-10 (CONVENTIONS.md rule 15). What
  stands in for it is the evaluator's red-flag rate on the two groups of the
  `emergency-theme` label (the theme tag only). Prompts of other themes can hold a red
  flag, so the rate outside the theme counts warranted flags too and is not an
  over-flagging rate. A label from the `cluster:` tags would be the physician label under
  another name, which the analysis plan keeps unseen, and Hard holds only 8 non-emergent
  prompts. The planned test (H7) reads the physician labels on the labelled cells. The
  paper reports the rate outside the theme as a descriptive check (section 4.4) and says
  that it is not an over-flagging rate.
- **No count of questions without a grader.** The second-person question rule was a
  keyword rule and was removed on 2026-10-10 (withdrawals W23). The cross-check of active
  inquiry is a judge classifier with a hand-label gate, specified in ANALYSIS_PLAN.md
  section 12 and not run (RUN_PLAN.md R17).
- **Three models have no tokenizer count.** `analyze --extract` counts reply tokens with
  the model's own tokenizer from the local Hugging Face cache and never downloads. Only
  Mistral-Small-24B's tokenizer is cached, so the MedGemma-27B, Med42-8B and
  BioMistral-7B cells have the engine's count for one-pass conditions only. To fill
  them: `huggingface-cli download <repo> tokenizer.json --revision <revision>` for each
  model record (MedGemma needs an accepted licence and a token), then `make extract`
  and `make numbers`. Two-pass conditions stored only their final reply, so their token
  counts leave out the first pass in every cell.

### Reproducibility gaps

- **The posted cells' environment is not recorded (REPRO-10, EVAL-12).** The code side is
  fixed (see Fixed below). What stays open is the record of the runs that produced the
  posted numbers: the exact `.venv-infer` versions (no pip freeze was kept;
  `requirements-infer.txt` gives the range and why; each eval JSON records only its
  transformers and bodhi-llm versions) and an overlap-check output (the gate came after
  these runs). The eval ids are synced: each non-HealthBench cell has its `eval_ids.json`,
  and the HealthBench cells used `02ac9a3:data/raw/hard_200_sample_ids.json`. New runs record all
  of this in their manifests (`bodhi_lora/manifest.py`).
- **The host's `multi_seed_summary.json` files are wrong.** They predate 55f6c7c and list
  their seeds as `["eval"]`, so each holds one seed. They are gitignored and unused;
  `make cells` recomputes from the per-seed files.

### Planned runs

- **The cost estimates rest on shared-host times.** docs/RUN_PLAN.md scales the stage
  times of the 2026-07 cells, measured on a shared host, to each cell record. The speed
  of serving a 4-bit base in vLLM, of a 14B or 27B judge, and of the judges' harm call is
  not measured; the plan gives a range or states the assumption. The first run's
  manifests and its `bench` stage replace the estimates (E20, C25).
- **The runner evaluates seed-invariant conditions for every seed.** With fixed eval ids
  and greedy decoding, `base` and `base+bodhi` give the same outputs for each seed, but
  the runner runs them five times. In the labelled cells that is nearly half of the time.
- **The eval stage prints scores to its log.** The no-interim-look rule
  (docs/ANALYSIS_PLAN.md section 2.5) therefore depends on the operator reading
  manifests, not logs.
- **No automatic check orders the plan before the results.** Convention 3 relies on a
  review step and the commits that the run manifests record. `tests/test_plan_coverage.py`
  checks only that each planned cell is named in docs/ANALYSIS_PLAN.md and that the plans
  name no cell without a record.
- **`analyze` does not compute every planned statistic yet.** The weighted kappa and the
  dimension contrasts under a judge (H8), the judges' harm grades (C33), the probe and
  bench outputs (E11, E20), stage times from the manifests and contrasts between cells
  (H9, H10, E24) are missing (docs/ARCHITECTURE.md section 7). Each is needed only once
  its cell has run. The paired-probe statistics (E14) and the declared contrasts of H5
  and H6 are built since 2026-10-10 and tested on synthetic tables and stub runs only.

### Repo state

- **Two tool trailers remain in public history (HYG-12).** The repository stays public,
  and the raters gave consent, so there is no privacy purge of history (D17, D18). A purge
  of the two trailers only is optional (BLOCKED.md item 4). The current tree is clean.
- **Temp branches may remain on the remote (HYG-15, HYG-11).** The first author deleted
  the remote `claude/*` and `worktree-agent*` branches on 2026-10-11 (HYG-12). Whether the
  temp branches and the branches merged into the mainline went too is not confirmed from
  this checkout. One of them, `results-temp`, holds the adapter of the 2026-04 MedGemma
  pilot (seed 42, 838 traces), not a submitted adapter; its weights match the local copy
  in the pilot cell's train folder. `scripts/maintenance/prune_branches.sh` lists them;
  `--apply` deletes them after the first author's yes (D16, BLOCKED.md item 4).
- **The remote slug misspells the name (`bohdi-lora`, D15).** Renaming is a remote
  action and is deferred.
- **`make smoke` stops after the smoke record changes.** The runner freezes a cell's
  record at its first run, in the scratch root too, so a changed smoke record fails
  against the old `outputs/smoke/` folder. Move that folder away and run again.

## Fixed

- **The hooks were not active in this clone (2026-10-11).** `core.hooksPath` pointed
  at the repository's old location, so `hooks/commit-msg` and `hooks/pre-push` did not
  run, and commit messages were checked by hand. It now points at `hooks/`, and a commit
  message that names a coding assistant is rejected. `hooks/pre-commit` was added the
  same day (BLOCKED.md item 5).
- **The report step named the cells it reported (2026-10-10).** Table 1, the per-cell
  tables and most main-body numbers came from `503b975:scripts/paper_stats.py`, which knew four
  condition names, the submitted cell and a token map of the posted cells, so a new cell
  reached only the appendix checks table. `python -m bodhi_lora report` (`make numbers`)
  now writes every macro and table from the cells' `stats.json`, and
  `configs/report.yaml` names the tables, groups and counts. Every number the paper
  printed stayed the same, except: two score differences print one more decimal (as the
  2026-05 cell's always did); set-aside rows carry a double dagger; and the second-grader
  check of Appendix H. That check compared the pilot's stored score (earned points over
  positive points) with the normalized score of the other run, so the scales seemed to
  differ; with one formula r is 0.70, not 0.42.
- **`make extract` stopped with a TypeError (2026-10-09).** The package function
  `bodhi_lora.analysis.extract` was hidden by its submodule of the same name once the
  submodule was imported, so `analyze --extract` called a module. The function is now
  `extract_cell`, and every recorded cell with outputs has been extracted.
- **The 2026-04 pilot's rubric scores did not recompute (2026-10-09).** Its eval files
  predate the `score_metric` field and store the benchmark's raw score. Extract now
  checks each file against the formula it stored and puts the normalized score in the
  table for every cell.
- **A request longer than a server's context window was retried forever (2026-10-09).**
  Traces, filter and grade counted it as a failed request, marked the stage incomplete
  and asked for a rerun, which failed the same way. The engine now raises
  `ContextLengthError` on vLLM's refusal, and the stages list such items in their
  manifests and skip them (`tests/test_pipeline_runner.py`, `tests/test_engine_vllm.py`).
- **A stub run could still write under `results/` (2026-10-09).** The runner kept stub
  and dry runs in the scratch root, but `--results-root results` got around it. It now
  refuses a results root inside the project's.

- **The code was not general (D22).** One registry (`configs/`), one runner
  (`python -m bodhi_lora run`), one results root (`results/<cell_id>/`) and no model,
  benchmark or path names in package code, each checked by a test or `make check`
  (docs/CONVENTIONS.md rules 7 and 11 to 13). The uniform analysis (rule 14) is
  `python -m bodhi_lora analyze`, and `bodhi_lora.report` turns its output into tables
  and macros through `make numbers`. The paper cites the cross-cell table (Table H.7)
  and the per-cell macros in sections 4.3 and 4.4. Table 1 and the per-cell detail tables
  still come from `make cells`; `make numbers` stops if the two sources disagree on a
  per-seed mean.
- **Added-cell raw outputs synced (PROV-1, PROV-2, PROV-3; 60c1a54, 5d5890e).** The
  outputs were copied read-only from the GPU host on 2026-10-08. `make cells` computes
  every added-cell value, the paired tests of the aggregate drop and the per-cell tau,
  trace counts and adapter recipe from them, and the paper takes every added-cell value
  from there. One posted value was wrong (MedQuAD Base hedging, withdrawals W21). Every
  cell used tau = 0.4.
- **#2** `format_example` mis-zipped single examples in TRL's probe call. It now handles
  both shapes (`bodhi_lora/pipeline/train.py`).
- **#4** Filtering and eval use one normalized score,
  `(earned - negative) / (positive - negative)`. The old fields stay for comparison.
- **#5** Rubric-grader parse failures are counted per trace and reported, not dropped
  silently. The epistemic grader still drops them from its means (EVAL-9, above).
- **#6, #7** The traces stage refuses to resume a trace file made with another model, protocol or ablation.
- **#23** Training checkpoints on step intervals and resumes from the latest one.
- **#65** `bodhi_lora/engine/vllm.py` sets `--enforce-eager` by default, because
  CUDA-graph capture with LoRA hangs in vLLM 0.9. The engine runs vLLM as a subprocess
  only; the Docker and TPU modes were removed on 2026-10-08.
- **#3** The filter grader (Qwen2.5-14B) and the evaluator (Llama-3.1-8B) come from
  different families. Med42-8B breaks this rule for its own cells (see above).
- **Dependency pins (REPRO-10, b2452bc).** `requirements-infer.txt` pins
  `vllm==0.9.2` and `torch==2.7.0` and bounds transformers below 4.54, and each model
  record in `configs/models/` pins a Hugging Face revision (e6e1f2d; the records replaced
  `02ac9a3:configs/families.yaml`). Every run manifest records the package versions it
  ran with.
- **Benchmark data pinning (REPRO-11, 92aed30).** Each benchmark record pins its source:
  a dataset commit, or a sha256 per downloaded file. `bodhi_lora/data/loaders.py` builds
  the rows from it, and `bodhi_lora/data/splits.py` draws the pool and the holdout with a
  seed. The head-of-file rule of the posted cells is the `head` split of their records.
- **Leakage gate (EVAL-12, 838e0a9).** The runner runs `bodhi_lora/data/leakage.py` on
  the graded traces of every cell before training and writes its report,
  `leakage.json`, into the cell's shared inputs folder (`bodhi_lora/paths.py`).
- **One record per cell.** `configs/cells/` lists every cell of the final grid, with the
  Med42-8B and ChatDoctor cells, and the runs that stopped early.
- **Train/eval separation (CLAIM-11).** Older docs said training used Hard prompts. It
  did not: all 1,000 HealthBench-Hard prompts are excluded from training, and the
  seed-7 training set of the submitted cell has no Hard prompt.

## History

The submitted MedGemma-27B cell trained on TPU. Stage 3 first used custom MaxText LoRA
code, which never ran end to end on a TPU. It was replaced by tunix and qwix, which
produced the submitted adapters. The rebuttal moved to one GPU pipeline
(`bodhi_lora/pipeline/train.py`) and deleted the TPU code from this branch. The TPU tree is
still in history, for example `c4b8e6f:tpu/launch_5seeds_tunix.sh` (the tip of
`origin/main` on 2026-10-07). The launchers that produced the submitted cell are kept in
`legacy/`. Issues that only concerned the MaxText or TPU path are closed as obsolete.
