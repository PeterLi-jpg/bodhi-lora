# Known issues

Open problems first, then the fixed ones, then a short history. IDs such as PROV-1 refer
to the audit of 2026-10-07; D-numbers refer to [docs/DECISIONS.md](docs/DECISIONS.md).
Issue numbers such as #5 are GitHub issues on `PeterLi-jpg/bohdi-lora`.

## Open

### Blocked on data outside the repo

- **Added-cell raw outputs are not synced (PROV-1).** The added cells exist only on the
  shared H100 host. Until the sync in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) section 7
  runs, their numbers come from `data/rebuttal_cells/posted_values.json` and carry the
  "transcribed from posted values" footnote. The per-cell tau (PROV-3) and the MedQA
  preamble version are in the same logs.
- **The submitted adapter config is not confirmed (HYG-4, D1).** Repo evidence says
  r=8, alpha=16. The `adapter_config.json` of the five submitted seeds on GCS must
  confirm it before the paper's hyperparameter table is final.
- **Returned physician grades are not in the repo (PROV-5, EVAL-5).** One of three
  raters returned. `scripts/compute_irr_kappa.py` read that file in place and wrote
  `results_modal/irr/kappa_results.json`, which names no rater. Nobody else can
  recompute it until the de-identified grades (grading_id to pass, fail or unsure) are
  committed. Strip the `physician_id` column first. Rater identities stay out of git.

### Measurement and design

- **Physician validation is incomplete (D5).** Agreement is below the pre-registered
  0.6 target, so every epistemic-behavior result is labeled LLM-graded.
- **Med42-8B shares the evaluator's family (EVAL-7, D6).** The cells are marked † and
  kept out of cross-model statements. `scripts/regrade_cross_family.py` is not run yet;
  it needs the synced outputs and a GPU.
- **No prior-work baseline has run (EVAL-8).** `scripts/baselines/` holds the prompt
  baselines; they need GPU inference on the existing eval sets.
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
  asking. It is untested (`rebuttal/analyses/FINDINGS_quality_drop.md`, section 5).
- **The discrimination labels were defined after the results (EVAL-10, D9).** Both
  labelings are reported, both marked post hoc and exploratory.
- **Calibration is not measured (#1, D3).** The eval JSON has Brier and ECE fields from
  a token-probability proxy. The paper does not report them.

### Reproducibility gaps

- **Dependency pins (REPRO-10).** The `.venv-infer` env for vLLM is not written down,
  vLLM is pinned only to `>=0.9.0,<0.10`, and no model or grader has a Hugging Face
  revision pin.
- **Benchmark data is not pinned (REPRO-11).** `scripts/build_benchmark_jsonl.py` reads
  the latest revision of each Hugging Face dataset, and the eval holdout is the first
  200 rows, not a seeded draw.
- **No leakage gate in the added-cell launcher (EVAL-12).** Generation excludes eval ids
  with `--exclude-ids`, but `rebuttal/launch/run_cell.sh` does not run
  `scripts/check_dataset_overlap.py`, and no overlap or contamination output is
  committed.
- **`rebuttal/launch/run_grid.sh` predates the final grid.** It has no Med42-8B or
  ChatDoctor cells. Use the cell table in REPRODUCIBILITY.md section 5.

### Repo state

- **Rater names, the author block and two tool trailers remain in public history
  (EVAL-6, HYG-1, HYG-12).** The current tree is clean. The purge is prepared in
  `scripts/maintenance/purge_history.md` and not run (D16).
- **Scratch and temp branches remain on the remote (HYG-12, HYG-15, HYG-11).** This
  includes `results-temp`, which holds MedGemma adapter weights and optimizer state.
  `scripts/maintenance/prune_branches.sh` lists them; `--apply` deletes them (D16).
- **The remote slug misspells the name (`bohdi-lora`, D15).** Renaming is a remote
  action and is deferred.

## Fixed

- **#2** `format_example` mis-zipped single examples in TRL's probe call. It now handles
  both shapes (`scripts/train_lora.py`).
- **#4** Filtering and eval use one normalized score,
  `(earned - negative) / (positive - negative)`. The old fields stay for comparison.
- **#5** Rubric-grader parse failures are counted per trace and reported, not dropped
  silently. The epistemic grader still drops them from its means (EVAL-9, above).
- **#6, #7** `generate_traces.py --resume-from` refuses rows made with other settings.
- **#23** Training checkpoints on step intervals and resumes from the latest one.
- **#65** `scripts/_vllm_engine.py` sets `--enforce-eager` the same way in Docker and
  subprocess mode. Eager mode stays the default, because CUDA-graph capture with LoRA
  hangs in vLLM 0.9.
- **#3** The filter grader (Qwen2.5-14B) and the evaluator (Llama-3.1-8B) come from
  different families. Med42-8B breaks this rule for its own cells (see above).
- **Train/eval separation (CLAIM-11).** Older docs said training used Hard prompts. It
  did not: all 1,000 HealthBench-Hard prompts are excluded from training, and the
  seed-7 training set of the submitted cell has no Hard prompt.

## History

The submitted MedGemma-27B cell trained on TPU. Stage 3 first used custom MaxText LoRA
code, which never ran end to end on a TPU. It was replaced by tunix and qwix, which
produced the submitted adapters. The rebuttal moved to one GPU pipeline
(`scripts/train_lora.py`) and deleted the TPU code from this branch. The TPU tree is
still in history, for example `c4b8e6f:tpu/launch_5seeds_tunix.sh` (the tip of
`origin/main` on 2026-10-07). The launchers that produced the submitted cell are kept in
`legacy/`. Issues that only concerned the MaxText or TPU path are closed as obsolete.
