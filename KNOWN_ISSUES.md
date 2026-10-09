# Known issues

Open problems first, then the fixed ones, then a short history. IDs such as PROV-1 refer
to the audit of 2026-10-07; D-numbers refer to [docs/DECISIONS.md](docs/DECISIONS.md).
Issue numbers such as #5 are GitHub issues on `PeterLi-jpg/bohdi-lora`.

## Open

### Blocked on data outside the repo

- **The added cells' full responses are not versioned (PROV-1).** The per-seed eval JSON
  (about 370 MB) is in the local copy, in the `results/*__2026-07/` cells, and on the
  shared H100 host, and it is gitignored. The repo tracks what the paper needs from it
  (`results/cross_cell/generated_values.json`) and the adapter configs, trainer states and
  eval ids. Rebuilding `generated_values.json` needs the local copy (`make cells`).
- **The submitted adapter config is not confirmed (HYG-4, D1).** Repo evidence says
  r=8, alpha=16. The GCS bucket that held the five submitted adapters returned 404 on
  2026-10-08, so the paper says the values come from the code and could not be checked.
  `legacy/modal_eval.py` copied each seed's `checkpoints/best/` into the Modal volume
  `bohdi-data-cache`, which may still hold them (BLOCKED.md item 3).
- **Returned physician grades are not in the repo (PROV-5, EVAL-5).** One of three
  raters returned. `scripts/compute_irr_kappa.py` read that file in place and wrote
  `results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/kappa_results.json`, which names no rater. Nobody else can
  recompute it until the de-identified grades (grading_id to pass, fail or unsure) are
  committed. Strip the `physician_id` column first. Rater identities stay out of git.

### Measurement and design

- **Physician validation is incomplete (D5).** The check grades HealthBench rubric
  criteria only, and agreement is below the pre-registered 0.6 target. No physician
  graded the seven behaviors, so every epistemic-behavior result is labeled LLM-graded.
- **The grader departs from its own anchors.** It gives context-seeking 2 to answers
  that ask no direct question, and hedging 2 to most answers it flags as blanket
  disclaimers (`context_seeking_anchor` and `blanket_disclaimers` in
  `paper/generated/numbers.json`).
- **Med42-8B shares the evaluator's family (EVAL-7, D6).** The cells are marked † and
  kept out of cross-model statements. `bodhi_lora/pipeline/grade.py` is not run yet;
  it needs a GPU.
- **No prior-work baseline has run (EVAL-8).** `bodhi_lora/protocols/` holds the prompt
  baselines; they need GPU inference on the existing eval sets.
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
  asking. It is untested (`rebuttal/analyses/FINDINGS_quality_drop.md`, section 5).
- **The discrimination labels were defined after the results (EVAL-10, D9).** Both
  labelings are reported, both marked post hoc and exploratory.
- **Calibration is not measured (#1, D3).** The eval JSON has Brier and ECE fields from
  a token-probability proxy. The paper does not report them.

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
