# Known issues

Open problems first, then the fixed ones, then a short history. IDs such as PROV-1 refer
to the audit of 2026-10-07 ([AUDIT_IDS.md](AUDIT_IDS.md) says what each one found);
D-numbers refer to [DECISIONS.md](DECISIONS.md). Issue numbers such as #5 are GitHub issues on `PeterLi-jpg/bohdi-lora`.

## Open

### Training

- **The adapters never trained on the end of a reply (found 2026-10-17).** When a
  tokenizer had no pad token, the training code padded with EOS
  (`bodhi_lora/pipeline/train.py` before 2026-10-17, and
  `579421f:scripts/train_lora.py:169-170` for the 2026-07 cells). The completion-only
  collator of TRL 0.11.4 builds its labels with transformers'
  `DataCollatorForLanguageModeling`, which gives every token with the pad id the label
  -100. So the EOS that ends each reply was out of the loss. On real training rows of R2
  and R5 with the Mistral-Small-24B tokenizer, the final `</s>` of every row had the
  label -100. Affected: every adapter on Mistral-Small-24B, BioMistral-7B or Med42-8B,
  that is, all 2026-07 adapters (their saved tokenizer configs say pad = EOS) and the
  2026-10 adapters of R2 and R5. R3 and R4 evaluate the R2 adapters. Not affected: the
  2026-05 MedGemma-27B cell, whose converter copied every reply token into the labels
  (`5bd9265:scripts/convert_traces_to_maxtext.py:217-219`; its loader cut rows longer than
  2,048 tokens before the reply ended: 96 of the 2,816 seed-7 rows by the Gemma 3 tokenizer, and
  3.7% by the comment in `5bd9265:configs/lora_medgemma27b_tunix.yaml`), and every Gemma, MedGemma,
  Qwen or Phi-4 tokenizer, which has its own pad token. What it did to the replies, by
  the content checks (a loop is a reply over 400 characters whose zlib ratio is below
  0.15; no score was read):
  - R5, HealthBench: `lora` 585 loops and 713 replies at the token limit of 1,000;
    `lora+ask-first` 829 and 831. `base` 1 and 9.
  - R5, MediQ paired: `lora` 1,607 loops and 1,880 at the limit of 2,000;
    `lora+ask-first` 1,785 and 1,789. `base` 0 and 0.
  - R2 adapters (R4, 3,875 replies each): `lora+bodhi` 70 loops and 229 at the limit,
    `lora+red-flags-first` 26 and 160, `lora` 2 and 83; `base` 0 and 45.
  - 2026-07 cells (their files do not flag the limit): BioMistral-7B `lora_bodhi` loops
    in 514 of 1,000 replies and `lora_no_wrapper` in 268, against 32 and 37 for the base;
    in the other cells at most 35 adapter replies of 1,000 loop. One base condition loops
    more than any of those: ChatDoctor with Med42-8B, 44 of 1,000 base replies against 0
    and 4 for its adapters, so there the base is the worse one.
  The code is fixed (see Fixed). Since 2026-10-18 the analysis counts these loops in every
  cell (`reply_health` in each stats.json; ANALYSIS_PLAN.md section 12), and the paper's
  appendix paragraph "What differs from the MedGemma-27B cell" says how the 2026-07
  adapters were trained, with those counts as macros. Open: the reruns of R2 to R5, which
  run on NRP (docs/NRP.md, DECISIONS.md D29).

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
  following. The fixed prompt (01b638e) was never run. The paper reports the cell's
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

- **The submitted cell's environment is not recorded (REPRO-10, EVAL-12).** The code side
  is fixed (see Fixed below). The 2026-05 cell ran on Modal with a vLLM range, and no
  freeze was kept (legacy/README.md). The 2026-07 cells are no longer open: since
  2026-10-13 `requirements-infer.lock.txt` and `requirements-train.lock.txt` are the full
  freezes of the two host envs that ran them, unchanged since 2026-07-23 (but for
  bitsandbytes in `.venv-infer`, added 2026-10-11), and the eval JSONs and adapters
  confirm the versions (`requirements-infer.txt` gives the evidence). Still open for all
  posted cells: an overlap-check output (the gate came after these runs). The eval ids are synced: each non-HealthBench cell has its `eval_ids.json`,
  and the HealthBench cells used `df10aa8:data/raw/hard_200_sample_ids.json`. New runs record all
  of this in their manifests (`bodhi_lora/manifest.py`).
- **The host's `multi_seed_summary.json` files are wrong.** They predate 2c13550 and list
  their seeds as `["eval"]`, so each holds one seed. They are gitignored and unused;
  `make cells` recomputes from the per-seed files.

### Planned runs

- **The cost estimates rest on shared-host times.** docs/RUN_PLAN.md scales the stage
  times of the 2026-07 cells, measured on a shared host, to each cell record. The speed
  of serving a 4-bit base in vLLM, of a 14B or 27B judge, and of the judges' harm call is
  not measured; the plan gives a range or states the assumption. The first run's
  manifests and its `bench` stage replace the estimates (E20, C25).
- **Seed-invariant conditions run once only within one shared eval.** Since 2026-10-12 a
  shared eval (docs/ARCHITECTURE.md section 4.2) runs each base condition of a fixed-id
  cell once and writes every seed's file from it. A run on several GPUs gives each seed
  its own eval step, and the `--no-shared-server` path evaluates each condition alone, so
  both still run them for every seed. The grade stage still grades each seed's copy of
  the same replies.
- **The serving changes of 2026-10-12 have run on the CPU stub only.** CUDA graphs on
  servers with no adapter, several adapters on one server (`--max-loras`), routing by
  model name, the 4-bit base under several adapters, and FlashAttention 2 for Gemma 2
  through the model record are tested on the stub engine and on the command lines they
  build, not on a GPU. R1 and R2 ran without them (commit 7e02348). What only a GPU run
  shows: whether graph capture works and how long it takes for each model (and with
  FlashAttention 2 for Gemma 2, and with the 4-bit base); whether the replies and grades
  of a server with graphs differ from those of an eager one (batch order already makes
  greedy replies differ slightly from run to run); whether base requests on a server
  with LoRA enabled give the replies of a base-only server; memory and speed with five
  adapters and the 4-bit base; and whether the first request to each adapter answers
  (the engine sends one at start, so a broken adapter stops the stage early).
- **A crash in a shared eval writes no partial result file.** One server makes the
  replies of every condition of a run, so a crash before the grading ends writes no
  result file. Each reply is on disk as it comes (`<condition>.replies.jsonl`), and a
  rerun makes only the missing ones. Since 2026-10-12 a condition with a failed reply or
  a failed grade gets no result file either, on both eval paths: its manifest says
  `incomplete` with the failures counted, the stage exits with an error, so the runner
  and a queue stop there, and a rerun makes only the missing replies. Up to then such a
  condition was written as complete with its failures counted (`n_failed_inference`),
  and the R1 and R2 eval manifests were written that way. A server that stops answering
  gets no more requests once each request it held has failed (`ServerDown` in
  `bodhi_lora/engine/vllm.py`), in every stage, so a hung eval server costs about two
  request timeouts, not one timeout per request. A request too long for a server's
  context is listed (`too_long_ids`) and skipped, as in the other stages.
- **The eval stage prints scores to its log.** The no-interim-look rule
  (docs/ANALYSIS_PLAN.md section 2.5) therefore depends on the operator reading
  manifests, not logs.
- **No automatic check orders the plan before the results.** Convention 3 relies on a
  review step and the commits that the run manifests record. `tests/test_plan_coverage.py`
  checks only that each planned cell is named in docs/ANALYSIS_PLAN.md and that the plans
  name no cell without a record.
- **`analyze` does not compute every planned statistic yet.** The judges' harm grades
  (C33) and the contrasts between cells (H9, H10, E24) are missing, and so is the tool
  that draws the 300 replies for R17's hand labels (docs/ARCHITECTURE.md section 7). Each
  is needed only once its cell has run. The paired-probe statistics (E14) and the
  declared contrasts of H5 and H6 (since 2026-10-10), and the H8 judge analysis and the
  probe, bench and stage-time readers (since 2026-10-11), are tested on synthetic tables
  and stub runs only.

### Repo state

- **Two tool trailers remain in public history (HYG-12).** The repository stays public,
  and the raters gave consent, so there is no privacy purge of history (D17, D18). On
  2026-10-11 the first author asked for a purge of the two trailers only. It rewrites the
  history of `main` and of this branch and needs a force-push, so it waits for his
  go-ahead on each (BLOCKED.md item 4.3). The recipe now rewrites the commit ids that
  tracked files cite before the push (`scripts/maintenance/purge_history.md`, step 7b), so
  `make check` and the pre-commit hook still pass afterwards. The current tree is clean.
- **Temp branches may remain on the remote (HYG-15, HYG-11).** The first author deleted
  the remote scratch branches on 2026-10-11 (HYG-12). Whether the
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

- **EOS out of the training loss (2026-10-17).** `set_pad_token` in
  `bodhi_lora/pipeline/train.py` pads with the model record's `chat_template.pad_token`
  and stops a run that would pad with EOS. The four records whose tokenizers have no
  separate pad token name one: `<pad>` (id 11) for Mistral-Small-24B, `<unk>` (id 0) for
  BioMistral-7B and Mistral-7B-Instruct (both fall back to bytes, so `<unk>` never
  occurs), and `<|reserved_special_token_2|>` (id 128004) for Med42-8B. Before any step,
  `check_replies_in_loss` runs the real collator on the first 16 tokenized training rows
  and stops when a reply token, the end-of-turn token included, is out of the loss. The
  smoke run passes through it (tests/test_pipeline_runner.py). Checked on 2026-10-17 on
  the GPU host with the real tokenizers of the six models that cells train and 16 R2 and
  R5 training rows: every reply token in the loss, and no row holds the pad id. The
  Mistral-7B-Instruct record also gained `response_template: "[/INST]"`, without which its
  template has no assistant header to find and training would have stopped.
- **The anonymous export kept the commit ids (2026-10-11).** The exported docs, tests and
  run manifests cited about 70 commits of this public repository, and GitHub finds a
  public commit by its hash, so the reviewer copy pointed at the authors. The paper
  already left hashes out for this reason (withdrawals W19). `scripts/export_anonymous.sh`
  now replaces every commit id of the repository with `<commit>` in the copy's text files
  (not in the response tables, whose 8-digit prompt ids could match by chance), and its
  gate fails on any that remain. On 2026-10-11 it replaced 596 ids in 273 files, and the
  gate passed.
- **The paper said a second copy of the adapters may exist (2026-10-09).** The Code
  availability paragraph and checklist items 4, 5, 11 and 13 said "we have not yet checked
  a second copy", and two comments in `paper/main.tex` pointed at the Modal volume. Since
  7ce22cc and 48814bf the paper says that the adapters were not retained (D19).
- **The hooks were not active in this clone (2026-10-11).** `core.hooksPath` pointed
  at the repository's old location, so `hooks/commit-msg` and `hooks/pre-push` did not
  run, and commit messages were checked by hand. It now points at `hooks/`, and a commit
  message that names a coding assistant is rejected. `hooks/pre-commit` was added the
  same day (BLOCKED.md item 5).
- **The report step named the cells it reported (2026-10-10).** Table 1, the per-cell
  tables and most main-body numbers came from `b533908:scripts/paper_stats.py`, which knew four
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
- **Added-cell raw outputs synced (PROV-1, PROV-2, PROV-3; 3b19a4c, b20f3f9).** The
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
- **#65** `bodhi_lora/engine/vllm.py` sets `--enforce-eager` on every server with an
  adapter, because CUDA-graph capture with LoRA hangs in vLLM 0.9. Since 2026-10-12 a
  server with no adapter captures CUDA graphs, and `serving.enforce_eager` in a record
  overrides either way (docs/ARCHITECTURE.md section 4.2). The engine runs vLLM as a
  subprocess only; the Docker and TPU modes were removed on 2026-10-08.
- **Gemma 2 crashed on its first request on an H100 (R1, 2026-10-11).** vLLM 0.10.2
  picks FlashAttention 3 there, which has no tanh softcapping. The run used a launcher
  prefix. Since 2026-10-12 the Gemma 2 model records carry
  `serving_env: {VLLM_FLASH_ATTN_VERSION: "2"}`, which the engine sets for their servers
  only.
- **#3** The filter grader (Qwen2.5-14B) and the evaluator (Llama-3.1-8B) come from
  different families. Med42-8B breaks this rule for its own cells (see above).
- **Dependency pins (REPRO-10, fc496d7).** `requirements-infer.txt` pins the env that
  ran the 2026-07 and 2026-10 cells: vllm 0.10.2, torch 2.8.0, transformers 4.56.2, with
  its full freeze in `requirements-infer.lock.txt` as constraints for `setup.sh`. Until
  2026-10-13 it pinned vllm 0.9.2, torch 2.7.0 and transformers below 4.54, a range that
  no run of the 2026-07 or 2026-10 cells used. Each model record in `configs/models/`
  pins a Hugging Face revision (481f998; the records replaced
  `df10aa8:configs/families.yaml`). Every run manifest records the package versions it
  ran with.
- **Manifests the runner writes named the runner's env (2026-10-13).** The runner runs in
  the training env and starts each stage in `python.infer` or `python.train`. The manifests
  it writes itself (the cell manifest, and the roll-up or pointer manifest in each seed's
  stage folders) recorded its own packages, so on the GPU host they said vllm 0.9.2, torch
  2.7.0 and transformers 4.57.6 for eval and grade work that ran under 0.10.2, 2.8.0 and
  4.56.2. Schema 3 records the env that ran the stage, and both envs on the cell
  manifest (`bodhi_lora/manifest.py`, `bodhi_lora/pipeline/runner.py`). The R1 to R6
  outputs (runs at commit a5aa4d6 and before) keep the old roll-ups; their per-part manifests
  (`<condition>.manifest.json`, `<grader>.manifest.json`, traces, filter, train, bench)
  were written by the stage itself and hold the right versions. No analysis reads the
  `packages` field.
- **Benchmark data pinning (REPRO-11, 28c5873).** Each benchmark record pins its source:
  a dataset commit, or a sha256 per downloaded file. `bodhi_lora/data/loaders.py` builds
  the rows from it, and `bodhi_lora/data/splits.py` draws the pool and the holdout with a
  seed. The head-of-file rule of the posted cells is the `head` split of their records.
- **Leakage gate (EVAL-12, 11cd835).** The runner runs `bodhi_lora/data/leakage.py` on
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
still in history, for example `8403e3b:tpu/launch_5seeds_tunix.sh` (the tip of
`origin/main` on 2026-10-07). The launchers that produced the submitted cell are kept in
`legacy/`. Issues that only concerned the MaxText or TPU path are closed as obsolete.
