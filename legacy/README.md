# legacy/: the code behind the submitted MedGemma numbers

The submitted MedGemma-27B numbers (`results/healthbench__medgemma-27b-text__bodhi__2026-05/`, moved from `02ac9a3:results_modal/`) came
from a TPU training path and a Modal eval path. Commit ef5b923 deleted both from
this branch. This directory restores them from history, unchanged except for
two reworded comments (section 1), so the submitted cell can be read and, in
principle, run again. The current GPU
pipeline (`scripts/`, `rebuttal/launch/`) is a different recipe. Section 4 lists
the differences.

Nothing here runs from this directory as it is. Section 5 says how to run it.

## 1. Files

Every file is the blob at commit 8262d49 (main, 2026-05-03 19:19 PDT), with
one exception: two code comments that named a tool were reworded
(`tpu/launch_5seeds_tunix.sh:52-53`, `scripts/train_lora_tunix.py:397-398`).
No code changed and every line number is the same; `git show 8262d49:<path>`
gives the original. 8262d49 is
the `git_sha` in the `run_metadata` of every submitted result file, because the
Modal eval cloned main at that commit. `modal_eval.py` was committed later
(e6acbff, 2026-05-04), with no change between its two commits.

| File | Git blob at 8262d49 | What it did |
|---|---|---|
| `modal_eval.py` | 8a3d309 | Stages 4 and 5 (the 2x2 eval and the epistemic grade) for the 5 seeds on Modal A100-80GB. It cloned main, patched three files at run time, pulled each adapter from GCS and wrote `02ac9a3:results_modal/` (now `results/healthbench__medgemma-27b-text__bodhi__2026-05/`). |
| `tpu/launch_5seeds_tunix.sh` | 2e5b571 | Stages 1 to 3 on five v6e-8 spot TPU VMs, one seed per VM: trace generation, Qwen filtering, tunix LoRA training and the export to a PEFT adapter. |
| `configs/lora_medgemma27b_tunix.yaml` | 9d899fe | The tunix training config. |
| `scripts/train_lora_tunix.py` | f92d792 | The tunix and qwix LoRA trainer. |
| `scripts/export_tunix_lora_to_peft.py` | 9ee6cd5 | Converts the orbax LoRA checkpoint to a PEFT adapter. |
| `scripts/convert_traces_to_maxtext.py` | 72d05e7 | Tokenizes train/val and masks the prompt tokens. The tunix trainer reads its output. |
| `scripts/maxtext_lora/dataset_loader.py` | 37dafdc | Batches the tokenized data for the trainer. |
| `scripts/prefetch_gemma_safetensors.py` | 3d21545 | Pre-downloads the MedGemma weights into a TPU host's HF cache. The launcher does not call it; `train_lora_tunix.py` downloads the weights itself. Moved here from `scripts/` on 2026-10-09. |
| `requirements.lock.txt` | (moved) | Formerly at the repo root. See section 6. |

The launcher also needs `tpu/setup_tpu.sh` and the rest of main at 8262d49.
They are not copied here; `git show 8262d49:<path>` prints any of them.

## 2. The submitted cell, stage by stage

| Stage | What ran | Where |
|---|---|---|
| 1 Generate | `generate_traces.py --datasets healthbench_hard healthbench --exclude-ids healthbench_hard.jsonl hard_200_sample_ids.json --use-bodhi`, MedGemma-27B-text-it in the vllm/vllm-tpu:latest docker image, max_model_len 8192, 64 slots. HealthBench has 5000 prompts and Hard is a subset of 1000, so 4000 prompts remain. One run, shared by all seeds through GCS. | `tpu/launch_5seeds_tunix.sh:395-402`, `:404-412` |
| 2 Filter | `filter_traces.py --grader-model Qwen/Qwen2.5-14B-Instruct --min-score 0.4 --val-ratio 0.1 --seed ${SEED}`. Run once per seed, so the seed changed the 90/10 train/val shuffle. | `tpu/launch_5seeds_tunix.sh:420-429` |
| 2.5 Leakage gate | `check_dataset_overlap.py --train-jsonl data/sft/train.jsonl --tag-overlap` | `tpu/launch_5seeds_tunix.sh:448-450` |
| 3 Train | `train_lora_tunix.py --config "${TRAIN_CONFIG}" --seed ${SEED}`. Skipped when an orbax step-250 checkpoint was already present. | `tpu/launch_5seeds_tunix.sh:521-530` |
| 3 Export | `export_tunix_lora_to_peft.py ... --r 8 --alpha 16 --dropout 0.0` | `tpu/launch_5seeds_tunix.sh:536-541` |
| 4 Eval | `eval_healthbench.py` on 4 configs, grader meta-llama/Llama-3.1-8B-Instruct, a separate `Random(seed).sample(200)` of the 1000 Hard prompts per seed | `modal_eval.py:475-512` |
| 5 Epistemic | `eval_epistemic.py` on the 4 outputs, same grader | `modal_eval.py:521-536` |

The TPU VMs ran `git fetch origin main` and `git reset --hard origin/main` at each
start and restart (`tpu/launch_5seeds_tunix.sh:263-271`). Each VM therefore ran
whatever main was when it last started. The training config changed many times
on 2026-05-03 (commits 9cbe28d to 9dab2b9), so a seed that restarted at a
different time may have trained with different settings. Only the trainer logs
on GCS can show which.

## 3. The submitted training recipe (D1)

The paper's hyperparameter table states r=16, alpha=32, all 7 projections,
3 epochs, effective batch 16, sequence length 4096. That is the GPU recipe in
`configs/lora_medgemma27b.yaml`, and the rebuttal configs copy it. The submitted
adapters came from the tunix path, with these values:

| Setting | Value | Source |
|---|---|---|
| Base model | google/medgemma-27b-text-it, bf16 | `configs/lora_medgemma27b_tunix.yaml:19`, `:28` |
| LoRA rank | 8 | `configs/lora_medgemma27b_tunix.yaml:37`; export `tpu/launch_5seeds_tunix.sh:540` |
| LoRA alpha | 16 | `configs/lora_medgemma27b_tunix.yaml:38`; export `tpu/launch_5seeds_tunix.sh:540` |
| LoRA dropout | 0.05 in training (export writes 0.0, inference only) | `configs/lora_medgemma27b_tunix.yaml:39`, `scripts/train_lora_tunix.py:365` |
| Target modules | q_einsum and kv_einsum (attention only), exported as q_proj, k_proj and v_proj; k and v share one lora_A | `configs/lora_medgemma27b_tunix.yaml:40-42`, `scripts/export_tunix_lora_to_peft.py:308-322` |
| Steps | max_steps 300 optimizer steps; the config comment counts this as about 6.2 epochs of about 3,100 rows | `configs/lora_medgemma27b_tunix.yaml:90`, comment `:65-68` |
| Exported step | the newest complete orbax checkpoint: 300, or 250 when the step-300 save was broken (seen on seeds 7, 99 and 101) | `scripts/export_tunix_lora_to_peft.py:341-377`, `scripts/train_lora_tunix.py:551`, `tpu/launch_5seeds_tunix.sh:512-523` |
| Effective batch | 64 per optimizer step: per-device batch 1, 8 rows per loader yield (1 x grad_accum 8), and 8 yields accumulated per update | `configs/lora_medgemma27b_tunix.yaml:45`, `:63`, comment `:53-56`; `scripts/train_lora_tunix.py:401-402` |
| Sequence length | 2048 in the trainer; the tokenizer step truncates at 4096 first | `configs/lora_medgemma27b_tunix.yaml:112`; `scripts/convert_traces_to_maxtext.py:72` |
| Optimizer | `optax.adamw(learning_rate)` with optax defaults (b1 0.9, b2 0.999, weight decay 1e-4) | `scripts/train_lora_tunix.py:449` |
| Schedule | constant, no warmup | `scripts/train_lora_tunix.py:449` |
| Learning rate | 1e-4 | `configs/lora_medgemma27b_tunix.yaml:64` |
| Eval interval | every 50 steps | `configs/lora_medgemma27b_tunix.yaml:94` |
| Hardware | one v6e-8 TPU VM per seed, mesh fsdp=8 x tp=1, block rematerialization | `tpu/launch_5seeds_tunix.sh:2`, `:74`; `scripts/train_lora_tunix.py:308-316`, `:163` |

Open points, all settled only by the files on GCS (see "Blocked" below):

- The launcher's default `TRAIN_CONFIG` is the smoke config
  (`tpu/launch_5seeds_tunix.sh:146`). The production run set it in the
  environment, which no file records. The GCS prefix `prod-v39-tunix-27b-neurips`
  (`modal_eval.py:54`) and the comment "v39 measured ..." at
  `configs/lora_medgemma27b_tunix.yaml:95-102` tie run v39 to this config.
- The sequence length moved between 1024, 1536, 1792 and 2048 during the run
  (commits b3f5ff8 to 363267f). Commit 624a3d4 says v39 hit an out-of-memory
  error at 2048. The config comment mentions a watchdog that halves the length
  on out-of-memory; no committed file contains it.
- Whether tunix counts `max_steps` in optimizer steps or in loader yields, and
  so whether the effective batch is 64 or 8, comes from the config comment.
  google-tunix was not pinned (`tpu/setup_tpu.sh` at 8262d49), so the version
  that ran is unknown.
- The audit's adapter-size check (60.47 MiB per seed with the tokenizer) agrees
  with r=8 on q/k/v and rules out r=16 on all 7 modules.

## 4. The submitted eval settings

| Setting | Value | Source |
|---|---|---|
| Hardware | Modal A100-80GB, one per seed | `modal_eval.py:290` |
| Context length | 4096 (patched from 8192) | `modal_eval.py:362-364` |
| Scheduler slots | 32 (patched from 64) | `modal_eval.py:373-375` |
| Client threads | 32 | `modal_eval.py:324` |
| Request timeout | 900 s (patched from 180) | `modal_eval.py:393-395` |
| Grader | meta-llama/Llama-3.1-8B-Instruct for both evals | `modal_eval.py:503`, `:530` |
| Eval prompts | per seed, `Random(seed).sample(200)` of the 1000 Hard prompts | `modal_eval.py:475-482` |
| Response length | 1024 new tokens without a wrapper; 2048 per BODHI pass | `8262d49:scripts/eval_healthbench.py:78`, `8262d49:scripts/_vllm_engine.py:438` |
| Packages | transformers 4.57.6, peft 0.19.1, bodhi-llm 0.1.4 (recorded in each result); torch `>=2.5,<2.9` and vllm `>=0.6.5,<0.10` (ranges) | `modal_eval.py:66-84` |

The vLLM range was resolved when the image was built (2026-05-03). pip takes
the newest version in range, and on PyPI that was 0.9.2 (2025-07-08; 0.10.0
followed on 2025-07-25). vLLM 0.9.2 requires torch==2.7.0. So the eval almost
certainly ran vLLM 0.9.2 with torch 2.7.0. No log records it. The paper's "vLLM
0.6.x series" is not supported.

The current eval scripts take these settings as flags, and their defaults equal
this table (`bodhi_lora/pipeline/eval.py`, `bodhi_lora/grading/epistemic.py`). The
per-seed draw of `bodhi_lora/data/splits.py` gives exactly the prompt
set of each submitted `base_no_wrapper.json`; this was checked for all 5 seeds.

## 5. Running the submitted cell again

1. `git worktree add ../bohdi-submitted 8262d49`. The launcher fetches
   `origin/main` on every VM (`tpu/launch_5seeds_tunix.sh:263-271`), so point it
   at a branch that stays at 8262d49, or remove that fetch, before a run.
2. Stages 1 to 3: `TRAIN_CONFIG=configs/lora_medgemma27b_tunix.yaml
   GCS_OUTPUT_PATH=gs://<bucket>/<run> bash tpu/launch_5seeds_tunix.sh` on a
   TPU v6e quota. Pin google-tunix and qwix first; section 3 explains why.
3. Stages 4 and 5: set `GCS_BASE` in `modal_eval.py:54` to the new run and
   `modal run modal_eval.py::run_all`.

Both steps cost money (TPU and Modal). Nothing in this lane ran them.

## 6. requirements.lock.txt

This file was at the repo root. It pins torch 2.11.0 and has no vllm, jax,
tunix or bitsandbytes entry, so it is not a freeze of any environment that
produced a reported number: the Modal eval image had torch below 2.9 and vllm,
and the TPU trainer needed jax and tunix. Its header names
`tpu/setup_tpu.sh` as a source of truth, so it belongs to the submission era.
It moved here so that the root no longer holds a lock that contradicts
`requirements.txt` (torch 2.7.0). The root now has `requirements.txt` (training
env) and `requirements-infer.txt` (vLLM env).

## 7. The posted rebuttal cells

The six posted cells (HealthBench x {Mistral-Small-24B, Med42-8B, BioMistral-7B};
ChatDoctor x {Mistral-Small-24B, Med42-8B}; MedQuAD x Mistral-Small-24B) and the
set-aside MedQA cell ran `rebuttal/launch/run_cell.sh` at 61dada9. That script
is now rewritten. `PROFILE=rebuttal-posted` (see `rebuttal/launch/_common.sh`)
restores its settings, so these cells stay describable:

| Setting | Posted cells | Submitted cell | Source (posted) |
|---|---|---|---|
| LoRA recipe | r=16, alpha=32, 7 projections, 3 epochs, effective batch 16, seq 4096, cosine with 5% warmup | section 3 | `rebuttal/configs/*.yaml` |
| Training precision | Mistral-24B: 4-bit NF4 QLoRA; Med42, BioMistral: bf16 | bf16 | `rebuttal/configs/lora_mistral_small_24b_qlora.yaml:16` |
| Eval base precision | bf16 for every cell, so the Mistral-24B adapters were trained on a 4-bit base and evaluated on a bf16 base. `EVAL_BASE_QUANT=match` now evaluates on the matching 4-bit base. | bf16 | `61dada9:scripts/_vllm_engine.py:236`, `:314` |
| Context length / slots | 8192 / 64 | 4096 / 32 | `61dada9:scripts/_vllm_engine.py:154`, `:250` |
| Client threads | 16 | 32 | `61dada9:scripts/eval_healthbench.py:380` |
| Grader | NousResearch/Meta-Llama-3.1-8B-Instruct (a re-upload; byte identity with the meta-llama weights is not checked here) | meta-llama/Llama-3.1-8B-Instruct | `61dada9:rebuttal/launch/run_cell.sh:38` |
| HealthBench eval prompts | one fixed set (`data/raw/hard_200_sample_ids.json`) for every seed; it shares 141 of 200 prompts with the submitted union | a separate 200 per seed | `61dada9:rebuttal/launch/run_cell.sh:97` |
| Other benchmarks | pool = first 4900 usable rows in HF order; holdout = first 200 ids of that file | n/a | `61dada9:rebuttal/launch/run_cell.sh:113`, `:116-121` |
| tau | 0.4 in every cell (`rebuttal/results/cell_recipe.json`). The rule re-filters a non-HealthBench cell at the 22nd-percentile tau when its train/raw is below 40%; no cell met it, and no log shows a second cut. MedQuAD kept 78.4% at 0.4, close to what a 22nd-percentile cut keeps, which caused an earlier misreading (`rebuttal/withdrawals.md` W11). | 0.4 | `61dada9:rebuttal/launch/run_cell.sh:159-201` |
| Train/val split | once per cell, seed 42 | once per seed | `61dada9:rebuttal/launch/run_cell.sh:146-157` |
| Hardware | one H100 80GB per stage | v6e-8 TPU (train), A100-80GB (eval) | `rebuttal/PLAN.md` |
| vLLM env | `.venv-infer`, versions not recorded | see section 4 | `61dada9:rebuttal/launch/run_cell.sh:18-22` |

The Med42-8B base is in the evaluator's family (Llama-3). The launcher now
refuses such a cell unless `--allow-same-family` is passed.
`bodhi_lora/pipeline/grade.py` re-grades a cell with a judge from a third
family.

## Blocked: what only remote files can settle

- `gs://bohdi-runs-tokyo-micron/prod-v39-tunix-27b-neurips/seed_<N>/`: the
  `checkpoints/best/adapter_config.json`, the orbax step that was exported, the
  `train.log` (sequence length, steps, losses) and `train.jsonl` / `val.jsonl`
  line counts of each seed. These confirm or correct section 3 and give the
  per-seed filter counts (PROV-11).
- The tunix, qwix and jax versions on the TPU VMs.
- On the H100 box (`/data/PROJECTS/bohdi/results_rebuttal/`): the filter logs
  with the tau each posted cell used, `eval_ids.json` of each cell, the built
  benchmark JSONLs and a `pip freeze` of `.venv` and `.venv-infer`.
