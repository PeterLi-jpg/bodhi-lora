# Known Issues

Triage of issues raised against this repo, with status for each.

Issue numbers below match GitHub issues on `PeterLi-jpg/bohdi-lora`.

## Fixed

### [#2] `format_example` batching bug — FIXED
TRL probes `formatting_func` on a single example first to decide whether it returns str or list[str]. The previous code unconditionally zipped `batch["messages"]` and `batch["response"]`, which on a single-example call silently zipped message dicts with *characters* of the response string — producing malformed training text. `format_example` now handles both shapes explicitly (`scripts/train_lora.py`).

### [#5] Silent grader parse failures — FIXED
`grade_trace` now returns a `parse_failures` count per trace. Both `filter_traces.py` and `eval_healthbench.py` surface the aggregate parse-failure rate in their summary output. Raw failed grader outputs are attached as `raw_parse_failures` for inspection.

### [#6, #7] Resume logic mixing settings — FIXED
`generate_traces.py --resume-from` now validates that existing rows in the target file were produced with the same `--model` and `--use-bodhi` settings. Mismatch aborts with a clear error. Override with `--force-resume` only when intentionally mixing is desired.

### [#23] Mid-epoch checkpoint resume — FIXED
Training now checkpoints on aligned step intervals instead of epoch boundaries. The default config uses matching `save_strategy: steps` and `eval_strategy: steps` values so `load_best_model_at_end=True` remains valid, and `scripts/train_lora.py` auto-resumes from the latest `checkpoint-*` directory under `--output-dir` when one is present.

## Open / blocked

### Stage 3b (MaxText LoRA training) — STRUCTURALLY BROKEN

`scripts/train_lora_maxtext.py` was committed as the Stage 3 training entry but has **never run end-to-end on a TPU VM**. Smoke runs v9–v11 each died in `setup_tpu.sh` before the trainer was reached, masking the issues below until the v167 setup hardening let setup pass. Confirmed against the vendored `third_party/maxtext` source.

**Bugs in `scripts/train_lora_maxtext.py`:**

| Line | Issue | Reality |
|---|---|---|
| 102 | `sys.path.insert(0, third_party/maxtext)` | should be `third_party/maxtext/src` (matches `scripts/convert_medgemma_to_maxtext.py:73`) |
| 107 | `from MaxText import pyconfig` | package is lowercase `maxtext`; pyconfig lives at `maxtext.configs.pyconfig` |
| 108 | `from MaxText.experimental.sft import sft_trainer` | no such module; SFT lives at `maxtext.trainers.post_train.sft.train_sft` |
| 219 | `dataset_loader.build_iterators(...)` | function does not exist anywhere — the `dataset_loader` alias in `scripts/maxtext_lora/__init__.py` resolves to `scripts/convert_traces_to_maxtext.py`, which is the offline JSONL writer, not an iterator builder |
| 252 | `sft_trainer.train(config=, model=, params=, train_iter=, eval_iter=, trainable_param_filter=, on_first_step=, ...)` | the actual upstream API is `train_sft.train(mt_config, goodput_recorder=None)` — takes one config object and constructs everything internally; none of these kwargs exist |
| 282 | `export_peft.write_adapter(orbax_checkpoint=, base_model_name=, rank=, alpha=, dropout=, variant=)` | function is `write_peft_adapter` with a different signature (takes pre-extracted `weights` dict). There is an `export(orbax_path, output_dir, base_model, settings)` end-to-end helper — call site must use either |

**Underlying architectural blocker — Python version mismatch:**

MaxText's SFT path delegates to **Tunix** (`from tunix.sft import peft_trainer` inside `train_sft.py`). `google-tunix` on PyPI requires **Python 3.11+**. TPU v6e VMs ship **Python 3.10** (and the rest of the bohdi-llm stack — vLLM-TPU container, torch_xla 2.7, optimum-tpu — is pinned to py3.10). So even with all five line-level fixes above, the import `from maxtext.trainers.post_train.sft import train_sft` would fail at module load with a missing-Tunix error.

**Why we can't just fall back to torch_xla:**

Per `docs/maxtext_migration.md` and commit `2460ac9`, Stage 3 on torch_xla 2.7 + FSDPv2 + Gemma-3-27B hangs on the first `xm.mark_step()` for 30+ min on v6e with no progress and no useful stderr. That's the whole reason MaxText was forked. Switching the launcher back to `tpu/launch_5seeds.sh` would replace "fast AttributeError" with "30+ min silent hang then nothing."

**Paths forward (require explicit decision):**

1. **Custom JAX training loop** — bypass Tunix, write a minimal SFT loop on top of MaxText's model + Optax. Multi-day work; needs live TPU iteration to validate. Avoids the py3.11 wall.
2. **Upgrade v6e VM image to Python 3.11** — multi-week infra; need to verify the rest of the bohdi-llm stack (vLLM-TPU, torch_xla, optimum-tpu) supports py3.11.
3. **Ship without Stage 3 LoRA results** — smoke and report Stages 1+2+4+5 only, with a stub or empty PEFT adapter at `checkpoints/seed_<N>/best/`. Validates ~80% of the pipeline; loses the LoRA-trained results.

Until one of these is taken, **`tpu/launch_5seeds_maxtext.sh` cannot complete Stage 3 on the current v6e image**.

## Documented, deferred to discussion

### [#1] Brier / ECE do not measure model calibration — PARTIALLY ADDRESSED
The old implementation used the evaluator's rubric score as "confidence" and compared it against rubric outcomes from the same grading pass. That was grader-internal consistency, not model calibration. **Action taken**: eval output now includes `brier_model_calibration` / `ece_model_calibration`, using the geometric mean next-token probability of the emitted response as a model-derived confidence proxy, while keeping the legacy `brier_grader_consistency` / `ece_grader_consistency` fields for backward comparison. **Still open for discussion**: whether the response-level logprob proxy is strong enough for the paper, or whether the final claim should move to a richer confidence protocol (verbalized confidence, abstention head, or similar).

### [#3] Same grader for filtering and final evaluation — ADDRESSED (asymmetric default)
The pipeline now ships an **asymmetric grader default**: `filter_traces.py` uses `Qwen/Qwen2.5-14B-Instruct` and `eval_healthbench.py` / `eval_epistemic.py` use `meta-llama/Llama-3.1-8B-Instruct`. Training-data selection and reported metrics are graded by **different model families**, so the "graded by your own evaluator" critique no longer applies to the headline numbers.

For an additional bias-control sweep, set `SECOND_GRADER_MODEL=...` (any family different from Llama, e.g. `Qwen/Qwen2.5-14B-Instruct`) — the launcher re-grades the four eval configs with the second grader and runs `scripts/grader_correlation.py` to report Spearman ρ vs. the primary Llama grader.

Both gated models need HF access:
- https://huggingface.co/Qwen/Qwen2.5-14B-Instruct (filter)
- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct (eval)

### [#4] Inconsistent filtering-score normalization — FIXED
Filtering and eval now use a normalized rubric score that accounts for both the positive ceiling and the negative penalty floor:

`normalized_score = (earned_points - negative_points) / (positive_points - negative_points)`

This makes a fixed threshold such as `--min-score 0.4` comparable across prompts with different penalty structure. The previous positive-only score is still emitted as `positive_score` for analysis. The alternate audit views (`absolute_point_score`, `positive_criteria_rate`) plus `--score-field` are still available in `filter_traces.py` so the old default can be re-derived without silently changing the new default.

### [#65] enforce-eager flag now opt-in for graph mode — RESOLVED
`scripts/_vllm_engine.py` previously hard-coded `--enforce-eager` in `_build_docker_cmd` while `_build_subprocess_cmd` silently omitted it, so the same code path produced different vLLM runtime modes depending on whether the host had a Docker daemon. Both builders now derive the flag from a new `enforce_eager: bool = True` kwarg on `VLLMEngine`, so the two run-modes stay in sync.

`enforce_eager=True` remains the production-eval default — turning it off triggers the ~130 min CUDA-graph capture hang on LoRA in vllm 0.9, which is still unresolved upstream. The new `scripts/latency_benchmark.py` defaults to `--enforce-eager` for all 4 configs (`base_no_wrapper`, `base_bodhi`, `lora_no_wrapper`, `lora_bodhi`) so the paper's base-vs-LoRA latency comparison stays apples-to-apples. Pass `--no-enforce-eager` only after the graph-capture hang is fixed; until then it pays the full capture cost on every config.

## Infrastructure for reporting and robustness

These aren't issues per se, but directly support the reviewer concerns that drive #1, #3, #4 and the paper overall.

### Bootstrap 95% CIs on every reported metric
`scripts/eval_ushape.py --bootstrap 1000` nonparametrically resamples the 200-prompt holdout 1000 times and reports 2.5th / 97.5th percentile CIs for mean and fail rate, stratified by tier and theme. Deterministic via `--bootstrap-seed` (default 42). `scripts/plot_ushape.py` renders the CIs as shaded bands and error bars.

### Multi-seed variance
`scripts/run_multi_seed.sh` reuses one set of generated traces and varies seed across filter split, LoRA init, and training order. `scripts/aggregate_seeds.py` combines per-seed eval JSONs into across-seed mean ± std (plus percentile CIs when `N >= 5`). Default seeds: `42 7 13 99 101`. Override with `SEEDS="..." bash scripts/run_multi_seed.sh`.

### HealthBench-only generalization experiment
Confirmed via `scripts/check_dataset_overlap.py` that HealthBench Hard is a strict subset of HealthBench full. To train on HealthBench without any Hard leakage, pass both the eval holdout and the full Hard JSONL to `--exclude-ids`:

```
python scripts/generate_traces.py \
    --datasets healthbench \
    --exclude-ids data/raw/healthbench_hard.jsonl data/raw/hard_200_sample_ids.json \
    ...
```

This produces ~4,000 training prompts from the Full set, with the 1,000 Hard prompts fully excluded. Compare against the main run (~4,800 prompts from Full + Hard minus the 200 eval).

### Optional QLoRA / DoRA / rsLoRA
`configs/*.yaml` now accept `model.quantization` in `{null, "4bit", "8bit"}` and `lora.variant` in `{"standard", "dora", "rslora"}`. DoRA is incompatible with quantization and is guarded with an upfront error. Existing configs leave both on defaults, so current runs behave identically.

## Process note

Fixes landing in `main` happen behind commits signed by the original authors. Deferred items (#1, #3, #4) should be resolved in a short methodology write-up before the paper eval is considered final.
