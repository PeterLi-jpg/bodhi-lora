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

## Documented, deferred to discussion

### [#1] Brier / ECE do not measure model calibration — PARTIALLY ADDRESSED
The old implementation used the evaluator's rubric score as "confidence" and compared it against rubric outcomes from the same grading pass. That was grader-internal consistency, not model calibration. **Action taken**: eval output now includes `brier_model_calibration` / `ece_model_calibration`, using the geometric mean next-token probability of the emitted response as a model-derived confidence proxy, while keeping the legacy `brier_grader_consistency` / `ece_grader_consistency` fields for backward comparison. **Still open for discussion**: whether the response-level logprob proxy is strong enough for the paper, or whether the final claim should move to a richer confidence protocol (verbalized confidence, abstention head, or similar).

### [#3] Same grader for filtering and final evaluation — OPEN
`filter_traces.py` and `eval_healthbench.py` default to the same grader family (`Qwen/Qwen2.5-14B-Instruct-AWQ`). This couples training-data selection to the evaluator used for claimed gains. **Mitigation paths**:
- Use a distinct grader for final eval (simplest — change `--grader-model` in `slurm/eval_lora.sh`)
- Report final results under a second independent grader and compare
- Keep filter grader ≠ eval grader as policy

Infrastructure now exists for an optional second grader pass via `SECOND_GRADER_MODEL=... sbatch slurm/eval_lora.sh`, plus `scripts/grader_correlation.py` to report Spearman correlation and large per-example disagreements. The paper claim remains open until a second grader is actually chosen and run.

Needs team agreement before any ablation is blessed as final.

### [#4] Inconsistent filtering-score normalization — OPEN
Score formula is `sum_of_met_points / sum_of_positive_points`. Negative rubric items contribute to the numerator as penalties but not to the denominator, so a fixed `--min-score 0.4` threshold is not comparable across prompts with different penalty structure. Data selection is therefore partially a function of rubric geometry, not just response quality.

`filter_traces.py` now exposes alternate score views (`absolute_point_score`, `positive_criteria_rate`) plus `--score-field` so this can be audited without silently changing the historical default. The main pipeline still defaults to `overall_score`, so the methodological choice is still unresolved.

**Options to consider**:
- Normalize by `sum(|points|)` so the score lives on a symmetric scale
- Apply per-prompt z-scoring before the filter
- Require a minimum fraction of positive items met, independent of points

Needs a call on which normalization to use, and a re-derivation of the filter threshold.

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
