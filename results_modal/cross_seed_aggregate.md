# Cross-seed aggregate (5 seeds: 7, 13, 42, 99, 101)

Mean ± std across 5 random seeds. Each seed evaluates its own deterministic 200-prompt subset of HealthBench Hard.

## Stage 4 — HealthBench Hard score

| Config | Mean | Std (across seeds) | Per-seed mean (7/13/42/99/101) |
|---|---|---|---|
| base_no_wrapper | 0.4591 | 0.0114 | 0.4594 / 0.4533 / 0.4723 / 0.4434 / 0.4673 |
| base_bodhi | 0.4630 | 0.0105 | 0.4791 / 0.4574 / 0.4601 / 0.4665 / 0.4516 |
| lora_no_wrapper | 0.4634 | 0.0096 | 0.4753 / 0.4493 / 0.4681 / 0.4628 / 0.4615 |
| lora_bodhi | 0.4554 | 0.0024 | 0.4555 / 0.4570 / 0.4575 / 0.4555 / 0.4515 |

## Stage 5 — Epistemic dimensions (mean ± std across seeds)

| Config | Uncertainty | Context | Scope | Specificity | Hedging | Q/resp | Active inq % | Red-flag % | Specific % | Scope-bounded % |
|---|---|---|---|---|---|---|---|---|---|---|
| base_no_wrapper | 1.94 ± 0.02 | 1.40 ± 0.03 | 1.91 ± 0.01 | 1.88 ± 0.02 | 1.86 ± 0.02 | 0.45 ± 0.11 | 17.5 ± 3.3 | 72.4 ± 0.8 | 61.2 ± 3.9 | 92.7 ± 1.1 |
| base_bodhi | 1.90 ± 0.02 | 1.77 ± 0.03 | 1.94 ± 0.02 | 1.88 ± 0.02 | 1.85 ± 0.02 | 1.25 ± 0.08 | 47.9 ± 2.9 | 78.8 ± 2.1 | 61.4 ± 4.0 | 95.4 ± 1.5 |
| lora_no_wrapper | 1.94 ± 0.01 | 1.75 ± 0.06 | 1.97 ± 0.01 | 1.92 ± 0.02 | 1.91 ± 0.02 | 1.16 ± 0.14 | 45.6 ± 3.9 | 78.3 ± 2.0 | 61.2 ± 4.3 | 97.6 ± 0.7 |
| lora_bodhi | 1.95 ± 0.02 | 1.90 ± 0.03 | 1.98 ± 0.02 | 1.92 ± 0.03 | 1.91 ± 0.03 | 1.71 ± 0.12 | 80.0 ± 6.7 | 57.7 ± 8.0 | 65.0 ± 2.3 | 98.1 ± 1.5 |

## Sample sizes

| Config | seed_7 n_results | seed_13 | seed_42 | seed_99 | seed_101 |
|---|---|---|---|---|---|
| base_no_wrapper | 200 | 200 | 200 | 200 | 200 |
| base_bodhi | 191 | 194 | 193 | 188 | 191 |
| lora_no_wrapper | 200 | 200 | 200 | 200 | 200 |
| lora_bodhi | 192 | 196 | 189 | 191 | 191 |

(200 = no inference failures. <200 = bodhi 2-pass requests that exceeded the 4096-token context cap; ~4-5%% per bodhi config.)