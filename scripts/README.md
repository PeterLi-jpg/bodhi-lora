# scripts/

The pipeline lives in the `bodhi_lora` package (`python -m bodhi_lora`). The scripts here
are repository tools and the analysis scripts that `make` runs until the analysis lane
moves them into the package. The TPU-era weight prefetch tool moved to `legacy/scripts/`
on 2026-10-09.

On 2026-10-08 the shell launchers that no reported number came from (`gpu/`, `slurm/`,
`run_all.sh`, `smoke.sh`, `run_multi_seed.sh`, `launch_gpu.sh`, `run_ablation_seeds.sh`,
`regrade_empty_seeds.sh`) and the manual tools `analyze_traces.py`, `chat.py`,
`compare_eval_runs.py`, `eval_high_stakes.py`, `preflight.py`,
`prepare_clinician_grading.py` and `regrade_with_secondary.py` were removed. Git history
keeps them. `draw_pipeline_figure.py` was removed the same day. It drew an old version of Figure 1
that said training excluded only a 200-prompt holdout. Training excludes all 1,000
HealthBench-Hard prompts, and Figure 1 is built from `paper/figures/pipeline_standalone.tex`.
