# scripts/

The pipeline and the uniform analysis live in the `bodhi_lora` package
(`python -m bodhi_lora run` and `analyze`). The scripts here are repository tools and the
paper scripts that `make numbers`, `make figures` and `make cells` run until
`python -m bodhi_lora report` reproduces their outputs. The TPU-era weight prefetch tool moved to `legacy/scripts/`
on 2026-10-09.

On 2026-10-08 the shell launchers that no reported number came from (`gpu/`, `slurm/`,
`run_all.sh`, `smoke.sh`, `run_multi_seed.sh`, `launch_gpu.sh`, `run_ablation_seeds.sh`,
`regrade_empty_seeds.sh`) and the manual tools `analyze_traces.py`, `chat.py`,
`compare_eval_runs.py`, `eval_high_stakes.py`, `preflight.py`,
`prepare_clinician_grading.py` and `regrade_with_secondary.py` were removed. Git history
keeps them. `draw_pipeline_figure.py` was removed the same day. It drew an old version of Figure 1
that said training excluded only a 200-prompt holdout. Training excludes all 1,000
HealthBench-Hard prompts, and Figure 1 is built from `paper/figures/pipeline_standalone.tex`.

On 2026-10-10 the hedge lexicon `_hedging_metric.py` and `analyze_hedge_rate.py`, which
used it, were removed with their tests. They decided whether a reply hedges by matching a
word list ("might", "see a doctor" and others), which misses negation, quotes and context
(docs/CONVENTIONS.md rule 15). Nothing called them, and no paper or reply number came from
them; the only output was the 2026-04 pilot's `hedge_rate.json`, archived in the move to
one results root (results/PROVENANCE.md). The evaluator's hedging-quality grade is the
measure. Git history keeps them.
