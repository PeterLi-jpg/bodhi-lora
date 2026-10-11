# scripts/

The pipeline, the uniform analysis and the report live in the `bodhi_lora` package
(`python -m bodhi_lora run`, `analyze` and `report`; `make numbers` is the report). The
scripts here are the tools that are not pipeline steps. Each one has a caller:

| Script | Run by | What it does |
|---|---|---|
| `check_doc_paths.py` | `make check`, CI, `hooks/pre-commit` | every repo path the docs and the paper name exists |
| `check_no_secrets.sh` | `make check`, CI, `hooks/pre-commit` | no token or key in a tracked file |
| `check_paper_macros.py` | `make paper`, CI | the paper uses only defined macros |
| `build_paper_figures.py` | `make figures` | every figure of the paper, with the two plot scripts below; Figure 1 from `paper/figures/pipeline_standalone.tex` |
| `plot_postprocessing.py` | `build_paper_figures.py` | the difficulty, axis and score-distribution figures, from the detail cell's response tables (`seed_*/table.csv`), and the forest plot from its `stats.json` |
| `plot_epistemic_figures.py` | `build_paper_figures.py` | Figure E.1 and the per-seed figure, from the detail cell's `analysis/stats.json` (configs/report.yaml `detail_cells`) |
| `compute_irr_kappa.py` | `make kappa` | physician agreement, from rater files and an answer key kept outside the repo |
| `make_irr_excel.py`, `make_tiebreaker_excel.py` | by hand (docs/BLOCKED.md item 6) | the physician grading sheets |
| `export_derived_inputs.py` | `make derived` | the tracked score copies, from the large ignored files they copy |
| `export_anonymous.sh` | by hand (convention 10) | the anonymous reviewer copy |
| `migrate_results.py` | ran once, 2026-10-09 | the move into one results root; every migrated manifest names it (results/PROVENANCE.md) |
| `maintenance/` | by hand (docs/BLOCKED.md item 4) | branch pruning, the history purge, and `remap_commit_ids.py`, which rewrites the commit ids that tracked files cite after the purge; goes once the purge is done |

## What went

Git history keeps every removed script.

- 2026-10-08: the shell launchers that no reported number came from (`df10aa8:gpu/`, `df10aa8:slurm/`,
  `df10aa8:run_all.sh`, `df10aa8:smoke.sh`, `run_multi_seed.sh`, `launch_gpu.sh`, `run_ablation_seeds.sh`,
  `regrade_empty_seeds.sh`) and the manual tools `analyze_traces.py`, `chat.py`,
  `compare_eval_runs.py`, `eval_high_stakes.py`, `preflight.py`,
  `prepare_clinician_grading.py` and `regrade_with_secondary.py`. `draw_pipeline_figure.py`
  went the same day: it drew an old Figure 1 that said training excluded only a
  200-prompt holdout. Training excludes all 1,000 HealthBench-Hard prompts, and Figure 1 is
  built from `paper/figures/pipeline_standalone.tex`.
- 2026-10-09: the TPU-era weight prefetch tool moved to `legacy/scripts/`.
- 2026-10-10: the hedge lexicon `_hedging_metric.py` and `analyze_hedge_rate.py`, with
  their tests. They decided whether a reply hedges by matching a word list, which misses
  negation, quotes and context (docs/CONVENTIONS.md rule 15). No paper or reply number came
  from them. The evaluator's hedging-quality grade is the measure.
- 2026-10-11: the scripts that no make target, command or live script called, with the
  tests that alone kept them alive. `aggregate_seeds.py` and `eval_ushape.py` summarized
  seeds for the 2026-07 launcher (`20f1ca1:scripts/aggregate_seeds.py`); the uniform
  analysis replaces them. `grader_correlation.py`, `plot_ushape.py` (`plot_postprocessing.py`
  draws the difficulty figure itself), `plot_training.py` (no figure uses it),
  `rubric_diff.py` and the shared-host GPU lease tool `gpu_lease.sh` had no caller.
- 2026-10-11, later: `aggregate_cross_seed.py` and its test. Figure E.1 now reads the
  detail cell's `stats.json`, the numbers the tables print, so no target needs the
  cross-seed summary (`92b2e76:scripts/aggregate_cross_seed.py` is the last version; its
  output `analysis/cross_seed_aggregate.*` stays as a moved record, results/PROVENANCE.md).
