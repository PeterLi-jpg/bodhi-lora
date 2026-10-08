# scripts/

Most scripts here run from a `make` target, a test or a launcher in
`rebuttal/launch/`. The tools below run from none of them. They are kept for manual use,
and no number in the paper comes from them. Check one before you use it: some describe
an earlier state of the project.

| Script | What it does | Status |
|---|---|---|
| `analyze_traces.py` | Summarizes a graded trace file: response lengths, theme counts and grader retry disagreement, with plots | Manual; reads the large ignored trace files |
| `compare_eval_runs.py` | Prints a table that compares two eval folders | Manual |
| `eval_high_stakes.py` | Runs the four conditions on high-stakes prompts (Appendix C harm review) | Scaffold. `data/raw/high_stakes_prompts.jsonl` holds placeholder prompts, not the physician-curated set the review needs |
| `prefetch_gemma_safetensors.py` | Pre-downloads MedGemma weights to a TPU host's cache | TPU path of the submitted cell; the TPU launcher is in `legacy/` |
| `prepare_clinician_grading.py` | Expands the clinician sheet into one row per rubric criterion (issue #73) | Superseded by `make_irr_excel.py`, which wrote the committed sheets |
| `regrade_empty_seeds.sh` | Grades again the seeds whose epistemic score file came out empty | One-off repair; the committed score files are complete |
| `regrade_with_secondary.py` | Grades the submitted responses again with a third-family grader | Superseded by `regrade_cross_family.py`, the regrade for the Med42-8B cells (D6) |
| `run_ablation_seeds.sh` | Ablation over BODHI components (issue #69) | Not run; the paper reports no ablation |

`draw_pipeline_figure.py` was removed on 2026-10-08. It drew an old version of Figure 1
that said training excluded only a 200-prompt holdout. Training excludes all 1,000
HealthBench-Hard prompts, and Figure 1 is built from `paper/figures/pipeline_standalone.tex`.
