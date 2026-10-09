# Generated numbers

`scripts/paper_stats.py` writes every file in this folder. Do not edit them by hand.
To rebuild them, run `make numbers` from the repo root. Two runs on the same inputs
give the same bytes.

## Files

- `numbers.json` holds every statistic at full precision (6 significant digits), plus
  the sha256 of each input under `meta.inputs`.
- `numbers.tex` holds one `\newcommand` per number. Each macro name starts with
  `\stat`, and a comment on each line names its source.
- `tab_*.tex` are complete `tabular` blocks. The paper keeps its own `table` float,
  caption and label, and `\input`s the block.

To use them in `paper/main.tex`, put `\input{generated/numbers}` in the preamble.
Then replace the body of a table with `\input{generated/tab_<name>}`.

## Inputs

`<M>` is `results/healthbench__medgemma-27b-text__bodhi__2026-05`, the submitted cell.

| Input | Tracked | Used for |
|---|---|---|
| `<M>/seed_*/eval/`, `<M>/seed_*/grade/` | yes | every number about the submitted cell |
| `<M>/validation/kappa_results.json` | yes | `tab_kappa`, `\statKappa*` |
| `<M>/analysis/{interference_mechanism,mimicry_split,second_person_questions}.json` | yes | leakage, discrimination, odds ratio and second-person macros |
| `<M>/analysis/stats.json` | yes (`make analyze`) | `\statTokensEngine*` |
| `<M>/seed_7/filter/filter_scores.json` | yes | `tab_tau_calibration`, `\statFilter*`, `\statTrainHardOverlapSeedSeven` |
| `results/cross_cell/generated_values.json` | yes (`make cells`) | `tab_cross_cell`, `tab_cell_detail_*`, `\statCell*`, `\statCells*` |
| `results/cross_cell/cell_recipe.json` | yes (`make cells`) | `tab_cell_filter`, `\statCellsLoRA*`, `\statRun*` |
| `rebuttal/posted_values.json` | yes | `\statPosted*`, and the fallback rows of the older cross-cell tables |
| `results/healthbench__medgemma-27b-text__bodhi__2026-04/analysis/qwen_grader_scores.json` | yes | `\statCrossGrader*` (the seed-42 run graded by Qwen2.5-14B) |
| `data/raw/healthbench/hard.jsonl` | no (public; `python -m bodhi_lora fetch healthbench`) | `tab_theme`, `\statDifficulty*`, `\statHardPrompts` |

The comments in `numbers.tex` and the keys of `numbers.json` under `meta.inputs` still
name most inputs by the paths they had when the numbers were first written
(`results_modal/`, `data/rebuttal_cells/`, `rebuttal/results/`, `data/derived/`).
`results/provenance.json` maps each old path to the file's place now.

The filter-score and grader-score files copy the scores, with prompt ids and no
response text, from two large ignored files. `make derived` writes them again from
those files, and each records the sha256 of its source.

`make numbers` stops with a message when HealthBench-Hard is missing. Run
`bash setup.sh` first, or pass `make numbers HB_HARD=<path>`. If you call the script
directly without an input, it prints `SKIP` and leaves out that table and its macros.
`make paper` then runs `scripts/check_paper_macros.py`, which names every macro the
paper uses that `numbers.tex` lacks.

## Statistical unit

In the submitted cell, a seed trains one adapter and draws its own 200 eval prompts.
All four conditions in a seed share that draw. In the added cells, every seed evaluates
one fixed set of 200 prompts. In every cell, each test and CI is over the 5 seed means,
paired across conditions, with a t distribution on 4 degrees of freedom. Pooled numbers (Cohen's
d, the length and theme tables) are descriptive. Their CIs come from a bootstrap
that resamples prompt ids. The docstring of `scripts/paper_stats.py` gives the
details.

## Cross-cell tables

`tab_cross_cell` (Table 1 of the paper) and `tab_cell_detail_<n>` (the per-cell detail
in the appendix) give every cell the same statistics. Their input is
`results/cross_cell/generated_values.json`, which `make cells` writes from the raw
outputs of each cell; it holds the per-seed means of every cell, the submitted one too.
One function (`uniform_section`) computes, for each cell, the mean of the seed means of
each condition and the paper's five contrasts. Each contrast gets the seed-level paired
t of the uniform analysis (`bodhi_lora/analysis/stats.py`, `paired_test`) and Holm over
the seven dimensions within the contrast (`holm`). The aggregate score is one secondary
test per contrast, outside the Holm family. A seed with no value in a condition leaves
that contrast, and `n_seeds` says so: seed 42 of the MedQA cell has no dimension grades.

The script stops when the submitted cell's per-seed means in `generated_values.json`
differ from the ones it computes from that cell's raw outputs.

A dagger marks a cell whose base model shares the evaluator's Llama-3 family (decision
D6). The counts (`\statCellsOutside*`, `\statCells*`) leave those cells and the
set-aside MedQA cell out, and include the submitted cell.

`tab_generality`, `tab_generality_aggregate` and `tab_cell_dimensions` are the older
cross-cell tables, kept until the paper stops using them. Their rows come from the same
file, with the submitted cell's first row from its raw outputs. Without
`generated_values.json` they fall back to the posted values of
`rebuttal/posted_values.json`, marked with an asterisk.

The script checks that the posted values of the submitted cell match its raw outputs
and stores the result in `numbers.json` under `generality.main_cell_check`.

## Token counts

`\statTokensEngine<condition>` come from the `cost` section of the submitted cell's
`analysis/stats.json`, which `make analyze` writes. The serving engine stored a token
count for the one-pass conditions only, so the two-pass conditions have no macro.
