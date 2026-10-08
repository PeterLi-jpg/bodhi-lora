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

| Input | Tracked | Used for |
|---|---|---|
| `results_modal/seed_*/` | yes | every number about the submitted MedGemma cell |
| `results_modal/irr/kappa_results.json` | yes | `tab_kappa`, `\statKappa*` |
| `rebuttal/results/*.json` | yes | leakage, discrimination, odds ratio and second-person macros |
| `data/rebuttal_cells/posted_values.json` | yes | `tab_generality`, `tab_generality_aggregate`, `\statCell*` |
| `data/rebuttal_cells/generated_values.json` | after `make cells` | replaces the posted values of each synced cell |
| `data/derived/seed_7_filter_scores.json` | yes | `tab_tau_calibration`, `\statFilter*`, `\statTrainHardOverlapSeedSeven` |
| `data/derived/seed_42_qwen_grader_scores.json` | yes | `\statCrossGrader*` (the seed-42 run graded by Qwen2.5-14B) |
| `data/raw/healthbench_hard.jsonl` | no (public; `setup.sh` downloads it) | `tab_theme`, `\statDifficulty*`, `\statHardPrompts` |

The two `data/derived/` files copy the scores, with prompt ids and no response text,
from two large ignored files: `data/sft_qwen_graded/seed_7_train.jsonl` (49 MB) and
`results/seed_42/eval/<condition>.json`. `make derived` writes them again from those
files, and each records the sha256 of its source.

`make numbers` stops with a message when HealthBench-Hard is missing. Run
`bash setup.sh` first, or pass `make numbers HB_HARD=<path>`. If you call the script
directly without an input, it prints `SKIP` and leaves out that table and its macros.
`make paper` then runs `scripts/check_paper_macros.py`, which names every macro the
paper uses that `numbers.tex` lacks.

## Statistical unit

A seed trains one adapter and draws its own 200 eval prompts. All four conditions in
a seed share that draw. So each test and CI is over the 5 seed means, paired across
conditions, with a t distribution on 4 degrees of freedom. Pooled numbers (Cohen's
d, the length and theme tables) are descriptive. Their CIs come from a bootstrap
that resamples prompt ids. The docstring of `scripts/paper_stats.py` gives the
details.

## Cross-cell tables

`tab_generality` and `tab_generality_aggregate` take the first row (the submitted
cell) from `results_modal/`. Each other row comes from
`data/rebuttal_cells/generated_values.json` when `make cells` has written that cell
from the synced raw outputs. Otherwise the row holds the posted values, transcribed
from the OpenReview replies into `data/rebuttal_cells/posted_values.json`, and carries
an asterisk. The raw outputs of the added cells are not in the repo yet (audit PROV-1).

A synced row gets a paired t interval on LoRA minus Base for the aggregate score. A
posted row has means only, so its interval reads "pending". A dagger marks a cell whose
base model shares the evaluator's Llama-3 family (decision D6). The `\statCells*`
counts leave those cells and the set-aside MedQA cell out, and include the submitted
cell.

The script checks that the posted values of the submitted cell match `results_modal/`
and stores the result in `numbers.json` under `generality.main_cell_check`.
