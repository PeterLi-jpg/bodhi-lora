# Generated numbers

`python -m bodhi_lora report` writes every file in this folder; `make numbers` (or
`make report`) runs it. Do not edit the files by hand. Two runs on the same inputs give
the same bytes (`tests/test_report.py` checks that the committed files are what the
report writes).

## Files

- `numbers.tex` holds one `\newcommand` per number. Each macro name starts with
  `\stat`, and a comment on each line names the statistic it comes from.
- `numbers.json` is the record of the report: the cells it read, the sha256 of each
  input, the tables and the macro count.
- `tab_*.tex` are complete `tabular` blocks. The paper keeps its own `table` float,
  caption and label, and `\input`s the block.

## Inputs

The report reads only what `make analyze` writes from the tracked response tables, so
it needs nothing outside git:

| Input | Used for |
|---|---|
| `results/<cell>/analysis/stats.json`, every analyzed cell | `\stat<token>*`, `tab_cell_<token>`, and the detail tables of the cells in `detail_cells` |
| `results/cross_cell/stats.json` | `\statGroup*`, the grader comparison (`\statPilotGrader*`) and the posted values the paper cites (`\statPosted*`, from `rebuttal/posted_values.json` through the posted-value check) |
| `configs/report.yaml` | which cells, groups, counts and tables; display names |

The analysis itself reads each cell's `seed_*/table.csv`, `analysis/prompts.csv`, its
filter and train records, and `validation/kappa_results.json` where a cell has one
(docs/ARCHITECTURE.md section 7).

## Statistical unit

The seed. Each test and interval is over the seed means, paired across conditions, with
a t distribution on n_seeds - 1 degrees of freedom. In the submitted cell each seed draws
its own 200 eval prompts; in the added cells every seed evaluates one fixed set of 200
prompts, so there the seed varies only training, and the prompt-clustered bootstrap
interval (`...BootCI`) is the one that covers prompt sampling. Pooled numbers (Cohen's d,
the length and theme tables) are descriptive, with intervals from a bootstrap that
resamples prompt ids (docs/ANALYSIS_PLAN.md section 2.1).

## Tables

| Table | What |
|---|---|
| `tab_cross_cell` | Table 1: active inquiry and four LoRA - Base contrasts in every cell of `report.cross_cells` (`cross_tables` in `configs/report.yaml`) |
| `tab_cross_cell_checks` | the post hoc checks of every cell (Table H.6) |
| `tab_cell_filter` | the filter and training record of each trained cell |
| `tab_cell_detail_<n>` | every cell's metrics in the four conditions, with two contrasts, four cells to a table |
| `tab_cell_<token>` | one per analyzed cell: every condition, the two adapter contrasts, labels, leakage, length |
| `tab_<kind>_<token>` | the detail tables of a cell in `detail_cells`: `conditions`, `per_seed`, `dimensions`, `contrasts`, `counts`, `worst_case`, `disclaimers`, `bootstrap`, `axes`, `length_quartiles`, `themes`, `gap`, `tau`, `kappa_<grader>` |

A dagger marks a cell whose base model shares the evaluator's family (decision D6), a
double dagger a cell set aside, with the reason in its record. "--" marks a value the
cell does not have.

## Macro names

Macro names start with the cell record's `report.token`, or with `Group` and the group's
token, then the statistic:

| Pattern | Value |
|---|---|
| `\stat<T>Seeds`, `SeedsGraded`, `Prompts`, `PromptsRepeated`, `PromptDraws`, `Completions<C>`, `CompletionDropPct<C>`, `Scored<C>` | counts |
| `\stat<T><Metric><C>`, with `SeedMin`, `SeedMax` | mean of seed means of a metric in condition C, and its lowest and highest seed |
| `\stat<T><Metric><A>vs<B>Diff`, `CI`, `P`, `PHolm`, `PHolmAll`, `D`, `DCI`, `BootCI`, `WorstCaseDiff`, `WorstCaseP`, `Seeds` | A minus B over seeds: mean difference, 95% t interval, paired t p, Holm over the seven dimensions within the contrast and over every contrast, pooled d and its bootstrap interval, the bootstrap interval of the pooled difference, the worst case for failed grades, and the seed count when a seed is missing |
| `\stat<T>DimsUp<A>vs<B>`, `DimsDown<A>vs<B>` | dimensions that rise or fall with Holm p below `alpha` |
| `\stat<T><Response><Label>PosRate<C>`, `NegRate<C>`, `Index<C>`, each with `CI` | rate of a 0/1 response on the label's positive and negative prompts, and their difference |
| `\stat<T>Leak<Protocol><Rule><C>`, with `N`, `NClean`, `RedFlagLeaked`, `RedFlagClean`, `CleanRedFlagQ<k>` | format leakage of a leak rule, and red flag by length quartile among clean replies |
| `\stat<T>BlanketDisclaimerCount<C>`, `Pct<C>`, `FisherP<C>` | blanket disclaimers and the Fisher test against Base |
| `\stat<T>AskersGap<C>`, `CI` | aggregate score of replies that ask minus replies that do not |
| `\stat<T>TokensEngine<C>`, `Tokens<A>vs<B>Pct`, `Chars<A>vs<B>Pct` | engine token counts (one-pass conditions); length differences in percent |
| `\stat<T>Filter*`, `Steps`, `Kappa*`, `Axis*`, `ShiftCorr*`, `LengthQuartileBound*`, `Label<L>Pop*` | the filter record, the adapter steps, physician agreement, rubric axes, shift correlations, length quartiles, population label counts |
| `\statGroup<G>Cells`, `\statGroup<G><Metric><A>vs<B>Up`, `Down`, `Measured`, `AbsMin`, `AbsMax`, `PMax`, `PHolmMax` | counts over a group of `configs/report.yaml` |
| `\statGroup<G><Count>`, with `Models` and `Benchmarks` | the `counts` and `compares` of `configs/report.yaml` |
| `\statGroup<G>Recipe*` | the adapter recipe and tau that every cell of a group shares |

`<T>` is the cell token (for example `HBMistralJul`), `<C>` a condition (`Base`,
`BaseBodhi`, `LoRA`, `LoRABodhi`), `<Metric>` a metric key in CamelCase
(`ActiveInquiry`, `RubricScore`, `RedFlagRate`), `<Label>` a label id in CamelCase. A
value the analysis lacks has no macro, so a paper that uses one fails
`scripts/check_paper_macros.py` (run by `make paper`) instead of printing a wrong number.
No macro comes from a keyword rule (docs/CONVENTIONS.md rule 15).

Token counts come from the model's own tokenizer, read from the local Hugging Face
cache at the pinned revision. Only the Mistral-Small-24B tokenizer is in that cache
now, so the MedGemma-27B, Med42-8B and BioMistral-7B cells have no tokenizer count; their
one-pass conditions have the engine's count. The two-pass conditions stored only their
final reply, so their token counts leave out the first pass.
