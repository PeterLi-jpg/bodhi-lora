# BODHI-LoRA

[![CI](https://github.com/PeterLi-jpg/bohdi-lora/actions/workflows/ci.yml/badge.svg)](https://github.com/PeterLi-jpg/bohdi-lora/actions/workflows/ci.yml)

Code, configs and results for the paper **"Ask Before You Answer: Distilling Clinical
Epistemic Behaviors into Model Weights"**.

The target setting is patient-facing, open-ended clinical question answering. The
method has four steps:

1. Run the BODHI two-pass chain-of-thought prompt on a base model to make
   demonstrations. BODHI stands for Balanced, Open-minded, Diagnostic, Humble, and
   Inquisitive (Arslan et al., BMJ Health & Care Informatics 33(1):e101877, 2026).
2. Grade each demonstration against its rubric with Qwen2.5-14B-Instruct.
3. Keep the demonstrations that score at or above the threshold tau.
4. Train a LoRA adapter on the kept demonstrations.

The evaluation is a 2x2 design: Base, Base with the BODHI prompt, LoRA, and LoRA with
the BODHI prompt. Each condition gets a rubric score (the paper's aggregate score) and
scores on seven epistemic behaviors: uncertainty acknowledgment, hedging quality,
specificity, active inquiry, context-seeking, red-flag identification and scope
bounding. Llama-3.1-8B-Instruct grades both.

The repo slug `bohdi-lora` misspells the name. The project name is BODHI everywhere
else. Renaming the remote repo is deferred (docs/DECISIONS.md, D15).

## Read this first

- **The epistemic-behavior scores are LLM-graded.** The physician check grades the
  HealthBench rubric criteria behind the rubric score, not the seven behaviors. One of
  three physician raters returned grades, and agreement with the LLM grader is below the
  target of 0.6 set in advance. The paper does not claim physician validation (D5). The
  other two raters are deferred (D20).
- **The adapter has a cost.** We report the seed-level paired difference LoRA minus
  Base, with a t(4) 95% interval, for every cell, computed from the raw outputs. The
  rubric score falls in all six added cells that are not set aside, and each drop
  survives Holm correction over those six. It rises in the seventh, MedQA, but that run
  used a prompt that told the model to ask (Base asked in nearly every response), so it
  measures instruction following and is set aside. We make no equivalence or
  non-inferiority claim (D2).
- **Med42-8B cells carry a dagger.** Med42-8B is a Llama-3 model, the same family as the
  evaluator. That breaks the paper's grader-separation rule, so these cells are not
  used in any cross-model statement (D6).
- **Calibration is not measured.** The word appears only as motivation (D3).
- **New runs wait.** The planned experiments ([docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)),
  the dated analysis plan that fixes how they are read
  ([docs/ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md)) and the ordered GPU runs with their
  cost ([docs/RUN_PLAN.md](docs/RUN_PLAN.md)) are written down. None has run: no GPU
  use until the first author says so (D21).

The open problems are in [KNOWN_ISSUES.md](KNOWN_ISSUES.md). The decisions behind these
points are in [docs/DECISIONS.md](docs/DECISIONS.md). What still needs the first author
is in [BLOCKED.md](BLOCKED.md).

## Layout

```
bodhi_lora/         the package: registry, engine, data, protocols, grading, pipeline, analysis, report
configs/            the registry: one YAML record per model, benchmark, protocol and cell
data/raw/           benchmark files, downloaded or built here (gitignored)
docs/               architecture, experiments, analysis plan, run plan, conventions, decisions, claims
hooks/              commit-msg and pre-push hooks (activate once, see below)
legacy/             frozen launchers of the 2026-05 MedGemma-27B cell (TPU training, Modal eval)
paper/              the paper source (paper/main.tex), paper/generated/ and the frozen submission
rebuttal/           the review record: posted replies, withdrawals, commitments, posted values
results/            one folder per cell, and cross_cell/ (results/PROVENANCE.md)
scripts/            repository tools: doc and secrets checks, anonymous export, physician sheets
tests/              pytest suite, no GPU needed
```

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains the design: the terms (cell,
protocol, condition, evaluator, judge), the records, the package, the results root and
the uniform analysis.

## Cells

A cell is one benchmark, one base model and one teaching protocol, run over a list of
seeds. Its id is `<benchmark>__<model>__<protocol>`, with a tag such as `__2026-07` for a
run made under earlier settings. `python -m bodhi_lora list cells` prints every cell with
its status and marks.

| Set | Cells | How it ran | Outputs |
|---|---|---|---|
| 2026-05 (submitted) | MedGemma-27B on HealthBench-Hard | tunix LoRA on 8x TPU v6e; eval on one A100 (Modal). Launchers in `legacy/`; the adapters are not retained (D19) | `results/healthbench__medgemma-27b-text__bodhi__2026-05/` |
| 2026-07 (added in the discussion) | Mistral-Small-24B, Med42-8B†, BioMistral-7B on HealthBench-Hard; Mistral-Small-24B and Med42-8B† on ChatDoctor; Mistral-Small-24B on MedQuAD; Mistral-Small-24B on MedQA (set aside) | the 2026-07 launcher (`61dada9:rebuttal/launch/run_cell.sh`) on H100 | the `results/*__2026-07/` cells (raw eval JSON gitignored) |
| Planned | the cells of docs/EXPERIMENTS.md | `python -m bodhi_lora run` | none yet; docs/RUN_PLAN.md orders them |

Records of past runs are frozen. The runner refuses to run them again, except to add
judge re-grades, so new outputs never mix with old ones; a new run of the same design is
a new cell. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) gives each cell's settings and how
the 2026-05 and 2026-07 settings differ.

## Reproduce the paper

Every number in the paper comes from code, and none is typed by hand
(docs/CONVENTIONS.md, rule 2). The same analysis code runs on every cell
(docs/CONVENTIONS.md, rule 14; docs/ARCHITECTURE.md section 7).

```bash
python -m bodhi_lora fetch healthbench   # the public HealthBench files, checked against their pins
make test        # pytest suite
make numbers     # paper/generated/: numbers.tex, numbers.json, tables
make figures     # paper/figures/
make paper       # make numbers and make figures, the macro check, then the LaTeX build
make analyze     # every statistic of every cell that has tracked response tables
make extract     # the response tables from the raw outputs (needs the gitignored files), then the statistics
```

| Output | Made from | Notes |
|---|---|---|
| `results/<cell_id>/seed_<n>/table.csv` | the raw eval and grade files of the seed (`make extract`) | one row per condition and prompt; the analysis reads nothing else |
| `results/<cell_id>/analysis/stats.json` | the cell's response tables (`make analyze`) | the seed is the unit: paired t over seeds, Holm over the seven behaviors within each contrast; prompt-level d is descriptive (D7) |
| `results/cross_cell/stats.json` | every cell's statistics | the cross-cell families of `configs/report.yaml` |
| `paper/generated/` | today `make numbers` (`scripts/paper_stats.py`); the two files above once `python -m bodhi_lora report` is built | `scripts/check_paper_macros.py` checks that the paper uses only defined macros |
| `results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/kappa_results.json` | `make kappa` on the returned rater file, kept outside the repo | per-row grades are not committed (KNOWN_ISSUES) |

`make analyze` exists and reproduces the 2026-05 cell's values
(`tests/test_analysis_equivalence.py`); only that cell has tracked tables so far. The
paper's numbers still come from `make numbers`, `make figures` and `make cells`, the
older scripts, until `python -m bodhi_lora report` is built (docs/ARCHITECTURE.md
section 5).

## Run the pipeline

Every setting is a record in `configs/` (docs/ARCHITECTURE.md section 3). The command
line runs a cell:

```bash
python -m bodhi_lora list cells                 # every cell, its status and marks
python -m bodhi_lora list cells --settings      # what differs between the cells
python -m bodhi_lora fetch healthbench          # download a benchmark, checked against its pins
python -m bodhi_lora run <cell_id> --dry-run    # print the GPU commands, run the rest on the CPU stub
python -m bodhi_lora run <cell_id> --gpus 0,1   # every stage, one seed per GPU
python -m bodhi_lora analyze <cell_id>          # the cell's statistics, from its tables
python -m bodhi_lora analyze <cell_id> --extract  # first the tables, from the raw outputs
make smoke                                      # the smoke cell on the CPU, training included
```

Setup, once per machine: `bash setup.sh` builds the training environment (`.venv`,
`requirements.txt`) and the vLLM environment (`.venv-infer`, `requirements-infer.txt`),
and `export HF_TOKEN=...` gives access to the gated models (MedGemma, Llama-3.1, Gemma).

A run writes `results/<cell_id>/`, with a manifest in every stage folder
(`bodhi_lora/manifest.py`): resolved settings, git commit and dirty flag, library
versions, model revisions, adapter hash, time, GPUs and token counts. GPU runs need the
first author's go-ahead (D21); [docs/RUN_PLAN.md](docs/RUN_PLAN.md) lists them.

## Where results live

Every result is in `results/`, one folder per cell, named by its cell id, with one folder
per seed and one per stage. [results/PROVENANCE.md](results/PROVENANCE.md) maps each old
location to its new one.

- Git tracks, in every cell: the manifests, `cell.yaml`, the eval id lists, the adapter
  configs and trainer states, the response tables and the analysis and validation
  folders. Git ignores traces, raw eval and grade JSON, adapter weights and logs. The
  rule depends on the kind of file, never on the cell.
- `results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/`: the blank physician
  grading sheets, the instructions and the answer key. Returned sheets are not in the
  repo.
- [RESULTS.md](RESULTS.md): what each result is, where it lives and its status.

## Data and leakage

`python -m bodhi_lora fetch <benchmark>` downloads or builds a benchmark into `data/raw/`,
checked against the pins of its record. HealthBench-Hard (1,000 prompts) is a subset of
HealthBench (5,000). No Hard prompt is ever trained on. The 2026-05 and 2026-07 cells
generated traces from the other 4,000 prompts; the planned cells also hold out the 679
prompts that carry the physician emergency or clinician-user labels, so those stay unseen
for the labelled probes (3,321 remain). The runner runs `bodhi_lora/data/leakage.py` on
every cell before training: no graded trace may be an eval or probe prompt, by id or by
text. On 2026-10-07 the seed-7 training set of the 2026-05 cell had 0 Hard prompts among
its 2,816 prompts and 0 overlap with the 667 eval prompts.

## Contributing

Activate the hooks once per clone. They block commit messages with tool trailers and
pushes of scratch branches:

```bash
git config core.hooksPath hooks
```

Before a pull request, run `make check` (doc paths, secrets scan, registry) and
`make test`. A new model, benchmark, protocol or cell is a new record in `configs/`, not
new code (docs/CONVENTIONS.md, rules 11 to 14). Reviewers get an anonymized copy made by
`scripts/export_anonymous.sh`, never this repo. See
[contributions/contributing.md](contributions/contributing.md) and
[docs/CONVENTIONS.md](docs/CONVENTIONS.md).

## License

MIT, see [LICENSE](LICENSE). Third-party code, data and models keep their own licenses;
see [NOTICE](NOTICE) and [docs/REUSE.md](docs/REUSE.md).

## Citation

There is no citation entry yet. One will follow publication.
