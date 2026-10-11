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

## Rebuild the paper's numbers on a CPU

No GPU and no token. The tracked results hold everything the paper reads, except the
public HealthBench files, which `fetch` downloads and checks against their pins.

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'
python -m bodhi_lora fetch healthbench
make analyze numbers figures      # then git status: nothing changed
```

`make paper` also builds `paper/build/main.pdf` (it needs `tectonic`). Install the
package editable (`pip install -e`): it finds `configs/` and `results/` next to its
folder. [docs/INSTALL.md](docs/INSTALL.md) covers the training and inference
environments for real runs.

| In the paper | File | Made by | From |
|---|---|---|---|
| Every number in the text (`\stat...` macros) | `paper/generated/numbers.tex` | `make numbers` | each cell's `results/<cell_id>/analysis/stats.json` and `results/cross_cell/stats.json` |
| Table 1, every cell | `paper/generated/tab_cross_cell.tex` | `make numbers` | the same two kinds of file, and the groups of `configs/report.yaml` |
| Appendix tables (B, E, G, H, J) | `paper/generated/tab_*.tex` | `make numbers` | the same |
| Figure 1 | `paper/figures/pipeline.pdf` (tracked) | `make figures` | `paper/figures/pipeline_standalone.tex` |
| Appendix figures (E, G, I) | `paper/figures/*.pdf` | `make figures` | the 2026-05 cell's `stats.json` and raw outputs |
| Each cell's statistics | `results/<cell_id>/analysis/stats.json` | `make analyze` | the cell's response tables, `results/<cell_id>/seed_<n>/table.csv` |
| The response tables | `results/<cell_id>/seed_<n>/table.csv` | `make extract` | the raw eval and grade files (gitignored, except the 2026-05 cell's) |
| Physician agreement (Appendix B) | `results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/kappa_results.json` | `make kappa` | the returned rater file and the answer key, both kept outside the repo |

[paper/NUMBERS.md](paper/NUMBERS.md) maps each number and claim of the paper to its
macro, the function that writes it and its input. Every number comes from code, none is
typed by hand (docs/CONVENTIONS.md, rule 2), and the same analysis runs on every cell
(rule 14). `make analyze` writes each `stats.json` again byte for byte
(`tests/test_analysis_committed.py`), and CI rebuilds `paper/generated/` from a clean
checkout and fails on any difference.

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
  the analysis plan that fixes how they are read, adopted on 2026-10-11
  ([docs/ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md)), and the ordered GPU runs with their
  cost ([docs/RUN_PLAN.md](docs/RUN_PLAN.md)) are written down. None has run: no GPU
  use until the first author says so (D21).

The open problems are in [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md). The decisions
behind these points are in [docs/DECISIONS.md](docs/DECISIONS.md). What still needs the
first author is in [docs/BLOCKED.md](docs/BLOCKED.md).

## Layout

```
bodhi_lora/          the package: registry, engine, data, protocols, grading, pipeline, analysis, report
configs/             the registry: one YAML record per model, benchmark, protocol and cell
data/raw/            benchmark files, downloaded or built here (gitignored)
docs/                install, architecture, experiments, analysis and run plans, conventions,
                     decisions, claims, known issues, what is blocked, reproducibility, audit IDs
hooks/               pre-commit, commit-msg and pre-push hooks (activate once, see below)
legacy/              record: the frozen launchers of the 2026-05 MedGemma-27B cell (TPU training, Modal eval)
paper/               the paper source (paper/main.tex), paper/generated/, and the frozen submission (record)
rebuttal/            record: the posted replies, withdrawals, commitments and posted values
results/             one folder per cell, and cross_cell/; results/README.md says what each holds
scripts/             repository tools: doc and secrets checks, figure scripts, physician sheets, export
tests/               pytest suite, no GPU needed
.github/workflows/   CI (checks, paper numbers and figures, tests) and shellcheck
pyproject.toml       the package and its pins (analysis, train, gpu and dev extras)
requirements-infer.txt   the separate vLLM environment (vLLM 0.9.2 cannot share an env with training)
Makefile, setup.sh   the make targets below, and the one-time setup of a GPU machine
```

`legacy/`, `rebuttal/` and `paper/submitted/` are records: they keep what ran or was
posted, and nothing in the package reads them. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
explains the design: the terms (cell, protocol, condition, evaluator, judge), the
records, the package, the results root and the uniform analysis.

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
judge re-grades, the judge classifier and the contamination probe, so new outputs never mix with old ones; a new run of the same design is
a new cell. [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) gives each cell's settings
and how the 2026-05 and 2026-07 settings differ.

## Run the pipeline

Every setting is a record in `configs/` (docs/ARCHITECTURE.md section 3). The command
line (`python -m bodhi_lora`, or `bodhi-lora` once installed) runs a cell:

```bash
python -m bodhi_lora list cells                 # every cell, its status and marks
python -m bodhi_lora list cells --settings      # what differs between the cells
python -m bodhi_lora fetch healthbench          # download a benchmark, checked against its pins
python -m bodhi_lora run <cell_id> --dry-run    # print the GPU commands, run the rest on the CPU stub
python -m bodhi_lora run <cell_id> --gpus 0,1   # every stage, one seed per GPU
python -m bodhi_lora analyze <cell_id>          # the cell's statistics, from its tables
python -m bodhi_lora analyze <cell_id> --extract  # first the tables, from the raw outputs
make smoke                                      # the smoke cell on the CPU, training included
python -m bodhi_lora.pipeline.eval --cell <cell_id> --seed 42 --condition lora+bodhi   # one stage alone
```

`make smoke` runs every stage of the smoke cell
(`configs/cells/tiny-rubric__tiny-random-llama__bodhi.yaml`) on the CPU in about two
minutes, with the stub engine in place of vLLM and a tiny public model trained for real.
It needs no GPU and no token, and writes to `outputs/smoke/` (ignored by git). Every
stage is also a module with the same command line, as in the last line above. A rerun
skips every stage whose manifest says complete, and the traces and filter stages resume
where they stopped.

Setup, once per GPU machine: `bash setup.sh` builds the training environment (`.venv`,
`pip install -e '.[train,gpu,dev]'`) and the vLLM environment (`.venv-infer`,
`requirements-infer.txt`), and `export HF_TOKEN=...` gives access to the gated models
(MedGemma, Llama-3.1, Gemma).

A run writes `results/<cell_id>/`, with a manifest in every stage folder
(`bodhi_lora/manifest.py`): resolved settings, git commit and dirty flag, library
versions, model revisions, adapter hash, time, GPUs and token counts. GPU runs need the
first author's go-ahead (D21); [docs/RUN_PLAN.md](docs/RUN_PLAN.md) lists them.

## Where results live

Every result is in `results/`, one folder per cell, named by its cell id, with one folder
per seed and one per stage. [results/README.md](results/README.md) says what each result
is and its status; [results/PROVENANCE.md](results/PROVENANCE.md) maps each old location
to its new one.

- Git tracks, in every cell: the manifests, `cell.yaml`, the eval id lists, the adapter
  configs and trainer states, the response tables and the analysis and validation
  folders. Git ignores traces, raw eval and grade JSON, adapter weights and logs. The
  rule depends on the kind of file, with one exception while open decision O2 stands
  (docs/ARCHITECTURE.md section 12): the 2026-05 cell's raw eval and grade JSON (25 files,
  about 80 MB) stay tracked, because `make figures` draws five of the paper's seven
  figures from them.
- `results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/`: the blank physician
  grading sheets and the instructions. The answer key and the returned sheets are not in
  the repo.
- The 2026-05 cell's `analysis/withdrawn/` holds the outputs of three keyword measures
  posted in the discussion and withdrawn since; the paper reads none of them.

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

Activate the hooks once per clone. They run the secrets and doc path checks before each
commit, and block commit messages with tool trailers and pushes of scratch branches:

```bash
git config core.hooksPath hooks
```

Before a pull request, run `make check` (doc paths, secrets scan, registry) and
`make test`. A new model, benchmark, protocol or cell is a new record in `configs/`, not
new code (docs/CONVENTIONS.md, rules 11 to 14). Reviewers get an anonymized copy made by
`scripts/export_anonymous.sh`, never this repo. See [CONTRIBUTING.md](CONTRIBUTING.md)
and [docs/CONVENTIONS.md](docs/CONVENTIONS.md).

## License

MIT, see [LICENSE](LICENSE). Third-party code, data and models keep their own licenses;
see [NOTICE](NOTICE) and [docs/REUSE.md](docs/REUSE.md).

## Citation

There is no citation entry yet. One will follow publication.
