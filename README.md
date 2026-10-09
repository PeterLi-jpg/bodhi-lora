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
the BODHI prompt. Each condition gets an aggregate rubric score and scores on seven
epistemic behaviors: uncertainty acknowledgment, hedging quality, specificity, active
inquiry, context-seeking, red-flag identification and scope bounding.
Llama-3.1-8B-Instruct grades both.

The repo slug `bohdi-lora` misspells the name. The project name is BODHI everywhere
else. Renaming the remote repo is deferred (docs/DECISIONS.md, D15).

## Read this first

- **The epistemic-behavior scores are LLM-graded.** The physician check grades the
  HealthBench rubric criteria behind the aggregate score, not the seven behaviors. One
  of three physician raters returned grades, and agreement with the LLM grader is below
  the target of 0.6 set in advance (in the repository, for the filter grader). The paper
  does not claim physician validation (D5).
- **The adapter has a cost.** We report the seed-level paired difference LoRA minus
  Base, with a t(4) 95% interval, for every cell, computed from the raw outputs. The
  aggregate score falls in all six added cells that are not set aside, and each drop
  survives Holm correction over those six. It rises in the seventh, MedQA, but that run
  used a prompt that told the model to ask (Base asked in nearly every response), so it
  measures instruction following and is set aside. We make no equivalence or
  non-inferiority claim (D2).
- **Med42-8B cells carry a dagger.** Med42-8B is a Llama-3 model, the same family as the
  evaluator. That breaks the paper's grader-separation rule, so these cells are not
  used in any cross-model statement (D6).
- **Calibration is not measured.** The word appears only as motivation (D3).

The full list of open problems is in [KNOWN_ISSUES.md](KNOWN_ISSUES.md). The decisions
behind these points are in [docs/DECISIONS.md](docs/DECISIONS.md).

## Layout

```
configs/            training configs for MedGemma-27B (GPU path) and the smoke model
data/
  irr_instructions/ physician grading instructions (raters are AnonClinician1-4)
  raw/              HealthBench eval ids; benchmark JSONL is downloaded or built here
  rebuttal_cells/   added-cell values from the raw outputs (generated_values.json) and as posted
docs/               conventions, decision log, reuse and licenses
hooks/              commit-msg and pre-push hooks (activate once, see below)
legacy/             frozen launchers of the submitted MedGemma-27B cell (TPU, Modal)
paper/              canonical paper source (paper/main.tex) and the frozen submission
rebuttal/           added-cell configs, launchers, analyses and the posted replies
results_modal/      submitted cell: per-seed eval and epistemic-score JSON
rebuttal/results/   JSON written by the rebuttal analyses
scripts/            pipeline stages, statistics, figures, IRR tools, repo checks
tests/              pytest suite, no GPU needed
```

## Two sets of cells

| Set | Cells | Pipeline | Raw outputs |
|---|---|---|---|
| Submitted | MedGemma-27B on HealthBench-Hard | tunix LoRA on 8x TPU v6e; eval on one A100 (Modal). Launchers in `legacy/` | `results_modal/`, committed |
| Added | Mistral-Small-24B, Med42-8B†, BioMistral-7B on HealthBench-Hard; Mistral-Small-24B and Med42-8B† on ChatDoctor; Mistral-Small-24B on MedQuAD; Mistral-Small-24B on MedQA (set aside) | GPU path: `bodhi_lora/pipeline/train.py` through the 2026-07 launcher (`61dada9:rebuttal/launch/run_cell.sh`), on H100 | copied from the GPU host into `results_rebuttal/` (eval JSON gitignored; per-seed values in `data/rebuttal_cells/generated_values.json`) |

The paper takes the added-cell numbers from `data/rebuttal_cells/generated_values.json`,
which `make cells` computes from the synced raw outputs.
`data/rebuttal_cells/posted_values.json` stays as the record of what was posted.
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) has the per-cell configs and the differences
between the two pipelines.

## Reproduce the paper

The rule is that every number in the paper comes from a script and none is typed by
hand (docs/CONVENTIONS.md, rule 2).

`make numbers` needs one file that is not in git: `data/raw/healthbench/hard.jsonl`,
the public HealthBench-Hard set. `python -m bodhi_lora fetch healthbench` downloads it. Every other input is
committed, including small copies of the filter scores and the second-grader scores
in `data/derived/` (paper/generated/README.md).

```bash
make test       # pytest suite
make numbers    # all tables and inline numbers, from results_modal/ and the rebuttal data
make figures    # all figures (scripts/build_paper_figures.py)
make paper      # numbers + figures, then compile paper/main.tex
```

| Paper item | Input | Notes |
|---|---|---|
| Main HealthBench-Hard table, per-seed table | `results_modal/seed_*/{base_no_wrapper,base_bodhi,lora_no_wrapper,lora_bodhi}.json` | seed is the unit: paired t over 5 seeds, t intervals |
| Epistemic-behavior table, effect sizes | `results_modal/seed_*/epistemic_scores.json` | Holm over the seven behaviors in each contrast; prompt-level d is descriptive, with a prompt-clustered bootstrap interval |
| Physician agreement table | `results_modal/irr/kappa_results.json` (`make kappa` on the returned file, read outside the repo) | per-row grades are not committed, so the value cannot be recomputed from the repo alone (KNOWN_ISSUES) |
| Discrimination analysis | `rebuttal/analyses/mimicry_split.py` | both labelings side by side, post hoc |
| Format-leakage analysis | `rebuttal/analyses/interference_mechanism.py` | post hoc diagnosis |
| Added-cell tables | `data/rebuttal_cells/generated_values.json` (`make cells` from `results_rebuttal/`) | Med42-8B rows marked † |
| Per-cell tau, trace counts, adapter recipe | `rebuttal/results/cell_recipe.json` (`make cells` from `logs/host/` and the adapter files) | PROV-3 |
| Hyperparameter tables | `legacy/` launcher, the train blocks of `configs/cells/`, the synced adapter files | one table per cell group; the submitted adapter config could not be checked against the saved files |

## Run the pipeline

Every setting is a record in `configs/`: models, benchmarks, protocols and cells
(docs/ARCHITECTURE.md section 3). A cell is one benchmark, one base model and one
protocol over seeds, and `python -m bodhi_lora` runs it:

```bash
python -m bodhi_lora list cells                 # every cell, its status and marks
python -m bodhi_lora list cells --settings      # what differs between the cells
python -m bodhi_lora fetch healthbench          # download a benchmark, checked against its pins
python -m bodhi_lora run <cell id> --gpus 0,1   # every stage, one seed per GPU
python -m bodhi_lora run <cell id> --dry-run    # print the GPU commands, run the rest on the CPU stub
make smoke                                      # the smoke cell on the CPU, training included
```

Setup, once per machine: `bash setup.sh` builds the training environment (`.venv`,
`requirements.txt`) and the vLLM environment (`.venv-infer`, `requirements-infer.txt`),
and `export HF_TOKEN=...` gives access to the gated models (MedGemma, Llama-3.1).

A run writes `results/<cell id>/`, with a manifest per stage (`bodhi_lora/manifest.py`):
resolved settings, git commit and dirty flag, library versions, model revisions, adapter
hash, time, GPUs and token counts. The runner refuses a cell that records a past run
(the 2026-05 and 2026-07 cells), so its outputs never mix with new ones; it only adds
judge re-grades to it (`--stages grade`).

## Where results live

- `results_modal/`: the submitted cell. Five seeds, four conditions, 200 prompts per
  seed, plus `epistemic_scores.json` per seed. `results_modal/cross_seed_aggregate.md`
  is a generated summary.
- `results_modal/irr/`: the blank physician grading sheets and the answer key. Returned
  sheets are not in the repo.
- `rebuttal/results/`: outputs of the rebuttal analyses.
- `results_rebuttal/`: the added cells, copied from the GPU host on 2026-10-08. Git
  tracks each cell's `eval_ids.json`, `adapter_config.json` and `trainer_state.json`. The
  per-seed eval JSON (full responses, about 370 MB), the host's `multi_seed_summary.json`
  (wrong seed list), raw generations and adapter weights are gitignored. `make cells`
  turns the eval JSON into `data/rebuttal_cells/generated_values.json`.
- `logs/host/`: the GPU host's logs (gitignored). `make cells` reads the filter logs.
- [RESULTS.md](RESULTS.md): what each result is, where it lives and its status.

## Data and leakage

`setup.sh` downloads HealthBench into `data/raw/`. HealthBench-Hard (1,000 prompts) is a
subset of HealthBench (5,000). All 1,000 Hard prompts are excluded from training by
`prompt_id`, so about 4,000 prompts feed trace generation. The submitted-cell launcher
ran a leakage gate before training, and the runner runs `bodhi_lora/data/leakage.py` on
every cell: no graded trace may be an eval or probe prompt, by id or by text.
On 2026-10-07 the seed-7 training set of the submitted cell had 0 Hard prompts among its
2,816 prompts and 0 overlap with the 667 eval prompts. ChatDoctor, MedQuAD and MedQA are
built into the same row format by `bodhi_lora/data/loaders.py`, from their benchmark records.

## Contributing

Activate the hooks once per clone. They block commit messages with tool trailers and
pushes of scratch branches:

```bash
git config core.hooksPath hooks
```

Before a pull request, run `pytest tests/`, `bash scripts/check_no_secrets.sh` and
`python scripts/check_doc_paths.py`. Reviewers get an anonymized copy made by
`scripts/export_anonymous.sh`, never this repo. See
[contributions/contributing.md](contributions/contributing.md) and the lab rules in
[docs/CONVENTIONS.md](docs/CONVENTIONS.md).

## License

MIT, see [LICENSE](LICENSE). Third-party code, data and models keep their own licenses;
see [NOTICE](NOTICE) and [docs/REUSE.md](docs/REUSE.md).

## Citation

There is no citation entry yet. One will follow publication.
