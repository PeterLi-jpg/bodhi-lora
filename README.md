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
  the pre-registered target of 0.6. The paper does not claim physician validation (D5).
- **The adapter has a cost.** We report the seed-level paired difference LoRA minus
  Base, with a t(4) 95% interval, for every cell. The aggregate score falls in six of
  the seven added cells. It rises only in the MedQA cell, which does not measure the
  target behavior. We make no equivalence or non-inferiority claim (D2).
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
  rebuttal_cells/   values transcribed from the posted rebuttal (posted_values.json)
docs/               conventions, decision log, reuse and licenses
gpu/                single-machine GPU launcher and accelerate configs
hooks/              commit-msg and pre-push hooks (activate once, see below)
legacy/             frozen launchers of the submitted MedGemma-27B cell (TPU, Modal)
paper/              canonical paper source (paper/main.tex) and the frozen submission
rebuttal/           added-cell configs, launchers, analyses and the posted replies
results_modal/      submitted cell: per-seed eval and epistemic-score JSON
rebuttal/results/   JSON written by the rebuttal analyses
scripts/            pipeline stages, statistics, figures, IRR tools, repo checks
slurm/              SBATCH wrappers for the pipeline stages
tests/              pytest suite, no GPU needed
```

## Two sets of cells

| Set | Cells | Pipeline | Raw outputs |
|---|---|---|---|
| Submitted | MedGemma-27B on HealthBench-Hard | tunix LoRA on 8x TPU v6e; eval on one A100 (Modal). Launchers in `legacy/` | `results_modal/`, committed |
| Added | Mistral-Small-24B, Med42-8B†, BioMistral-7B on HealthBench-Hard; Mistral-Small-24B and Med42-8B† on ChatDoctor; Mistral-Small-24B on MedQuAD; Mistral-Small-24B on MedQA (set aside) | GPU path: `scripts/train_lora.py` through `rebuttal/launch/run_cell.sh`, on H100 | on the GPU host, not yet synced |

Until the added-cell outputs are synced, the paper typesets their numbers from
`data/rebuttal_cells/posted_values.json` with the footnote "transcribed from posted
values; raw outputs pending sync". [REPRODUCIBILITY.md](REPRODUCIBILITY.md) has the
per-cell configs and the differences between the two pipelines.

## Reproduce the paper

The rule is that every number in the paper comes from a script and none is typed by
hand (docs/CONVENTIONS.md, rule 2).

`make numbers` needs one file that is not in git: `data/raw/healthbench_hard.jsonl`,
the public HealthBench-Hard set. `bash setup.sh` downloads it. Every other input is
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
| Added-cell table | `results_rebuttal/` after sync, else `data/rebuttal_cells/posted_values.json` | Med42-8B rows marked † |
| Hyperparameter tables | `legacy/` launcher, `rebuttal/configs/*.yaml` | one table per cell group; the submitted adapter config is pending confirmation |

## Run the pipeline

Setup, once per machine:

```bash
python3 -m venv .venv && source .venv/bin/activate
bash setup.sh                  # installs requirements.txt, runs pip check, downloads HealthBench
export HF_TOKEN=...            # gated models: MedGemma, Llama-3.1, Mistral
bash smoke.sh                  # all four stages on a small model, under 10 minutes on one GPU
```

One added cell, five seeds:

```bash
MODEL=mistralai/Mistral-Small-24B-Instruct-2501 \
CONFIG=rebuttal/configs/lora_mistral_small_24b_qlora.yaml \
BENCH=healthbench SEEDS="42 7 13 99 101" \
bash rebuttal/launch/run_cell.sh
```

The cell writes to `results_rebuttal/<bench>__<model>/`. `scripts/run_manifest.py` records
what ran: resolved config, git commit and dirty flag, library versions, model revisions
and adapter hash. Prior-work prompt baselines live in `scripts/baselines/`. The Med42-8B
regrade with a judge outside the Llama, Qwen and base-model families is
`scripts/regrade_cross_family.py`.

## Where results live

- `results_modal/`: the submitted cell. Five seeds, four conditions, 200 prompts per
  seed, plus `epistemic_scores.json` per seed. `results_modal/cross_seed_aggregate.md`
  is a generated summary.
- `results_modal/irr/`: the blank physician grading sheets and the answer key. Returned
  sheets are not in the repo.
- `rebuttal/results/`: outputs of the rebuttal analyses.
- `results_rebuttal/`: the added cells, once synced. Raw generations (`*.jsonl`) and
  adapter weights stay out of git; per-seed eval JSON, summaries, adapter configs and
  trainer states go in.
- [RESULTS.md](RESULTS.md): what each result is, where it lives and its status.

## Data and leakage

`setup.sh` downloads HealthBench into `data/raw/`. HealthBench-Hard (1,000 prompts) is a
subset of HealthBench (5,000). All 1,000 Hard prompts are excluded from training by
`prompt_id` (`--exclude-ids`), so about 4,000 prompts feed trace generation. The
submitted-cell launcher ran `scripts/check_dataset_overlap.py` as a gate before
training, and `rebuttal/launch/run_cell.sh` runs the same gate on every cell.
On 2026-10-07 the seed-7 training set of the submitted cell had 0 Hard prompts among its
2,816 prompts and 0 overlap with the 667 eval prompts. ChatDoctor, MedQuAD and MedQA are
converted to the same format by `scripts/build_benchmark_jsonl.py`.

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
