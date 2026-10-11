# Results

This file says what each result is, where its data lives, what makes it and what state
it is in. It holds no effect sizes, p-values or scores. Those come from `make analyze`
and, for the paper, `make numbers` (see [README.md](../README.md)), so they cannot drift
from the data.

Status words used below:

- **raw**: the numbers regenerate from the raw outputs. `make extract` turns each seed's
  raw eval and grade files into its tracked response table
  (`results/<cell_id>/seed_<n>/table.csv`), and `make analyze` computes every statistic
  from the tables. The raw files of every cell are gitignored and kept in the local copy
  (open decision O2): about 80 MB for the 2026-05 cell, which history keeps at 31e4fd8,
  and about 370 MB for the 2026-07 cells.
- **posted**: the numbers exist only as posted rebuttal text, transcribed into
  `rebuttal/posted_values.json`. Since the sync of 2026-10-08 no cell is in this
  state; the file stays as the record of what was posted.
- **planned**: the cell or analysis is designed and its reading is fixed in
  [docs/ANALYSIS_PLAN.md](../docs/ANALYSIS_PLAN.md), but it has not run.
  [docs/RUN_PLAN.md](../docs/RUN_PLAN.md) gives its run.
- **blocked**: an input is not in the repo. [docs/KNOWN_ISSUES.md](../docs/KNOWN_ISSUES.md) says
  what unblocks it.

## Cells

† Med42-8B is a Llama-3 model, the same family as the evaluator. This breaks the
paper's grader-separation rule, so its cells are not used in any cross-model statement.

| Base model (family) | Benchmark | Status | Data |
|---|---|---|---|
| MedGemma-27B (Gemma) | HealthBench-Hard | raw | `results/healthbench__medgemma-27b-text__bodhi__2026-05/` |
| Mistral-Small-24B (Mistral) | HealthBench-Hard | raw | `results/healthbench__mistral-small-24b__bodhi__2026-07/` |
| Med42-8B † (Llama-3) | HealthBench-Hard | raw | `results/healthbench__med42-8b__bodhi__2026-07/` |
| BioMistral-7B (Mistral) | HealthBench-Hard | raw | `results/healthbench__biomistral-7b__bodhi__2026-07/` |
| Mistral-Small-24B (Mistral) | ChatDoctor | raw | `results/chatdoctor__mistral-small-24b__bodhi__2026-07/` |
| Med42-8B † (Llama-3) | ChatDoctor | raw | `results/chatdoctor__med42-8b__bodhi__2026-07/` |
| Mistral-Small-24B (Mistral) | MedQuAD | raw | `results/medquad__mistral-small-24b__bodhi__2026-07/` |
| Mistral-Small-24B (Mistral) | MedQA-USMLE | raw, set aside (preamble run; dimensions in 4 of 5 seeds) | `results/medqa-open-preamble__mistral-small-24b__bodhi__2026-07/` |

Three more host runs left folders and are not cells of the paper: MedGemma-27B on
MedQA and on MedQuAD (eval ids only; trace generation failed on 2026-07-27) and
BioMistral-7B on MedQA (a one-seed smoke run). A fourth run left no folder on the host:
Gemma-2-9B on HealthBench-Hard was stopped in trace generation on 2026-07-27, while its
vLLM server was starting; its logs are in `results/healthbench__gemma-2-9b__bodhi__2026-07/`.
All four have records in `configs/cells/`. `results/cross_cell/cell_recipe.json` records
the status of all four (the last under `runs_without_folder`), and Appendix J of the
paper names them.

All cells are in the main evaluation of the paper (D12), including these two:

- **BioMistral-7B** is a null result: the adapter does not raise active inquiry. One
  hypothesis is that the BODHI prompt itself fails to make this base ask, so there is
  little to distill. That hypothesis is not tested.
- **MedQA-USMLE** was run and set aside because it does not measure the target. It is a
  single-best-answer exam: each vignette is complete and the rubric grades agreement
  with the keyed answer, so asking is not the right behavior there. An earlier version
  of the cell used a preamble that told the model to ask for missing information, which
  cued the behavior under test. The preamble in `bodhi_lora/data/loaders.py` is
  now neutral. The synced MedQA outputs are that earlier run: they predate the fix
  (035d85f), and Base asks in nearly every response. No run without the preamble exists.
  Seed 42's epistemic grading found none of the MedQA prompts, so the seven dimensions
  have four seeds. The paper's Table 1 gives the cell's dimensions from those four seeds
  and its aggregate score from all five, set aside (withdrawals W12).

## Planned cells

[docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md) lists every planned cell and the reviewer
point it answers. All of them share one set of settings (EXPERIMENTS.md E13), so the
same analysis applies to each. None has run (D21). The first ones in
[docs/RUN_PLAN.md](../docs/RUN_PLAN.md) are `healthbench__mistral-small-24b__bodhi` (a
widely used general base, with the prompt baselines and red flags first),
`mediq-paired__mistral-small-24b__bodhi` (the calibration against mimicry probe on its
adapters) and `healthbench__mistral-small-24b__bodhi__labelled` (the physician-labelled
probes for over-flagging and clinician-posed prompts).

## What the submitted cell holds

Checked on 2026-10-07 against the files now in `results/healthbench__medgemma-27b-text__bodhi__2026-05/`:

- 5 seeds (7, 13, 42, 99, 101). Each seed draws its own 200 prompts from the 1,000
  HealthBench-Hard prompts.
- 667 unique prompts across the five draws. Seeds share prompts, so the seed is the
  unit of inference (D7).
- HealthBench responses per condition: Base 1,000; Base+BODHI 957; LoRA 1,000;
  LoRA+BODHI 959. The BODHI conditions lose responses when the two-pass prompt runs past
  the 4,096-token context of the eval server.
- Responses with a parsed epistemic grade per condition: 998, 947, 985 and 928 in the
  same order. The rest are grader parse failures (D13).

To recheck the counts:

```bash
python -c "import json,glob; print(len({r['prompt_id'] for f in glob.glob('results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_*/eval/base.json') for r in json.load(open(f))['results']}))"
```

## Results and their status

| Result | Made by | Status | Rule |
|---|---|---|---|
| Rubric score, LoRA minus Base, every cell | `make analyze` | raw | seed-level paired difference with a t(4) 95% interval; no equivalence claim (D2, D7) |
| Seven epistemic behaviors | `make analyze`, from the evaluator's grade files | raw | LLM-graded; paired t over seeds, Holm over the seven within each contrast; prompt-level d is descriptive (D5, D7) |
| Physician agreement (kappa) | `make kappa` (`scripts/compute_irr_kappa.py`), output in `results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/kappa_results.json` | raw output committed; per-row grades not in the repo | 1 of 3 raters returned; below the pre-registered 0.6 target; prompt-clustered bootstrap interval (D5); the other two raters are deferred (D20) |
| Blanket disclaimers | `make analyze` | raw | counts per condition with a two-sided Fisher exact p, in the results only (D10) |
| Grader parse failures | `make analyze` | raw, every cell | rate per condition for both graders, and a worst case that scores failures as not met (D13) |
| Discrimination with the theme-tag labelling | `make analyze` (`needs-info-theme`) | raw on every HealthBench cell; the paper cites LoRA minus Base in each (section 4.4, Table H.7) | post hoc and exploratory (D9); no cell shows a gain above zero; the rubric-keyword labelling was removed on 2026-10-10 (withdrawals W24, CONVENTIONS rule 15) |
| Discrimination with the physician labels (E02) | `make analyze` (`needs-info-physician`, family F6 in `configs/report.yaml`) | raw on every HealthBench cell since 2026-10-11, after the plan's adoption; the paper cites it (section 4.4, Table C.1) | the secondary test of calibration against mimicry (ANALYSIS_PLAN.md section 5.2): not detected in any of the three F6 cells, and with BioMistral-7B the index falls |
| Red-flag drop with LoRA plus the BODHI prompt | `make analyze` (format leakage: two of the protocol's own Pass 1 headers) | raw | format leakage, a post hoc diagnosis (D11); the posted nine-marker rule is withdrawn (W25) |
| Where the rubric-score drop comes from | `make analyze` (rubric axes with Holm over the axes, and the askers gap) | raw; the paper cites the axes of every cell (Table H.2) | exploratory (ANALYSIS_PLAN.md section 4); the internal note is `rebuttal/responses/drafts/FINDINGS_quality_drop.md` |
| Harm proxy (negative rubric criteria met) | `make analyze` (`neg_any`, `neg_points_lost`) | raw, every recorded cell with rubric grades (extracted 2026-10-09); the paper cites it for every cell since 2026-10-11 (Table C.1, section 4.1) | LLM-graded proxy, secondary (E06, ANALYSIS_PLAN.md section 10); not the physician harm review (C33) |
| Red-flag rate by the emergency theme (stands in for the C30 check) | `make analyze` (the evaluator's red-flag rate on the `emergency-theme` groups) | raw on every HealthBench cell; post hoc; the paper cites it as a descriptive check (section 4.4, Table H.6), which the first author accepted with the analysis plan on 2026-10-11 | descriptive, not an over-flagging rate: prompts of other themes can hold a red flag (ANALYSIS_PLAN.md section 12, 2026-10-10); the over-flagging test is H7 on the physician labels. The escalation keyword rule and the `red-flag-at-stake` label it used were removed |
| Grader-free question rate (E03) | none since 2026-10-10 | withdrawn: the second-person keyword rule is gone (W23) | the cross-check is a judge classifier with a hand-label gate (ANALYSIS_PLAN.md section 12, 2026-10-10); its `classify` stage is built, the hand labels are not made, and it has not run (RUN_PLAN.md R17) |
| Prior-work prompt baselines | `bodhi_lora/protocols/` | planned: all three built and stub-tested; RUN_PLAN.md R2 | compared only inside a cell that ran them (ANALYSIS_PLAN.md H4) |
| Med42-8B re-grade outside the Llama and Qwen families | `bodhi_lora/pipeline/grade.py` | planned: built, needs a GPU; RUN_PLAN.md R1 | D6, ANALYSIS_PLAN.md H8 |
| Judges on the other recorded cells, with their harm grades | `bodhi_lora/pipeline/grade.py` | planned: built, needs a GPU; RUN_PLAN.md R6 | C26, ANALYSIS_PLAN.md H8; the harm grades are an LLM proxy, not C33 |
| Contamination probe | `bodhi_lora/pipeline/probe.py` | planned: built, needs a GPU; RUN_PLAN.md R11 | Appendix A only (E11, ANALYSIS_PLAN.md section 10) |
| Per-cell tau, trace counts and adapter recipe | the filter logs and the adapter and trainer files of each cell | raw (`results/cross_cell/cell_recipe.json`) | PROV-3 |
| Measured cost per configuration | the run manifests and the `bench` stage | planned (E20); the 2026-07 host logs give shared-host times only (RUN_PLAN.md section 3.1) | C25 |
| Training-loss figure | none | removed from the paper | it came from an early run on 838 traces, not the reported adapters (D14) |
| Calibration (Brier, ECE) | fields in the raw eval JSON | not reported | calibration is not a measured property (D3) |
