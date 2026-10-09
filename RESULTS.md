# Results

This file says what each result is, where its data lives, which script makes it and
what state it is in. It holds no effect sizes, p-values or scores. Those come from
`make numbers` (see [README.md](README.md)), so they cannot drift from the data.

Status words used below:

- **raw**: the numbers regenerate from the raw outputs. For the submitted cell the raw
  outputs are in git (`results/healthbench__medgemma-27b-text__bodhi__2026-05/`). For the added cells the per-seed eval JSON is in the
  local copy of the host's outputs, in the `results/*__2026-07/` cells (gitignored,
  about 370 MB); `make cells` turns it into the tracked
  `results/cross_cell/generated_values.json`, which `make numbers` reads.
- **posted**: the numbers exist only as posted rebuttal text, transcribed into
  `rebuttal/posted_values.json`. Since the sync of 2026-10-08 no cell is in this
  state; the file stays as the record of what was posted.
- **blocked**: an input is not in the repo. [KNOWN_ISSUES.md](KNOWN_ISSUES.md) says
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
  have four seeds; the paper reports the cell's aggregate score and leaves the dimensions
  out of Table 3 (withdrawals W12).

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
| Aggregate rubric score, LoRA minus Base, every cell | `make cells`, `make numbers` | raw | seed-level paired difference with a t(4) 95% interval; no equivalence claim (D2, D7) |
| Seven epistemic behaviors | `make numbers` from `results/*/seed_*/grade/epistemic__*.json` (added cells through `make cells`) | raw | LLM-graded; paired t over seeds, Holm over the seven within each contrast; prompt-level d is descriptive (D5, D7) |
| Physician agreement (kappa) | `scripts/compute_irr_kappa.py` (`make kappa`), output in `results/healthbench__medgemma-27b-text__bodhi__2026-05/validation/kappa_results.json` | raw output committed; per-row grades not in the repo | 1 of 3 raters returned; below the pre-registered 0.6 target; prompt-clustered bootstrap interval (D5) |
| Blanket disclaimers | `make numbers` | raw | counts per condition with a two-sided Fisher exact p, in the results only (D10) |
| Grader parse failures | `make numbers` | raw (submitted); not tabulated for the added cells | rate per condition for both graders, and a worst case that scores failures as not met (D13) |
| Discrimination (asks when information is missing, holds back when it is not) | `rebuttal/analyses/mimicry_split.py`, output in `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/` | raw | rubric-based and theme-only labelings side by side, both post hoc and exploratory (D9) |
| Red-flag drop with LoRA plus the BODHI prompt | `rebuttal/analyses/interference_mechanism.py`, output in `results/healthbench__medgemma-27b-text__bodhi__2026-05/analysis/` | raw | format leakage, a post hoc diagnosis (D11) |
| Where the aggregate drop comes from | `rebuttal/responses/drafts/FINDINGS_quality_drop.md`; the paired tests are `rebuttal/analyses/quality_drop.py` | tests raw; per-axis analysis from ad hoc code, not in the repo | per-axis analysis on the added cells; one open problem (KNOWN_ISSUES) |
| Prior-work prompt baselines | `bodhi_lora/protocols/` | not run (needs GPU inference) | same eval sets and grader as the cells |
| Med42-8B regrade outside the Llama and Qwen families | `bodhi_lora/pipeline/grade.py` | not run (needs a GPU; the outputs are in the local copy and on the host) | D6 |
| Per-cell tau, trace counts and adapter recipe | `rebuttal/analyses/cell_recipe.py` (`make cells`), output `results/cross_cell/cell_recipe.json` | raw (host logs and synced adapter files) | PROV-3 |
| Training-loss figure | none | removed from the paper | it came from an early run on 838 traces, not the reported adapters (D14) |
| Calibration (Brier, ECE) | fields in `results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_*/eval/*.json` | not reported | calibration is not a measured property (D3) |
