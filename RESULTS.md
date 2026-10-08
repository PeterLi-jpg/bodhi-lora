# Results

This file says what each result is, where its data lives, which script makes it and
what state it is in. It holds no effect sizes, p-values or scores. Those come from
`make numbers` (see [README.md](README.md)), so they cannot drift from the data.

Status words used below:

- **raw**: the raw outputs are in the repo, and the numbers regenerate from them.
- **posted**: the numbers are transcribed from the posted rebuttal replies into
  `data/rebuttal_cells/posted_values.json`. The paper marks them "transcribed from
  posted values; raw outputs pending sync".
- **blocked**: an input is not in the repo. [KNOWN_ISSUES.md](KNOWN_ISSUES.md) says
  what unblocks it.

## Cells

† Med42-8B is a Llama-3 model, the same family as the evaluator. This breaks the
paper's grader-separation rule, so its cells are not used in any cross-model statement.

| Base model (family) | Benchmark | Status | Data |
|---|---|---|---|
| MedGemma-27B (Gemma) | HealthBench-Hard | raw | `results_modal/` |
| Mistral-Small-24B (Mistral) | HealthBench-Hard | posted | `results_rebuttal/healthbench__mistralai_Mistral-Small-24B-Instruct-2501/` after sync |
| Med42-8B † (Llama-3) | HealthBench-Hard | posted | `results_rebuttal/healthbench__m42-health_Llama3-Med42-8B/` after sync |
| BioMistral-7B (Mistral) | HealthBench-Hard | posted | `results_rebuttal/healthbench__BioMistral_BioMistral-7B/` after sync |
| Mistral-Small-24B (Mistral) | ChatDoctor | posted | `results_rebuttal/chatdoctor__mistralai_Mistral-Small-24B-Instruct-2501/` after sync |
| Med42-8B † (Llama-3) | ChatDoctor | posted | `results_rebuttal/chatdoctor__m42-health_Llama3-Med42-8B/` after sync |
| Mistral-Small-24B (Mistral) | MedQuAD | posted | `results_rebuttal/medquad__mistralai_Mistral-Small-24B-Instruct-2501/` after sync |
| Mistral-Small-24B (Mistral) | MedQA-USMLE | run, set aside | `results_rebuttal/medqa__mistralai_Mistral-Small-24B-Instruct-2501/` after sync |

All cells are in the main evaluation of the paper (D12), including these two:

- **BioMistral-7B** is a null result: the adapter does not raise active inquiry. One
  hypothesis is that the BODHI prompt itself fails to make this base ask, so there is
  little to distill. That hypothesis is not tested.
- **MedQA-USMLE** was run and set aside because it does not measure the target. It is a
  single-best-answer exam: each vignette is complete and the rubric grades agreement
  with the keyed answer, so asking is not the right behavior there. An earlier version
  of the cell used a preamble that told the model to ask for missing information, which
  cued the behavior under test. The preamble in `scripts/build_benchmark_jsonl.py` is
  now neutral. Which preamble produced the posted MedQA values is recorded only in the
  GPU-host logs, which are not synced yet.

## What the submitted cell holds

Checked on 2026-10-07 against `results_modal/`:

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
python -c "import json,glob; print(len({r['prompt_id'] for f in glob.glob('results_modal/seed_*/base_no_wrapper.json') for r in json.load(open(f))['results']}))"
```

## Results and their status

| Result | Made by | Status | Rule |
|---|---|---|---|
| Aggregate rubric score, LoRA minus Base, every cell | `make numbers` | raw (submitted), posted (added) | seed-level paired difference with a t(4) 95% interval; no equivalence claim (D2, D7) |
| Seven epistemic behaviors | `make numbers` from `epistemic_scores.json` | raw (submitted), posted (added) | LLM-graded; paired t over seeds, Holm over the seven within each contrast; prompt-level d is descriptive (D5, D7) |
| Physician agreement (kappa) | `scripts/compute_irr_kappa.py` | blocked: returned grades are not in the repo | 1 of 3 raters returned; below the pre-registered 0.6 target; prompt-clustered bootstrap interval (D5) |
| Blanket disclaimers | `make numbers` | raw | counts per condition with a two-sided Fisher exact p, in the results only (D10) |
| Grader parse failures | `make numbers` | raw (submitted), posted (added) | rate per condition for both graders, and a worst case that scores failures as not met (D13) |
| Discrimination (asks when information is missing, holds back when it is not) | `rebuttal/analyses/mimicry_split.py`, output in `rebuttal/results/` | raw | rubric-based and theme-only labelings side by side, both post hoc and exploratory (D9) |
| Red-flag drop with LoRA plus the BODHI prompt | `rebuttal/analyses/interference_mechanism.py`, output in `rebuttal/results/` | raw | format leakage, a post hoc diagnosis (D11) |
| Where the aggregate drop comes from | `rebuttal/analyses/FINDINGS_quality_drop.md` | posted | per-axis analysis on the added cells; one open problem (KNOWN_ISSUES) |
| Prior-work prompt baselines | `scripts/baselines/` | not run (needs GPU inference) | same eval sets and grader as the cells |
| Med42-8B regrade outside the Llama and Qwen families | `scripts/regrade_cross_family.py` | not run (needs the synced outputs and a GPU) | D6 |
| Training-loss figure | none | removed from the paper | it came from an early run on 838 traces, not the reported adapters (D14) |
| Calibration (Brier, ECE) | fields in `results_modal/seed_*/*.json` | not reported | calibration is not a measured property (D3) |
