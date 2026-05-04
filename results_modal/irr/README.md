# Inter-rater reliability (issue #73 + dual-grader validation)

Format mirrors the Sebastian-processed template at
`/Users/anqipeterli/Downloads/clinician_1-processed.xlsx`. Each Excel has
three sheets — Grading, Instructions - Primary, Instructions - Tiebreaker —
with the full grading instructions and a worked example baked into the file
itself.

## Files in this directory

```
Grading-template.xlsx     master template, do not edit
Grading-1.xlsx            ⎫
Grading-2.xlsx            ⎬ 3 identical reviewer copies. Rename to
Grading-3.xlsx            ⎭ Grading-<your-name>.xlsx before sending.
answer_key.xlsx           DO NOT SEND. Maps grading_id → source/config/llm_score
                          for downstream κ analysis.
```

After all 3 reviewers return their `Grading-*.xlsx` files, drop them in this
directory and run:

```
python scripts/make_tiebreaker_excel.py    # → tiebreaker_disputes.xlsx
python scripts/compute_irr_kappa.py        # → kappa_results.{json,md}
```

## Sample design

- **50 responses** (50 = 40 eval + 10 filter), expanded to
  **662 criterion-level grading rows** (one row per
  (response, rubric criterion) pair).
- **40 eval rows** = 10 prompts × 4 Stage-4 configs (paired same-prompt design;
  validates Llama-3.1-8B eval grader).
- **10 filter rows** = 10 Qwen-graded BODHI training traces (validates the
  Stage-2 filter grader).
- Reviewers blind to source AND configuration; both are recoverable only via
  `answer_key.xlsx`.

## Reviewer instructions (inside each Excel)

- `physician_grade` ∈ {pass, fail, unsure}. Excel data validation enforces.
- `physician_confidence` ∈ {1, 2, 3}. Excel data validation enforces.
- Orange `is_avoid_item=YES` rows: criteria the AI should NOT exhibit. Read
  carefully — `pass` means the AI correctly avoided the harmful behavior.

## Stats reported by compute_irr_kappa.py

- **Inter-physician κ** (3 pairwise, averaged) — upper bound.
- **LLM grader vs physician consensus κ** — issue #73's headline number.
  Reported separately for `eval_llama` (Llama-3.1-8B) and `filter_qwen`
  (Qwen-14B) sources.
- **Per-config κ breakdown** for the 4 Stage-4 conditions.

## Data sources

- `results_modal/seed_<N>/<config>.json` — 5-seed Modal eval (Llama-graded)
- `data/sft_qwen_graded/seed_7_train.jsonl` — Qwen-filtered BODHI traces
- `data/raw/healthbench_hard.jsonl`, `data/raw/healthbench.jsonl` — rubrics
