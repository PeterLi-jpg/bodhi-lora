# Inter-rater reliability (issue #73 + dual-grader validation)

Format mirrors the processed clinician_1 template
(`clinician_1-processed.xlsx`, kept outside the repo). Each Excel has
three sheets (Grading, Instructions - Primary, Instructions - Tiebreaker),
with the full grading instructions and a worked example baked into the file
itself.

## Files in this directory

```
Grading-template.xlsx     master template, do not edit
Grading-reviewer1.xlsx    → AnonClinician1
Grading-reviewer2.xlsx    → AnonClinician2
Grading-reviewer3.xlsx    → AnonClinician3
```

The answer key (`answer_key.xlsx`, sha256
d2994f85d822b402c18535a2ccba2f08e818294d5a8e50fad1584b2e3034e0a4) maps grading_id →
source/config/llm_score for the κ analysis. It unblinds the sheets, so it is no longer in
the repo: git stopped tracking it on 2026-10-11, a copy stays outside
the repo, and `.gitignore` covers it. Git history still holds it (it was tracked from
2026-05-04), so anyone can unblind these three sheets; KNOWN_ISSUES.md says what that
means for the two deferred raters.

Returned `Grading-*.xlsx` files can carry rater names. Keep them outside the
repo and pass their paths. The scripts read them in place and never copy them:

```
python scripts/make_tiebreaker_excel.py --rater-files <r1.xlsx> <r2.xlsx> <r3.xlsx>   # → Grading-tiebreaker.xlsx
make kappa RATER_FILES="<r1.xlsx> ..." ANSWER_KEY=<answer_key.xlsx>   # → kappa_results.json
```

`kappa_results.json` names raters only as rater_A, rater_B, ... So far one rater
has returned grades, and `kappa_results.json` holds the result for that rater.

## Sample design

- **50 responses** (50 = 40 eval + 10 filter), expanded to
  **662 criterion-level grading rows** (one row per
  (response, rubric criterion) pair).
- **40 eval rows** = 10 prompts × 4 Stage-4 configs (paired same-prompt design;
  validates Llama-3.1-8B eval grader).
- **10 filter rows** = 10 Qwen-graded BODHI training traces (validates the
  Stage-2 filter grader).
- Reviewers blind to source AND configuration; both are recoverable only via
  the answer key.

## Reviewer instructions (inside each Excel)

- `physician_grade` ∈ {pass, fail, unsure}. Excel data validation enforces.
- `physician_confidence` ∈ {1, 2, 3}. Excel data validation enforces.
- Orange `is_avoid_item=YES` rows: criteria the AI should NOT exhibit. Read
  carefully: `pass` means the AI correctly avoided the harmful behavior.

## Stats reported by compute_irr_kappa.py

- **Inter-physician κ** (3 pairwise, averaged): upper bound.
- **LLM grader vs physician consensus κ**: issue #73's headline number.
  Reported separately for `eval_llama` (Llama-3.1-8B) and `filter_qwen`
  (Qwen-14B) sources.
- **Per-config κ breakdown** for the 4 Stage-4 conditions.

## Data sources

- `../seed_<N>/eval/<condition>.json`: 5-seed Modal eval (Llama-graded)
- `../seed_7/filter/train.jsonl`: Qwen-filtered BODHI traces
- `data/raw/healthbench/hard.jsonl`, `data/raw/healthbench/full.jsonl`: rubrics
