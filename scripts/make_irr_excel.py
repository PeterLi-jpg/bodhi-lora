"""Per-criterion blind IRR generator (matches clinician_1-processed.xlsx format).

Each Excel has 3 sheets:

  - **Grading**: one row per (response, criterion). Clinicians fill three
    yellow columns: ``physician_id``, ``physician_grade`` (pass/fail/unsure),
    ``physician_confidence`` (1/2/3), plus optional ``notes``.
  - **Instructions - Primary**: workflow + grading rules (verbatim from
    the Sebastian-processed template at data/irr_instructions/primary.json).
  - **Instructions - Tiebreaker**: instructions for the adjudicator (used
    only after Stage-1 reviewers return; verbatim from
    data/irr_instructions/tiebreaker.json).

Sampling
--------
50 total responses chosen to fit the example's "~2 hours per clinician" budget:
  - 40 eval responses (10 per Stage-4 config × 4 configs, paired across configs
    for stronger per-config κ — Llama-3.1-8B-graded MedGemma responses)
  - 10 filter responses (Qwen-14B-graded BODHI training traces)

Each response carries its own rubric criteria (8-12 typical), so total grading
rows ≈ 50 × ~10 = ~500 rows per clinician. The is_avoid_item column flags
criteria where the model should NOT exhibit the behavior (negative-points
rubric items) — orange-highlighted in the Excel.

Output files (in results_modal/irr/):
  Grading-template.xlsx              master copy, do not edit
  Grading-1.xlsx, Grading-2.xlsx,    3 identical reviewer copies, rename to
  Grading-3.xlsx                     Grading-<reviewer-name>.xlsx before sending
  answer_key.xlsx                    KEEP LOCAL — row→source/config/llm_score
  README.md                          short workflow summary

Run scripts/make_tiebreaker_excel.py AFTER 3 reviewers return.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results_modal"
HEALTHBENCH_PATH = ROOT / "data" / "raw" / "healthbench_hard.jsonl"
HEALTHBENCH_FULL_PATH = ROOT / "data" / "raw" / "healthbench.jsonl"
FILTER_TRACES_PATH = ROOT / "data" / "sft_qwen_graded" / "seed_7_train.jsonl"
INSTR_DIR = ROOT / "data" / "irr_instructions"
OUT_DIR = RESULTS_DIR / "irr"

SEEDS = (7, 13, 42, 99, 101)
CONFIGS = ("base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi")

# Sized to match the example's "~2 hours per clinician" budget.
EVAL_PROMPTS_PER_CONFIG = 10    # 10 prompts × 4 configs = 40 eval responses
FILTER_RESPONSES = 10           # 10 Qwen-graded BODHI traces
SHUFFLE_SEED = 42

# Issue #73 clinical leads. Update if the reviewer roster changes.
REVIEWER_NAMES = ("zineb", "ash", "hillary")

# Style constants pulled byte-exact from clinician_1-processed.xlsx.
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(color="FFFFFF", bold=True)
AVOID_FILL = PatternFill("solid", fgColor="FCE4D6")    # orange — avoid rows
ZEBRA_A_FILL = PatternFill("solid", fgColor="D9E1F2")  # light blue (every other normal row)
ZEBRA_B_FILL = PatternFill("solid", fgColor="FFFFFF")  # white
INPUT_HIGHLIGHT_FILL = PatternFill("solid", fgColor="FFF7CC")  # yellow on input cols

GRADING_COLS = [
    ("grading_id",            14),
    ("response_id",           10),
    ("prompt_id",             38),
    ("criterion_number",       8),
    ("is_avoid_item",         12),
    ("criterion_text",        60),
    ("physician_id",          14),
    ("physician_grade",       14),
    ("physician_confidence",  12),
    ("notes",                 30),
    ("_prompt",               40),
    ("_response",             60),
]
INPUT_COLS = ("physician_id", "physician_grade", "physician_confidence", "notes")


def _format_messages(messages: list[dict]) -> str:
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if not content:
            continue
        lines.append(f"[{role.upper()}]\n{content}")
    return "\n\n".join(lines)


def load_healthbench_index(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            out[ex["prompt_id"]] = {
                "prompt_messages": ex["prompt"],
                "rubrics": ex.get("rubrics", []),
            }
    return out


def load_per_seed_results(seed: int) -> dict[str, list[dict]]:
    seed_dir = RESULTS_DIR / f"seed_{seed}"
    out: dict[str, list[dict]] = {}
    for cfg in CONFIGS:
        with open(seed_dir / f"{cfg}.json") as f:
            d = json.load(f)
        out[cfg] = d["results"]
    return out


def load_filter_traces(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_eval_responses(per_seed: dict[int, dict[str, list[dict]]],
                        prompts_per_config: int) -> list[dict]:
    """Pick prompts that exist across all 4 configs, evenly distributed across seeds.

    Returns one row per (seed, prompt_id, config) triple — total =
    prompts_per_config × len(CONFIGS). Each row has the prompt_id, the model's
    response, and the LLM score so we can build per-criterion rows in
    build_grading_rows().
    """
    rng = random.Random(SHUFFLE_SEED)
    base = prompts_per_config // len(SEEDS)
    extra = prompts_per_config - base * len(SEEDS)
    per_seed_n = {s: base + (1 if i < extra else 0) for i, s in enumerate(SEEDS)}

    rows: list[dict] = []
    for seed in SEEDS:
        n = per_seed_n[seed]
        ids_per_cfg = [
            {r["prompt_id"] for r in per_seed[seed][cfg]} for cfg in CONFIGS
        ]
        common = sorted(set.intersection(*ids_per_cfg))
        if len(common) < n:
            raise SystemExit(f"seed {seed}: only {len(common)} prompts cover all 4 configs (need {n})")
        for pid in rng.sample(common, n):
            for cfg in CONFIGS:
                for r in per_seed[seed][cfg]:
                    if r["prompt_id"] == pid:
                        rows.append({
                            "_source": "eval_llama",
                            "_seed": seed,
                            "_config": cfg,
                            "_llm_score": r.get("score"),
                            "prompt_id": pid,
                            "response": r.get("response", ""),
                        })
                        break
    return rows


def pick_filter_responses(filter_traces: list[dict], n: int) -> list[dict]:
    rng = random.Random(SHUFFLE_SEED + 2)
    sample = rng.sample(filter_traces, n)
    rows: list[dict] = []
    for tr in sample:
        rows.append({
            "_source": "filter_qwen",
            "_seed": None,
            "_config": "filter_trace",
            "_llm_score": tr.get("grade", {}).get("normalized_score"),
            "prompt_id": tr.get("prompt_id"),
            "response": tr.get("response", ""),
            "_filter_messages": tr.get("messages") or [],   # fallback prompt source
        })
    return rows


def assemble_responses(eval_rows: list[dict], filter_rows: list[dict]) -> list[dict]:
    """Mix and shuffle, then assign sequential R001..R0NN response_ids."""
    rows = eval_rows + filter_rows
    rng = random.Random(SHUFFLE_SEED + 1)
    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["_response_id"] = f"R{i:03d}"
    return rows


def build_grading_rows(responses: list[dict], hb: dict[str, dict]) -> list[dict]:
    """Expand each response into N rows — one per rubric criterion.

    Each grading row carries a unique grading_id like R001_c01, plus
    is_avoid_item flag (YES if the rubric criterion has negative points,
    "no" otherwise). HealthBench rubrics for filter rows come from the
    full-HealthBench index; eval rows come from HealthBench Hard.
    """
    rows: list[dict] = []
    for resp in responses:
        pid = resp["prompt_id"]
        hb_row = hb.get(pid, {})
        rubrics = hb_row.get("rubrics", [])
        # Prompt text — prefer the indexed prompt; fall back to the filter
        # trace's own messages (filter traces from the wider HealthBench may
        # not be in either index even after merging Hard + full).
        prompt_msgs = hb_row.get("prompt_messages") or resp.get("_filter_messages") or []
        prompt_text = _format_messages(prompt_msgs)
        response_text = resp.get("response", "")

        for j, r in enumerate(rubrics, 1):
            pts = r.get("points", 0)
            is_avoid = pts < 0
            rows.append({
                # Visible columns (in this order):
                "grading_id":           f"{resp['_response_id']}_c{j:02d}",
                "response_id":          resp["_response_id"],
                "prompt_id":            pid,
                "criterion_number":     j,
                "is_avoid_item":        "YES" if is_avoid else "no",
                "criterion_text":       r.get("criterion", ""),
                "physician_id":         "",
                "physician_grade":      "",
                "physician_confidence": "",
                "notes":                "",
                "_prompt":              prompt_text,
                "_response":            response_text,
                # Hidden — preserved for the answer key only:
                "__source":             resp["_source"],
                "__config":             resp["_config"],
                "__seed":               resp["_seed"],
                "__llm_score":          resp["_llm_score"],
            })
    return rows


def write_grading_template(grading_rows: list[dict], out_path: Path) -> None:
    """Write the 3-sheet workbook with exact styling matching the example."""
    wb = Workbook()
    # Default sheet → Grading.
    ws = wb.active
    ws.title = "Grading"

    headers = [c[0] for c in GRADING_COLS]
    widths = {c[0]: c[1] for c in GRADING_COLS}

    # Header row.
    for i, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=i, value=h)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(i)].width = widths[h]

    # Data rows.
    n_normal_seen = 0
    for row_idx, r in enumerate(grading_rows, 2):
        is_avoid = r["is_avoid_item"] == "YES"
        if is_avoid:
            row_fill = AVOID_FILL
        else:
            row_fill = ZEBRA_A_FILL if (n_normal_seen % 2 == 0) else ZEBRA_B_FILL
            n_normal_seen += 1

        for col_idx, h in enumerate(headers, 1):
            v = r.get(h, "")
            cell = ws.cell(row=row_idx, column=col_idx, value=v)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.fill = row_fill

        # Yellow tint over the input cells (overrides zebra/avoid for visibility).
        for col_idx, h in enumerate(headers, 1):
            if h in INPUT_COLS:
                ws.cell(row=row_idx, column=col_idx).fill = INPUT_HIGHLIGHT_FILL

        # Generous row height so the long _prompt/_response cells don't truncate.
        ws.row_dimensions[row_idx].height = 80

    ws.freeze_panes = "A2"

    # Data validation: physician_grade ∈ {pass, fail, unsure}.
    grade_col = get_column_letter(headers.index("physician_grade") + 1)
    dv_grade = DataValidation(
        type="list",
        formula1='"pass,fail,unsure"',
        allow_blank=True,
        showErrorMessage=True,
        errorTitle="Invalid grade",
        error="Enter pass, fail, or unsure (lowercase). See Instructions tab.",
    )
    dv_grade.add(f"{grade_col}2:{grade_col}{ws.max_row}")
    ws.add_data_validation(dv_grade)

    # Data validation: physician_confidence ∈ {1, 2, 3}.
    conf_col = get_column_letter(headers.index("physician_confidence") + 1)
    dv_conf = DataValidation(
        type="whole",
        operator="between",
        formula1=1,
        formula2=3,
        allow_blank=True,
        showErrorMessage=True,
        errorTitle="Invalid confidence",
        error="Enter 1 (low), 2 (medium), or 3 (high).",
    )
    dv_conf.add(f"{conf_col}2:{conf_col}{ws.max_row}")
    ws.add_data_validation(dv_conf)

    # Instructions tabs.
    _write_instructions_sheet(wb, "Instructions - Primary",
                              INSTR_DIR / "primary.json")
    _write_instructions_sheet(wb, "Instructions - Tiebreaker",
                              INSTR_DIR / "tiebreaker.json")

    wb.save(out_path)


def _write_instructions_sheet(wb: Workbook, sheet_name: str, json_path: Path) -> None:
    rows = json.loads(json_path.read_text())
    ws = wb.create_sheet(sheet_name)
    ws.column_dimensions["A"].width = 110
    for row_idx, entry in enumerate(rows, 1):
        text = entry.get("text", "")
        bold = bool(entry.get("bold", False))
        cell = ws.cell(row=row_idx, column=1, value=text)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        if bold:
            cell.font = Font(bold=True, size=12)


def write_answer_key(grading_rows: list[dict], out_path: Path) -> None:
    """Unblinding key — keep local. Rows: every grading_id with source/config/llm_score."""
    df = pd.DataFrame([{
        "grading_id":  r["grading_id"],
        "response_id": r["response_id"],
        "source":      r["__source"],
        "seed":        r["__seed"],
        "config":      r["__config"],
        "criterion_number": r["criterion_number"],
        "is_avoid_item":    r["is_avoid_item"],
        "llm_score":   r["__llm_score"],
    } for r in grading_rows])
    df.to_excel(out_path, index=False, sheet_name="answer_key")


def write_readme(out_path: Path, n_responses: int, n_grading_rows: int,
                 n_eval: int, n_filter: int) -> None:
    out_path.write_text(f"""# Inter-rater reliability (issue #73 + dual-grader validation)

Format mirrors the Sebastian-processed template at
`/Users/anqipeterli/Downloads/clinician_1-processed.xlsx`. Each Excel has
three sheets — Grading, Instructions - Primary, Instructions - Tiebreaker —
with the full grading instructions and a worked example baked into the file
itself.

## Files in this directory

```
Grading-template.xlsx     master template, do not edit
Grading-zineb.xlsx        → email to Zineb
Grading-ash.xlsx          → email to Ash (Doulla)
Grading-hillary.xlsx      → email to Hillary
answer_key.xlsx           DO NOT SEND. Maps grading_id → source/config/llm_score
                          for downstream κ analysis.
```

After all 3 reviewers return their `Grading-*.xlsx` files, drop them in this
directory and run:

```
python scripts/make_tiebreaker_excel.py    # → tiebreaker_disputes.xlsx
python scripts/compute_irr_kappa.py        # → kappa_results.{{json,md}}
```

## Sample design

- **{n_responses} responses** (50 = 40 eval + 10 filter), expanded to
  **{n_grading_rows} criterion-level grading rows** (one row per
  (response, rubric criterion) pair).
- **40 eval rows** = 10 prompts × 4 Stage-4 configs (paired same-prompt design;
  validates Llama-3.1-8B eval grader).
- **10 filter rows** = 10 Qwen-graded BODHI training traces (validates the
  Stage-2 filter grader).
- Reviewers blind to source AND configuration; both are recoverable only via
  `answer_key.xlsx`.

## Reviewer instructions (inside each Excel)

- `physician_grade` ∈ {{pass, fail, unsure}}. Excel data validation enforces.
- `physician_confidence` ∈ {{1, 2, 3}}. Excel data validation enforces.
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
""")


def main() -> None:
    for p in (HEALTHBENCH_PATH, HEALTHBENCH_FULL_PATH, FILTER_TRACES_PATH,
              INSTR_DIR / "primary.json", INSTR_DIR / "tiebreaker.json"):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading rubrics (HealthBench Hard + full)...")
    hb = load_healthbench_index(HEALTHBENCH_PATH)
    hb_full = load_healthbench_index(HEALTHBENCH_FULL_PATH)
    for pid, row in hb_full.items():
        hb.setdefault(pid, row)
    print(f"  combined rubric index: {len(hb)} prompts")

    print("loading per-seed eval results...")
    per_seed = {s: load_per_seed_results(s) for s in SEEDS}

    print("loading filter traces...")
    filter_traces = load_filter_traces(FILTER_TRACES_PATH)
    print(f"  {len(filter_traces)} traces available")

    print(f"sampling eval: {EVAL_PROMPTS_PER_CONFIG} prompts × {len(CONFIGS)} configs = {EVAL_PROMPTS_PER_CONFIG * len(CONFIGS)} responses")
    eval_rows = pick_eval_responses(per_seed, EVAL_PROMPTS_PER_CONFIG)

    print(f"sampling filter: {FILTER_RESPONSES} responses")
    filter_rows = pick_filter_responses(filter_traces, FILTER_RESPONSES)

    responses = assemble_responses(eval_rows, filter_rows)
    print(f"  {len(responses)} total responses ({len(eval_rows)} eval + {len(filter_rows)} filter)")

    grading_rows = build_grading_rows(responses, hb)
    print(f"  {len(grading_rows)} grading rows (criterion-level)")

    template_path = OUT_DIR / "Grading-template.xlsx"
    print(f"writing grading template -> {template_path}")
    write_grading_template(grading_rows, template_path)

    # 3 identical reviewer copies, named for the issue #73 clinical leads.
    # If reviewer roster changes, edit REVIEWER_NAMES below or rename the
    # files manually before sending — find_reviewer_files() in the
    # tiebreaker/kappa scripts globs Grading-*.xlsx so any naming works.
    for name in REVIEWER_NAMES:
        copy_path = OUT_DIR / f"Grading-{name}.xlsx"
        copy_path.write_bytes(template_path.read_bytes())
        print(f"  wrote reviewer copy -> {copy_path}")

    answer_path = OUT_DIR / "answer_key.xlsx"
    print(f"writing answer key       -> {answer_path}")
    write_answer_key(grading_rows, answer_path)

    readme_path = OUT_DIR / "README.md"
    print(f"writing README           -> {readme_path}")
    write_readme(readme_path, n_responses=len(responses),
                 n_grading_rows=len(grading_rows),
                 n_eval=len(eval_rows), n_filter=len(filter_rows))

    # Sanity print.
    by_source = {"eval_llama": 0, "filter_qwen": 0}
    for r in grading_rows:
        by_source[r["__source"]] += 1
    print()
    print("grading rows by source:")
    for k, v in by_source.items():
        print(f"  {k}: {v}")
    n_avoid = sum(1 for r in grading_rows if r["is_avoid_item"] == "YES")
    print(f"avoid-rows (is_avoid_item=YES): {n_avoid} ({n_avoid / len(grading_rows):.1%})")


if __name__ == "__main__":
    main()
