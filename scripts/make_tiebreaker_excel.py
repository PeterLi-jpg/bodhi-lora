"""Stage 2 IRR helper: build a tiebreaker Excel from the 3 reviewer files.

Reads ``results_modal/irr/Grading-*.xlsx`` (NOT Grading-template, NOT
Grading-tiebreaker), joins by ``grading_id``, and emits ONE Excel containing
only the rows where the 3 primary reviewers split. The adjudicator (Felipe /
or whoever you assign) fills in the Tiebreaker tab.

Dispute logic
-------------
Per the example's Instructions-Tiebreaker:

  "For most criteria, at least two of the three agreed — those are
   resolved automatically by majority vote. Where all three disagreed
   (a genuine three-way split), your vote decides the final answer."

So we flag a row for tiebreaker IFF the three reviewers all gave DIFFERENT
non-blank grades among {pass, fail, unsure} (i.e., one of each — the only
way to be a three-way split with three categories). Two-vs-one disagreements
are resolved by majority and are NOT shown to the adjudicator.

Output
------
``results_modal/irr/Grading-tiebreaker.xlsx`` with two sheets:
  - Tiebreaker    (one row per disputed criterion, with the 3 prior grades
                   anonymized as reviewer_1/2/3 in the order the files were
                   loaded; adjudicator fills tiebreaker_grade + confidence
                   + required notes)
  - Instructions - Tiebreaker  (verbatim from data/irr_instructions/tiebreaker.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parent.parent
IRR_DIR = ROOT / "results_modal" / "irr"
INSTR_DIR = ROOT / "data" / "irr_instructions"

GRADE_VALUES = {"pass", "fail", "unsure"}

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(color="FFFFFF", bold=True)
AVOID_FILL = PatternFill("solid", fgColor="FCE4D6")
ZEBRA_A = PatternFill("solid", fgColor="D9E1F2")
ZEBRA_B = PatternFill("solid", fgColor="FFFFFF")
INPUT_FILL = PatternFill("solid", fgColor="FFF7CC")

TIEBREAKER_COLS = [
    ("grading_id",            14),
    ("response_id",           10),
    ("prompt_id",             38),
    ("criterion_number",       8),
    ("is_avoid_item",         12),
    ("criterion_text",        60),
    ("reviewer_1_grade",      14),
    ("reviewer_2_grade",      14),
    ("reviewer_3_grade",      14),
    ("tiebreaker_id",         14),
    ("tiebreaker_grade",      14),
    ("tiebreaker_confidence", 12),
    ("tiebreaker_notes",      40),
    ("_prompt",               40),
    ("_response",             60),
]
INPUT_COLS = ("tiebreaker_id", "tiebreaker_grade", "tiebreaker_confidence", "tiebreaker_notes")


def find_reviewer_files() -> list[Path]:
    """All Grading-*.xlsx except the template + tiebreaker outputs."""
    out: list[Path] = []
    for p in sorted(IRR_DIR.glob("Grading-*.xlsx")):
        stem = p.stem.lower()
        if stem in ("grading-template", "grading-tiebreaker"):
            continue
        out.append(p)
    return out


def load_reviewer(path: Path) -> pd.DataFrame:
    """Read one reviewer's Grading sheet, normalize grade values."""
    df = pd.read_excel(path, sheet_name="Grading")
    needed = {"grading_id", "physician_grade", "physician_confidence",
              "physician_id", "notes", "_prompt", "_response",
              "is_avoid_item", "criterion_text", "criterion_number",
              "response_id", "prompt_id"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"{path.name}: missing columns {missing}")
    df["physician_grade"] = (
        df["physician_grade"].astype(str).str.strip().str.lower().replace({"nan": ""})
    )
    return df


def is_three_way_split(grades: list[str]) -> bool:
    """All three different, all in {pass, fail, unsure}."""
    if len(grades) != 3:
        return False
    if not all(g in GRADE_VALUES for g in grades):
        return False
    return len(set(grades)) == 3


def build_disputes(reviewers: list[pd.DataFrame]) -> pd.DataFrame:
    """Return one row per criterion where the 3 reviewers gave 3 different grades."""
    if len(reviewers) != 3:
        raise SystemExit(f"need exactly 3 reviewer files, got {len(reviewers)}")

    a, b, c = reviewers
    base = a[[
        "grading_id", "response_id", "prompt_id", "criterion_number",
        "is_avoid_item", "criterion_text", "_prompt", "_response",
    ]].copy()
    merged = (
        base
        .merge(a[["grading_id", "physician_grade"]].rename(columns={"physician_grade": "reviewer_1_grade"}), on="grading_id")
        .merge(b[["grading_id", "physician_grade"]].rename(columns={"physician_grade": "reviewer_2_grade"}), on="grading_id")
        .merge(c[["grading_id", "physician_grade"]].rename(columns={"physician_grade": "reviewer_3_grade"}), on="grading_id")
    )

    def is_split(row) -> bool:
        return is_three_way_split([
            row["reviewer_1_grade"],
            row["reviewer_2_grade"],
            row["reviewer_3_grade"],
        ])

    disputes = merged[merged.apply(is_split, axis=1)].copy()
    disputes["tiebreaker_id"] = ""
    disputes["tiebreaker_grade"] = ""
    disputes["tiebreaker_confidence"] = ""
    disputes["tiebreaker_notes"] = ""
    return disputes


def write_tiebreaker_workbook(disputes: pd.DataFrame, out_path: Path) -> None:
    """Two sheets: Tiebreaker + Instructions - Tiebreaker (verbatim)."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Tiebreaker"

    headers = [c[0] for c in TIEBREAKER_COLS]
    widths = {c[0]: c[1] for c in TIEBREAKER_COLS}

    for i, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=i, value=h)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(i)].width = widths[h]

    n_normal_seen = 0
    for row_idx, (_, r) in enumerate(disputes.iterrows(), 2):
        is_avoid = str(r.get("is_avoid_item", "")).strip().upper() == "YES"
        if is_avoid:
            row_fill = AVOID_FILL
        else:
            row_fill = ZEBRA_A if (n_normal_seen % 2 == 0) else ZEBRA_B
            n_normal_seen += 1
        for col_idx, h in enumerate(headers, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=r.get(h, ""))
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.fill = row_fill
        for col_idx, h in enumerate(headers, 1):
            if h in INPUT_COLS:
                ws.cell(row=row_idx, column=col_idx).fill = INPUT_FILL
        ws.row_dimensions[row_idx].height = 80

    ws.freeze_panes = "A2"

    # Data validation: tiebreaker_grade ∈ {pass, fail} only — no unsure here.
    grade_col = get_column_letter(headers.index("tiebreaker_grade") + 1)
    dv = DataValidation(
        type="list",
        formula1='"pass,fail"',
        allow_blank=True,
        showErrorMessage=True,
        errorTitle="Invalid grade",
        error="Adjudicator must choose pass or fail (no unsure). See Instructions tab.",
    )
    dv.add(f"{grade_col}2:{grade_col}{ws.max_row}")
    ws.add_data_validation(dv)

    conf_col = get_column_letter(headers.index("tiebreaker_confidence") + 1)
    dv_conf = DataValidation(
        type="whole",
        operator="between",
        formula1=1,
        formula2=3,
        allow_blank=True,
        showErrorMessage=True,
        errorTitle="Invalid confidence",
        error="Enter 1, 2, or 3.",
    )
    dv_conf.add(f"{conf_col}2:{conf_col}{ws.max_row}")
    ws.add_data_validation(dv_conf)

    # Instructions sheet — verbatim.
    instr_path = INSTR_DIR / "tiebreaker.json"
    rows = json.loads(instr_path.read_text())
    ws2 = wb.create_sheet("Instructions - Tiebreaker")
    ws2.column_dimensions["A"].width = 110
    for i, entry in enumerate(rows, 1):
        cell = ws2.cell(row=i, column=1, value=entry.get("text", ""))
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        if entry.get("bold"):
            cell.font = Font(bold=True, size=12)

    wb.save(out_path)


def main() -> None:
    files = find_reviewer_files()
    if len(files) != 3:
        raise SystemExit(
            f"Found {len(files)} reviewer files in {IRR_DIR}, expected exactly 3 "
            "(Grading-<name1>.xlsx, Grading-<name2>.xlsx, Grading-<name3>.xlsx). "
            "If reviewers are still grading, wait until all three return."
        )

    print("loading reviewer files:")
    reviewers = []
    for p in files:
        df = load_reviewer(p)
        scored = (df["physician_grade"].isin(GRADE_VALUES)).sum()
        print(f"  {p.name}: {len(df)} rows, {scored} scored")
        reviewers.append(df)

    disputes = build_disputes(reviewers)
    print(f"\nfound {len(disputes)} 3-way-split rows (need adjudicator)")

    if len(disputes) == 0:
        marker = IRR_DIR / "Grading-tiebreaker.NONE.md"
        marker.write_text(
            "# No tiebreaker rows\n\n"
            "All criteria had at least 2-of-3 agreement (or were unscored by "
            "any reviewer). No adjudicator round needed; majority vote "
            "resolves every criterion.\n"
        )
        print(f"  wrote marker: {marker}")
        return

    out_path = IRR_DIR / "Grading-tiebreaker.xlsx"
    write_tiebreaker_workbook(disputes, out_path)
    print(f"\nwrote {out_path}")
    print(f"  send to adjudicator (Felipe per issue #73, or whoever you assign)")


if __name__ == "__main__":
    main()
