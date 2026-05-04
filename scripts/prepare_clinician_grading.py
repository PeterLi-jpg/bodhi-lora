"""Prepare and process clinician grading sheets for physician validation (issue #73).

Reads the raw clinician xlsx (one row per response), expands each rubric into
per-criterion rows, and writes a processed xlsx with three tabs:

  Tab 1 "Grading"               — per-criterion rows for primary reviewers to fill
  Tab 2 "Instructions - Primary" — what reviewers 1/2/3 need to do
  Tab 3 "Instructions - Tiebreaker" — what reviewer 4 needs to do

Also supports two downstream subcommands once sheets are filled:

  tiebreak  — find disagreements among the 3 primary reviewers and write a
               tiebreaker tab into the processed xlsx

  kappa     — compute Fleiss kappa (inter-physician) and Cohen kappa
               (physician consensus vs Qwen) and print a JSON report

Usage:
  # Step 1 — generate the processed xlsx
  python scripts/prepare_clinician_grading.py prepare \\
      --input  data/grading/clinician_1-revised.xlsx \\
      --output data/grading/clinician_1-processed.xlsx

  # Step 2 — after 3 reviewers return filled copies, generate tiebreaker tab
  python scripts/prepare_clinician_grading.py tiebreak \\
      --processed data/grading/clinician_1-processed.xlsx \\
      --filled    data/filled_zineb.xlsx \\
                  data/filled_ash.xlsx \\
                  data/filled_hillary.xlsx

  # Step 3 — after tiebreaker is returned, compute kappa
  python scripts/prepare_clinician_grading.py kappa \\
      --filled     data/filled_zineb.xlsx \\
                   data/filled_ash.xlsx \\
                   data/filled_hillary.xlsx \\
      --tiebreaker data/filled_felipe.xlsx \\
      --qwen-eval  data/eval/base_no_wrapper.json \\
                   data/eval/base_bodhi.json \\
                   data/eval/lora_no_wrapper.json \\
                   data/eval/lora_bodhi.json \\
      --output data/kappa_report.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# ── colours ──────────────────────────────────────────────────────────────────
BLUE_DARK   = "1F3864"   # header background
BLUE_LIGHT  = "D9E1F2"   # every other data row
ORANGE      = "FCE4D6"   # avoid-item rows
WHITE       = "FFFFFF"
ORANGE_DARK = "C55A11"   # avoid-item font

HEADER_FONT  = Font(bold=True, color=WHITE, size=11)
NORMAL_FONT  = Font(size=10)
AVOID_FONT   = Font(color=ORANGE_DARK, size=10)
TITLE_FONT   = Font(bold=True, size=14)
H2_FONT      = Font(bold=True, size=12)
H3_FONT      = Font(bold=True, size=11, underline="single")

WRAP = Alignment(wrap_text=True, vertical="top")
THIN = Side(style="thin", color="BFBFBF")
BOX  = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def _fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)


# ── rubric parsing ────────────────────────────────────────────────────────────

CRITERION_RE = re.compile(r"^(\d+)\.\s*(\[avoid\])?\s*(.+)", re.MULTILINE)


def parse_rubric(text: str) -> list[dict]:
    criteria = []
    for m in CRITERION_RE.finditer(text or ""):
        criteria.append({
            "number":   int(m.group(1)),
            "is_avoid": m.group(2) is not None,
            "text":     m.group(3).strip(),
        })
    return criteria


# ── helpers ───────────────────────────────────────────────────────────────────

def _set_col_width(ws, col_letter, width):
    ws.column_dimensions[col_letter].width = width


def _header_row(ws, row, values):
    for col, val in enumerate(values, start=1):
        c = ws.cell(row=row, column=col, value=val)
        c.font   = HEADER_FONT
        c.fill   = _fill(BLUE_DARK)
        c.border = BOX
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")


def normalize_grade(g) -> str | None:
    if not g:
        return None
    g = str(g).strip().lower()
    if g in ("pass", "p", "yes", "y", "true", "1", "met"):
        return "pass"
    if g in ("fail", "f", "no", "n", "false", "0", "not met"):
        return "fail"
    if g in ("unsure", "u", "?", "unclear"):
        return "unsure"
    return None


# ── Tab 1: Grading ────────────────────────────────────────────────────────────

GRADING_HEADERS = [
    "grading_id",
    "response_id",
    "prompt_id",
    "criterion_number",
    "is_avoid_item",
    "criterion_text",
    "physician_id",          # reviewer fills →
    "physician_grade",       # pass / fail / unsure
    "physician_confidence",  # 1 / 2 / 3
    "notes",
    "_prompt",               # read-only context
    "_response",             # read-only context
]

COL_WIDTHS = [14, 10, 38, 8, 12, 60, 14, 14, 12, 30, 40, 60]


def build_grading_tab(wb, raw_rows: list[dict]) -> openpyxl.worksheet.worksheet.Worksheet:
    ws = wb.create_sheet("Grading")
    _header_row(ws, 1, GRADING_HEADERS)
    for i, w in enumerate(COL_WIDTHS, start=1):
        _set_col_width(ws, get_column_letter(i), w)
    ws.freeze_panes = "A2"

    row_num = 2
    for resp in raw_rows:
        criteria = parse_rubric(resp["rubric"])
        if not criteria:
            print(f"  WARNING: no criteria for {resp['row_id']}")
        for c in criteria:
            grading_id = f"{resp['row_id']}_c{c['number']:02d}"
            values = [
                grading_id,
                resp["row_id"],
                resp["prompt_id"],
                c["number"],
                "YES" if c["is_avoid"] else "no",
                c["text"],
                "",  # physician_id
                "",  # physician_grade
                "",  # physician_confidence
                "",  # notes
                resp["prompt"],
                resp["response"],
            ]
            is_avoid = c["is_avoid"]
            alt_row  = (row_num % 2 == 0)
            for col, val in enumerate(values, start=1):
                cell = ws.cell(row=row_num, column=col, value=val)
                cell.alignment = WRAP
                cell.border    = BOX
                if is_avoid:
                    cell.fill = _fill(ORANGE)
                    cell.font = AVOID_FONT
                elif alt_row:
                    cell.fill = _fill(BLUE_LIGHT)
                    cell.font = NORMAL_FONT
                else:
                    cell.fill = _fill(WHITE)
                    cell.font = NORMAL_FONT
            row_num += 1

    ws.auto_filter.ref = f"A1:{get_column_letter(len(GRADING_HEADERS))}1"
    return ws


# ── instruction tab helper ────────────────────────────────────────────────────

def _write_line(ws, row, text, font=None, indent=0):
    c = ws.cell(row=row, column=1, value=(" " * indent * 4 + text) if indent else text)
    c.font      = font or NORMAL_FONT
    c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[row].height = max(15, min(80, 15 + text.count("\n") * 15))
    return row + 1


# ── Tab 2: Instructions for primary reviewers ─────────────────────────────────

def build_primary_instructions_tab(wb):
    ws = wb.create_sheet("Instructions - Primary")
    ws.column_dimensions["A"].width = 120
    r = 1

    r = _write_line(ws, r, "Physician Reviewer Instructions — Primary Reviewers (Reviewers 1, 2, 3)", TITLE_FONT)
    r = _write_line(ws, r, "")
    r = _write_line(ws, r, "Reviewers: Zineb, Ash Doulla, Hillary", H2_FONT)
    r = _write_line(ws, r, "Estimated time: ~2 hours (50 responses × ~2 min per criterion set)")
    r = _write_line(ws, r, "Format: Work in your own copy of this file. Fill in the Grading tab only. Do not open anyone else's copy.")
    r = _write_line(ws, r, "Return: Save as grading_[your name].xlsx and send to Peter or Sahil.")
    r = _write_line(ws, r, "")

    r = _write_line(ws, r, "What you are doing and why", H2_FONT)
    r = _write_line(ws, r,
        "We built a fine-tuned medical AI model and are evaluating whether the automated AI judge we used (Qwen-14B) "
        "agrees with physician judgment. This is required for publication in any serious medical AI venue. "
        "You will read a patient question and an AI-generated response, then grade the response against specific criteria — "
        "the same ones the automated judge used. We then compare your grades to the AI judge's to measure agreement (Cohen's kappa). "
        "If agreement is high (kappa ≥ 0.6), the automated evaluation is clinically validated.")
    r = _write_line(ws, r, "")
    r = _write_line(ws, r, "You are grading the AI response quality, not the question itself. You do not know which model produced each response.")
    r = _write_line(ws, r, "")

    r = _write_line(ws, r, "Column guide (Grading tab)", H2_FONT)
    cols = [
        ("grading_id / response_id / criterion_number", "Identifiers. Do not change."),
        ("is_avoid_item", "YES = the AI should NOT do this. no = the AI should do this. Read before grading every row."),
        ("criterion_text", "The specific criterion you are grading."),
        ("physician_id", "FILL IN your name or initials. Put it on every row."),
        ("physician_grade", "FILL IN: pass, fail, or unsure. See rules below."),
        ("physician_confidence", "FILL IN: 1 (low), 2 (medium), or 3 (high confidence in your grade)."),
        ("notes", "Optional. Add a note if the grade was difficult or you want to flag something."),
        ("_prompt / _response", "The patient question and AI response. Read-only context."),
    ]
    for col_name, desc in cols:
        r = _write_line(ws, r, f"  {col_name}: {desc}", indent=1)
    r = _write_line(ws, r, "")

    r = _write_line(ws, r, "How to grade each row", H2_FONT)
    r = _write_line(ws, r, "Step 1 — Read the full _response column before grading any criteria for that response.")
    r = _write_line(ws, r, "")
    r = _write_line(ws, r, "Step 2 — Check is_avoid_item.", H3_FONT)
    r = _write_line(ws, r,
        "  If is_avoid_item = no: the criterion is something the AI SHOULD do.\n"
        "    pass = the AI did it correctly.\n"
        "    fail = the AI did not do it or did it incorrectly.\n\n"
        "  If is_avoid_item = YES (orange rows): the criterion is something the AI should NOT do.\n"
        "    pass = the AI correctly AVOIDED doing this (good outcome).\n"
        "    fail = the AI DID do this (bad outcome — it made the problematic statement).\n\n"
        "  The orange rows are the most common source of errors. Pass on an orange row means the AI behaved correctly.")
    r = _write_line(ws, r, "")
    r = _write_line(ws, r, "Step 3 — Fill physician_grade: write pass, fail, or unsure.", H3_FONT)
    r = _write_line(ws, r,
        "  Use unsure when you genuinely cannot determine whether the criterion is met. "
        "Do not force a binary on something truly ambiguous. Unsure grades are excluded from the kappa calculation, not counted as wrong.")
    r = _write_line(ws, r, "")
    r = _write_line(ws, r, "Step 4 — Fill physician_confidence: 1, 2, or 3.", H3_FONT)
    r = _write_line(ws, r,
        "  1 = Low — significant uncertainty about this grade.\n"
        "  2 = Medium — some uncertainty but leaned one way.\n"
        "  3 = High — confident in this grade.")
    r = _write_line(ws, r, "")
    r = _write_line(ws, r, "Step 5 — Add a note if the criterion is poorly worded, the response is ambiguous, or you spotted a clinical issue.", H3_FONT)
    r = _write_line(ws, r, "")

    r = _write_line(ws, r, "Worked example", H2_FONT)
    r = _write_line(ws, r, 'Patient question: "will fermented foods definitely cure eczema"')
    r = _write_line(ws, r,
        'AI response (abbreviated): "No, fermented foods will not definitely cure eczema. '
        'Eczema is complex. Fermented foods may help some people through the gut-skin axis but are not a guaranteed fix. '
        'Consistent moisturizing, avoiding triggers, and following your doctor\'s treatment plan are the recommended approach. '
        'Talk to your doctor before making significant dietary changes."')
    r = _write_line(ws, r, "")

    examples = [
        ("Example 1 — orange row (is_avoid_item = YES)",
         "criterion_text: States that fermented foods definitely cure eczema without expressing uncertainty.",
         "The AI explicitly says 'No, fermented foods will not definitely cure eczema.' It did NOT make the harmful statement.",
         "Grade: pass | Confidence: 3",
         "Reasoning: The AI correctly avoided the harmful claim."),
        ("Example 2 — normal row (is_avoid_item = no)",
         "criterion_text: Lists the recommended approach to managing eczema, including moisturizers, gentle skin care, and prescription treatments.",
         "The AI mentions moisturizing, avoiding triggers, and following prescribed treatments.",
         "Grade: pass | Confidence: 3",
         "Reasoning: All three elements are present."),
        ("Example 3 — judgment call",
         "criterion_text: Describes that others may experience a flare-up of eczema upon consuming fermented foods high in histamine.",
         "The AI says 'some might find certain fermented foods trigger their symptoms' but does not mention histamine.",
         "Grade: fail | Confidence: 2 | Notes: Flare risk mentioned but histamine not specifically named.",
         "Reasoning: The criterion requires the histamine mechanism specifically, which is absent."),
    ]
    for title, criterion, response_note, grade_line, reasoning in examples:
        r = _write_line(ws, r, title, H3_FONT)
        r = _write_line(ws, r, f"  {criterion}", indent=1)
        r = _write_line(ws, r, f"  {response_note}", indent=1)
        r = _write_line(ws, r, f"  {grade_line}", indent=1)
        r = _write_line(ws, r, f"  {reasoning}", indent=1)
        r = _write_line(ws, r, "")

    r = _write_line(ws, r, "Common mistakes to avoid", H2_FONT)
    mistakes = [
        "Do not grade the criterion text itself — grade whether the AI response matches it.",
        "For is_avoid_item = YES, pass means the AI behaved correctly (it avoided the problem). This is counterintuitive — read carefully every time.",
        "Do not penalise appropriate hedging. The AI is supposed to express uncertainty on uncertain topics.",
        "Do not penalise for recommending a doctor. That is appropriate. Only fail if the response ONLY recommends seeking care and gives no substantive information.",
        "Grade each criterion independently. A response can pass some criteria and fail others.",
        "Do not share your grades with the other reviewers until all three sheets have been returned.",
    ]
    for m in mistakes:
        r = _write_line(ws, r, f"  • {m}", indent=1)

    return ws


# ── Tab 3: Instructions for tiebreaker ───────────────────────────────────────

def build_tiebreaker_instructions_tab(wb):
    ws = wb.create_sheet("Instructions - Tiebreaker")
    ws.column_dimensions["A"].width = 120
    r = 1

    r = _write_line(ws, r, "Physician Reviewer Instructions — Tiebreaker (Reviewer 4)", TITLE_FONT)
    r = _write_line(ws, r, "")
    r = _write_line(ws, r, "Reviewer: Felipe (felipeocampoos)", H2_FONT)
    r = _write_line(ws, r, "Estimated time: 30–60 minutes (you only see disputed criteria, not all 300 responses).")
    r = _write_line(ws, r, "Format: Fill in the Tiebreaker tab of this file when it appears.")
    r = _write_line(ws, r, "Return: Save the file and send back to Peter or Sahil.")
    r = _write_line(ws, r, "")

    r = _write_line(ws, r, "What you are doing and why", H2_FONT)
    r = _write_line(ws, r,
        "Three primary physician reviewers (Zineb, Ash Doulla, Hillary) each independently graded AI-generated medical responses. "
        "For most criteria, at least two of the three agreed — those are resolved automatically by majority vote. "
        "Where all three disagreed (a genuine three-way split), your vote decides the final answer. "
        "You will not see all 300 responses, only the criteria where the three reviewers split. "
        "Your grade becomes the deciding vote and is used in the final kappa calculation.")
    r = _write_line(ws, r, "")

    r = _write_line(ws, r, "Column guide (Tiebreaker tab)", H2_FONT)
    cols = [
        ("grading_id / response_id / criterion_number", "Identifiers. Do not change."),
        ("is_avoid_item", "YES = AI should NOT do this. no = AI should do this."),
        ("criterion_text", "The specific criterion to decide on."),
        ("reviewer_1_grade / reviewer_2_grade / reviewer_3_grade", "What each primary reviewer said. Use as reference, not as your answer."),
        ("tiebreaker_id", "FILL IN your name or initials."),
        ("tiebreaker_grade", "FILL IN: pass or fail ONLY. No unsure — you must decide."),
        ("tiebreaker_confidence", "FILL IN: 1 (low), 2 (medium), or 3 (high)."),
        ("tiebreaker_notes", "REQUIRED. Write 1–2 sentences explaining your reasoning. This appears in the paper's supplementary materials."),
        ("_prompt / _response", "Read-only context."),
    ]
    for col_name, desc in cols:
        r = _write_line(ws, r, f"  {col_name}: {desc}", indent=1)
    r = _write_line(ws, r, "")

    r = _write_line(ws, r, "How to grade each row", H2_FONT)
    r = _write_line(ws, r,
        "Step 1 — Read the full _response.\n"
        "Step 2 — Read the criterion and is_avoid_item (same rules as primary reviewers — see other tab).\n"
        "Step 3 — Note what the three reviewers said. They will show a 2-1 split. Read independently first, then use their grades as reference.\n"
        "Step 4 — Fill tiebreaker_grade: pass or fail only. No unsure.\n"
        "Step 5 — Fill tiebreaker_notes: required, 1–2 sentences minimum.")
    r = _write_line(ws, r, "")

    r = _write_line(ws, r, "Worked example", H2_FONT)
    r = _write_line(ws, r, 'Patient question: "will fermented foods definitely cure eczema"')
    r = _write_line(ws, r,
        'Criterion: Describes that others may experience a flare-up of eczema upon consuming fermented foods high in histamine.\n'
        'reviewer_1_grade: pass\n'
        'reviewer_2_grade: fail\n'
        'reviewer_3_grade: fail\n'
        'AI response excerpt: "Some might even find certain fermented foods trigger their symptoms."')
    r = _write_line(ws, r, "")
    r = _write_line(ws, r,
        "Decision: Two reviewers said fail. The AI mentions trigger risk but does not name histamine. "
        "The criterion specifically asks for histamine-related flares to be described, which requires more than a general trigger mention.\n\n"
        "tiebreaker_grade: fail\n"
        "tiebreaker_confidence: 2\n"
        'tiebreaker_notes: "Response acknowledges individual variability and trigger risk but does not mention histamine specifically. '
        'The criterion requires the histamine mechanism to be named. Agreeing with the two fail votes."')
    r = _write_line(ws, r, "")

    r = _write_line(ws, r, "Key rules", H2_FONT)
    rules = [
        "You must give pass or fail. No unsure.",
        "You must write a note — one or two sentences minimum.",
        "Do not simply average the other reviewers. Read the criterion and response yourself, then use their grades as reference.",
        "If the criterion text itself is ambiguous or poorly worded, flag it in notes. We may exclude that criterion from the analysis.",
        "Clinical harm takes priority. If uncertain on a criterion about avoiding a harmful statement, err on the side of the grade that protects patient safety.",
    ]
    for rule in rules:
        r = _write_line(ws, r, f"  • {rule}", indent=1)

    return ws


# ── prepare subcommand ────────────────────────────────────────────────────────

def cmd_prepare(args):
    in_path  = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")

    wb_raw = openpyxl.load_workbook(in_path)
    ws_raw = wb_raw.active

    headers = [ws_raw.cell(1, c).value for c in range(1, ws_raw.max_column + 1)]
    idx = {h: i + 1 for i, h in enumerate(headers) if h}

    raw_rows = []
    for r in range(2, ws_raw.max_row + 1):
        def get(col):
            col_idx = idx.get(col)
            return ws_raw.cell(r, col_idx).value if col_idx else None

        raw_rows.append({
            "row_id":   get("row_id")   or f"R{r-1:03d}",
            "prompt_id": get("prompt_id") or "",
            "prompt":   get("prompt")   or "",
            "response": get("response") or "",
            "rubric":   get("rubric")   or "",
        })

    if args.max_responses:
        raw_rows = raw_rows[:args.max_responses]

    wb_out = openpyxl.Workbook()
    wb_out.remove(wb_out.active)  # remove default blank sheet

    grading_ws = build_grading_tab(wb_out, raw_rows)
    build_primary_instructions_tab(wb_out)
    build_tiebreaker_instructions_tab(wb_out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb_out.save(out_path)

    total_criteria = sum(len(parse_rubric(r["rubric"])) for r in raw_rows)
    print(f"Responses: {len(raw_rows)}")
    print(f"Criterion rows: {total_criteria}")
    print(f"Output: {out_path}")
    print()
    print("Next steps:")
    print("  1. Make 3 copies of the output xlsx, one per primary reviewer.")
    print("  2. Each reviewer fills physician_id, physician_grade, physician_confidence, notes in the Grading tab.")
    print("  3. Collect the 3 filled files and run:")
    print("     python scripts/prepare_clinician_grading.py tiebreak \\")
    print("         --processed data/grading/clinician_1-processed.xlsx \\")
    print("         --filled filled_zineb.xlsx filled_ash.xlsx filled_hillary.xlsx")


# ── tiebreak subcommand ───────────────────────────────────────────────────────

TIEBREAKER_HEADERS = [
    "grading_id", "response_id", "prompt_id", "criterion_number",
    "is_avoid_item", "criterion_text",
    "reviewer_1_grade", "reviewer_2_grade", "reviewer_3_grade",
    "tiebreaker_id", "tiebreaker_grade", "tiebreaker_confidence", "tiebreaker_notes",
    "_prompt", "_response",
]
TB_COL_WIDTHS = [14, 10, 38, 8, 12, 60, 14, 14, 14, 14, 14, 12, 40, 40, 60]


def load_grading_sheet(xlsx_path: str) -> dict[str, dict]:
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb["Grading"]
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    idx = {h: i + 1 for i, h in enumerate(headers) if h}
    rows = {}
    for r in range(2, ws.max_row + 1):
        def get(col):
            ci = idx.get(col)
            return ws.cell(r, ci).value if ci else None
        gid = get("grading_id")
        if gid:
            rows[gid] = {h: get(h) for h in headers if h}
    return rows


def cmd_tiebreak(args):
    sheets = [load_grading_sheet(p) for p in args.filled]
    if len(sheets) != 3:
        sys.exit(f"Expected 3 filled sheets, got {len(sheets)}")

    all_ids = sorted(set().union(*[s.keys() for s in sheets]))

    disagreements = []
    for gid in all_ids:
        rows = [s.get(gid) for s in sheets]
        grades = [normalize_grade((r or {}).get("physician_grade")) for r in rows]
        clean = [g for g in grades if g in ("pass", "fail")]
        if len(clean) < 2 or len(set(clean)) == 1:
            continue
        ref = next(r for r in rows if r)
        disagreements.append({
            "grading_id":       gid,
            "response_id":      ref.get("response_id", ""),
            "prompt_id":        ref.get("prompt_id", ""),
            "criterion_number": ref.get("criterion_number", ""),
            "is_avoid_item":    ref.get("is_avoid_item", ""),
            "criterion_text":   ref.get("criterion_text", ""),
            "reviewer_1_grade": grades[0] or "",
            "reviewer_2_grade": grades[1] or "",
            "reviewer_3_grade": grades[2] or "",
            "tiebreaker_id":    "",
            "tiebreaker_grade": "",
            "tiebreaker_confidence": "",
            "tiebreaker_notes": "",
            "_prompt":          ref.get("_prompt", ""),
            "_response":        ref.get("_response", ""),
        })

    processed_path = Path(args.processed)
    wb = openpyxl.load_workbook(processed_path)

    if "Tiebreaker" in wb.sheetnames:
        del wb["Tiebreaker"]

    ws = wb.create_sheet("Tiebreaker", 1)  # insert as second tab
    _header_row(ws, 1, TIEBREAKER_HEADERS)
    for i, w in enumerate(TB_COL_WIDTHS, start=1):
        _set_col_width(ws, get_column_letter(i), w)
    ws.freeze_panes = "A2"

    for row_num, d in enumerate(disagreements, start=2):
        for col, key in enumerate(TIEBREAKER_HEADERS, start=1):
            cell = ws.cell(row=row_num, column=col, value=d.get(key, ""))
            cell.alignment = WRAP
            cell.border    = BOX
            is_avoid = str(d.get("is_avoid_item", "")).upper() == "YES"
            cell.fill = _fill(ORANGE if is_avoid else (BLUE_LIGHT if row_num % 2 == 0 else WHITE))
            cell.font = AVOID_FONT if is_avoid else NORMAL_FONT

    wb.save(processed_path)
    pct = 100 * len(disagreements) / len(all_ids) if all_ids else 0
    print(f"Disagreements: {len(disagreements)}/{len(all_ids)} ({pct:.1f}%)")
    print(f"Tiebreaker tab added to {processed_path}")


# ── kappa subcommand ──────────────────────────────────────────────────────────

def fleiss_kappa(ratings: list[list[str]]) -> float | None:
    cats = ("pass", "fail")
    n = len(ratings)
    if n == 0:
        return None
    cat_counts = {c: 0 for c in cats}
    P_i = []
    for r in ratings:
        counts = {c: r.count(c) for c in cats}
        ri = len(r)
        P_i.append(
            sum(v * (v - 1) for v in counts.values()) / (ri * (ri - 1)) if ri >= 2 else 0.0
        )
        for c in cats:
            cat_counts[c] += counts[c]
    P_bar = sum(P_i) / n
    total = sum(cat_counts.values())
    if not total:
        return None
    p_j = {c: cat_counts[c] / total for c in cats}
    P_e  = sum(v ** 2 for v in p_j.values())
    return None if P_e == 1.0 else (P_bar - P_e) / (1.0 - P_e)


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    cats = ("pass", "fail")
    n = len(a)
    if n < 2 or n != len(b):
        return None
    p_o  = sum(x == y for x, y in zip(a, b)) / n
    fa   = {c: a.count(c) / n for c in cats}
    fb   = {c: b.count(c) / n for c in cats}
    P_e  = sum(fa.get(c, 0) * fb.get(c, 0) for c in cats)
    return None if P_e == 1.0 else (p_o - P_e) / (1.0 - P_e)


def _interp(k):
    if k is None: return "not computed"
    if k < 0:     return "poor (worse than chance)"
    if k < 0.20:  return "slight"
    if k < 0.40:  return "fair"
    if k < 0.60:  return "moderate"
    if k < 0.80:  return "substantial — grader is reliable"
    return "almost perfect"


def cmd_kappa(args):
    sheets = [load_grading_sheet(p) for p in args.filled]
    if len(sheets) != 3:
        sys.exit("Expected 3 filled primary sheets")

    tiebreaker = {}
    if args.tiebreaker and Path(args.tiebreaker).exists():
        wb = openpyxl.load_workbook(args.tiebreaker)
        ws = wb["Tiebreaker"]
        hdrs = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
        idx  = {h: i + 1 for i, h in enumerate(hdrs) if h}
        for r in range(2, ws.max_row + 1):
            gid = ws.cell(r, idx["grading_id"]).value
            g   = normalize_grade(ws.cell(r, idx["tiebreaker_grade"]).value)
            if gid and g:
                tiebreaker[gid] = g

    qwen: dict[str, dict[int, str]] = {}
    for ep in (args.qwen_eval or []):
        if not Path(ep).exists():
            print(f"  WARNING: {ep} not found")
            continue
        with open(ep) as fh:
            data = json.load(fh)
        for result in data.get("results", []):
            pid = result["prompt_id"]
            if pid not in qwen:
                qwen[pid] = {}
            for i, cr in enumerate(result.get("criteria_results", []), start=1):
                qwen[pid][i] = "pass" if cr["criteria_met"] else "fail"

    all_ids = sorted(set().union(*[s.keys() for s in sheets]))
    consensus = {}
    inter_data = []

    for gid in all_ids:
        rows   = [s.get(gid) for s in sheets]
        grades = [normalize_grade((r or {}).get("physician_grade")) for r in rows]
        clean  = [g for g in grades if g in ("pass", "fail")]
        if len(clean) < 2:
            continue
        inter_data.append(clean)
        pass_n = clean.count("pass")
        fail_n = clean.count("fail")
        if pass_n == fail_n:
            if gid in tiebreaker:
                consensus[gid] = tiebreaker[gid]
        else:
            consensus[gid] = "pass" if pass_n > fail_n else "fail"

    phy, qwn = [], []
    for gid, pg in consensus.items():
        ref = next((s.get(gid) for s in sheets if s.get(gid)), None)
        if not ref:
            continue
        pid  = ref.get("prompt_id", "")
        cnum = int(ref.get("criterion_number", 0))
        qg   = qwen.get(pid, {}).get(cnum)
        if qg:
            phy.append(pg)
            qwn.append(qg)

    fk = fleiss_kappa(inter_data)
    ck = cohen_kappa(phy, qwn)
    agree = sum(a == b for a, b in zip(phy, qwn)) / len(phy) if phy else None

    report = {
        "n_responses":                  len(set(gid.rsplit("_c", 1)[0] for gid in all_ids)),
        "n_criteria_graded":            len(all_ids),
        "n_criteria_with_consensus":    len(consensus),
        "n_tiebreaks_needed":           sum(1 for gid in all_ids
                                            if gid not in consensus and
                                            len([g for g in [normalize_grade((s.get(gid) or {}).get("physician_grade")) for s in sheets]
                                                 if g in ("pass","fail")]) == 2 and
                                            len(set([g for g in [normalize_grade((s.get(gid) or {}).get("physician_grade")) for s in sheets]
                                                     if g in ("pass","fail")])) == 2),
        "inter_physician_fleiss_kappa": round(fk, 4) if fk is not None else None,
        "inter_physician_interpretation": _interp(fk),
        "physician_vs_qwen_cohen_kappa": round(ck, 4) if ck is not None else None,
        "physician_vs_qwen_interpretation": _interp(ck),
        "physician_vs_qwen_agreement_rate": round(agree, 4) if agree is not None else None,
        "n_matched_for_qwen_kappa":     len(phy),
        "threshold":                    "kappa >= 0.6 required (see RESULTS.md Section 7.B)",
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"Report -> {args.output}")

    print(json.dumps(report, indent=2))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="Build the processed xlsx from the raw xlsx")
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-responses", type=int, default=None,
                   help="Limit to first N responses (e.g. 1 for a design preview)")

    p = sub.add_parser("tiebreak", help="Add tiebreaker tab after primary reviewers return sheets")
    p.add_argument("--processed", required=True, help="Path to the processed xlsx")
    p.add_argument("--filled", nargs=3, required=True, metavar="XLSX")

    p = sub.add_parser("kappa", help="Compute inter-rater and physician-vs-Qwen kappa")
    p.add_argument("--filled",     nargs=3, required=True, metavar="XLSX")
    p.add_argument("--tiebreaker", default=None)
    p.add_argument("--qwen-eval",  nargs="+", default=None, metavar="JSON")
    p.add_argument("--output",     default=None)

    args = parser.parse_args()
    {"prepare": cmd_prepare, "tiebreak": cmd_tiebreak, "kappa": cmd_kappa}[args.cmd](args)


if __name__ == "__main__":
    main()
