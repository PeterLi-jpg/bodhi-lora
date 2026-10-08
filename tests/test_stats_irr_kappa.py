"""Tests for scripts/compute_irr_kappa.py on a small synthetic fixture.

The returned physician files stay outside the repo, so these tests build the same
spreadsheet layout (template, answer key, rater sheets) and a matching
results_modal-style grader output in a temp dir.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import compute_irr_kappa as irr  # noqa: E402

# Two prompts, one response each (seed 7, base_no_wrapper), four criteria each.
# points < 0 marks an avoid criterion; met is the grader's criteria_met.
CRITERIA = {
    "p1": [(5, True), (3, False), (-4, False), (-2, True)],
    "p2": [(4, True), (2, True), (-3, False), (6, False)],
}


def _build(tmp_path, rater_grades, extra_raters=()):
    results = tmp_path / "results"
    (results / "seed_7").mkdir(parents=True)
    out_results, sheet_rows, key_rows = [], [], []
    for r_i, (pid, crits) in enumerate(CRITERIA.items(), 1):
        cr = [{"criterion": f"{pid} criterion {j}", "points": pts, "criteria_met": met}
              for j, (pts, met) in enumerate(crits, 1)]
        out_results.append({"prompt_id": pid, "criteria_results": cr})
        for j, (pts, _) in enumerate(crits, 1):
            gid = f"R{r_i:03d}_c{j:02d}"
            avoid = "YES" if pts < 0 else "no"
            sheet_rows.append({"grading_id": gid, "response_id": f"R{r_i:03d}", "prompt_id": pid,
                               "criterion_number": j, "is_avoid_item": avoid,
                               "criterion_text": f"{pid} criterion {j}"})
            key_rows.append({"grading_id": gid, "response_id": f"R{r_i:03d}", "source": "eval_llama",
                             "seed": 7.0, "config": "base_no_wrapper", "criterion_number": j,
                             "is_avoid_item": avoid, "llm_score": 0.5})
    (results / "seed_7" / "base_no_wrapper.json").write_text(json.dumps({"results": out_results}))

    template = tmp_path / "Grading-template.xlsx"
    with pd.ExcelWriter(template) as w:
        pd.DataFrame(sheet_rows).assign(physician_grade=None).to_excel(w, sheet_name="Grading", index=False)
    key = tmp_path / "answer_key.xlsx"
    pd.DataFrame(key_rows).to_excel(key, index=False)

    files = []
    for k, grades in enumerate([rater_grades, *extra_raters]):
        f = tmp_path / f"Grading-person{k}.xlsx"
        with pd.ExcelWriter(f) as w:
            pd.DataFrame(sheet_rows).assign(physician_grade=grades).to_excel(w, sheet_name="Grading", index=False)
        files.append(f)
    return files, key, template, results


def test_avoid_rows_flip_and_kappa_matches_sklearn(tmp_path):
    sk = pytest.importorskip("sklearn.metrics")
    grades = ["pass", "fail", "pass", "pass", "pass", "pass", "unsure", "fail"]
    files, key, template, results = _build(tmp_path, grades)
    out = irr.run(files, key, template, results, None, None)

    # Expected LLM verdicts: positive -> met, avoid -> not met.
    llm = []
    for crits in CRITERIA.values():
        llm += ["pass" if (met != (pts < 0)) else "fail" for pts, met in crits]
    keep = [i for i, g in enumerate(grades) if g != "unsure"]
    phys = [grades[i] for i in keep]
    llm = [llm[i] for i in keep]
    block = out["llm_vs_physician"]["eval_llama"]["all"]
    assert block["n"] == 7
    assert block["kappa"] == pytest.approx(sk.cohen_kappa_score(phys, llm))
    assert block["agree"] == pytest.approx(np.mean([a == b for a, b in zip(phys, llm)]))
    assert block["n_prompts"] == 2
    # Three avoid rows, one graded unsure, so two count. R001_c04 is avoid with
    # criteria_met=True: the grader saw the harm, so the LLM verdict is "fail".
    neg = out["llm_vs_physician"]["eval_llama"]["negative"]
    assert neg["n"] == 2
    assert neg["llm_pass"] == pytest.approx(0.5)
    assert out["raters"] == ["rater_A"]
    assert out["inter_physician"]["pairs"] == []


def test_output_carries_no_file_names(tmp_path):
    files, key, template, results = _build(tmp_path, ["pass"] * 8)
    out = irr.run(files, key, template, results, None, None)
    text = json.dumps(out)
    assert "person0" not in text and "xlsx" not in text and str(tmp_path) not in text


def test_blank_rater_file_is_refused(tmp_path):
    files, key, template, results = _build(tmp_path, [None] * 8)
    with pytest.raises(SystemExit) as exc:
        irr.run(files, key, template, results, None, None)
    assert "blank template" in str(exc.value)
    assert "rater_A" in str(exc.value)


def test_consensus_rules():
    # One rater: every graded row counts, including a single vote.
    assert irr.derive_consensus([{"a": "pass", "b": "unsure"}], {}) == {"a": "pass", "b": "unsure"}
    # Three raters: majority wins; a three-way split goes to the adjudicator.
    raters = [{"a": "pass", "b": "pass", "c": "pass"},
              {"a": "pass", "b": "fail", "c": "fail"},
              {"a": "fail", "b": "unsure", "c": "fail"}]
    assert irr.derive_consensus(raters, {"b": "fail"}) == {"a": "pass", "b": "fail", "c": "fail"}
    assert "b" not in irr.derive_consensus(raters, {})


def test_two_raters_give_pairwise_kappa(tmp_path):
    a = ["pass", "fail", "pass", "pass", "pass", "pass", "fail", "fail"]
    b = ["pass", "fail", "fail", "pass", "pass", "pass", "fail", "pass"]
    files, key, template, results = _build(tmp_path, a, extra_raters=[b])
    out = irr.run(files, key, template, results, None, None)
    pair = out["inter_physician"]["pairs"][0]
    assert pair["pair"] == "rater_A-rater_B"
    assert pair["n"] == 8
    cm = irr._table(a, b)
    po = (cm[0, 0] + cm[1, 1]) / 8
    pe = (cm[0].sum() * cm[:, 0].sum() + cm[1].sum() * cm[:, 1].sum()) / 64
    assert pair["kappa"] == pytest.approx((po - pe) / (1 - pe))


def test_bootstrap_ci_brackets_the_estimate(tmp_path):
    grades = ["pass", "fail", "pass", "pass", "pass", "pass", "fail", "fail"]
    files, key, template, results = _build(tmp_path, grades)
    block = irr.run(files, key, template, results, None, None)["llm_vs_physician"]["eval_llama"]["all"]
    lo, hi = block["ci95"]
    assert lo <= block["kappa"] <= hi


def test_committed_kappa_results_are_deidentified():
    """The committed output names raters only by label and confirms the paper's 0.35 / n=509."""
    path = ROOT / "results_modal" / "irr" / "kappa_results.json"
    if not path.is_file():
        pytest.skip("kappa_results.json not committed yet")
    d = json.loads(path.read_text())
    assert d["raters"] == ["rater_A"]
    ev = d["llm_vs_physician"]["eval_llama"]["all"]
    assert ev["n"] == 509
    assert round(ev["kappa"], 2) == 0.35
    text = path.read_text().lower()
    assert "xlsx" not in text and "/users" not in text and "downloads" not in text
