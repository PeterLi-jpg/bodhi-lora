"""The report of the uniform analysis (bodhi_lora.report) on the committed stats.json
files: macro names come from each cell's record, every value comes from stats.json, the
cross-cell tables have the same columns for every cell, a value a cell lacks is "--",
never a number, and paper/generated/ is what the report writes."""
import json
import re

import pytest

pytest.importorskip("pandas")

from bodhi_lora.analysis import analyzable, report_config  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR, REPO_ROOT, project  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402
from bodhi_lora.report import Report, report, write  # noqa: E402
from bodhi_lora.report.cross import cross_label  # noqa: E402
from bodhi_lora.report.latex import fmt, fmt_pm, fmt_range, token  # noqa: E402

CROSS_TABLE = "tab_cross_cell_checks"
GENERATED = REPO_ROOT / "paper" / "generated"


@pytest.fixture(scope="module")
def setup():
    reg, cfg, proj = load_registry(CONFIGS_DIR), report_config(), project(CONFIGS_DIR)
    if not all((proj.cell(c).analysis() / "stats.json").is_file() for c in cfg.cross_cells):
        pytest.skip("the cells' stats.json are not in this checkout")
    return reg, cfg, proj, report(reg, cfg, proj)


def _stats(proj, cid):
    return json.loads((proj.cell(cid).analysis() / "stats.json").read_text())


def test_range_prints_both_ends_of_an_asymmetric_interval():
    assert fmt_range(0.1, [-0.005, 0.201], 3) == "\\ensuremath{+0.100\\;[-0.005,\\,+0.201]}"
    assert fmt_range(0.1, [-0.005, 0.201], 3, stack=True) == \
        "\\shortstack[r]{\\ensuremath{+0.100}\\\\\\ensuremath{[-0.005,\\,+0.201]}}"
    assert fmt_range(0.1, None, 3) == "\\ensuremath{+0.100}"
    assert fmt_range(None, [0, 1], 3) == "--"


def test_cross_label_splits_benchmark_from_model():
    assert cross_label("ChatDoctor, Med42-8B", "$^{\\dagger}$") == \
        "\\shortstack[l]{ChatDoctor,\\\\Med42-8B$^{\\dagger}$}"
    assert cross_label("smoke_cell") == "smoke\\_cell"


def test_tokens():
    assert token("needs-info-theme") == "NeedsInfoTheme"
    assert token("pass1-headers") == "PassOneHeaders"
    assert token("red_flag_rate") == "RedFlagRate"


def test_macro_names_are_letters_unique_and_named_by_the_cell(setup):
    reg, cfg, proj, rep = setup
    names = [m[0] for m in rep.macros]
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[A-Za-z]+", n) for n in names)
    tokens = {reg.cells[c].report.token for c in rep.record["cells"]}
    tokens |= {"Group", "Posted", "CrossCells"} | {g.token for g in cfg.grader_comparisons}
    assert all(any(n.startswith(t) for t in tokens) for n in names)
    # every macro names its source in stats.json
    assert all(src for _, _, src in rep.macros)


def test_every_analyzed_cell_gets_a_table(setup):
    reg, cfg, proj, rep = setup
    have = [c for c in analyzable(reg, cfg) if (proj.cell(c).analysis() / "stats.json").is_file()]
    assert sorted(rep.record["cells"]) == sorted(have)
    for c in have:
        assert f"tab_cell_{reg.cells[c].report.token}" in rep.tables


def test_values_come_from_stats(setup):
    reg, cfg, proj, rep = setup
    macros = {n: v for n, v, _ in rep.macros}
    for cid in cfg.cross_cells:
        s, t = _stats(proj, cid), reg.cells[cid].report.token
        r = s["conditions"]["active_inquiry"]["lora"]
        if r["seed_mean"] is not None:
            assert macros[f"{t}ActiveInquiryLoRA"] == fmt(r["seed_mean"] * 100, 1)
        d = s["contrasts"]["lora - base"]["metrics"]["rubric_score"]
        assert macros[f"{t}RubricScoreLoRAvsBaseDiff"] == fmt(d["diff"], 3, sign=True)
        over = s["discrimination"]["red_flag_rate"]["emergency-theme"]
        name = f"{t}RedFlagRateEmergencyThemeNegRateLoRA"
        if "not_measured" in over:
            assert name not in macros
        else:
            assert macros[name] == fmt(over["conditions"]["lora"]["negative_rate"] * 100, 1)


def test_cross_table_has_the_same_columns_for_every_cell(setup):
    reg, cfg, proj, rep = setup
    text = rep.tables[CROSS_TABLE]
    body = text.split("\\midrule", 1)[1]
    rows = [line for line in body.splitlines() if line.endswith("\\\\")]
    assert len(rows) == len(cfg.cross_cells)
    columns = cfg.cross_tables[CROSS_TABLE].columns
    assert {line.count(" & ") for line in rows} == {len(columns)}
    for cid in cfg.cross_cells:
        s = _stats(proj, cid)
        line = next(r for r in rows if r.startswith(cross_label(reg.cells[cid].report.label)[:-1]))
        cells = line.removesuffix(" \\\\").split(" & ")[1:]
        # The red-flag rate outside the emergency theme: a seed-level t interval, plus or
        # minus its half width; "--" where the benchmark has no emergency-theme label.
        over = s["discrimination"]["red_flag_rate"]["emergency-theme"]
        i = next(i for i, c in enumerate(columns) if c.value.startswith("discrimination.red_flag_rate"))
        if "not_measured" in over:
            assert cells[i] == "--"
        else:
            t = over["contrasts"]["lora - base"]["negative_rate"]["seed_test"]
            assert cells[i] == fmt_pm(t["diff"] * 100, [x * 100 for x in t["ci95"]], 1)
        # The askers gap: a bootstrap interval, printed with both ends.
        i = next(i for i, c in enumerate(columns) if c.value == "askers_gap.lora.gap")
        g = s["askers_gap"]["lora"]
        assert cells[i] == fmt_range(g["gap"], g["gap_ci95"], 3, sign=True, stack=True)


def test_a_set_aside_cell_states_why(setup):
    reg, cfg, proj, rep = setup
    for cid in rep.record["cells"]:
        cell = reg.cells[cid]
        if cell.report.set_aside:
            table = rep.tables[f"tab_cell_{cell.report.token}"]
            assert "set aside: " in table and cell.report.label in table


def test_report_is_deterministic(setup):
    reg, cfg, proj, rep = setup
    again = report(reg, cfg, proj)
    assert again.macros == rep.macros and again.tables == rep.tables


def test_table_one_has_a_row_for_every_cross_cell(setup):
    """Table 1 (tab_cross_cell) comes from the report config, so a cell put in cross_cells
    gets its row with no code change (the report step was not general before 2026-10-10)."""
    reg, cfg, proj, rep = setup
    text = rep.tables["tab_cross_cell"]
    rows = [line for line in text.split("\\midrule", 1)[1].splitlines() if line.endswith("\\\\")]
    assert len(rows) == len(cfg.cross_cells)
    width = len(cfg.cross_tables["tab_cross_cell"].columns) + len(cfg.cross_tables["tab_cross_cell"].label_columns)
    assert {line.count(" & ") + 1 for line in rows} == {width}


def test_the_committed_paper_numbers_are_what_the_report_writes(setup, tmp_path):
    """paper/generated/ is current: a fresh report writes the same bytes, and no other
    tab_*.tex is left over. CI checks the same with make numbers and git diff."""
    reg, cfg, proj, rep = setup
    written = write(rep, tmp_path)
    for name in written:
        assert (tmp_path / name).read_bytes() == (GENERATED / name).read_bytes(), name
    assert sorted(p.name for p in GENERATED.glob("tab_*.tex")) == sorted(n for n in written if n.startswith("tab_"))


def test_every_macro_names_its_source_and_no_absolute_path(setup, tmp_path):
    reg, cfg, proj, rep = setup
    write(rep, tmp_path)
    for path in tmp_path.iterdir():
        text = path.read_text()
        assert "/Users/" not in text and "/home/" not in text, path.name
    for line in (tmp_path / "numbers.tex").read_text().splitlines():
        if line.startswith("\\newcommand"):
            assert re.search(r"\} % \S", line), line


def test_no_paper_number_comes_from_a_keyword_analysis(setup):
    """The three discussion scripts decide meaning by keywords (CONVENTIONS.md rule 15). Their
    outputs are a record of withdrawn values (withdrawals W23 to W25): the report reads none
    of them, and no package code names them."""
    reg, cfg, proj, rep = setup
    # Their output files; a posted value the paper cites where it withdraws it names the
    # posted record (\\statPosted...), not these files.
    record = ("mimicry_split.json", "interference_mechanism.json", "second_person_questions.json",
              "rebuttal/analyses")
    assert not any(r in src for _, _, src in rep.macros for r in record)
    assert not any(r in path for path in rep.record["inputs"] for r in record)
    for path in sorted((REPO_ROOT / "bodhi_lora").rglob("*.py")):
        text = path.read_text()
        assert not any(r in text for r in record), path


def test_a_macro_name_is_letters_and_made_once():
    rep = Report()
    with pytest.raises(ValueError):
        rep.add("Seed7Mean", "1", "x")
    rep.add("Ok", "1", "x")
    with pytest.raises(ValueError):
        rep.add("Ok", "2", "x")
    rep.add("Missing", "--", "x")
    assert [m[0] for m in rep.macros] == ["Ok"]
