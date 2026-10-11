"""Cross-cell statistics: the groups of configs/report.yaml select cells by their records,
and a group's summary counts directions and runs Holm over the group's cells."""
import pytest

pytest.importorskip("statsmodels")

from statsmodels.stats.multitest import multipletests  # noqa: E402

from bodhi_lora.analysis import report_config  # noqa: E402
from bodhi_lora.analysis.cross import group_summary, members  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    reg = load_registry(CONFIGS_DIR)
    cfg = report_config()
    return reg, cfg, [reg.cells[c] for c in cfg.cross_cells]


def test_groups_follow_the_records(setup):
    reg, cfg, cells = setup
    group = {gid: members(g, cells, reg) for gid, g in cfg.groups.items()}
    waived = {c.id for c in cells if c.grading.same_family_waiver}
    set_aside = {c.id for c in cells if c.report.set_aside}
    assert waived and set_aside
    assert not (set(group["outside-family"]) & (waived | set_aside))
    assert not (set(group["added-kept"]) & set_aside)
    assert set(group["set-aside"]) == set_aside
    assert set(group["added"]) == {c.id for c in cells if c.report.status == "complete"} - set(group["submitted"])


def _stats(diff, p, protocol="bodhi"):
    m = {"diff": diff, "p": p}
    contrasts = {gc.contrast.replace("{protocol}", protocol): {"metrics": {k: m for k in gc.metrics}}
                 for gc in report_config().group_contrasts}
    return {"contrasts": contrasts,
            "recipe": {"seeds_trained": [1, 2], "r": 16}, "filter": {"recorded": {"tau": 0.4}},
            "askers_gap": {"lora": {"gap": diff}}}


def test_a_new_complete_cell_joins_the_rule_groups_but_not_the_added_ones(setup):
    """A cell run later and put in cross_cells must not change what the paper says about
    the cells added in the 2026-07 discussion; it does join a statement about every cell."""
    reg, cfg, cells = setup
    new = reg.cells["healthbench__mistral-small-24b__bodhi"]
    new = new.model_copy(update={"report": new.report.model_copy(update={"status": "complete"})})
    group = {gid: members(g, cells + [new], reg) for gid, g in cfg.groups.items()}
    assert new.id in group["outside-family"]
    for gid in cfg.groups:
        if gid.startswith("added"):
            assert new.id not in group[gid], gid
    assert group["added"] == [c.id for c in cells if c.id != "healthbench__medgemma-27b-text__bodhi__2026-05"]


def test_group_summary_counts_directions_and_runs_holm_over_cells(setup):
    reg, cfg, _ = setup
    a, b, c = cfg.cross_cells[:3]
    all_stats = {a: _stats(0.2, 0.01), b: _stats(-0.1, 0.04), c: _stats(0.05, 0.5)}
    out = group_summary([a, b, c], all_stats, cfg, reg)
    gc = cfg.group_contrasts[0]
    r = out["contrasts"][gc.contrast][gc.metrics[0]]
    assert (r["n_positive"], r["n_negative"], r["n_measured"]) == (2, 1, 3)
    expected = multipletests([0.01, 0.04, 0.5], method="holm")[1]
    assert [r["p_holm"][x] for x in (a, b, c)] == pytest.approx(list(expected))
    assert out["recipe"]["adapters"] == 6 and out["recipe"]["r"] == 16 and out["recipe"]["tau"] == 0.4
    assert out["askers_gap"]["lora"] == {"n_measured": 3, "n_positive": 2, "n_negative": 1}


def test_a_group_contrast_reads_each_cells_own_protocol(setup):
    """"lora - base+{protocol}" is the adapter against its own teacher in every cell: a
    cell on another protocol is measured, not left out because its keys differ."""
    reg, cfg, _ = setup
    other = "healthbench__mistral-small-24b__ask-first"
    assert reg.cells[other].protocol != "bodhi"
    teacher = [gc for gc in cfg.group_contrasts if "{protocol}" in gc.contrast]
    assert teacher, "report.yaml has no group contrast against the cell's own teacher"
    all_stats = {other: _stats(0.1, 0.01, protocol=reg.cells[other].protocol)}
    out = group_summary([other], all_stats, cfg, reg)
    for gc in teacher:
        assert out["contrasts"][gc.contrast][gc.metrics[0]]["n_measured"] == 1, gc.contrast
