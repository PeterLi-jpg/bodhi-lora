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


def _stats(diff, p):
    m = {"diff": diff, "p": p}
    contrasts = {gc.contrast: {"metrics": {k: m for k in gc.metrics}} for gc in report_config().group_contrasts}
    return {"contrasts": contrasts,
            "recipe": {"seeds_trained": [1, 2], "r": 16}, "filter": {"recorded": {"tau": 0.4}},
            "askers_gap": {"lora": {"gap": diff}}}


def test_group_summary_counts_directions_and_runs_holm_over_cells(setup):
    _, cfg, _ = setup
    all_stats = {"a": _stats(0.2, 0.01), "b": _stats(-0.1, 0.04), "c": _stats(0.05, 0.5)}
    out = group_summary(["a", "b", "c"], all_stats, cfg)
    gc = cfg.group_contrasts[0]
    r = out["contrasts"][gc.contrast][gc.metrics[0]]
    assert (r["n_positive"], r["n_negative"], r["n_measured"]) == (2, 1, 3)
    expected = multipletests([0.01, 0.04, 0.5], method="holm")[1]
    assert [r["p_holm"][c] for c in "abc"] == pytest.approx(list(expected))
    assert out["recipe"]["adapters"] == 6 and out["recipe"]["r"] == 16 and out["recipe"]["tau"] == 0.4
    assert out["askers_gap"]["lora"] == {"n_measured": 3, "n_positive": 2, "n_negative": 1}
