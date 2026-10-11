"""Tests for the paper figure scripts (Figure 2 layout and byte-stable PDFs)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import plot_epistemic_figures as pef  # noqa: E402

CELL = "healthbench__medgemma-27b-text__bodhi__2026-05"
STATS = ROOT / "results" / CELL / "analysis" / "stats.json"


def _figure_input():
    stats = json.loads(STATS.read_text())
    return stats, pef.conditions_of(stats, "bodhi")


def test_figure_2_has_two_panels_with_their_own_axes(tmp_path):
    """FzpA-W3: 0-2 grades and percentage rates must not share one axis."""
    stats, conds = _figure_input()
    fig = pef.plot_epistemic_bar(stats, conds, tmp_path / "epistemic_bar.pdf")
    ax_a, ax_b = fig.axes
    assert ax_a.get_ylabel() == "Score (0–2)"
    assert ax_b.get_ylabel() == "Rate (%)"
    assert ax_a.get_ylim()[1] < 3
    assert ax_b.get_ylim()[1] > 100
    assert [t.get_text() for t in ax_b.get_xticklabels()][0] == "Active inquiry"
    # Every 0-2 metric is in panel A and every rate in panel B, by the analysis' own kinds.
    assert all(stats["metrics"][k]["kind"] == "score02" for k, _ in pef.PANEL_A_METRICS)
    assert all(stats["metrics"][k]["kind"] == "rate" for k, _ in pef.PANEL_B_METRICS)


def test_figure_2_draws_the_tables_seed_means(tmp_path):
    """GEN-8: the bars are the seed means of stats.json, the numbers the tables print."""
    stats, conds = _figure_input()
    fig = pef.plot_epistemic_bar(stats, conds, tmp_path / "epistemic_bar.pdf")
    heights = sorted(round(p.get_height(), 9) for p in fig.axes[1].patches)
    expected = sorted(round(stats["conditions"][k][c]["seed_mean"] * 100, 9)
                      for k, _ in pef.PANEL_B_METRICS for c, _, _ in conds)
    assert heights == expected
    assert [label for _, label, _ in conds] == ["Base", "Base + BODHI", "LoRA", "LoRA + BODHI"]


def test_figure_pdfs_are_byte_identical_across_runs(tmp_path):
    stats, conds = _figure_input()
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    pef.plot_epistemic_bar(stats, conds, a)
    pef.plot_epistemic_bar(stats, conds, b)
    assert a.read_bytes() == b.read_bytes()


def test_figure_builder_pins_build_env(monkeypatch):
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    from scripts import build_paper_figures as bpf
    env = bpf._build_env()
    assert env["SOURCE_DATE_EPOCH"] == "0"
    assert env["PYTHONHASHSEED"] == "0"


def test_forest_plot_draws_the_clustered_bootstrap_cis(tmp_path, monkeypatch):
    """D7: the forest plot shows the analysis' prompt-clustered CIs, not a normal approximation."""
    from scripts import plot_postprocessing as pp
    numbers = ROOT / "results" / "healthbench__medgemma-27b-text__bodhi__2026-05" / "analysis" / "stats.json"
    contrasts = json.loads(numbers.read_text())["contrasts"]
    drawn = []
    real_plot = pp.plt.Axes.plot

    def spy(self, xs, ys, *a, **k):
        drawn.append(tuple(xs))
        return real_plot(self, xs, ys, *a, **k)

    monkeypatch.setattr(pp.plt.Axes, "plot", spy)
    pp.plot_effect_sizes(numbers, tmp_path / "forest.pdf")
    assert (tmp_path / "forest.pdf").stat().st_size > 0
    rf = contrasts["lora+bodhi - lora"]["metrics"]["red_flag"]["d_ci95"]
    assert tuple(rf) in drawn
    # Three comparisons x seven dimensions.
    assert len(drawn) == 21
