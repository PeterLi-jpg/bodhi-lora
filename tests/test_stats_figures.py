"""Tests for the paper figure scripts (Figure 2 layout and byte-stable PDFs)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import plot_epistemic_figures as pef  # noqa: E402

AGG = ROOT / "results_modal" / "cross_seed_aggregate.json"


def test_figure_2_has_two_panels_with_their_own_axes(tmp_path):
    """FzpA-W3: 0-2 grades and percentage rates must not share one axis."""
    ep = json.loads(AGG.read_text())["stage5_epistemic"]
    fig = pef.plot_epistemic_bar(ep, tmp_path / "epistemic_bar.pdf")
    ax_a, ax_b = fig.axes
    assert ax_a.get_ylabel() == "Score (0–2)"
    assert ax_b.get_ylabel() == "Rate (%)"
    assert ax_a.get_ylim()[1] < 3
    assert ax_b.get_ylim()[1] > 100
    assert [t.get_text() for t in ax_b.get_xticklabels()][0] == "Active inquiry"
    # Every 0-2 metric is in panel A and every rate in panel B.
    assert all(k.endswith("_mean") for k, _ in pef.PANEL_A_METRICS)
    assert all(k.endswith("_rate") for k, _ in pef.PANEL_B_METRICS)


def test_figure_pdfs_are_byte_identical_across_runs(tmp_path):
    ep = json.loads(AGG.read_text())["stage5_epistemic"]
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    pef.plot_epistemic_bar(ep, a)
    pef.plot_epistemic_bar(ep, b)
    assert a.read_bytes() == b.read_bytes()


def test_figure_builder_pins_build_env(monkeypatch):
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    from scripts import build_paper_figures as bpf
    env = bpf._build_env()
    assert env["SOURCE_DATE_EPOCH"] == "0"
    assert env["PYTHONHASHSEED"] == "0"
