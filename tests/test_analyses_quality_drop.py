"""Tests for rebuttal/analyses/quality_drop.py on a synthetic cells tree.

The real cells live on the shared H100 box (PROV-1). These fixtures mirror the
layout rebuttal/launch/run_cell.sh writes, so the script runs unchanged once the
data is synced.
"""

import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "quality_drop", ROOT / "rebuttal" / "analyses" / "quality_drop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_cell(root, name, base, lora, extra_seed_without_lora=False):
    for i, (b, l) in enumerate(zip(base, lora)):
        ev = root / name / f"seed_{[7, 13, 42, 99, 101][i]}" / "eval"
        ev.mkdir(parents=True)
        (ev / "base_no_wrapper.json").write_text(json.dumps({"mean": b}))
        (ev / "lora_no_wrapper.json").write_text(json.dumps({"mean": l}))
    if extra_seed_without_lora:
        ev = root / name / "seed_5" / "eval"
        ev.mkdir(parents=True)
        (ev / "base_no_wrapper.json").write_text(json.dumps({"mean": 0.9}))


def test_paired_t_and_holm_match_hand_computation(tmp_path, monkeypatch):
    qd = _load()
    base_a = [0.42, 0.43, 0.41, 0.44, 0.42]
    lora_a = [0.40, 0.42, 0.40, 0.42, 0.41]
    base_b = [0.50, 0.52, 0.49, 0.51, 0.50]
    lora_b = [0.50, 0.51, 0.50, 0.50, 0.51]
    _write_cell(tmp_path / "cells", "healthbench__a", base_a, lora_a, extra_seed_without_lora=True)
    _write_cell(tmp_path / "cells", "chatdoctor__b", base_b, lora_b)
    out = tmp_path / "quality_drop.json"
    monkeypatch.setattr(sys, "argv", ["quality_drop.py", "--cells-root", str(tmp_path / "cells"),
                                      "--out", str(out)])
    qd.main()
    cells = json.loads(out.read_text())["cells"]

    # Seed 5 has no LoRA output, so it is left out rather than paired with nothing.
    assert cells["healthbench__a"]["seeds"] == [7, 13, 42, 99, 101]
    d = [l - b for b, l in zip(base_a, lora_a)]
    t = statistics.mean(d) / (statistics.stdev(d) / math.sqrt(5))
    assert cells["healthbench__a"]["t"] == pytest.approx(t)
    assert cells["healthbench__a"]["diff"] == pytest.approx(statistics.mean(d))
    half = 2.7764451051977987 * statistics.stdev(d) / math.sqrt(5)  # t(0.975, 4)
    assert cells["healthbench__a"]["ci95"] == pytest.approx([statistics.mean(d) - half,
                                                             statistics.mean(d) + half])
    # Holm with two cells: smaller p times 2, larger p times 1, kept monotone.
    p = sorted([cells["healthbench__a"]["p"], cells["chatdoctor__b"]["p"]])
    holm = sorted([cells["healthbench__a"]["p_holm"], cells["chatdoctor__b"]["p_holm"]])
    assert holm[0] == pytest.approx(min(1.0, 2 * p[0]))
    assert holm[1] == pytest.approx(max(min(1.0, 2 * p[0]), p[1]))


def test_posted_t_values_pass_holm():
    """The six t values in FINDINGS_quality_drop.md (df=4) all give p < 0.02, and
    all six survive Holm at 0.05, as posted in posted/HomM_3_followup_part2.md:14."""
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    ts = [-5.47, -6.20, -4.01, -5.34, -10.51, -9.02]
    p = [2 * stats.t.sf(abs(t), 4) for t in ts]
    assert max(p) < 0.02
    assert multipletests(p, method="holm")[0].all()


def test_missing_cells_root_exits_with_message(tmp_path, monkeypatch):
    qd = _load()
    monkeypatch.setattr(sys, "argv", ["quality_drop.py", "--cells-root", str(tmp_path / "nope")])
    with pytest.raises(SystemExit) as exc:
        qd.main()
    assert "PROV-1" in str(exc.value)
