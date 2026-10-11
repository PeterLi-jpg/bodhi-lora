"""Tests for rebuttal/analyses/mimicry_split.py.

The unit tests run anywhere. The golden test reruns the script on the
submitted cell's committed grade files and pins the values quoted to reviewer LxMF; it needs
data/raw/healthbench/hard.jsonl (gitignored, fetched by python -m bodhi_lora fetch healthbench)
and skips without it.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HARD = ROOT / "data" / "raw" / "healthbench" / "hard.jsonl"
# The script reads the 2026-05 cell's raw eval and grade JSON, which git ignores since
# 2026-10-11 (open decision O2), so a checkout without them skips the golden values.
RAW_PRESENT = any((ROOT / "results" / "healthbench__medgemma-27b-text__bodhi__2026-05").glob("seed_*/eval/base.json"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "mimicry_split", ROOT / "rebuttal" / "analyses" / "mimicry_split.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_odds_ratio_matches_hand_computation():
    ms = _load()
    # Base rates from the committed output: 18.0% vs 15.9% -> OR 1.16 (posted).
    cd, sc = 0.1800518134715026, 0.1592920353982301
    expected = (cd / (1 - cd)) / (sc / (1 - sc))
    assert ms.odds_ratio(cd, sc) == pytest.approx(expected)
    assert round(ms.odds_ratio(cd, sc), 2) == 1.16


def test_odds_ratio_is_nan_at_the_edges():
    ms = _load()
    assert math.isnan(ms.odds_ratio(0.0, 0.5))
    assert math.isnan(ms.odds_ratio(0.5, 1.0))


def test_discrimination_on_a_toy_index():
    """Two context-dependent prompts asked 3/4 times, one self-contained asked 1/4."""
    ms = _load()
    pid_map = {
        "a": {"group": "context_dependent", "ai": [True, True, False, True]},
        "b": {"group": "context_dependent", "ai": [True, True, True, False]},
        "c": {"group": "self_contained", "ai": [True, False, False, False]},
        "d": {"group": None, "ai": [True, True]},  # unlabelled prompt is ignored
    }
    disc, cd, sc = ms.discrimination(pid_map, ["a", "b", "c", "d"])
    assert cd == pytest.approx(6 / 8)
    assert sc == pytest.approx(1 / 4)
    assert disc == pytest.approx(0.5)


def test_bootstrap_resamples_prompts_and_returns_or_ci():
    ms = _load()
    idx = {cfg: {"a": {"group": "context_dependent", "ai": [True, False]},
                 "b": {"group": "self_contained", "ai": [False, True]},
                 "c": {"group": "context_dependent", "ai": [True, True]},
                 "e": {"group": "self_contained", "ai": [False, False]}}
           for cfg in ms.CONFIG_ORDER}
    ci_cfg, diff_ci, or_ci = ms.bootstrap_disc(idx, ["a", "b", "c", "e"], B=200, seed=0)
    assert set(or_ci) == set(ms.CONFIG_ORDER)
    for lo, hi in or_ci.values():
        assert lo <= hi
    # Identical configs: the LoRA - Base difference is 0 in every resample.
    assert diff_ci == (0.0, 0.0)


@pytest.mark.skipif(not HARD.is_file(), reason="data/raw/healthbench/hard.jsonl not present")
@pytest.mark.skipif(not RAW_PRESENT, reason="the 2026-05 raw eval files are not in this checkout")
def test_golden_values_on_committed_results(tmp_path, monkeypatch):
    ms = _load()
    out = tmp_path / "mimicry_split.json"
    monkeypatch.setattr(sys, "argv", ["mimicry_split.py", "--out", str(out)])
    ms.main()
    d = json.loads(out.read_text())
    assert d["n_eval_prompts"] == 667
    prim = d["primary_theme_or_rubric"]
    assert prim["groups"] == {"context_dependent": 508, "self_contained": 158, "null": 1}
    assert d["robustness_theme_only"]["groups"]["context_dependent"] == 127
    # Posted to LxMF (posted/LxMF_1_rebuttal.md:26): ORs 1.16 / 1.54 / 1.56.
    ors = {c: round(prim["configs"][c]["odds_ratio"], 2) for c in ms.CONFIG_ORDER}
    assert ors["base_no_wrapper"] == 1.16
    assert ors["lora_no_wrapper"] == 1.54
    assert ors["base_bodhi"] == 1.56
    # Posted at posted/LxMF_1_rebuttal.md:28: LoRA - Base discrimination +8.4pp [-1.2, 17.9].
    assert round(100 * prim["lora_minus_base_disc"], 1) == 8.4
    assert [round(100 * x, 1) for x in prim["lora_minus_base_disc_ci95"]] == [-1.2, 17.9]
    assert round(100 * d["robustness_theme_only"]["lora_minus_base_disc"], 1) == 10.0
