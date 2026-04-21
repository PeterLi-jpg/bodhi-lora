"""Tests for scripts/plot_training.py — requires matplotlib."""

import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")  # skip if matplotlib not installed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.plot_training import series_from_history, load_history


# ---------------------------------------------------------------------------
# series_from_history
# ---------------------------------------------------------------------------

def test_series_extracts_train_loss():
    history = [
        {"step": 10, "loss": 1.5},
        {"step": 20, "loss": 1.2},
        {"step": 30, "loss": 0.9},
    ]
    steps, losses = series_from_history(history, "loss")
    assert steps == [10, 20, 30]
    assert losses == [1.5, 1.2, 0.9]


def test_series_extracts_eval_loss():
    history = [
        {"step": 50, "eval_loss": 1.1},
        {"step": 100, "eval_loss": 0.8},
    ]
    steps, losses = series_from_history(history, "eval_loss")
    assert steps == [50, 100]
    assert losses == [1.1, 0.8]


def test_series_skips_entries_without_key():
    history = [
        {"step": 10, "loss": 1.5},
        {"step": 20, "eval_loss": 1.1},  # no "loss" key
        {"step": 30, "loss": 0.9},
    ]
    steps, losses = series_from_history(history, "loss")
    assert steps == [10, 30]
    assert losses == [1.5, 0.9]


def test_series_skips_entries_without_step():
    history = [
        {"loss": 1.5},  # no step
        {"step": 20, "loss": 1.2},
    ]
    steps, losses = series_from_history(history, "loss")
    assert steps == [20]
    assert losses == [1.2]


def test_series_returns_empty_for_missing_key():
    history = [{"step": 10, "loss": 1.0}]
    steps, losses = series_from_history(history, "eval_loss")
    assert steps == []
    assert losses == []


def test_series_empty_history():
    steps, losses = series_from_history([], "loss")
    assert steps == []
    assert losses == []


# ---------------------------------------------------------------------------
# load_history
# ---------------------------------------------------------------------------

def test_load_history(tmp_path):
    import json
    state = {
        "log_history": [
            {"step": 1, "loss": 2.0},
            {"step": 2, "loss": 1.8},
        ]
    }
    p = tmp_path / "trainer_state.json"
    p.write_text(json.dumps(state))
    history = load_history(str(p))
    assert len(history) == 2
    assert history[0]["loss"] == 2.0


def test_load_history_missing_key(tmp_path):
    import json
    p = tmp_path / "trainer_state.json"
    p.write_text(json.dumps({}))
    assert load_history(str(p)) == []
