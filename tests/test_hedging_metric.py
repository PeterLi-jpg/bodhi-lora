"""Pure-stdlib tests for scripts/_hedging_metric.py — no LLM, no GPU."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Add scripts/ for the bare-name `_hedging_metric` import the script itself uses.
sys.path.insert(0, str(ROOT / "scripts"))

from scripts._hedging_metric import HEDGE_LEXICON, compute_hedge_rate


def test_hedge_rate_zero_for_direct_text():
    result = compute_hedge_rate("Take ibuprofen 200mg every 4 hours.")
    assert result["hedge_count"] == 0
    assert result["hedge_rate"] == 0.0


def test_hedge_rate_positive_for_hedged_text():
    # "might" + "consider consulting" both appear -> at least 2 distinct phrases.
    result = compute_hedge_rate("You might want to consider consulting a doctor.")
    assert result["hedge_count"] >= 2
    assert result["hedge_rate"] > 0


def test_lexicon_is_exposed():
    assert isinstance(HEDGE_LEXICON, tuple)
    assert len(HEDGE_LEXICON) > 5


def test_phrase_counted_once():
    # Each distinct phrase counted once per response, even if repeated.
    result = compute_hedge_rate("might might might.")
    assert result["hedge_count"] == 1


def test_matched_phrases_returned():
    result = compute_hedge_rate("You might want to consider consulting a doctor.")
    assert "might" in result["matched_phrases"]
