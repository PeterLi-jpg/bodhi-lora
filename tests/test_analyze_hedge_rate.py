"""Tests for scripts/analyze_hedge_rate.py (issue #64).

These tests cover the lexical classifier itself (no model needed).  The pipe
that reads JSONL and writes the summary file is intentionally not tested
because it is a thin I/O wrapper around the classifier — keeping it fast
keeps it on every CI run.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyze_hedge_rate import (
    is_hedged,
    quartile_buckets,
    quartile_label,
)


# ---- is_hedged ------------------------------------------------------------

class TestIsHedged:
    def test_hedged_disclaimer(self):
        assert is_hedged("I am not a doctor, but you should consult a physician.")

    def test_hedged_call_911(self):
        assert is_hedged("If you have chest pain, call 911 immediately.")
        assert is_hedged("If you have chest pain, call 9-1-1 immediately.")

    def test_hedged_seek_attention(self):
        assert is_hedged("Please seek medical attention right away.")
        assert is_hedged("You should seek emergency care.")

    def test_hedged_not_substitute(self):
        assert is_hedged("This information is not a substitute for medical advice.")

    def test_hedged_idk(self):
        assert is_hedged("I don't know what's causing this.")
        assert is_hedged("I'm not certain about that diagnosis.")

    def test_not_hedged_direct_answer(self):
        assert not is_hedged("The most likely cause is acute pancreatitis.")

    def test_not_hedged_empty(self):
        assert not is_hedged("")
        assert not is_hedged(None)

    def test_word_boundary_consultation_does_not_match(self):
        # 'consult' regex requires "consult <a/your/with a> doctor/...".
        # The bare word 'consultation' alone should NOT match — the
        # classifier should under-count, not over-count.
        assert not is_hedged("She had a consultation last Tuesday.")

    def test_word_boundary_consult_with_doctor_matches(self):
        # But the full hedge phrase ("consult a doctor") SHOULD match.
        assert is_hedged("Please consult a doctor about this.")


# ---- quartile helpers ------------------------------------------------------

class TestQuartiles:
    def test_quartile_buckets_returns_three_cutoffs(self):
        scores = [0.1, 0.3, 0.4, 0.6, 0.8, 0.9, 1.0, 0.5, 0.2, 0.7]
        cutoffs = quartile_buckets(scores)
        assert cutoffs is not None
        q1, q2, q3 = cutoffs
        assert q1 < q2 < q3

    def test_quartile_buckets_too_few_returns_none(self):
        assert quartile_buckets([0.5, 0.6]) is None
        assert quartile_buckets([]) is None

    def test_quartile_buckets_skips_none(self):
        scores = [0.1, None, 0.5, 0.7, 0.9]
        cutoffs = quartile_buckets(scores)
        assert cutoffs is not None  # 4 valid values -> Q1/Q2/Q3 defined

    def test_quartile_label_lowest(self):
        # Quartiles for 1..4: q1=1.75, q2=2.5, q3=3.25
        q1, q2, q3 = 1.75, 2.5, 3.25
        assert quartile_label(1.0, q1, q2, q3) == "Q1 (lowest)"
        assert quartile_label(2.0, q1, q2, q3) == "Q2"
        assert quartile_label(3.0, q1, q2, q3) == "Q3"
        assert quartile_label(4.0, q1, q2, q3) == "Q4 (highest)"

    def test_quartile_label_none(self):
        assert quartile_label(None, 0.0, 0.5, 1.0) is None
