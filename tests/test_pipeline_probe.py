"""The contamination probe's pure helpers (bodhi_lora/pipeline/probe.py, issue #72)."""

from bodhi_lora.pipeline import probe as cp


def test_compute_prefix_overlap_tokens_basic():
    # All match.
    assert cp.compute_prefix_overlap_tokens([1, 2, 3], [1, 2, 3]) == 3
    # Diverge after 2.
    assert cp.compute_prefix_overlap_tokens([1, 2, 9, 4], [1, 2, 3, 4]) == 2
    # No overlap at all.
    assert cp.compute_prefix_overlap_tokens([5, 6], [7, 8]) == 0
    # Empty inputs.
    assert cp.compute_prefix_overlap_tokens([], [1, 2]) == 0
    # Different lengths but match up to shorter.
    assert cp.compute_prefix_overlap_tokens([1, 2], [1, 2, 3, 4]) == 2


def test_normalize_for_exact_match_strips_case_and_whitespace():
    assert cp.normalize_for_exact_match("Hello   World") == "hello world"
    assert cp.normalize_for_exact_match("  HELLO\nworld\t") == "hello world"
    assert cp.normalize_for_exact_match("Hello World") == cp.normalize_for_exact_match(
        "hello   world"
    )


def test_extract_user_prompt_text_concatenates_user_turns():
    msgs = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second turn"},
    ]
    assert cp.extract_user_prompt_text(msgs) == "first turn\nsecond turn"


def test_two_proportion_z_pvalue_extremes():
    # Identical proportions -> p-value close to 1.
    p_same = cp.two_proportion_z_pvalue(5, 100, 5, 100)
    assert p_same == 1.0 or p_same > 0.99
    # Strong signal -> p-value small.
    p_strong = cp.two_proportion_z_pvalue(80, 100, 5, 100)
    assert p_strong < 0.01
    # Zero variance edge case (both empty match counts, full denominators).
    assert cp.two_proportion_z_pvalue(0, 100, 0, 100) == 1.0


def test_aggregate_handles_empty_and_populated():
    empty = cp.aggregate([], n_skipped=3)
    assert empty["n"] == 0
    assert empty["n_skipped"] == 3
    assert empty["exact_match_rate"] is None

    samples = [
        {"exact_match": True, "prefix_overlap_tokens": 10},
        {"exact_match": False, "prefix_overlap_tokens": 4},
    ]
    agg = cp.aggregate(samples, n_skipped=1)
    assert agg["n"] == 2
    assert agg["exact_match_rate"] == 0.5
    assert agg["mean_prefix_overlap_tokens"] == 7.0
