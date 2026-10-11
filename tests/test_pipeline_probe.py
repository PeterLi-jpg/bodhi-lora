"""The contamination probe (bodhi_lora/pipeline/probe.py, issue #72): its helpers, and a
probe on the stub engine whose replies recall some texts verbatim."""
import pytest

from bodhi_lora.engine import StubEngine

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


def test_aggregate_handles_empty_and_populated():
    empty = cp.aggregate([], n_rows=3, n_tokenized=3)
    assert empty["n"] == 0
    assert empty["n_rows_in_set"] == 3
    assert empty["exact_match_rate"] is None

    samples = [
        {"exact_match": True, "prefix_overlap_tokens": 10},
        {"exact_match": False, "prefix_overlap_tokens": 4},
    ]
    agg = cp.aggregate(samples, n_rows=9, n_tokenized=4)
    assert agg["n"] == 2 and agg["n_rows_tokenized"] == 4
    assert agg["exact_match_rate"] == 0.5 and agg["n_exact_match"] == 1
    assert agg["mean_prefix_overlap_tokens"] == 7.0
    # The summary is tracked in git, so it carries no per-item text (open decision O8).
    assert "samples" not in agg


def test_strip_template_removes_only_the_fixed_words_around_the_fields():
    tpl = "You are advising a clinician colleague.\n\n{question}"
    assert cp.strip_template("You are advising a clinician colleague.\n\nWhat is X?", tpl) == "What is X?"
    assert cp.strip_template("What is X?", tpl) == "What is X?"
    assert cp.strip_template("Q: a b. Answer briefly.", "Q: {question} Answer briefly.") == "a b."
    assert cp.strip_template("{x} stays", None) == "{x} stays"
    assert cp.strip_template("ctx\n\nq", "{context}\n\n{question}") == "ctx\n\nq"


def test_draw_takes_only_long_enough_rows_in_a_seeded_order():
    eng = StubEngine("m")
    items = [(f"r{i}", " ".join(["w"] * (i + 1))) for i in range(10)]
    drawn, seen = cp.draw_long_enough(items, eng, need=6, n=3, seed=1, name="t")
    assert len(drawn) == 3 and all(len(ids) >= 6 for _, _, ids in drawn)
    assert drawn == cp.draw_long_enough(list(reversed(items)), StubEngine("m"), 6, 3, 1, "t")[0]
    assert seen >= 3
    with pytest.raises(SystemExit, match="needs 8"):
        cp.draw_long_enough(items, eng, need=6, n=8, seed=1, name="t")


def test_probe_one_counts_a_verbatim_continuation_as_memorised():
    text = "the patient has a cough and a fever since monday night"

    def recall(messages):
        # Continue the probed text verbatim, as a memorising model would.
        return " ".join(text[len(messages[-1]["content"]):].split()[:3])

    eng = StubEngine("m", reply=recall)
    ids = eng.tokenize(text)
    hit = cp.probe_one("a", ids, eng, prefix_tokens=4, continuation_tokens=3)
    assert hit["exact_match"] and hit["prefix_overlap_tokens"] == 3
    assert hit["ground_truth_continuation"] == "cough and a"
    eng._reply = "something else entirely"
    miss = cp.probe_one("b", ids, eng, 4, 3)
    assert not miss["exact_match"] and miss["prefix_overlap_tokens"] == 0
