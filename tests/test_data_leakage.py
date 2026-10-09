"""The leakage gate (bodhi_lora/data/leakage.py): held-out prompts never reach training."""
import pytest

from bodhi_lora.data import LeakageError, Split, check_leakage, normalize_text

SPLIT = Split(name="t", train=["t1", "t2", "t3"], eval=["e1", "e2", "e3"], probe=["p1"])
TEXTS = {"e1": "How long should a fever last?", "e2": "Is my rash an allergy?",
         "e3": "What dose of ibuprofen?", "p1": "Chest pain at rest."}


def test_clean_training_set_passes_and_reports():
    report = check_leakage({"t1": "a question", "t2": "another"}, SPLIT, TEXTS,
                           {42: ["e1", "e2"], 7: ["e2", "e3"]}, SPLIT.eval)
    assert report["passed"] and report["id_leak"] == 0 and report["text_duplicates"] == 0
    assert report["n_held_out"] == 4 and report["eval_ids"]["42"]["n"] == 2


def test_a_held_out_id_in_training_fails():
    with pytest.raises(LeakageError, match="held out") as e:
        check_leakage({"t1": "x", "e2": "y"}, SPLIT, TEXTS, {42: ["e1"]}, SPLIT.eval)
    assert e.value.report["id_leak_first"] == ["e2"] and not e.value.report["passed"]


def test_a_probe_row_in_training_fails():
    with pytest.raises(LeakageError):
        check_leakage({"p1": "z"}, SPLIT, TEXTS, {42: ["e1"]}, SPLIT.eval)


def test_a_near_copy_under_another_id_fails():
    trained = {"t1": "  how LONG should a   fever last?  "}
    with pytest.raises(LeakageError, match="copy a held-out") as e:
        check_leakage(trained, SPLIT, TEXTS, {42: ["e1"]}, SPLIT.eval)
    assert e.value.report["text_duplicates_first"] == ["t1"]


def test_eval_ids_outside_the_population_fail():
    with pytest.raises(LeakageError, match="outside the eval population"):
        check_leakage({}, SPLIT, TEXTS, {42: ["e1", "t1"]}, SPLIT.eval)


def test_normalize_text():
    assert normalize_text("  A  b\nC ") == "a b c"
