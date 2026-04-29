"""Pure-logic regression tests for training, eval, and dataset helpers."""

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def import_with_mocks(module_name, mocked_modules, monkeypatch):
    for mocked_name in mocked_modules:
        monkeypatch.setitem(sys.modules, mocked_name, MagicMock())
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = " | ".join(f"{msg['role']}:{msg['content']}" for msg in messages)
        if add_generation_prompt:
            rendered += " | assistant:"
        return rendered


def test_format_example_single_example(monkeypatch):
    train_lora = import_with_mocks(
        "scripts.train_lora",
        ["torch", "datasets", "transformers", "trl", "peft"],
        monkeypatch,
    )
    train_lora._tokenizer = FakeTokenizer()

    result = train_lora.format_example(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "response": "hello",
        }
    )

    assert result == "user:hi | assistant:hello"


def test_format_example_batched_examples(monkeypatch):
    train_lora = import_with_mocks(
        "scripts.train_lora",
        ["torch", "datasets", "transformers", "trl", "peft"],
        monkeypatch,
    )
    train_lora._tokenizer = FakeTokenizer()

    result = train_lora.format_example(
        {
            "messages": [
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "second"}],
            ],
            "response": ["one", "two"],
        }
    )

    assert result == [
        "user:first | assistant:one",
        "user:second | assistant:two",
    ]


def test_load_exclude_ids_accepts_json_and_jsonl(monkeypatch, tmp_path):
    generate_traces = import_with_mocks(
        "scripts.generate_traces",
        ["torch", "transformers", "tqdm"],
        monkeypatch,
    )

    list_json = tmp_path / "ids.json"
    list_json.write_text(json.dumps(["a", "b"]))

    dict_json = tmp_path / "ids_dict.json"
    dict_json.write_text(json.dumps({"prompt_ids": ["c"]}))

    jsonl_file = tmp_path / "rows.jsonl"
    jsonl_file.write_text(
        "\n".join(
            [
                json.dumps({"prompt_id": "d"}),
                json.dumps({"prompt_id": "e"}),
            ]
        )
    )

    exclude_ids = generate_traces.load_exclude_ids(
        [str(list_json), str(dict_json), str(jsonl_file)]
    )

    assert exclude_ids == {"a", "b", "c", "d", "e"}


def test_compute_tertile_cutoffs_and_tiers():
    from scripts.eval_ushape import compute_tertile_cutoffs, tier_of

    meta = {
        "p1": {"pos_points": 1},
        "p2": {"pos_points": 2},
        "p3": {"pos_points": 3},
        "p4": {"pos_points": 4},
        "p5": {"pos_points": 5},
        "p6": {"pos_points": 6},
    }

    q1, q2 = compute_tertile_cutoffs(meta)

    assert q1 < q2
    assert tier_of(1, q1, q2) == "easy"
    assert tier_of(3, q1, q2) == "medium"
    assert tier_of(6, q1, q2) == "hard"


def test_compute_tertile_cutoffs_restricts_to_subset():
    from scripts.eval_ushape import compute_tertile_cutoffs

    meta = {
        "p1": {"pos_points": 1},
        "p2": {"pos_points": 2},
        "p3": {"pos_points": 3},
        "p4": {"pos_points": 100},
    }

    q1, q2 = compute_tertile_cutoffs(meta, restrict_to={"p1", "p2", "p3"})

    assert q2 < 100


def test_summarize_handles_empty_scores():
    from scripts.eval_ushape import summarize

    assert summarize([], fail_threshold=0.4) == {"n": 0}


def test_summarize_handles_degenerate_scores():
    from scripts.eval_ushape import summarize

    summary = summarize([0.5, 0.5, 0.5], fail_threshold=0.4)

    assert summary["mean"] == 0.5
    assert summary["median"] == 0.5
    assert summary["fail_rate"] == 0.0


def test_summarize_bootstrap_is_seeded():
    from scripts.eval_ushape import summarize

    scores = [0.1, 0.4, 0.8, 0.9]
    summary_a = summarize(
        scores,
        fail_threshold=0.4,
        bootstrap=200,
        rng=np.random.default_rng(42),
    )
    summary_b = summarize(
        scores,
        fail_threshold=0.4,
        bootstrap=200,
        rng=np.random.default_rng(42),
    )

    assert summary_a == summary_b


def test_vllm_engine_enforce_eager_flag():
    """Issue #65: --enforce-eager must be conditional on the kwarg.

    Default True preserves the old behavior (graph-capture hang on LoRA);
    explicit False is needed for the latency benchmark to compare graph mode.
    Both docker and subprocess builds must agree, otherwise the run-mode
    silently changes the eval profile.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from _vllm_engine import VLLMEngine

    eng_default = VLLMEngine(model="google/gemma-3-4b-it", tp_size=1)
    assert eng_default._enforce_eager is True
    assert "--enforce-eager" in eng_default._build_docker_cmd()
    assert "--enforce-eager" in eng_default._build_subprocess_cmd()

    eng_off = VLLMEngine(model="google/gemma-3-4b-it", tp_size=1, enforce_eager=False)
    assert eng_off._enforce_eager is False
    assert "--enforce-eager" not in eng_off._build_docker_cmd()
    assert "--enforce-eager" not in eng_off._build_subprocess_cmd()


# ── contamination_probe.py (issue #72) ───────────────────────────────────

def _import_contamination_probe(monkeypatch):
    return import_with_mocks(
        "scripts.contamination_probe",
        ["transformers", "tqdm"],
        monkeypatch,
    )


def test_compute_prefix_overlap_tokens_basic(monkeypatch):
    cp = _import_contamination_probe(monkeypatch)

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


def test_normalize_for_exact_match_strips_case_and_whitespace(monkeypatch):
    cp = _import_contamination_probe(monkeypatch)

    assert cp.normalize_for_exact_match("Hello   World") == "hello world"
    assert cp.normalize_for_exact_match("  HELLO\nworld\t") == "hello world"
    assert cp.normalize_for_exact_match("Hello World") == cp.normalize_for_exact_match(
        "hello   world"
    )


def test_extract_user_prompt_text_concatenates_user_turns(monkeypatch):
    cp = _import_contamination_probe(monkeypatch)

    msgs = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second turn"},
    ]
    assert cp.extract_user_prompt_text(msgs) == "first turn\nsecond turn"


def test_two_proportion_z_pvalue_extremes(monkeypatch):
    cp = _import_contamination_probe(monkeypatch)

    # Identical proportions -> p-value close to 1.
    p_same = cp.two_proportion_z_pvalue(5, 100, 5, 100)
    assert p_same == 1.0 or p_same > 0.99
    # Strong signal -> p-value small.
    p_strong = cp.two_proportion_z_pvalue(80, 100, 5, 100)
    assert p_strong < 0.01
    # Zero variance edge case (both empty match counts, full denominators).
    assert cp.two_proportion_z_pvalue(0, 100, 0, 100) == 1.0


def test_aggregate_handles_empty_and_populated(monkeypatch):
    cp = _import_contamination_probe(monkeypatch)

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
