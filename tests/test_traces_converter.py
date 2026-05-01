"""Tests for scripts/convert_traces_to_maxtext.py.

These tests exercise the JSONL -> MaxText converter end-to-end against
the 5-row fixture in tests/fixtures/. We don't require the real Gemma
tokenizer to be importable on the dev box (transformers + sklearn +
pyarrow are heavy and the user's local env may have ABI conflicts), so
we substitute a small in-process FakeTokenizer that mimics Gemma-3's
chat template ("<start_of_turn>...<end_of_turn>"). The masking and
schema contract are language-agnostic, so the fake covers every code
path that matters.

The pattern (FakeTokenizer + monkeypatch) matches
tests/test_pure_logic.py's existing approach for ``train_lora.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import convert_traces_to_maxtext as conv  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "sft_traces_5rows.jsonl"


# ---------------------------------------------------------------------
# A minimal Gemma-3-flavoured tokenizer.
#
# Deliberately tiny so we control exactly what the masking boundary
# looks like. The chat template renders to:
#   <bos><start_of_turn>user\n{content}<end_of_turn>\n
#   <start_of_turn>model\n{content}<end_of_turn>\n
# add_generation_prompt=True suffixes "<start_of_turn>model\n" without
# the closing <end_of_turn>. That diff IS the response template — same
# contract train_lora.find_response_template() relies on.
# ---------------------------------------------------------------------
class FakeGemmaTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    add_bos_token = True

    BOS = "<bos>"
    GEN_PREFIX = "<start_of_turn>model\n"

    # Real Gemma renames "assistant" -> "model" in the chat template.
    # Mirror that here so the response-template diff yields exactly
    # "<start_of_turn>model\n", matching production Gemma behavior.
    _ROLE_MAP = {"assistant": "model"}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [self.BOS]
        for msg in messages:
            role = self._ROLE_MAP.get(msg["role"], msg["role"])
            parts.append(
                f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>\n"
            )
        if add_generation_prompt:
            parts.append(self.GEN_PREFIX)
        rendered = "".join(parts)
        if tokenize:
            return self._tokenize_str(rendered)
        return rendered

    # Word-level "tokenization" splits on whitespace and a few special-token
    # boundaries. Good enough to verify length math; the real model's BPE
    # is irrelevant to the masking logic.
    def __call__(self, text, add_special_tokens=True, return_attention_mask=False):
        ids = self._tokenize_str(text)
        return {"input_ids": ids}

    def _tokenize_str(self, s):
        # Walk the string, splitting on each special-token bracket so the
        # boundary tokens stay aligned with the chat-template diff. The
        # resulting list is "stable" under prefix concatenation, which
        # is what tokenize_with_completion_mask depends on.
        out: List[int] = []
        # Use string identity -> int via a shared vocab so the mock is
        # deterministic across calls.
        for tok in self._split(s):
            out.append(self._vocab(tok))
        return out

    _vocab_cache: Dict[str, int] = {}

    @classmethod
    def _vocab(cls, tok: str) -> int:
        if tok not in cls._vocab_cache:
            cls._vocab_cache[tok] = len(cls._vocab_cache) + 1
        return cls._vocab_cache[tok]

    @staticmethod
    def _split(s: str) -> List[str]:
        # Replace special tokens with sentinels so they survive whitespace
        # split, then split on whitespace.
        for special in ("<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>", "\n"):
            s = s.replace(special, f" {special} ")
        return [t for t in s.split(" ") if t]


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_detect_response_template_matches_train_lora():
    """The boundary string must be exactly '<start_of_turn>model\\n'.

    train_lora.py's DataCollatorForCompletionOnlyLM uses the same string
    to mask prompt tokens — if this drifts, the two trainers will mask
    different spans.
    """
    tok = FakeGemmaTokenizer()
    template = conv.detect_response_template(tok)
    assert template == "<start_of_turn>model\n"


def test_detect_response_template_raises_on_bad_template():
    """If the template can't be diffed, fail loudly rather than silently
    train on prompt tokens."""
    class BrokenTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            # No diff between the two — emulates a tokenizer whose
            # chat template doesn't add a generation prompt header.
            return "same string"

    with pytest.raises(ValueError, match="Could not auto-detect"):
        conv.detect_response_template(BrokenTokenizer())


def test_build_full_conversation_appends_assistant():
    msgs = [{"role": "user", "content": "hi"}]
    full = conv.build_full_conversation(msgs, "hello")
    assert full == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    # Original list must not be mutated — the converter is called per
    # row and mutating would corrupt the caller's data on retry.
    assert msgs == [{"role": "user", "content": "hi"}]


def test_tokenize_with_completion_mask_masks_prompt_only():
    """Prompt tokens must be -100; assistant tokens must equal input_ids.

    This is the core contract that lets MaxText compute loss only over
    the response — same masking semantics train_lora.py enforces via
    DataCollatorForCompletionOnlyLM(response_template=...).
    """
    tok = FakeGemmaTokenizer()
    template = conv.detect_response_template(tok)
    full = [
        {"role": "user", "content": "what is 2+2"},
        {"role": "assistant", "content": "four"},
    ]
    input_ids, labels = conv.tokenize_with_completion_mask(
        tok, full, template, max_length=128
    )
    assert len(input_ids) == len(labels)
    # At least one token is masked (the user turn) and at least one
    # token is unmasked (the assistant content).
    assert conv.LABEL_IGNORE_ID in labels
    assert any(lab != conv.LABEL_IGNORE_ID for lab in labels)
    # Every unmasked label MUST equal its input_id — we do not relabel
    # anything, only mask.
    for inp, lab in zip(input_ids, labels):
        if lab != conv.LABEL_IGNORE_ID:
            assert lab == inp

    # The last unmasked token should be from the assistant content
    # ("four" or its <end_of_turn>\n suffix), not from the user turn.
    rendered = tok.apply_chat_template(full, tokenize=False)
    boundary = rendered.rfind(template)
    assert boundary > 0
    completion_str = rendered[boundary + len(template):]
    assert "four" in completion_str
    # And the prompt-side string must NOT contain "four" — i.e. the
    # answer is only present in the unmasked region.
    assert "four" not in rendered[:boundary + len(template)]


def test_tokenize_with_completion_mask_multi_turn_uses_last_assistant():
    """In a multi-turn conversation only the LAST assistant turn is trained on.

    This matches DataCollatorForCompletionOnlyLM's behavior, which uses
    the response_template to find the LAST occurrence of the
    assistant-turn header. Earlier assistant turns are part of the
    prompt and must be masked.
    """
    tok = FakeGemmaTokenizer()
    template = conv.detect_response_template(tok)
    full = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "INTERMEDIATE_ANSWER"},
        {"role": "user", "content": "follow-up"},
        {"role": "assistant", "content": "FINAL_ANSWER"},
    ]
    input_ids, labels = conv.tokenize_with_completion_mask(
        tok, full, template, max_length=256
    )
    # Decode (cheaply) by mapping ids back to their tokens.
    inv = {v: k for k, v in FakeGemmaTokenizer._vocab_cache.items()}
    unmasked = [inv[i] for i, lab in zip(input_ids, labels)
                if lab != conv.LABEL_IGNORE_ID]
    assert "FINAL_ANSWER" in unmasked
    # The intermediate assistant content must NOT be in the unmasked region.
    assert "INTERMEDIATE_ANSWER" not in unmasked


def test_tokenize_with_completion_mask_truncates_at_max_length():
    tok = FakeGemmaTokenizer()
    template = conv.detect_response_template(tok)
    full = [
        {"role": "user", "content": "tell me a long story"},
        {"role": "assistant", "content": " ".join(["word"] * 50)},
    ]
    input_ids, labels = conv.tokenize_with_completion_mask(
        tok, full, template, max_length=10
    )
    assert len(input_ids) == 10
    assert len(labels) == 10


def test_convert_split_preserves_aux_fields(tmp_path):
    """Every input field except messages/response must survive into the
    MaxText row, so downstream join keys (prompt_id, source_dataset,
    tags, ablate_component) keep working."""
    rows = [json.loads(l) for l in FIXTURE.read_text().splitlines() if l.strip()]
    assert len(rows) == 5

    tok = FakeGemmaTokenizer()
    template = conv.detect_response_template(tok)
    maxtext_rows, tokenized_rows = conv.convert_split(
        rows, tok, template, max_length=512, src_path=FIXTURE
    )
    assert len(maxtext_rows) == 5
    assert len(tokenized_rows) == 5
    for src, dst in zip(rows, maxtext_rows):
        # messages now ends with the assistant turn; response is dropped
        # because it lives inside that assistant turn.
        assert dst["messages"][-1] == {"role": "assistant", "content": src["response"]}
        assert "response" not in dst
        # All other fields preserved verbatim.
        for k, v in src.items():
            if k in ("messages", "response"):
                continue
            assert dst[k] == v, f"field {k!r} not preserved"
    # prompt_id is mirrored into the tokenized sidecar so launchers can
    # join it back to the maxtext row.
    for src, tok_row in zip(rows, tokenized_rows):
        assert tok_row["prompt_id"] == src["prompt_id"]
        assert isinstance(tok_row["input_ids"], list)
        assert isinstance(tok_row["labels"], list)
        assert len(tok_row["input_ids"]) == len(tok_row["labels"])


def test_convert_split_validates_required_fields():
    """Missing messages/response must raise with a helpful pointer to
    the offending row, not a generic KeyError."""
    tok = FakeGemmaTokenizer()
    template = conv.detect_response_template(tok)
    bad = [{"messages": [{"role": "user", "content": "x"}]}]  # no 'response'
    with pytest.raises(ValueError, match="missing required field"):
        conv.convert_split(bad, tok, template, 128, Path("inline"))


def test_main_writes_all_artifacts(tmp_path, monkeypatch):
    """End-to-end: feed the 5-row fixture in for both --train and --val
    and check we get all six expected files plus the right counts."""
    monkeypatch.setattr(conv, "_load_tokenizer", lambda name: FakeGemmaTokenizer())

    out = tmp_path / "maxtext_out"
    rc = conv.main([
        "--train", str(FIXTURE),
        "--val", str(FIXTURE),
        "--tokenizer", "google/medgemma-27b-text-it",
        "--output-dir", str(out),
        "--max-seq-length", "256",
    ])
    assert rc == 0

    expected = [
        "train.jsonl", "val.jsonl",
        "train.tokenized.jsonl", "val.tokenized.jsonl",
        "metadata.json",
    ]
    for name in expected:
        assert (out / name).exists(), f"{name} not written"

    # MaxText rows: one per fixture row, each carrying a 'messages'
    # column whose final entry is the assistant response.
    train_rows = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    assert len(train_rows) == 5
    for row in train_rows:
        assert "messages" in row
        assert row["messages"][-1]["role"] == "assistant"
        # Auxiliary join keys we expect downstream eval to use.
        assert "prompt_id" in row
        assert "source_dataset" in row

    # Tokenized sidecar: same row count, each with input_ids and labels
    # that agree on length and have at least one masked + one unmasked
    # token.
    tok_rows = [json.loads(l)
                for l in (out / "train.tokenized.jsonl").read_text().splitlines()]
    assert len(tok_rows) == 5
    for tr in tok_rows:
        assert len(tr["input_ids"]) == len(tr["labels"])
        assert conv.LABEL_IGNORE_ID in tr["labels"]
        assert any(lab != conv.LABEL_IGNORE_ID for lab in tr["labels"])

    # metadata.json: counts agree, response_template detected, columns
    # advertised so launchers can sanity-check the artefact.
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["counts"] == {"train": 5, "val": 5}
    assert meta["response_template"] == "<start_of_turn>model\n"
    assert meta["maxtext_dataset_columns"] == ["messages"]
    assert meta["max_seq_length"] == 256


def test_help_does_not_require_transformers(monkeypatch):
    """--help must work on a laptop without HF installed; the import is
    deferred to actual conversion. This guards against accidentally
    moving ``from transformers import AutoTokenizer`` to module scope."""
    parser = conv.build_arg_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    # argparse exits 0 on --help.
    assert excinfo.value.code == 0
