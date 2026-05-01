"""Convert filtered BOHDI SFT JSONL into MaxText's HF SFT input format.

Stage 3 (LoRA fine-tune of MedGemma-27B) is migrating from
PyTorch+torch_xla (``train_lora.py``) to a forked MaxText. MaxText's HF
input pipeline (``dataset_type=hf``, ``train_data_columns=['messages']``)
expects each row to carry a single ``messages`` column whose value is the
full conversation INCLUDING the assistant response — see
``src/maxtext/input_pipeline/hf_data_processing.py`` and
``src/maxtext/input_pipeline/input_pipeline_utils.apply_chat_template``
in upstream MaxText. The ``SFTPromptMasking`` op then walks the rendered
chat template, masks all prompt tokens (including the
``<start_of_turn>model\\n`` generation-prompt prefix) to the pad/unk id,
and trains only on the assistant content.

Our existing ``scripts/train_lora.py`` does the same thing via
``trl.DataCollatorForCompletionOnlyLM(response_template=...)``: it asks
``find_response_template()`` for the diff between
``add_generation_prompt=True`` and ``False`` (which is exactly
``<start_of_turn>model\\n`` for Gemma) and masks every label token before
that marker. The two pipelines therefore agree on what counts as
"prompt" vs "completion", and this converter just has to reshape the
source rows so MaxText can see them.

What this script writes into ``--output-dir``:

  * ``train.jsonl`` / ``val.jsonl`` — MaxText-ready rows with a
    ``messages`` column (assistant turn appended). Auxiliary fields from
    the input (``prompt_id``, ``source_dataset``, ``tags``, etc.) are
    preserved as extra columns so the launcher can join back later.
    Pointed at via ``hf_path=<output_dir>`` in MaxText's config.

  * ``train.tokenized.jsonl`` / ``val.tokenized.jsonl`` — sidecar with
    pre-tokenized ``input_ids`` / ``labels`` (prompt tokens replaced by
    -100). Only the assistant tokens contribute to loss — same masking
    contract ``train_lora.py`` enforces. Useful for local sanity checks
    and for the launcher to ship a "ready to load" tensor file when we
    want to skip MaxText's HF tokenization stage.

  * ``metadata.json`` — row counts, tokenizer name, max length, the
    detected response template. Lets the launcher fail fast if it picked
    up the wrong artefact.

The script touches no JAX / no MaxText — it's pure HF + json so it can
run on a laptop before launching the TPU job.

Usage:
    python scripts/convert_traces_to_maxtext.py \\
        --train data/sft/train.jsonl \\
        --val data/sft/val.jsonl \\
        --tokenizer google/medgemma-27b-text-it \\
        --output-dir data/sft/maxtext
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# 4096 matches the YAML configs in ``configs/*.yaml``.
DEFAULT_MAX_SEQ_LENGTH = 4096

# Matches torch.nn.CrossEntropyLoss's default ``ignore_index=-100`` and HF
# Trainer's convention so the sidecar can be loaded as-is by either stack.
LABEL_IGNORE_ID = -100


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{line_no} is not valid JSON ({e}); "
                    "graded SFT JSONL should have one JSON object per line."
                ) from e
    return rows


def _validate_row(row: Dict[str, Any], path: Path, line_no: int) -> None:
    """Require the SFT fields we use; pass everything else through.

    Mirrors ``train_lora.load_sft_jsonl``: only ``messages`` and
    ``response`` are mandatory.
    """
    if "messages" not in row or "response" not in row:
        missing = [k for k in ("messages", "response") if k not in row]
        raise ValueError(
            f"{path}:{line_no} missing required field(s) {missing}. "
            f"Row keys: {sorted(row.keys())}"
        )
    if not isinstance(row["messages"], list) or not row["messages"]:
        raise ValueError(
            f"{path}:{line_no} 'messages' must be a non-empty list of "
            f"{{role, content}} dicts; got {type(row['messages']).__name__}."
        )
    if not isinstance(row["response"], str):
        raise ValueError(
            f"{path}:{line_no} 'response' must be a string; "
            f"got {type(row['response']).__name__}."
        )


def detect_response_template(tokenizer) -> str:
    """Find the assistant-turn header by diffing
    ``add_generation_prompt`` False vs. True.

    Byte-for-byte identical to ``scripts.train_lora.find_response_template``
    so the masking contract stays in sync between the two trainers. For
    Gemma-3 returns ``"<start_of_turn>model\\n"``; for Llama-3 returns
    ``"<|start_header_id|>assistant<|end_header_id|>\\n\\n"``.
    """
    dummy = [{"role": "user", "content": "hi"}]
    without_gen = tokenizer.apply_chat_template(
        dummy, tokenize=False, add_generation_prompt=False
    )
    with_gen = tokenizer.apply_chat_template(
        dummy, tokenize=False, add_generation_prompt=True
    )
    if with_gen.startswith(without_gen):
        template = with_gen[len(without_gen):]
        if template.strip():
            return template
    raise ValueError(
        "Could not auto-detect response template — the tokenizer's chat "
        "template does not append a clean assistant-turn header to "
        "add_generation_prompt=True. "
        f"without_gen={without_gen!r} with_gen={with_gen!r}. "
        "Extend detect_response_template() to handle this template family."
    )


def build_full_conversation(messages: List[Dict[str, str]],
                            response: str) -> List[Dict[str, str]]:
    """Return ``messages`` with the assistant response appended.

    MaxText's ``apply_chat_template`` only treats trailing-assistant
    content as the completion to learn on, so the assistant turn must
    already be in the messages list when we write the row. Mirrors
    ``train_lora.format_example``.
    """
    full = list(messages)  # don't mutate the caller's list
    full.append({"role": "assistant", "content": response})
    return full


def tokenize_with_completion_mask(
    tokenizer,
    full_messages: List[Dict[str, str]],
    response_template: str,
    max_length: int,
) -> Tuple[List[int], List[int]]:
    """Tokenize the rendered chat and mask everything before the final
    assistant turn (-100 in ``labels``).

    Returns ``(input_ids, labels)`` truncated to ``max_length``. Same
    masking contract as ``train_lora.py``'s
    ``DataCollatorForCompletionOnlyLM(response_template=...)``.
    """
    rendered = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    # ``rfind`` so multi-turn conversations only train on the LAST
    # assistant turn — earlier assistant turns become part of the prompt.
    boundary = rendered.rfind(response_template)
    if boundary < 0:
        # Without a boundary we'd silently train on every token; fail
        # fast rather than burn a multi-hour TPU run on misaligned data.
        raise ValueError(
            f"Response template {response_template!r} not found in "
            "rendered chat. Check that detect_response_template() and "
            "the tokenizer's chat template agree."
        )
    prompt_str = rendered[: boundary + len(response_template)]

    # ``add_special_tokens=False`` because the chat template already
    # emits the model's special tokens (BOS for Gemma) inline; HF would
    # otherwise prepend a second BOS and shift the boundary by one.
    # Same reason train_lora.py sets tokenizer.add_bos_token = False.
    full_ids: List[int] = tokenizer(
        rendered, add_special_tokens=False, return_attention_mask=False
    )["input_ids"]
    prompt_ids: List[int] = tokenizer(
        prompt_str, add_special_tokens=False, return_attention_mask=False
    )["input_ids"]

    n_prompt = len(prompt_ids)
    if n_prompt > len(full_ids):
        # Only happens if the tokenizer isn't prefix-stable (aggressive
        # whitespace normalization, etc.). Bail rather than emit garbage.
        raise ValueError(
            "Prompt prefix tokenizes to more tokens than the full "
            f"sequence ({n_prompt} > {len(full_ids)}); tokenizer is "
            "not prefix-stable, refusing to write a misaligned mask."
        )

    # Truncate AFTER masking so prompt tokens stay masked even when the
    # response itself spills past max_length.
    input_ids = full_ids[:max_length]
    labels = list(input_ids)
    for i in range(min(n_prompt, len(labels))):
        labels[i] = LABEL_IGNORE_ID
    return input_ids, labels


def convert_split(
    rows: List[Dict[str, Any]],
    tokenizer,
    response_template: str,
    max_length: int,
    src_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build (maxtext_rows, tokenized_rows) for one split.

    ``src_path`` is only used for error messages (it points at the
    JSONL the rows came from).
    """
    maxtext_rows: List[Dict[str, Any]] = []
    tokenized_rows: List[Dict[str, Any]] = []
    for row_idx, row in enumerate(rows, start=1):
        _validate_row(row, src_path, row_idx)
        full_messages = build_full_conversation(row["messages"], row["response"])

        # Pass auxiliary fields through unchanged so prompt_id /
        # source_dataset / tags survive for downstream joins. MaxText
        # ignores extra columns at training time (it reads only
        # ``train_data_columns``); the launcher and eval scripts use
        # them. ``response`` is now folded into the assistant turn
        # inside ``messages`` and would be redundant if kept.
        maxtext_row: Dict[str, Any] = {"messages": full_messages}
        for k, v in row.items():
            if k in ("messages", "response"):
                continue
            maxtext_row[k] = v
        maxtext_rows.append(maxtext_row)

        input_ids, labels = tokenize_with_completion_mask(
            tokenizer, full_messages, response_template, max_length
        )
        tok_row: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
        }
        if "prompt_id" in row:
            tok_row["prompt_id"] = row["prompt_id"]
        tokenized_rows.append(tok_row)

    return maxtext_rows, tokenized_rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def _load_tokenizer(name: str):
    """Lazy HF import so ``--help`` works without transformers installed
    and tests can monkeypatch the tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        # Same fallback as train_lora.py — Gemma's tokenizer doesn't
        # define a separate pad token, so we use eos for padding. It
        # only matters for the tokenized sidecar; MaxText pads with its
        # own pad/unk handling.
        tokenizer.pad_token = tokenizer.eos_token
    # Match train_lora.py: the chat template emits BOS itself; if the
    # tokenizer ALSO prepends BOS we'd get "<bos><bos>..." and shift the
    # masking boundary. Disable here to keep the prompt prefix-stable.
    tokenizer.add_bos_token = False
    return tokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert filtered BOHDI SFT JSONL into MaxText-ready files. "
            "Writes train.jsonl, val.jsonl (MaxText HF SFT format), the "
            "*.tokenized.jsonl sidecars, and a metadata.json into "
            "--output-dir."
        ),
    )
    parser.add_argument(
        "--train", required=True, type=Path,
        help="Path to graded train SFT JSONL (e.g. data/sft/train.jsonl).",
    )
    parser.add_argument(
        "--val", required=True, type=Path,
        help="Path to graded val SFT JSONL (e.g. data/sft/val.jsonl).",
    )
    parser.add_argument(
        "--tokenizer", required=True,
        help=(
            "HF tokenizer name (e.g. google/medgemma-27b-text-it). "
            "Must be the SAME tokenizer the MaxText training run loads, "
            "otherwise the pre-tokenized sidecar will be misaligned."
        ),
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory to write the converted files into.",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
        help=(
            f"Truncate the tokenized sidecar at this many tokens "
            f"(default {DEFAULT_MAX_SEQ_LENGTH}). MaxText's max_target_length "
            "must be set to the same value at training time."
        ),
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.train.exists():
        raise FileNotFoundError(f"--train file not found: {args.train}")
    if not args.val.exists():
        raise FileNotFoundError(f"--val file not found: {args.val}")

    tokenizer = _load_tokenizer(args.tokenizer)
    response_template = detect_response_template(tokenizer)
    print(f"Detected response template: {response_template!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}

    for split_name, src in (("train", args.train), ("val", args.val)):
        rows = _read_jsonl(src)
        maxtext_rows, tokenized_rows = convert_split(
            rows, tokenizer, response_template, args.max_seq_length, src
        )
        n_mt = write_jsonl(args.output_dir / f"{split_name}.jsonl", maxtext_rows)
        n_tok = write_jsonl(
            args.output_dir / f"{split_name}.tokenized.jsonl", tokenized_rows
        )
        # Counts must agree — if not, _read_jsonl skipped rows we
        # serialized, or vice versa, and we want to know.
        assert n_mt == n_tok == len(rows), (
            f"{split_name}: row-count mismatch "
            f"(read {len(rows)}, wrote {n_mt} maxtext / {n_tok} tokenized)"
        )
        counts[split_name] = n_mt
        print(f"{split_name}: {n_mt} rows -> {args.output_dir}/{split_name}.jsonl")

    metadata = {
        "tokenizer": args.tokenizer,
        "response_template": response_template,
        "max_seq_length": args.max_seq_length,
        "label_ignore_id": LABEL_IGNORE_ID,
        "counts": counts,
        "source_train": str(args.train),
        "source_val": str(args.val),
        # Emit the schema explicitly so downstream code can sanity-check
        # before launching a multi-hour TPU run.
        "maxtext_dataset_columns": ["messages"],
        "tokenized_sidecar_columns": ["input_ids", "labels", "prompt_id"],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Wrote metadata: {args.output_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
