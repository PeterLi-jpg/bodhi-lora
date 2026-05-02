"""Pre-tokenized JSONL → batch iterator for the MaxText LoRA trainer.

Stage 3b's training loop in ``scripts/train_lora_maxtext.py`` calls
``dataset_loader.build_iterators(...)`` to get back a pair of generators
plus a per-epoch step count.  The upstream MaxText input pipeline reads
TFDS / HF datasets with its own tokenization stack; we deliberately
bypass that here because Stage 2 (``scripts/convert_traces_to_maxtext.py``)
already wrote pre-tokenized JSONL with the exact ``-100``-masked labels
we need, and re-tokenizing on the TPU would just re-run the same code
the converter already ran on a CPU.

Reads ``<dataset_dir>/train.tokenized.jsonl`` and ``val.tokenized.jsonl``
(written alongside the maxtext_rows by the converter — see the
``tokenized_sidecar_columns`` field of ``metadata.json``). Each row:

    {"input_ids": List[int], "labels": List[int], "prompt_id": str}

with ``labels[i] == LABEL_IGNORE_ID`` (-100) on prompt tokens. We pad
both ``input_ids`` and ``labels`` to ``max_seq_length`` (right-pad with
``pad_id`` for inputs and -100 for labels), construct an explicit
``loss_mask`` so the trainer never has to special-case the ignore id,
and yield numpy batches of shape ``(global_batch_size, max_seq_length)``.

Notes for callers:
  - Iterators are infinite for train (re-shuffle each epoch with the
    same RNG seed so resumes are deterministic) and finite for eval
    (one full pass, no shuffle).
  - ``global_batch_size = per_device_batch_size * gradient_accumulation_steps``.
    The trainer slices into per-device chunks itself.
  - This module imports only stdlib + numpy. No JAX, no transformers,
    no flax — keeps tests runnable on a CPU dev box.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np


# Same constant the converter writes (see scripts/convert_traces_to_maxtext.py).
LABEL_IGNORE_ID = -100

# Default pad token id. The converter doesn't write a pad column (its
# rows are variable-length); we pad here at batch time. The id only
# matters for the input_ids tensor, never for the loss: loss is masked
# at every pad position (loss_mask=0 wherever labels==-100), so the
# value the model sees at a pad slot has no gradient effect. We use 0
# because it's a valid token id in every Gemma vocab; any non-special
# id would do. Override via the ``pad_id`` kwarg if your tokenizer
# requires a specific pad id.
DEFAULT_PAD_ID = 0



def _read_tokenized_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read all rows from a tokenized-sidecar JSONL.

    Strict: any malformed line aborts (rather than silently dropping
    rows from training data, which would be much harder to notice
    after the fact).
    """
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{line_no}: malformed JSONL row: {e}"
                )
    if not rows:
        raise ValueError(f"{path}: 0 rows after parsing — refusing to train on empty dataset")
    return rows


def _validate_row(row: Dict[str, Any], path: Path, idx: int) -> None:
    """Confirm the row has the columns the trainer needs."""
    for key in ("input_ids", "labels"):
        if key not in row:
            raise ValueError(
                f"{path}: row {idx} missing required field {key!r}. "
                "Re-run scripts/convert_traces_to_maxtext.py — it must "
                "have written the tokenized sidecar (the *.tokenized.jsonl "
                "filename, not *.jsonl)."
            )
    if len(row["input_ids"]) != len(row["labels"]):
        raise ValueError(
            f"{path}: row {idx} input_ids/labels length mismatch "
            f"({len(row['input_ids'])} vs {len(row['labels'])})."
        )


def _pad_row(
    row: Dict[str, Any],
    max_seq_length: int,
    pad_id: int,
) -> Dict[str, np.ndarray]:
    """Right-pad ``input_ids`` / ``labels`` to ``max_seq_length`` and
    derive ``loss_mask``.

    Truncates if longer than ``max_seq_length`` (the converter already
    truncates at its own ``--max-seq-length`` flag, so this is the
    final safety net).
    """
    ids = row["input_ids"][:max_seq_length]
    labels = row["labels"][:max_seq_length]
    pad_n = max_seq_length - len(ids)
    if pad_n > 0:
        ids = ids + [pad_id] * pad_n
        labels = labels + [LABEL_IGNORE_ID] * pad_n
    return {
        "input_ids": np.asarray(ids, dtype=np.int32),
        "labels": np.asarray(labels, dtype=np.int32),
        "loss_mask": np.asarray(
            [1 if lab != LABEL_IGNORE_ID else 0 for lab in labels],
            dtype=np.float32,
        ),
    }


def _stack_batch(rows: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Stack per-row arrays into ``(B, T)`` batched arrays."""
    return {
        "input_ids": np.stack([r["input_ids"] for r in rows], axis=0),
        "labels": np.stack([r["labels"] for r in rows], axis=0),
        "loss_mask": np.stack([r["loss_mask"] for r in rows], axis=0),
    }


def _shuffled_indices(n: int, rng: random.Random) -> List[int]:
    idx = list(range(n))
    rng.shuffle(idx)
    return idx


def _train_iter(
    rows: List[Dict[str, Any]],
    global_batch_size: int,
    max_seq_length: int,
    pad_id: int,
    seed: int,
) -> Iterator[Dict[str, np.ndarray]]:
    """Infinite shuffled iterator. Drops a partial last batch each epoch
    so every batch has exactly ``global_batch_size`` rows."""
    rng = random.Random(seed)
    epoch = 0
    while True:
        idx = _shuffled_indices(len(rows), rng)
        for start in range(0, len(idx) - global_batch_size + 1, global_batch_size):
            chunk = [rows[i] for i in idx[start:start + global_batch_size]]
            padded = [_pad_row(r, max_seq_length, pad_id) for r in chunk]
            yield _stack_batch(padded)
        epoch += 1
        # Re-derive RNG from seed + epoch so a preempt resume that
        # restarts the iterator gets the same shuffles. The training
        # loop tracks step count separately, so it can fast-forward if
        # needed.
        rng = random.Random(seed + epoch)


def _eval_iter(
    rows: List[Dict[str, Any]],
    global_batch_size: int,
    max_seq_length: int,
    pad_id: int,
) -> Iterator[Dict[str, np.ndarray]]:
    """One-pass eval iterator, no shuffle. Drops the last partial batch
    so every yielded batch has exactly ``global_batch_size`` rows
    (matches the train shape, keeps jit cache hot)."""
    for start in range(0, len(rows) - global_batch_size + 1, global_batch_size):
        chunk = rows[start:start + global_batch_size]
        padded = [_pad_row(r, max_seq_length, pad_id) for r in chunk]
        yield _stack_batch(padded)


def _resolve_path(dataset_dir: str, override: str | None, split: str) -> Path:
    """Pick the tokenized-sidecar path for a split.

    ``override`` is the explicit ``--train-file`` / ``--val-file`` from
    the trainer's YAML — usually ``data/sft/train.jsonl`` (the
    raw-messages split). The matching tokenized sidecar is
    ``data/sft/maxtext/<split>.tokenized.jsonl`` written by the
    converter to ``dataset_dir``. We always prefer the tokenized one
    here; ``override`` is accepted for forward-compat but currently
    only used in error messages.
    """
    cand = Path(dataset_dir) / f"{split}.tokenized.jsonl"
    if cand.is_file():
        return cand
    raise FileNotFoundError(
        f"tokenized sidecar not found at {cand}. The trainer reads the "
        "pre-tokenized rows produced by convert_traces_to_maxtext.py "
        f"(file pattern: <dataset_dir>/<split>.tokenized.jsonl). "
        f"override hint: train-file YAML key was {override!r}, but the "
        "loader keys off dataset_dir, not train_file."
    )


def build_iterators(
    *,
    dataset_dir: str,
    train_file: str | None,
    val_file: str | None,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    max_seq_length: int,
    seed: int,
    pad_id: int = DEFAULT_PAD_ID,
) -> Tuple[Iterator[Dict[str, np.ndarray]], Iterator[Dict[str, np.ndarray]], int]:
    """Build the (train_iter, eval_iter, steps_per_epoch) triple the
    trainer expects.

    Args:
        dataset_dir: directory containing ``train.tokenized.jsonl`` and
            ``val.tokenized.jsonl`` (output of
            ``scripts/convert_traces_to_maxtext.py``).
        train_file / val_file: the YAML's data.train_file / data.val_file
            paths. Currently only used in error messages — the actual
            loader resolves files under ``dataset_dir`` because that's
            where the converter wrote the tokenized sidecars.
        per_device_batch_size, gradient_accumulation_steps: combined to
            give ``global_batch_size``. Must divide the train row count
            (we drop the partial last batch each epoch to keep the
            jit'd train_step's batch shape constant).
        max_seq_length: pad/truncate every row to this length.
        seed: shuffles deterministically. ``seed + epoch`` derives each
            epoch's shuffle so a multi-epoch run is reproducible.
        pad_id: token id used for right-pad on inputs. Loss is masked
            on pad positions so the value doesn't affect training.

    Returns:
        ``(train_iter, eval_iter, steps_per_epoch)``.
        ``train_iter`` is infinite; the trainer caps via ``total_steps =
        num_epochs * steps_per_epoch``. ``eval_iter`` is finite (one
        pass over the val set).

    Raises:
        FileNotFoundError if the tokenized sidecars are missing. This
        should normally have been caught earlier by
        ``train_lora_maxtext.py``'s own existence check, but having the
        same guard here keeps unit tests honest.
        ValueError if either split has fewer rows than
        ``global_batch_size``. Both iterators drop the partial last
        batch, so a too-small split would yield zero batches — training
        would crash on the empty step count and eval would silently
        emit no line at all. We fail fast here with a fix-it message
        pointing at per_device_batch_size / gradient_accumulation_steps
        / data.val_ratio.
    """
    train_path = _resolve_path(dataset_dir, train_file, "train")
    val_path = _resolve_path(dataset_dir, val_file, "val")

    train_rows = _read_tokenized_jsonl(train_path)
    val_rows = _read_tokenized_jsonl(val_path)

    for i, row in enumerate(train_rows, 1):
        _validate_row(row, train_path, i)
    for i, row in enumerate(val_rows, 1):
        _validate_row(row, val_path, i)

    global_batch_size = per_device_batch_size * gradient_accumulation_steps
    if global_batch_size <= 0:
        raise ValueError(
            f"global_batch_size must be positive, got "
            f"per_device={per_device_batch_size} * accum={gradient_accumulation_steps}"
        )

    steps_per_epoch = len(train_rows) // global_batch_size
    if steps_per_epoch <= 0:
        raise ValueError(
            f"train set ({len(train_rows)} rows) too small for "
            f"global_batch_size={global_batch_size}; reduce "
            "per_device_batch_size or gradient_accumulation_steps."
        )

    # Mirror of the train guard. _eval_iter drops partial batches, so a
    # val split smaller than global_batch_size would yield zero batches
    # and the trainer's eval_losses list would stay empty — no eval line
    # logged, silent. Fail loudly instead.
    if len(val_rows) < global_batch_size:
        raise ValueError(
            f"val set ({len(val_rows)} rows) too small for global_batch_size={global_batch_size}; "
            "eval would yield zero batches. Either reduce per_device_batch_size * gradient_accumulation_steps "
            "or use a larger val set (consider lowering data.val_ratio in the YAML)."
        )

    train = _train_iter(train_rows, global_batch_size, max_seq_length, pad_id, seed)
    eval_ = _eval_iter(val_rows, global_batch_size, max_seq_length, pad_id)
    return train, eval_, steps_per_epoch
