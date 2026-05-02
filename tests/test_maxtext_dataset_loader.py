"""Tests for scripts/maxtext_lora/dataset_loader.py.

Pure-numpy code path — no JAX, no TPU. Confirms the JSONL → batched
arrays handoff produces the shapes/dtypes the trainer's jit'd
train_step expects, and that the loss_mask honors -100 on prompt
tokens (the masking contract from convert_traces_to_maxtext.py).
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.maxtext_lora import dataset_loader as dl


def _write_tokenized(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """Eight train rows + four val rows, shorter than max_seq_length so
    we can test the padding path."""
    train_rows = [
        {
            "input_ids": [10, 11, 12, 13, 14],
            # First three tokens are "prompt" (-100 masked); response is 13, 14.
            "labels": [-100, -100, -100, 13, 14],
            "prompt_id": f"train_{i}",
        }
        for i in range(8)
    ]
    val_rows = [
        {
            "input_ids": [20, 21, 22, 23],
            "labels": [-100, -100, 22, 23],
            "prompt_id": f"val_{i}",
        }
        for i in range(4)
    ]
    _write_tokenized(tmp_path / "train.tokenized.jsonl", train_rows)
    _write_tokenized(tmp_path / "val.tokenized.jsonl", val_rows)
    return tmp_path


def test_build_iterators_shapes_and_steps(fake_dataset: Path) -> None:
    """Train iter yields (B, T) batches; steps_per_epoch matches drop-last."""
    train_iter, eval_iter, steps_per_epoch = dl.build_iterators(
        dataset_dir=str(fake_dataset),
        train_file=None,
        val_file=None,
        per_device_batch_size=2,
        gradient_accumulation_steps=2,  # global batch = 4
        max_seq_length=8,
        seed=42,
    )
    assert steps_per_epoch == 8 // 4  # 2

    batch = next(train_iter)
    assert batch["input_ids"].shape == (4, 8)
    assert batch["labels"].shape == (4, 8)
    assert batch["loss_mask"].shape == (4, 8)
    assert batch["input_ids"].dtype == np.int32
    assert batch["labels"].dtype == np.int32
    assert batch["loss_mask"].dtype == np.float32


def test_padding_and_loss_mask(fake_dataset: Path) -> None:
    """Pad tokens get loss_mask=0; prompt tokens (-100 in labels) get
    loss_mask=0; response tokens get loss_mask=1."""
    train_iter, _, _ = dl.build_iterators(
        dataset_dir=str(fake_dataset),
        train_file=None,
        val_file=None,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,  # global batch = 1
        max_seq_length=8,
        seed=0,
    )
    batch = next(train_iter)
    # Each train row has 5 tokens, so positions 5..7 are padding.
    # Labels[3]=13, [4]=14 are real (mask=1); [0..2] are -100 (mask=0);
    # [5..7] are pad (-100, mask=0).
    expected_mask = [0, 0, 0, 1, 1, 0, 0, 0]
    assert list(batch["loss_mask"][0]) == expected_mask
    # Loss mask aligns with labels != -100.
    assert ((batch["labels"][0] == -100) == (batch["loss_mask"][0] == 0)).all()


def test_eval_iter_finite_no_shuffle(fake_dataset: Path) -> None:
    """Eval iter visits each row at most once, in order. With 4 val rows
    and global_batch=2, we should get exactly 2 batches."""
    _, eval_iter, _ = dl.build_iterators(
        dataset_dir=str(fake_dataset),
        train_file=None,
        val_file=None,
        per_device_batch_size=1,
        gradient_accumulation_steps=2,  # global batch = 2
        max_seq_length=8,
        seed=0,
    )
    batches = list(eval_iter)
    assert len(batches) == 2
    # Concatenated should preserve original order: prompt_ids val_0..3
    # appear by their input_ids head (20..23 starting tokens).
    starts = np.concatenate([b["input_ids"][:, 0] for b in batches])
    assert list(starts) == [20, 20, 20, 20]


def test_train_iter_is_infinite(fake_dataset: Path) -> None:
    """Pull more steps than steps_per_epoch — the iterator must keep
    going (re-shuffling each epoch under a derived seed)."""
    train_iter, _, steps_per_epoch = dl.build_iterators(
        dataset_dir=str(fake_dataset),
        train_file=None,
        val_file=None,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,  # global batch = 1
        max_seq_length=8,
        seed=42,
    )
    # 8 rows, batch=1 -> 8 steps/epoch. Pull 2 epochs' worth.
    pulled = [next(train_iter) for _ in range(2 * steps_per_epoch)]
    assert len(pulled) == 16


def test_seed_determinism(tmp_path: Path) -> None:
    """Same seed -> same shuffle order. Use rows with distinguishable
    first tokens so we can read the shuffle order off the first column
    of each batch."""
    rows = [
        {"input_ids": [100 + i, 0, 0], "labels": [-100, 0, 0]}
        for i in range(8)
    ]
    _write_tokenized(tmp_path / "train.tokenized.jsonl", rows)
    _write_tokenized(tmp_path / "val.tokenized.jsonl", rows[:4])

    def epoch_order(seed: int) -> list[int]:
        it, _, steps = dl.build_iterators(
            dataset_dir=str(tmp_path),
            train_file=None,
            val_file=None,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,  # global=1 -> 8 steps/epoch
            max_seq_length=4,
            seed=seed,
        )
        return [int(next(it)["input_ids"][0, 0]) for _ in range(steps)]

    assert epoch_order(7) == epoch_order(7)
    # 8! = 40320 possible orders — collision probability between two
    # different seeds is negligible.
    assert epoch_order(7) != epoch_order(99)


def test_missing_tokenized_file_raises(tmp_path: Path) -> None:
    """If the tokenized sidecar doesn't exist, fail fast with a clear
    pointer back to the converter."""
    # Empty dataset_dir
    with pytest.raises(FileNotFoundError, match="tokenized sidecar"):
        dl.build_iterators(
            dataset_dir=str(tmp_path),
            train_file=None,
            val_file=None,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            max_seq_length=8,
            seed=0,
        )


def test_malformed_jsonl_raises(tmp_path: Path) -> None:
    """A truncated/garbage line aborts loading rather than silently
    dropping rows."""
    (tmp_path / "train.tokenized.jsonl").write_text(
        '{"input_ids": [1,2,3], "labels": [-100,-100,3]}\n'
        "this is not json\n"
    )
    (tmp_path / "val.tokenized.jsonl").write_text(
        '{"input_ids": [1], "labels": [-100]}\n'
    )
    with pytest.raises(ValueError, match="malformed JSONL"):
        dl.build_iterators(
            dataset_dir=str(tmp_path),
            train_file=None,
            val_file=None,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            max_seq_length=8,
            seed=0,
        )


def test_too_small_for_batch_raises(tmp_path: Path) -> None:
    """Train set smaller than global batch must error, not yield zero
    steps and corrupt the trainer's compile cache."""
    _write_tokenized(
        tmp_path / "train.tokenized.jsonl",
        [{"input_ids": [1, 2], "labels": [-100, 2]}],
    )
    _write_tokenized(
        tmp_path / "val.tokenized.jsonl",
        [{"input_ids": [1, 2], "labels": [-100, 2]}],
    )
    with pytest.raises(ValueError, match="too small"):
        dl.build_iterators(
            dataset_dir=str(tmp_path),
            train_file=None,
            val_file=None,
            per_device_batch_size=4,
            gradient_accumulation_steps=4,  # global=16, train has 1 row
            max_seq_length=8,
            seed=0,
        )


def test_val_too_small_raises(tmp_path: Path) -> None:
    """Val set smaller than global_batch_size must error. _eval_iter
    drops partial last batches, so a too-small val split would yield
    zero batches and the trainer would silently log no eval line.
    Symmetric with the train-too-small guard."""
    # Train must clear its own guard (>= global_batch_size rows) so that
    # the val guard is the one that fires.
    _write_tokenized(
        tmp_path / "train.tokenized.jsonl",
        [
            {"input_ids": [1, 2], "labels": [-100, 2]},
            {"input_ids": [3, 4], "labels": [-100, 4]},
            {"input_ids": [5, 6], "labels": [-100, 6]},
            {"input_ids": [7, 8], "labels": [-100, 8]},
        ],
    )
    _write_tokenized(
        tmp_path / "val.tokenized.jsonl",
        [{"input_ids": [9, 10], "labels": [-100, 10]}],
    )
    with pytest.raises(ValueError, match=r"val set .* too small"):
        dl.build_iterators(
            dataset_dir=str(tmp_path),
            train_file=None,
            val_file=None,
            per_device_batch_size=2,
            gradient_accumulation_steps=2,  # global=4, val has 1 row
            max_seq_length=8,
            seed=0,
        )


def test_input_ids_labels_length_mismatch(tmp_path: Path) -> None:
    """A row with mismatched input_ids/labels lengths is corrupt — the
    converter never emits this — bail rather than train on noise."""
    _write_tokenized(
        tmp_path / "train.tokenized.jsonl",
        [{"input_ids": [1, 2, 3, 4], "labels": [-100, 2, 3]}],  # 4 vs 3
    )
    _write_tokenized(
        tmp_path / "val.tokenized.jsonl",
        [{"input_ids": [1, 2], "labels": [-100, 2]}],
    )
    with pytest.raises(ValueError, match="length mismatch"):
        dl.build_iterators(
            dataset_dir=str(tmp_path),
            train_file=None,
            val_file=None,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            max_seq_length=8,
            seed=0,
        )


def test_build_iterators_default_format_is_maxtext(fake_dataset: Path) -> None:
    """The output_format kwarg defaults to 'maxtext' so existing callers
    keep getting dict batches with no behavior change."""
    sig = inspect.signature(dl.build_iterators)
    assert sig.parameters["output_format"].default == "maxtext"

    # Functional check: no output_format -> dict batches with the legacy keys.
    train_iter, _, _ = dl.build_iterators(
        dataset_dir=str(fake_dataset),
        train_file=None,
        val_file=None,
        per_device_batch_size=2,
        gradient_accumulation_steps=2,
        max_seq_length=8,
        seed=42,
    )
    batch = next(train_iter)
    assert isinstance(batch, dict)
    assert set(batch.keys()) == {"input_ids", "labels", "loss_mask"}


@pytest.mark.skipif(
    importlib.util.find_spec("tunix") is None,
    reason="tunix not installed in this environment",
)
def test_build_iterators_tunix_format_yields_TrainingInput(fake_dataset: Path) -> None:
    """output_format='tunix' wraps each batch in a TrainingInput with
    input_tokens=input_ids and a boolean input_mask derived from labels
    != LABEL_IGNORE_ID."""
    # Sibling tests (test_filter_traces) install a MagicMock as
    # sys.modules['tqdm'] so they can collect on dev boxes without
    # tqdm. Tunix's transitive import chain hits
    # ``from tqdm.contrib.concurrent import thread_map`` which then
    # blows up because the mock isn't a real package. Heal sys.modules
    # by dropping any non-real tqdm entry before importing tunix —
    # leaves the real package free to import normally.
    import sys

    if "tqdm" in sys.modules:
        mod = sys.modules["tqdm"]
        if not getattr(mod, "__file__", None):
            del sys.modules["tqdm"]

    from tunix.sft.peft_trainer import TrainingInput

    train_iter, eval_iter, steps_per_epoch = dl.build_iterators(
        dataset_dir=str(fake_dataset),
        train_file=None,
        val_file=None,
        per_device_batch_size=2,
        gradient_accumulation_steps=2,  # global batch = 4
        max_seq_length=8,
        seed=42,
        output_format="tunix",
    )
    assert steps_per_epoch == 2

    batch = next(train_iter)
    assert isinstance(batch, TrainingInput)
    assert batch.input_tokens.shape == (4, 8)
    assert batch.input_mask.shape == (4, 8)
    assert batch.input_tokens.dtype == np.int32
    assert batch.input_mask.dtype == np.bool_
    # Mask must mark non-(-100) label positions as True.
    # Each train row has labels [-100,-100,-100,13,14] padded with -100 to 8;
    # so positions 3,4 are True and the rest False.
    expected_mask = np.array(
        [[False, False, False, True, True, False, False, False]] * 4
    )
    assert np.array_equal(batch.input_mask, expected_mask)

    # Eval iterator is also wrapped.
    eval_batch = next(eval_iter)
    assert isinstance(eval_batch, TrainingInput)


def test_build_iterators_invalid_format_raises(fake_dataset: Path) -> None:
    """Anything other than 'maxtext' / 'tunix' must error with a clear
    message — silently falling back would hide config typos."""
    with pytest.raises(ValueError, match="Unknown output_format"):
        dl.build_iterators(
            dataset_dir=str(fake_dataset),
            train_file=None,
            val_file=None,
            per_device_batch_size=2,
            gradient_accumulation_steps=2,
            max_seq_length=8,
            seed=0,
            output_format="garbage",
        )
