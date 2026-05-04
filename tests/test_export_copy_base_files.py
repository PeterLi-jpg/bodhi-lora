"""Tests for the base-model file copy step of the tunix LoRA exporter.

The exporter writes ``adapter_config.json`` + ``adapter_model.safetensors``
into the adapter directory, but Stage 4 (vLLM-TPU eval) calls
``AutoTokenizer.from_pretrained(adapter_dir)`` and needs ``config.json``
plus the full tokenizer set in that same directory. ``copy_base_model_files``
pulls those files from the base model's HF snapshot.

These tests verify the copy logic without touching the network: we
monkeypatch ``huggingface_hub.snapshot_download`` to return a local fake
snapshot directory and assert the right files land in the output dir
with the expected log lines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Files we expect copy_base_model_files to consider, in source order.
EXPECTED_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


def _make_fake_snapshot(snapshot_dir: Path, files: dict[str, bytes]) -> None:
    """Populate ``snapshot_dir`` with ``files`` (name -> content)."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (snapshot_dir / name).write_bytes(content)


def test_copy_base_model_files_copies_present_skips_absent(tmp_path, capsys, monkeypatch):
    """Files present in the snapshot should land in output_dir; absent ones
    should be skipped with a "no <file> in base, skipping" log line.
    """
    from scripts import export_tunix_lora_to_peft as exporter

    # Build a fake snapshot that has SOME but not all base files. Notably
    # missing: generation_config.json and chat_template.jinja, to exercise
    # the skip-on-absent path.
    snapshot_dir = tmp_path / "snapshot"
    fake_files = {
        "config.json": b'{"model_type": "gemma3"}',
        # Sentencepiece is binary; non-utf8 bytes are fine for the copy.
        "tokenizer.model": b"\x00\x01\x02fake-sentencepiece",
        "tokenizer.json": b'{"version": "1.0"}',
        "tokenizer_config.json": b'{"tokenizer_class": "GemmaTokenizer"}',
        "special_tokens_map.json": b'{"bos_token": "<bos>"}',
        "added_tokens.json": b"{}",
    }
    _make_fake_snapshot(snapshot_dir, fake_files)

    # Stub snapshot_download at the import site (the function does a local
    # ``from huggingface_hub import snapshot_download``, so we monkeypatch
    # the module attribute itself).
    import huggingface_hub  # noqa: F401  (ensure module exists; if not, the test will skip)

    def fake_snapshot_download(repo_id, allow_patterns=None, **kwargs):
        # Caller should ask only for the small files set; ``snapshot_download``
        # ignores patterns it doesn't match, so we don't need to filter --
        # just hand back the dir.
        assert repo_id == "google/medgemma-27b-text-it"
        # The exporter passes allow_patterns as a list of file names.
        assert allow_patterns is not None
        return str(snapshot_dir)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", fake_snapshot_download
    )

    output_dir = tmp_path / "adapter"
    output_dir.mkdir()

    exporter.copy_base_model_files(
        output_dir=output_dir,
        base_model="google/medgemma-27b-text-it",
    )

    # Files that existed in the snapshot must now exist in the adapter
    # dir, byte-for-byte equal to source.
    for name, content in fake_files.items():
        dst = output_dir / name
        assert dst.exists(), f"expected {name} to be copied into adapter dir"
        assert dst.read_bytes() == content, f"{name} contents do not match source"

    # Files that did NOT exist in the snapshot must not exist in the
    # adapter dir (we didn't put them there beforehand).
    for missing in ("generation_config.json", "chat_template.jinja"):
        assert not (output_dir / missing).exists(), (
            f"{missing} should not have been created when absent from snapshot"
        )

    captured = capsys.readouterr()
    log = captured.out

    # Every present file should have a [exporter] copied <file> line.
    for name in fake_files:
        assert f"[exporter] copied {name}" in log, (
            f"missing 'copied {name}' log line; got:\n{log}"
        )

    # Every absent file should have a (no <file> in base, skipping) line.
    for missing in ("generation_config.json", "chat_template.jinja"):
        assert f"[exporter] (no {missing} in base, skipping)" in log, (
            f"missing skip log line for {missing}; got:\n{log}"
        )


def test_copy_base_model_files_overwrites_existing(tmp_path, monkeypatch):
    """Stale trainer-exported tokenizer files in the adapter dir must be
    overwritten by the base-model copies (the bug we hit was a stale /
    incomplete tokenizer.model in the adapter dir).
    """
    from scripts import export_tunix_lora_to_peft as exporter

    snapshot_dir = tmp_path / "snapshot"
    fresh = b"FRESH-FROM-BASE"
    _make_fake_snapshot(
        snapshot_dir,
        {
            "config.json": fresh,
            "tokenizer.model": fresh,
        },
    )

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo_id, allow_patterns=None, **kw: str(snapshot_dir),
    )

    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    # Pre-existing stale copies (the bug condition).
    (output_dir / "config.json").write_bytes(b"STALE")
    (output_dir / "tokenizer.model").write_bytes(b"STALE")

    exporter.copy_base_model_files(
        output_dir=output_dir,
        base_model="google/medgemma-27b-text-it",
    )

    assert (output_dir / "config.json").read_bytes() == fresh
    assert (output_dir / "tokenizer.model").read_bytes() == fresh


def test_copy_base_model_files_resolves_symlinks(tmp_path, monkeypatch):
    """HF cache snapshots are symlinks into a blobs/ dir. The destination
    must be a regular file with the dereferenced bytes, so that deleting
    the cache later doesn't break the exported adapter.
    """
    from scripts import export_tunix_lora_to_peft as exporter

    blobs_dir = tmp_path / "blobs"
    blobs_dir.mkdir()
    blob = blobs_dir / "deadbeef"
    blob.write_bytes(b"real-config-bytes")

    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    # Symlink into blobs, mirroring the HF cache layout.
    (snapshot_dir / "config.json").symlink_to(blob)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo_id, allow_patterns=None, **kw: str(snapshot_dir),
    )

    output_dir = tmp_path / "adapter"
    output_dir.mkdir()

    exporter.copy_base_model_files(
        output_dir=output_dir,
        base_model="some/repo",
    )

    dst = output_dir / "config.json"
    assert dst.exists()
    # The destination must be a regular file (not a symlink) and contain
    # the dereferenced contents.
    assert not dst.is_symlink(), "destination should be a regular file"
    assert dst.read_bytes() == b"real-config-bytes"


def test_copy_base_model_files_swallows_snapshot_download_errors(tmp_path, capsys, monkeypatch):
    """When snapshot_download raises (no network, gated repo, etc.) the
    function must NOT crash the export. The safetensors + adapter_config
    blobs were already written before this call; we just log loudly.
    """
    from scripts import export_tunix_lora_to_peft as exporter

    def boom(repo_id, allow_patterns=None, **kw):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)

    output_dir = tmp_path / "adapter"
    output_dir.mkdir()

    # Should not raise.
    exporter.copy_base_model_files(
        output_dir=output_dir,
        base_model="some/repo",
    )

    log = capsys.readouterr().out
    assert "snapshot_download" in log
    assert "simulated network failure" in log
    # Nothing should have landed in the adapter dir.
    assert list(output_dir.iterdir()) == []
