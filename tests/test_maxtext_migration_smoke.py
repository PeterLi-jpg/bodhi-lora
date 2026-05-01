"""Smoke checks for the MaxText migration (Stage 3 LoRA fine-tune).

This is Unit 9 of the migration: it verifies that the modules from
Units 2-7 import cleanly and that the entry-point scripts build their
argparsers without crashing. We deliberately do NOT run conversion or
training here, those need a TPU VM and live data, and are covered by
Phase 2 of the migration plan.

Each test pytest-skips with a clear reason if the module under test
isn't merged yet, so this file is safe to land before the upstream
units do.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _import_or_skip(module_name):
    """Import module_name or pytest-skip with a clear "not merged yet" reason.

    We catch ImportError specifically so a real bug inside the module
    (e.g. SyntaxError, NameError at import time) still surfaces as a
    test failure rather than a silent skip.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        pytest.skip(f"{module_name} not available yet ({e}); upstream unit not merged")


def _help_runs(script_path):
    """Run ``python <script_path> --help`` and assert it exits 0.

    Uses a subprocess so we don't have to deal with argparse calling
    sys.exit() in-process. The script lives under scripts/, so we run
    it as a path rather than as a -m module to dodge any package-init
    side effects.
    """
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{script_path} --help exited {result.returncode}\n"
        f"stdout: {result.stdout[-500:]}\n"
        f"stderr: {result.stderr[-500:]}"
    )


# -- Unit 7: training entry point --


def test_train_lora_maxtext_help():
    """Unit 7: scripts/train_lora_maxtext.py --help works."""
    script = ROOT / "scripts" / "train_lora_maxtext.py"
    if not script.exists():
        pytest.skip(f"{script} not present; Unit 7 not merged")
    _help_runs(script)


# -- Units 2 & 3: maxtext_lora package internals --


def test_maxtext_lora_layer_imports():
    """Unit 2: scripts.maxtext_lora.layer imports cleanly."""
    _import_or_skip("scripts.maxtext_lora.layer")


def test_maxtext_lora_injector_imports():
    """Unit 3: scripts.maxtext_lora.injector imports cleanly."""
    _import_or_skip("scripts.maxtext_lora.injector")


# -- Units 4-6: conversion + export entry points --


def test_convert_medgemma_to_maxtext_help():
    """Unit 4: scripts/convert_medgemma_to_maxtext.py --help works."""
    script = ROOT / "scripts" / "convert_medgemma_to_maxtext.py"
    if not script.exists():
        pytest.skip(f"{script} not present; Unit 4 not merged")
    _help_runs(script)


def test_convert_traces_to_maxtext_help():
    """Unit 5: scripts/convert_traces_to_maxtext.py --help works."""
    script = ROOT / "scripts" / "convert_traces_to_maxtext.py"
    if not script.exists():
        pytest.skip(f"{script} not present; Unit 5 not merged")
    _help_runs(script)


def test_export_maxtext_lora_to_peft_help():
    """Unit 6: scripts/export_maxtext_lora_to_peft.py --help works.

    This script writes the HF PEFT adapter that Stage 4 reads. The
    contract it must satisfy is documented in
    docs/maxtext_migration.md.

    The script imports torch + safetensors + peft at module load time, so
    --help fails on dev/CI environments without those installed.
    Skip cleanly in that case rather than fail the smoke test.
    """
    script = ROOT / "scripts" / "export_maxtext_lora_to_peft.py"
    if not script.exists():
        pytest.skip(f"{script} not present; Unit 6 not merged")
    pytest.importorskip("torch", reason="torch not installed; --help imports torch")
    pytest.importorskip("safetensors", reason="safetensors not installed")
    pytest.importorskip("peft", reason="peft not installed; --help imports peft")
    _help_runs(script)


# -- doc sanity check --


def test_migration_doc_exists_and_is_readable():
    """The migration doc itself compiles cleanly (read-as-text).

    No real markdown linter in the repo, so this just confirms the
    file is non-empty and decodes as UTF-8, which is what the unit
    spec asks for.
    """
    doc = ROOT / "docs" / "maxtext_migration.md"
    assert doc.exists(), f"{doc} missing"
    text = doc.read_text(encoding="utf-8")
    assert len(text) > 0, f"{doc} is empty"
    # Spot-check the doc covers the five required sections.
    for marker in (
        "Why we forked MaxText",
        "What changed",
        "Run plan",
        "Checkpoint format",
        "Acceptance criterion",
    ):
        assert marker in text, f"{doc} missing required section: {marker!r}"
