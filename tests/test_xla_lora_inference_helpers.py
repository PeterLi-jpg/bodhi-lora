"""Tests for the small handful of pure-logic helpers in scripts/_xla_lora_inference.py.

The bulk of _xla_lora_inference.py is the merge-then-serve pipeline that
needs torch / transformers / peft / vllm-tpu, so it can only run on a TPU
host with those packages. The constructor and lifecycle-state code, however,
are plain Python: argument validation, path normalization, env-var defaulting,
and the rmtree branch in stop(). This module pins those.

If a future refactor moves all of this logic into the merge step (so the
class genuinely has no pure-logic surface area left), this file should
be replaced with a single skip-with-reason test documenting the gap.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))


def _load_module(monkeypatch):
    """Import scripts._xla_lora_inference with the heavy/optional deps mocked.

    The module imports VLLMEngine from _vllm_engine at module level, so we
    mock that too — none of the constructor tests actually instantiate
    VLLMEngine, but the import has to succeed.
    """
    import importlib

    # _vllm_engine itself imports requests and other things at module load
    # time on some configs; replace it with a mock module exporting a
    # MagicMock VLLMEngine class.
    mock_vllm = MagicMock()
    mock_vllm.VLLMEngine = MagicMock()
    monkeypatch.setitem(sys.modules, "_vllm_engine", mock_vllm)

    sys.modules.pop("_xla_lora_inference", None)
    sys.modules.pop("scripts._xla_lora_inference", None)
    return importlib.import_module("_xla_lora_inference")


# ── constructor: argument validation ───────────────────────────────────────

def test_constructor_requires_lora_path(monkeypatch):
    """The class is meaningless without a LoRA adapter — for base-model
    serving the caller should use VLLMEngine directly. The constructor
    fails fast with ValueError so a misconfigured launcher script can't
    spin up a 30-minute merge step that does nothing."""
    mod = _load_module(monkeypatch)

    with pytest.raises(ValueError, match="lora_path"):
        mod.XLALoRAEngine(model="google/medgemma-27b-text-it")

    with pytest.raises(ValueError, match="lora_path"):
        mod.XLALoRAEngine(model="google/medgemma-27b-text-it", lora_path="")

    with pytest.raises(ValueError, match="lora_path"):
        mod.XLALoRAEngine(model="google/medgemma-27b-text-it", lora_path=None)


def test_constructor_normalizes_relative_lora_path(monkeypatch, tmp_path):
    """lora_path is stored via os.path.realpath so downstream consumers
    (the inner VLLMEngine, the rmtree call) get an absolute path. A
    relative path on launch + a cwd change between start() and stop()
    used to leak the merge dir; pinning realpath stops that regression."""
    mod = _load_module(monkeypatch)

    rel = tmp_path / "adapter"
    rel.mkdir()
    monkeypatch.chdir(tmp_path)

    eng = mod.XLALoRAEngine(model="google/medgemma-27b-text-it",
                            lora_path="adapter")

    assert os.path.isabs(eng.lora_path)
    assert eng.lora_path == os.path.realpath(str(rel))


def test_constructor_stores_forwarded_kwargs(monkeypatch):
    """The constructor's job is mostly attribute storage; pin that the
    fields the inner VLLMEngine needs are kept verbatim. A typo here would
    silently use the default max_model_len/tp_size on TPU, which can cause
    OOM or graph-capture differences."""
    mod = _load_module(monkeypatch)

    eng = mod.XLALoRAEngine(
        model="base-model",
        tp_size=8,
        max_model_len=8192,
        lora_path="/tmp/adapter",
        port=9001,
        hf_token="hf_dummy",
        enforce_eager=False,
        merged_dir="/mnt/cache",
    )
    assert eng.base_model == "base-model"
    assert eng.tp_size == 8
    assert eng.max_model_len == 8192
    assert eng.port == 9001
    assert eng.hf_token == "hf_dummy"
    assert eng.enforce_eager is False
    assert eng._merged_dir_root == "/mnt/cache"
    # lifecycle attrs default to None until start()
    assert eng._merged_path is None
    assert eng._inner is None


def test_constructor_falls_back_to_env_hf_token(monkeypatch):
    """When the caller doesn't pass hf_token, we read $HF_TOKEN. The
    fallback is what keeps the launcher scripts portable: they all set
    HF_TOKEN once and let every subprocess pick it up."""
    mod = _load_module(monkeypatch)

    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    eng = mod.XLALoRAEngine(model="m", lora_path="/tmp/adapter")
    assert eng.hf_token == "hf_from_env"

    # Explicit empty string -> env fallback (matches the `or` semantics).
    eng2 = mod.XLALoRAEngine(model="m", lora_path="/tmp/adapter", hf_token="")
    assert eng2.hf_token == "hf_from_env"


def test_constructor_hf_token_default_when_no_env(monkeypatch):
    """If neither caller nor env supplies a token, hf_token is the empty
    string (not None) — the inner VLLMEngine treats falsy as 'public model'."""
    mod = _load_module(monkeypatch)

    monkeypatch.delenv("HF_TOKEN", raising=False)
    eng = mod.XLALoRAEngine(model="m", lora_path="/tmp/adapter")
    assert eng.hf_token == ""


# ── stop(): rmtree branch is reachable without a real merge ────────────────

def test_stop_cleans_up_merged_path_when_no_inner(monkeypatch, tmp_path):
    """stop() must rmtree the merged dir even if start() never set up the
    inner VLLMEngine (e.g. start() crashed mid-merge after _merged_path was
    assigned). Pinning this prevents the 54 GB checkpoint from leaking
    onto the boot disk on every preemption."""
    mod = _load_module(monkeypatch)

    eng = mod.XLALoRAEngine(model="m", lora_path="/tmp/adapter")

    # Simulate "merge succeeded, then engine startup raised".
    fake_merged = tmp_path / "bodhi_merged_x"
    fake_merged.mkdir()
    (fake_merged / "weights.safetensors").write_bytes(b"placeholder")
    eng._merged_path = str(fake_merged)
    eng._inner = None

    eng.stop()

    assert not fake_merged.exists()
    assert eng._merged_path is None


def test_stop_is_idempotent_when_nothing_set(monkeypatch):
    """Calling stop() twice (or before start()) must be a no-op, not an
    AttributeError. The eval scripts call stop() in finally blocks."""
    mod = _load_module(monkeypatch)

    eng = mod.XLALoRAEngine(model="m", lora_path="/tmp/adapter")
    # No _merged_path, no _inner. Should silently return.
    eng.stop()
    eng.stop()
    assert eng._merged_path is None
    assert eng._inner is None


def test_stop_calls_inner_stop_then_rmtree(monkeypatch, tmp_path):
    """Order matters: tear down the vLLM server first (releases TPU chips),
    then rm the merge dir. If we rm'd first, vLLM's open file handles
    would prevent the rmtree from completing on the boot disk."""
    mod = _load_module(monkeypatch)

    eng = mod.XLALoRAEngine(model="m", lora_path="/tmp/adapter")

    call_order = []
    inner = MagicMock()
    inner.stop.side_effect = lambda: call_order.append("inner_stop")

    fake_merged = tmp_path / "merged"
    fake_merged.mkdir()
    eng._merged_path = str(fake_merged)
    eng._inner = inner

    # rmtree happens via shutil; observe via the directory disappearing.
    eng.stop()

    assert call_order == ["inner_stop"]
    assert not fake_merged.exists()
    assert eng._inner is None
    assert eng._merged_path is None


# ── context-manager protocol passes through to start/stop ──────────────────

def test_context_manager_calls_start_and_stop(monkeypatch):
    """The with-statement contract is part of the public API used by
    scripts/eval_healthbench.py — pin it so a refactor can't accidentally
    drop __enter__ / __exit__ and leave callers with no resource
    management."""
    mod = _load_module(monkeypatch)

    eng = mod.XLALoRAEngine(model="m", lora_path="/tmp/adapter")

    started = MagicMock(return_value=eng)
    stopped = MagicMock()
    eng.start = started
    eng.stop = stopped

    with eng as got:
        assert got is eng
    started.assert_called_once_with()
    stopped.assert_called_once_with()
