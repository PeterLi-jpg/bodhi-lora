"""Unit tests for the MedGemma -> MaxText converter (Unit 4).

We don't download the real 54 GB MedGemma weights here. Instead we exercise
two things:

1. The pure layer-name remap: MedGemma's text-only `Gemma3ForCausalLM`
   state_dict keys (`model.layers.X.*`, `model.embed_tokens.weight`,
   `model.norm.weight`) map onto the multimodal naming
   (`model.language_model.layers.X.*`) that MaxText's gemma3-27b mapping
   expects. We feed in a tiny 1-layer fake state_dict and assert every key
   ends up where MaxText would look for it.

2. The CLI / argparse plumbing: --help, default model name, --output
   required, sensible MaxText arg construction.

If MaxText isn't on PYTHONPATH (the common local-dev case), `run()` returns
exit code 2 with a clear message; we test that too instead of skipping.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def converter():
    """Import the converter module fresh, mocking heavy deps if absent."""
    sys.modules.pop("scripts.convert_medgemma_to_maxtext", None)
    return importlib.import_module("scripts.convert_medgemma_to_maxtext")


def _fake_text_only_state_dict(num_layers: int = 1, hidden: int = 8) -> dict:
    """Mimic the keys `Gemma3ForCausalLM.state_dict()` produces for medgemma."""
    rng = np.random.default_rng(0)
    sd = {
        "model.embed_tokens.weight": rng.standard_normal((32, hidden), dtype=np.float32),
        "model.norm.weight": rng.standard_normal((hidden,), dtype=np.float32),
        "lm_head.weight": rng.standard_normal((32, hidden), dtype=np.float32),
    }
    inner_keys = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    for i in range(num_layers):
        for k in inner_keys:
            sd[f"model.layers.{i}.{k}"] = rng.standard_normal((hidden,), dtype=np.float32)
    return sd


def test_remap_text_only_to_multimodal_layers(converter):
    """Every `model.layers.{i}.X` key gets a `language_model.` prefix injected."""
    sd = _fake_text_only_state_dict(num_layers=2, hidden=4)
    out = converter.remap_text_only_to_multimodal(sd)

    # Sanity: keys aren't lost or duplicated.
    assert len(out) == len(sd)

    # Layer params relocated.
    for i in range(2):
        old = f"model.layers.{i}.self_attn.q_proj.weight"
        new = f"model.language_model.layers.{i}.self_attn.q_proj.weight"
        assert old not in out, "old text-only key should be gone"
        assert new in out, "remapped multimodal-style key should exist"
        # And it should be the same tensor: equality, not just shape.
        assert np.array_equal(out[new], sd[old])


def test_remap_text_only_to_multimodal_embed_and_norm(converter):
    """Embedding and final-norm keys also get re-prefixed."""
    sd = _fake_text_only_state_dict(num_layers=1, hidden=4)
    out = converter.remap_text_only_to_multimodal(sd)

    assert "model.embed_tokens.weight" not in out
    assert "model.norm.weight" not in out
    assert "model.language_model.embed_tokens.weight" in out
    assert "model.language_model.norm.weight" in out
    assert np.array_equal(
        out["model.language_model.embed_tokens.weight"],
        sd["model.embed_tokens.weight"],
    )


def test_remap_passes_through_lm_head_unchanged(converter):
    """`lm_head.weight` is NOT under `model.` so it must pass through verbatim."""
    sd = _fake_text_only_state_dict(num_layers=1, hidden=4)
    out = converter.remap_text_only_to_multimodal(sd)

    assert "lm_head.weight" in out
    assert np.array_equal(out["lm_head.weight"], sd["lm_head.weight"])


def test_remap_is_idempotent_on_already_multimodal_keys(converter):
    """Running the remap twice should be a no-op the second time."""
    sd = _fake_text_only_state_dict(num_layers=1, hidden=4)
    once = converter.remap_text_only_to_multimodal(sd)
    twice = converter.remap_text_only_to_multimodal(once)
    assert set(once.keys()) == set(twice.keys())
    for k in once:
        assert np.array_equal(once[k], twice[k])


def test_parse_args_defaults(converter):
    """--output is required; --hf-path defaults to the public MedGemma repo."""
    with pytest.raises(SystemExit):
        converter.parse_args([])  # missing --output should error

    ns = converter.parse_args(["--output", "/tmp/mt-out"])
    assert ns.output == "/tmp/mt-out"
    assert ns.hf_path == "google/medgemma-27b-text-it"
    assert ns.model_name == "gemma3-27b"
    assert ns.save_dtype == "bfloat16"
    assert ns.lazy_load is False


def test_parse_args_overrides(converter):
    ns = converter.parse_args([
        "--hf-path", "/local/medgemma",
        "--output", "gs://my-bucket/mt",
        "--save-dtype", "float32",
        "--lazy-load",
    ])
    assert ns.hf_path == "/local/medgemma"
    assert ns.output == "gs://my-bucket/mt"
    assert ns.save_dtype == "float32"
    assert ns.lazy_load is True


def test_build_maxtext_args_shape(converter):
    """The forwarded MaxText args carry the expected key=value overrides."""
    args = converter._build_maxtext_args(
        model_name="gemma3-27b",
        output="/tmp/out",
        hf_token="hf_xxx",
        extra=["per_device_batch_size=1"],
    )
    # arg[0] is the program name; everything after is parsed by pyconfig.
    body = args[1:]
    assert body[0].endswith("base.yml")
    assert "model_name=gemma3-27b" in body
    assert "base_output_directory=/tmp/out" in body
    assert "hardware=cpu" in body
    assert "skip_jax_distributed_system=True" in body
    assert "scan_layers=False" in body
    assert "use_multimodal=False" in body
    assert "hf_access_token=hf_xxx" in body
    assert "per_device_batch_size=1" in body


def test_run_returns_2_when_maxtext_missing(converter, monkeypatch, capsys):
    """If MaxText can't be imported, run() prints a clear error and returns 2."""
    import builtins

    # Drop any cached maxtext modules so the import statement re-runs.
    for k in list(sys.modules):
        if k == "maxtext" or k.startswith("maxtext."):
            monkeypatch.delitem(sys.modules, k, raising=False)

    real_import = builtins.__import__

    def _blocked_import(name, *a, **kw):
        if name == "maxtext" or name.startswith("maxtext."):
            raise ImportError(f"blocked-by-test: {name}")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    ns = converter.parse_args(["--output", "/tmp/out"])
    rc = converter.run(ns)
    assert rc == 2
    assert "MaxText is not importable" in capsys.readouterr().err


def test_run_invokes_maxtext_with_patched_loader(converter, monkeypatch):
    """End-to-end (with mocked MaxText): run() patches the HF loader before
    calling to_maxtext.main(), and restores the original after main returns.

    To prove the patching works, we capture (inside main()'s side_effect)
    what the bound loader produces when called with a fake state_dict.
    """

    # Build fake MaxText modules.
    fake_ckpt_utils = MagicMock()
    fake_ckpt_utils.load_hf_dict_from_transformers = MagicMock(
        name="orig-on-utils",
        return_value=_fake_text_only_state_dict(num_layers=1, hidden=4),
    )

    fake_to_maxtext = MagicMock()
    # Mirror the real to_maxtext: it `from ...utils import load_hf_dict_from_transformers`,
    # so the bound name on the to_maxtext module starts as the same object.
    fake_to_maxtext.load_hf_dict_from_transformers = (
        fake_ckpt_utils.load_hf_dict_from_transformers
    )

    captured = {}

    def _main_side_effect(args, **kwargs):
        # The patched loader should be reachable both via the utils module
        # (where run() rebinds it) and via to_maxtext (where the bound name
        # was rebound). Real to_maxtext.main calls the bound name, so check
        # *that* one in particular.
        bound = fake_to_maxtext.load_hf_dict_from_transformers
        captured["sd"] = bound("any-repo", token="ignored")
        captured["args"] = list(args)
        captured["kwargs"] = dict(kwargs)
        return None

    fake_to_maxtext.main = MagicMock(side_effect=_main_side_effect)

    fake_pkg = MagicMock()
    fake_pkg.checkpoint_conversion = MagicMock()
    fake_pkg.checkpoint_conversion.to_maxtext = fake_to_maxtext
    fake_pkg.checkpoint_conversion.utils = MagicMock()
    fake_pkg.checkpoint_conversion.utils.utils = fake_ckpt_utils

    monkeypatch.setitem(sys.modules, "maxtext", fake_pkg)
    monkeypatch.setitem(sys.modules, "maxtext.checkpoint_conversion", fake_pkg.checkpoint_conversion)
    monkeypatch.setitem(
        sys.modules,
        "maxtext.checkpoint_conversion.to_maxtext",
        fake_to_maxtext,
    )
    monkeypatch.setitem(
        sys.modules,
        "maxtext.checkpoint_conversion.utils",
        fake_pkg.checkpoint_conversion.utils,
    )
    monkeypatch.setitem(
        sys.modules,
        "maxtext.checkpoint_conversion.utils.utils",
        fake_ckpt_utils,
    )

    ns = converter.parse_args([
        "--output", "/tmp/out",
        "--hf-path", "google/medgemma-27b-text-it",
        "--hf-token", "hf_test_token",
    ])
    rc = converter.run(ns)
    assert rc == 0

    fake_to_maxtext.main.assert_called_once()
    assert captured["kwargs"]["hf_model_path"] == "google/medgemma-27b-text-it"
    assert captured["kwargs"]["lazy_load_tensors"] is False
    assert captured["kwargs"]["eager_load_method"] == "transformers"

    # The state_dict returned through the patched loader is remapped.
    sd = captured["sd"]
    assert "model.language_model.embed_tokens.weight" in sd
    assert "model.language_model.layers.0.self_attn.q_proj.weight" in sd
    assert "model.embed_tokens.weight" not in sd
    assert "model.layers.0.self_attn.q_proj.weight" not in sd
    # Things that don't match the prefix list pass through.
    assert "lm_head.weight" in sd

    # After run() returns, the original loader binding is restored on both
    # the utils module and the to_maxtext module.
    orig = fake_ckpt_utils.load_hf_dict_from_transformers
    assert fake_to_maxtext.load_hf_dict_from_transformers is orig
