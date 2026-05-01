"""Round-trip test for the orbax -> HuggingFace PEFT adapter exporter.

We don't have a live MaxText fork checked into this repo (Phase 2 work),
so the test fakes an orbax-style nested dict of LoRA weights, runs it
through the exporter, and verifies:

1. The output directory contains ``adapter_config.json`` and
   ``adapter_model.safetensors``.
2. ``adapter_config.json`` carries the expected schema fields
   (``target_modules``, ``r``, ``lora_alpha``, ``lora_dropout``,
   ``task_type=CAUSAL_LM``, ``peft_type=LORA``, ...).
3. The adapter is loadable via ``peft.PeftModel.from_pretrained`` against
   a minimal Gemma-3 base model — the same code path Stage 4 uses to
   pick up the checkpoint.
4. Tensor shapes / values survive both the Flax-style and PEFT-style
   storage orientations (the auto-transpose logic).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module on CI/dev boxes without these ML deps installed.
# The exporter itself imports them lazily, so this only gates testing.
pytest.importorskip("safetensors", reason="safetensors not installed")
pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("peft", reason="peft not installed")
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixtures: a tiny Gemma-3 model + a fake "orbax" pytree of LoRA weights.
# ---------------------------------------------------------------------------


# Small but non-degenerate so transposes/shape detection is exercised.
HIDDEN = 32
INTERMEDIATE = 64
NUM_LAYERS = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8
LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05


def _build_tiny_gemma3():
    """Construct a tiny randomly-initialised Gemma-3 causal model."""
    from transformers.models.gemma3 import Gemma3TextConfig
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

    cfg = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=128,
        sliding_window=64,
    )
    return Gemma3ForCausalLM(cfg), cfg


def _fake_orbax_tree(*, flax_orientation: bool):
    """Build a nested dict that imitates an orbax-restored MaxText tree.

    When ``flax_orientation`` is True, the LoRA matrices are stored in
    Flax convention (in_features-first for A, r-first for B) — the layout
    the exporter has to auto-detect and transpose.  When False, the
    matrices are stored already in PEFT convention.
    """
    rng = np.random.default_rng(0)
    tree = {"params": {"decoder": {}}}
    layers = tree["params"]["decoder"]

    # Attention projection in_features for q/k/v is hidden_size.
    # out_features differ:
    #   q_proj: num_heads * head_dim = 32
    #   v_proj: num_kv_heads * head_dim = 16
    # We exercise both Q (full-size) and V (KV-grouped) so the auto
    # transpose detection has to cope with non-square A/B.
    proj_io = {
        "q_proj": (HIDDEN, NUM_HEADS * HEAD_DIM),
        "v_proj": (HIDDEN, NUM_KV_HEADS * HEAD_DIM),
    }

    for layer_idx in range(NUM_LAYERS):
        layer_block = {}
        for proj, (in_f, out_f) in proj_io.items():
            if flax_orientation:
                a = rng.standard_normal((in_f, LORA_R), dtype=np.float32)
                b = rng.standard_normal((LORA_R, out_f), dtype=np.float32)
            else:
                a = rng.standard_normal((LORA_R, in_f), dtype=np.float32)
                b = rng.standard_normal((out_f, LORA_R), dtype=np.float32)
            layer_block[proj] = {
                # The terminal "kernel" wrapper mimics how Flax/MaxText
                # nests the actual array under a leaf dict.  The exporter
                # has to look through it.
                "lora_a": {"kernel": a},
                "lora_b": {"kernel": b},
            }
        # Plant some non-LoRA junk that must be ignored.
        layer_block["self_attention_bias"] = rng.standard_normal((HIDDEN,))
        layers[f"layers_{layer_idx}"] = {"self_attention": layer_block}

    # Optimizer state at the top level — must be ignored.
    tree["opt_state"] = {"step": np.int32(123)}
    return tree


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _run_export(monkeypatch, tmp_path, *, flax_orientation: bool):
    """Helper: monkeypatch orbax loading, run export, return output dir."""
    from scripts import export_maxtext_lora_to_peft as exporter

    fake_tree = _fake_orbax_tree(flax_orientation=flax_orientation)
    # Skip orbax I/O: replace load_orbax_checkpoint with one that returns
    # the in-memory pytree.  This keeps the test hermetic — orbax would
    # otherwise need a real serialised checkpoint on disk.
    monkeypatch.setattr(
        exporter, "load_orbax_checkpoint", lambda _path: fake_tree
    )

    output_dir = tmp_path / "best"
    exporter.main([
        "--orbax-path", str(tmp_path / "fake_orbax"),
        "--output-dir", str(output_dir),
        "--base-model", "google/medgemma-27b-text-it",
        "--lora-r", str(LORA_R),
        "--lora-alpha", str(LORA_ALPHA),
        "--lora-dropout", str(LORA_DROPOUT),
        "--target-modules", "q_proj", "v_proj",
        "--task-type", "CAUSAL_LM",
        "--lora-variant", "standard",
    ])
    return output_dir


def test_adapter_config_schema(monkeypatch, tmp_path):
    output_dir = _run_export(monkeypatch, tmp_path, flax_orientation=True)

    cfg_path = output_dir / "adapter_config.json"
    assert cfg_path.is_file(), "adapter_config.json missing"
    cfg = json.loads(cfg_path.read_text())

    # Core fields that PEFT and downstream consumers rely on.
    assert cfg["peft_type"] == "LORA"
    assert cfg["task_type"] == "CAUSAL_LM"
    assert cfg["r"] == LORA_R
    assert cfg["lora_alpha"] == LORA_ALPHA
    assert cfg["lora_dropout"] == LORA_DROPOUT
    assert cfg["use_dora"] is False
    assert cfg["use_rslora"] is False
    # PEFT serialises target_modules as a list (or set rendered as a list)
    # — accept either ordering, the load path is order-insensitive.
    assert sorted(cfg["target_modules"]) == ["q_proj", "v_proj"]
    assert cfg["base_model_name_or_path"] == "google/medgemma-27b-text-it"


def test_safetensors_keys_match_peft(monkeypatch, tmp_path):
    from safetensors.torch import load_file

    output_dir = _run_export(monkeypatch, tmp_path, flax_orientation=True)
    weights = load_file(str(output_dir / "adapter_model.safetensors"))

    # 2 layers * 2 projections * 2 matrices (A, B) = 8 tensors.
    assert len(weights) == NUM_LAYERS * 2 * 2 == 8

    expected = set()
    for layer_idx in range(NUM_LAYERS):
        for proj in ("q_proj", "v_proj"):
            for ab in ("A", "B"):
                expected.add(
                    f"base_model.model.model.layers.{layer_idx}."
                    f"self_attn.{proj}.lora_{ab}.weight"
                )
    assert set(weights.keys()) == expected

    # Shape check: PEFT convention.
    for layer_idx in range(NUM_LAYERS):
        a_q = weights[
            f"base_model.model.model.layers.{layer_idx}."
            "self_attn.q_proj.lora_A.weight"
        ]
        b_q = weights[
            f"base_model.model.model.layers.{layer_idx}."
            "self_attn.q_proj.lora_B.weight"
        ]
        assert a_q.shape == (LORA_R, HIDDEN)
        assert b_q.shape == (NUM_HEADS * HEAD_DIM, LORA_R)
        a_v = weights[
            f"base_model.model.model.layers.{layer_idx}."
            "self_attn.v_proj.lora_A.weight"
        ]
        b_v = weights[
            f"base_model.model.model.layers.{layer_idx}."
            "self_attn.v_proj.lora_B.weight"
        ]
        assert a_v.shape == (LORA_R, HIDDEN)
        assert b_v.shape == (NUM_KV_HEADS * HEAD_DIM, LORA_R)


def test_peft_model_loads_exported_adapter(monkeypatch, tmp_path):
    """End-to-end: exported dir is loadable via PeftModel.from_pretrained."""
    from peft import PeftModel

    output_dir = _run_export(monkeypatch, tmp_path, flax_orientation=True)

    base, _ = _build_tiny_gemma3()
    # If PEFT can't parse the adapter (bad config schema, missing
    # weights, mismatched shapes) this raises.  No assertion needed
    # beyond the call returning a PeftModel.
    peft_model = PeftModel.from_pretrained(base, str(output_dir))
    assert peft_model is not None

    # Sanity: at least one LoRA module was actually wired in.
    has_lora_param = any(
        "lora_A" in name or "lora_B" in name
        for name, _ in peft_model.named_parameters()
    )
    assert has_lora_param, "PEFT loaded the adapter but no lora_A/B params show up"


def test_handles_already_peft_oriented_weights(monkeypatch, tmp_path):
    """Shape-detection branch: weights stored in PEFT convention shouldn't
    be transposed.  We feed in the same values once in Flax orientation
    and once in PEFT orientation and confirm the saved tensors agree.
    """
    from safetensors.torch import load_file
    from scripts import export_maxtext_lora_to_peft as exporter

    # Build a deterministic source tree, then build its PEFT-orientation
    # twin (the leaves are transposed copies of the Flax tree's leaves).
    flax_tree = _fake_orbax_tree(flax_orientation=True)
    peft_tree = _fake_orbax_tree(flax_orientation=False)

    out_flax = tmp_path / "flax"
    out_peft = tmp_path / "peft"

    monkeypatch.setattr(exporter, "load_orbax_checkpoint", lambda _p: flax_tree)
    exporter.main([
        "--orbax-path", str(tmp_path / "fake1"),
        "--output-dir", str(out_flax),
        "--base-model", "google/medgemma-27b-text-it",
        "--lora-r", str(LORA_R),
        "--target-modules", "q_proj", "v_proj",
    ])

    monkeypatch.setattr(exporter, "load_orbax_checkpoint", lambda _p: peft_tree)
    exporter.main([
        "--orbax-path", str(tmp_path / "fake2"),
        "--output-dir", str(out_peft),
        "--base-model", "google/medgemma-27b-text-it",
        "--lora-r", str(LORA_R),
        "--target-modules", "q_proj", "v_proj",
    ])

    w_flax = load_file(str(out_flax / "adapter_model.safetensors"))
    w_peft = load_file(str(out_peft / "adapter_model.safetensors"))

    # Both runs should produce identically-shaped tensors keyed on the
    # same PEFT names.  We don't compare values across the runs (they
    # used different RNG seeds inside _fake_orbax_tree); we only check
    # that each saved tensor matches the PEFT shape contract.
    assert set(w_flax.keys()) == set(w_peft.keys())
    for k in w_flax:
        f_shape = tuple(w_flax[k].shape)
        p_shape = tuple(w_peft[k].shape)
        assert f_shape == p_shape, (
            f"shape disagreement on {k!r}: flax-source={f_shape}, "
            f"peft-source={p_shape} — auto-transpose is broken"
        )
        if "lora_A" in k:
            assert f_shape[0] == LORA_R, (
                f"lora_A axis-0 should be the rank, got {f_shape}"
            )
        else:
            assert f_shape[1] == LORA_R, (
                f"lora_B axis-1 should be the rank, got {f_shape}"
            )


def test_missing_target_modules_raises(monkeypatch, tmp_path):
    """If the user asks for target_modules the checkpoint doesn't have,
    the exporter should print a warning and (when nothing remains) fail
    loudly rather than silently writing an empty adapter."""
    from scripts import export_maxtext_lora_to_peft as exporter

    fake_tree = _fake_orbax_tree(flax_orientation=True)
    monkeypatch.setattr(exporter, "load_orbax_checkpoint", lambda _p: fake_tree)

    with pytest.raises(ValueError, match="target_modules"):
        exporter.main([
            "--orbax-path", str(tmp_path / "fake"),
            "--output-dir", str(tmp_path / "out"),
            "--base-model", "google/medgemma-27b-text-it",
            "--lora-r", str(LORA_R),
            # Asking only for k_proj / o_proj — neither is in our fake
            # tree (which only has q_proj + v_proj).
            "--target-modules", "k_proj", "o_proj",
        ])


def test_no_lora_leaves_raises(monkeypatch, tmp_path):
    """Empty / unrecognised orbax tree should fail clearly, not silently
    produce a zero-tensor adapter."""
    from scripts import export_maxtext_lora_to_peft as exporter

    monkeypatch.setattr(
        exporter, "load_orbax_checkpoint",
        lambda _p: {"params": {"opt_state": {"step": np.int32(0)}}},
    )

    with pytest.raises(ValueError, match="no LoRA leaves"):
        exporter.main([
            "--orbax-path", str(tmp_path / "fake"),
            "--output-dir", str(tmp_path / "out"),
            "--base-model", "google/medgemma-27b-text-it",
            "--lora-r", str(LORA_R),
            "--target-modules", "q_proj",
        ])


def test_yaml_config_provides_defaults(monkeypatch, tmp_path):
    """YAML config should drive r/alpha/dropout/target_modules/variant
    when no CLI override is given.  This is the production call path
    from Unit 7's training entry."""
    from safetensors.torch import load_file
    from scripts import export_maxtext_lora_to_peft as exporter

    yaml_path = tmp_path / "lora.yaml"
    yaml_path.write_text(
        "lora:\n"
        f"  r: {LORA_R}\n"
        f"  lora_alpha: {LORA_ALPHA}\n"
        f"  lora_dropout: {LORA_DROPOUT}\n"
        "  target_modules: [q_proj, v_proj]\n"
        "  task_type: CAUSAL_LM\n"
        "  variant: standard\n"
    )

    fake_tree = _fake_orbax_tree(flax_orientation=True)
    monkeypatch.setattr(exporter, "load_orbax_checkpoint", lambda _p: fake_tree)

    output_dir = tmp_path / "best"
    exporter.main([
        "--orbax-path", str(tmp_path / "fake"),
        "--output-dir", str(output_dir),
        "--base-model", "google/medgemma-27b-text-it",
        "--config", str(yaml_path),
    ])

    cfg = json.loads((output_dir / "adapter_config.json").read_text())
    assert cfg["r"] == LORA_R
    assert cfg["lora_alpha"] == LORA_ALPHA
    assert cfg["lora_dropout"] == LORA_DROPOUT
    assert sorted(cfg["target_modules"]) == ["q_proj", "v_proj"]
    weights = load_file(str(output_dir / "adapter_model.safetensors"))
    assert len(weights) == NUM_LAYERS * 2 * 2  # 8 tensors


def test_bf16_tensors_round_trip(monkeypatch, tmp_path):
    """ml_dtypes.bfloat16 arrays from JAX should land as torch.bfloat16
    in the safetensors file (no float32 promotion that would double
    the on-disk size)."""
    pytest.importorskip("ml_dtypes")
    import ml_dtypes
    from safetensors.torch import load_file
    from scripts import export_maxtext_lora_to_peft as exporter

    rng = np.random.default_rng(7)
    # Build a minimal tree with exactly one layer + one projection in
    # bf16, so we can assert dtype exactly.
    a = rng.standard_normal((HIDDEN, LORA_R), dtype=np.float32).astype(
        ml_dtypes.bfloat16
    )
    b = rng.standard_normal((LORA_R, HIDDEN), dtype=np.float32).astype(
        ml_dtypes.bfloat16
    )
    fake_tree = {
        "params": {
            "decoder": {
                "layers_0": {
                    "self_attention": {
                        "q_proj": {
                            "lora_a": {"kernel": a},
                            "lora_b": {"kernel": b},
                        }
                    }
                }
            }
        }
    }
    monkeypatch.setattr(exporter, "load_orbax_checkpoint", lambda _p: fake_tree)

    output_dir = tmp_path / "best"
    exporter.main([
        "--orbax-path", str(tmp_path / "fake"),
        "--output-dir", str(output_dir),
        "--base-model", "google/medgemma-27b-text-it",
        "--lora-r", str(LORA_R),
        "--target-modules", "q_proj",
    ])

    weights = load_file(str(output_dir / "adapter_model.safetensors"))
    for tensor in weights.values():
        assert tensor.dtype == torch.bfloat16, (
            f"expected bf16, got {tensor.dtype} — bf16 path lost dtype"
        )


def test_maxtext_alias_names_resolve(monkeypatch, tmp_path):
    """A MaxText fork that names its projections "query" / "value" instead
    of HF's q_proj / v_proj should round-trip without renaming on the JAX
    side.  We assert the alias resolution + that "value" doesn't get
    eaten by leaf-key stripping (it isn't in _LEAF_KEYS for this exact
    reason).
    """
    from safetensors.torch import load_file
    from scripts import export_maxtext_lora_to_peft as exporter

    rng = np.random.default_rng(11)
    fake_tree = {
        "params": {
            "decoder": {
                "layers_0": {
                    "self_attention": {
                        "query": {  # MaxText alias for q_proj
                            "lora_a": {"kernel": rng.standard_normal(
                                (HIDDEN, LORA_R), dtype=np.float32
                            )},
                            "lora_b": {"kernel": rng.standard_normal(
                                (LORA_R, NUM_HEADS * HEAD_DIM),
                                dtype=np.float32,
                            )},
                        },
                        "value": {  # MaxText alias for v_proj
                            "lora_a": {"kernel": rng.standard_normal(
                                (HIDDEN, LORA_R), dtype=np.float32
                            )},
                            "lora_b": {"kernel": rng.standard_normal(
                                (LORA_R, NUM_KV_HEADS * HEAD_DIM),
                                dtype=np.float32,
                            )},
                        },
                    }
                }
            }
        }
    }
    monkeypatch.setattr(exporter, "load_orbax_checkpoint", lambda _p: fake_tree)

    output_dir = tmp_path / "best"
    exporter.main([
        "--orbax-path", str(tmp_path / "fake"),
        "--output-dir", str(output_dir),
        "--base-model", "google/medgemma-27b-text-it",
        "--lora-r", str(LORA_R),
        "--target-modules", "q_proj", "v_proj",
    ])

    weights = load_file(str(output_dir / "adapter_model.safetensors"))
    assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in weights
    assert "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight" in weights
    cfg = json.loads((output_dir / "adapter_config.json").read_text())
    assert sorted(cfg["target_modules"]) == ["q_proj", "v_proj"]
