"""Round-trip tests for the tunix + qwix orbax -> PEFT adapter exporter.

These tests synthesise an in-memory pytree shaped like what a tunix orbax
restore produces for a qwix-LoRA-wrapped Gemma-3 model, monkeypatch the
loader to return it, and inspect the PEFT directory the exporter writes.

We don't need a real orbax checkpoint on disk and we don't need torch /
peft / transformers; the exporter is deliberately torch-free so tests
run on the same TPU venv that hosts the trainer (which has neither).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _require_export_deps():
    """Per-test gate: skip cleanly if the deps the exporter touches are
    not installed.  Numpy is always available; safetensors needs to be
    importable for ``write_peft_adapter`` to actually write tensors.
    """
    pytest.importorskip("safetensors", reason="safetensors not installed")


# ---------------------------------------------------------------------------
# Tunix-shaped fixtures (separate q_einsum + kv_einsum case + qkv variant).
# ---------------------------------------------------------------------------

# Small but non-degenerate so transposes are exercised on different axes.
EMBED_DIM = 32
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8
NUM_LAYERS = 2
LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.0


def _fake_separate_qkv_tree(seed: int = 0):
    """Build a qwix-on-tunix orbax tree with separate q_einsum + kv_einsum.

    Mirrors the smoke config (num_heads != num_kv_heads, so q is its own
    einsum and k+v are packed into kv_einsum on axis 0 = [k, v]).
    """
    rng = np.random.default_rng(seed)
    layers: dict = {}
    for layer_idx in range(NUM_LAYERS):
        # q_einsum.w shape (N, D, H); lora_a (D, R); lora_b (R, N, H)
        q_a = rng.standard_normal((EMBED_DIM, LORA_R), dtype=np.float32)
        q_b = rng.standard_normal((LORA_R, NUM_HEADS, HEAD_DIM), dtype=np.float32)
        # kv_einsum.w shape (2, K, D, H); lora_a (D, R); lora_b (R, 2, K, H)
        kv_a = rng.standard_normal((EMBED_DIM, LORA_R), dtype=np.float32)
        kv_b = rng.standard_normal(
            (LORA_R, 2, NUM_KV_HEADS, HEAD_DIM), dtype=np.float32
        )
        layers[f"layers.{layer_idx}"] = {
            "attn": {
                "q_einsum": {"w_lora_a": q_a, "w_lora_b": q_b},
                "kv_einsum": {"w_lora_a": kv_a, "w_lora_b": kv_b},
            }
        }
    # Drop in some non-LoRA junk that must be ignored.
    layers["embedder"] = {"input_embedding": rng.standard_normal((64, EMBED_DIM))}
    return layers


def _fake_fused_qkv_tree(seed: int = 1):
    """Build an orbax tree with a fused qkv_einsum (num_heads == num_kv_heads).

    Used to verify the q+k+v split path; not the smoke config but a path
    we'd hit if the trainer ever runs against a non-GQA model.
    """
    rng = np.random.default_rng(seed)
    n = NUM_HEADS
    layers: dict = {}
    for layer_idx in range(NUM_LAYERS):
        a = rng.standard_normal((EMBED_DIM, LORA_R), dtype=np.float32)
        b = rng.standard_normal((LORA_R, 3, n, HEAD_DIM), dtype=np.float32)
        layers[f"layers.{layer_idx}"] = {
            "attn": {
                "qkv_einsum": {"w_lora_a": a, "w_lora_b": b},
            }
        }
    return layers


# ---------------------------------------------------------------------------
# Tests for path mapping
# ---------------------------------------------------------------------------


def test_path_mapping_q_einsum(tmp_path, monkeypatch):
    """q_einsum.w_lora_a/b should produce q_proj.lora_A/B keys only."""
    _require_export_deps()
    from safetensors.numpy import load_file
    from scripts import export_tunix_lora_to_peft as exporter

    # Single-layer tree with only q_einsum (no kv).
    rng = np.random.default_rng(0)
    tree = {
        "layers.0": {
            "attn": {
                "q_einsum": {
                    "w_lora_a": rng.standard_normal(
                        (EMBED_DIM, LORA_R), dtype=np.float32
                    ),
                    "w_lora_b": rng.standard_normal(
                        (LORA_R, NUM_HEADS, HEAD_DIM), dtype=np.float32
                    ),
                }
            }
        }
    }
    monkeypatch.setattr(exporter, "load_orbax_checkpoint", lambda _p: tree)

    output_dir = tmp_path / "out"
    exporter.main([
        "--orbax-dir", str(tmp_path / "fake"),
        "--output-dir", str(output_dir),
        "--base-model-name", "google/medgemma-27b-text-it",
        "--r", str(LORA_R),
        "--alpha", str(LORA_ALPHA),
        "--dropout", "0.0",
    ])

    weights = load_file(str(output_dir / "adapter_model.safetensors"))
    assert (
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in weights
    )
    assert (
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight" in weights
    )
    # No k_proj / v_proj keys when only q_einsum was present.
    assert not any("k_proj" in k for k in weights)
    assert not any("v_proj" in k for k in weights)

    # PEFT shape contract.
    a = weights[
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    ]
    b = weights[
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
    ]
    assert a.shape == (LORA_R, EMBED_DIM)
    assert b.shape == (NUM_HEADS * HEAD_DIM, LORA_R)


def test_path_mapping_kv_einsum_splits_to_k_and_v(tmp_path, monkeypatch):
    """kv_einsum.w_lora_a/b should fan out to BOTH k_proj and v_proj."""
    _require_export_deps()
    from safetensors.numpy import load_file
    from scripts import export_tunix_lora_to_peft as exporter

    tree = _fake_separate_qkv_tree(seed=0)
    monkeypatch.setattr(exporter, "load_orbax_checkpoint", lambda _p: tree)

    output_dir = tmp_path / "out"
    exporter.main([
        "--orbax-dir", str(tmp_path / "fake"),
        "--output-dir", str(output_dir),
        "--base-model-name", "google/medgemma-27b-text-it",
        "--r", str(LORA_R),
        "--alpha", str(LORA_ALPHA),
    ])

    weights = load_file(str(output_dir / "adapter_model.safetensors"))

    # Every layer should have q_proj, k_proj, v_proj entries with both
    # lora_A and lora_B.
    for layer_idx in range(NUM_LAYERS):
        for proj in ("q_proj", "k_proj", "v_proj"):
            for ab in ("A", "B"):
                key = (
                    f"base_model.model.model.layers.{layer_idx}."
                    f"self_attn.{proj}.lora_{ab}.weight"
                )
                assert key in weights, f"missing {key}"

    # k_proj.lora_A and v_proj.lora_A should be identical (kv_einsum
    # shares lora_a between the two output projections in the packed
    # qwix decomposition).
    for layer_idx in range(NUM_LAYERS):
        k_a = weights[
            f"base_model.model.model.layers.{layer_idx}."
            "self_attn.k_proj.lora_A.weight"
        ]
        v_a = weights[
            f"base_model.model.model.layers.{layer_idx}."
            "self_attn.v_proj.lora_A.weight"
        ]
        assert np.array_equal(k_a, v_a), (
            "k_proj.lora_A and v_proj.lora_A must come from the same "
            "kv_einsum lora_a tensor"
        )
        # PEFT shapes.
        assert k_a.shape == (LORA_R, EMBED_DIM)
        k_b = weights[
            f"base_model.model.model.layers.{layer_idx}."
            "self_attn.k_proj.lora_B.weight"
        ]
        v_b = weights[
            f"base_model.model.model.layers.{layer_idx}."
            "self_attn.v_proj.lora_B.weight"
        ]
        assert k_b.shape == (NUM_KV_HEADS * HEAD_DIM, LORA_R)
        assert v_b.shape == (NUM_KV_HEADS * HEAD_DIM, LORA_R)
        # k_b and v_b come from different slices of axis 1, so they must
        # NOT be equal (modulo astronomically rare coincidence; at
        # f32 random, never).
        assert not np.array_equal(k_b, v_b), (
            "k_proj.lora_B and v_proj.lora_B should hold different "
            "halves of the kv_einsum lora_b packing"
        )


def test_kv_einsum_split_values_match_packing(tmp_path, monkeypatch):
    """The k/v split must come from axis-1 indices 0 and 1 of lora_b.

    This guards against silently swapping K and V (or transposing the
    wrong axis); both would still produce shape-correct adapters that
    just compute the wrong thing at inference time.
    """
    _require_export_deps()
    from safetensors.numpy import load_file
    from scripts import export_tunix_lora_to_peft as exporter

    rng = np.random.default_rng(42)
    kv_a = rng.standard_normal((EMBED_DIM, LORA_R), dtype=np.float32)
    kv_b = rng.standard_normal(
        (LORA_R, 2, NUM_KV_HEADS, HEAD_DIM), dtype=np.float32
    )
    tree = {
        "layers.0": {
            "attn": {
                "kv_einsum": {"w_lora_a": kv_a, "w_lora_b": kv_b},
            }
        }
    }
    monkeypatch.setattr(exporter, "load_orbax_checkpoint", lambda _p: tree)

    output_dir = tmp_path / "out"
    exporter.main([
        "--orbax-dir", str(tmp_path / "fake"),
        "--output-dir", str(output_dir),
        "--base-model-name", "google/medgemma-27b-text-it",
        "--r", str(LORA_R),
    ])

    weights = load_file(str(output_dir / "adapter_model.safetensors"))

    # k_proj.lora_B should be kv_b[:, 0, :, :] reshaped+transposed.
    expected_k = np.transpose(kv_b[:, 0], (1, 2, 0)).reshape(
        NUM_KV_HEADS * HEAD_DIM, LORA_R
    )
    expected_v = np.transpose(kv_b[:, 1], (1, 2, 0)).reshape(
        NUM_KV_HEADS * HEAD_DIM, LORA_R
    )
    k_b = weights[
        "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight"
    ]
    v_b = weights[
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight"
    ]
    np.testing.assert_array_equal(k_b, expected_k)
    np.testing.assert_array_equal(v_b, expected_v)

    # lora_A on both sides should be the transpose of the shared kv_a.
    expected_a = kv_a.T
    for proj in ("k_proj", "v_proj"):
        a = weights[
            f"base_model.model.model.layers.0.self_attn.{proj}.lora_A.weight"
        ]
        np.testing.assert_array_equal(a, expected_a)


def test_path_mapping_qkv_einsum_splits_three_ways(tmp_path, monkeypatch):
    """qkv_einsum.w_lora_b should fan out to q_proj, k_proj, v_proj."""
    _require_export_deps()
    from safetensors.numpy import load_file
    from scripts import export_tunix_lora_to_peft as exporter

    tree = _fake_fused_qkv_tree(seed=2)
    monkeypatch.setattr(exporter, "load_orbax_checkpoint", lambda _p: tree)

    output_dir = tmp_path / "out"
    exporter.main([
        "--orbax-dir", str(tmp_path / "fake"),
        "--output-dir", str(output_dir),
        "--base-model-name", "google/medgemma-27b-text-it",
        "--r", str(LORA_R),
    ])

    weights = load_file(str(output_dir / "adapter_model.safetensors"))
    # Each layer produces 6 tensors (3 projections * 2 factors).
    assert len(weights) == NUM_LAYERS * 3 * 2

    # All three projections for layer 0 share the same lora_A.
    a_q = weights[
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    ]
    a_k = weights[
        "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight"
    ]
    a_v = weights[
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight"
    ]
    assert np.array_equal(a_q, a_k)
    assert np.array_equal(a_q, a_v)


# ---------------------------------------------------------------------------
# Tests for adapter_config.json formatting
# ---------------------------------------------------------------------------


def test_adapter_config_json_format(tmp_path, monkeypatch):
    """adapter_config.json must carry the schema PEFT.from_pretrained checks."""
    _require_export_deps()
    from scripts import export_tunix_lora_to_peft as exporter

    tree = _fake_separate_qkv_tree(seed=3)
    monkeypatch.setattr(exporter, "load_orbax_checkpoint", lambda _p: tree)

    output_dir = tmp_path / "out"
    exporter.main([
        "--orbax-dir", str(tmp_path / "fake"),
        "--output-dir", str(output_dir),
        "--base-model-name", "google/medgemma-27b-text-it",
        "--r", str(LORA_R),
        "--alpha", str(LORA_ALPHA),
        "--dropout", str(LORA_DROPOUT),
    ])

    cfg_path = output_dir / "adapter_config.json"
    assert cfg_path.is_file()
    cfg = json.loads(cfg_path.read_text())

    assert cfg["peft_type"] == "LORA"
    assert cfg["task_type"] == "CAUSAL_LM"
    assert cfg["r"] == LORA_R
    assert cfg["lora_alpha"] == LORA_ALPHA
    assert cfg["lora_dropout"] == LORA_DROPOUT
    assert cfg["base_model_name_or_path"] == "google/medgemma-27b-text-it"
    assert cfg["bias"] == "none"
    assert cfg["fan_in_fan_out"] is False
    assert cfg["use_dora"] is False
    assert cfg["use_rslora"] is False
    # Smoke uses q_proj + k_proj + v_proj.
    assert sorted(cfg["target_modules"]) == ["k_proj", "q_proj", "v_proj"]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_no_lora_leaves_raises(tmp_path, monkeypatch):
    """A pytree with no qwix LoRA leaves should fail loudly."""
    _require_export_deps()
    from scripts import export_tunix_lora_to_peft as exporter

    monkeypatch.setattr(
        exporter, "load_orbax_checkpoint",
        lambda _p: {"opt_state": {"step": np.int32(0)}},
    )
    with pytest.raises(ValueError, match="no qwix LoRA leaves"):
        exporter.main([
            "--orbax-dir", str(tmp_path / "fake"),
            "--output-dir", str(tmp_path / "out"),
            "--base-model-name", "google/medgemma-27b-text-it",
        ])


def test_missing_lora_b_raises(tmp_path, monkeypatch):
    """If lora_a is present but lora_b is missing for a layer, error out."""
    _require_export_deps()
    from scripts import export_tunix_lora_to_peft as exporter

    rng = np.random.default_rng(0)
    tree = {
        "layers.0": {
            "attn": {
                "q_einsum": {
                    "w_lora_a": rng.standard_normal(
                        (EMBED_DIM, LORA_R), dtype=np.float32
                    ),
                    # w_lora_b deliberately missing
                },
            }
        }
    }
    monkeypatch.setattr(exporter, "load_orbax_checkpoint", lambda _p: tree)
    with pytest.raises(ValueError, match="missing lora_b"):
        exporter.main([
            "--orbax-dir", str(tmp_path / "fake"),
            "--output-dir", str(tmp_path / "out"),
            "--base-model-name", "google/medgemma-27b-text-it",
        ])


def test_bf16_round_trip(tmp_path, monkeypatch):
    """ml_dtypes.bfloat16 inputs should land as bf16 in the safetensors file."""
    _require_export_deps()
    pytest.importorskip("ml_dtypes")
    import ml_dtypes
    from safetensors.numpy import load_file
    from scripts import export_tunix_lora_to_peft as exporter

    rng = np.random.default_rng(7)
    a = rng.standard_normal((EMBED_DIM, LORA_R), dtype=np.float32).astype(
        ml_dtypes.bfloat16
    )
    b = rng.standard_normal(
        (LORA_R, NUM_HEADS, HEAD_DIM), dtype=np.float32
    ).astype(ml_dtypes.bfloat16)
    tree = {
        "layers.0": {
            "attn": {
                "q_einsum": {"w_lora_a": a, "w_lora_b": b},
            }
        }
    }
    monkeypatch.setattr(exporter, "load_orbax_checkpoint", lambda _p: tree)

    output_dir = tmp_path / "out"
    exporter.main([
        "--orbax-dir", str(tmp_path / "fake"),
        "--output-dir", str(output_dir),
        "--base-model-name", "google/medgemma-27b-text-it",
        "--r", str(LORA_R),
    ])

    weights = load_file(str(output_dir / "adapter_model.safetensors"))
    for tensor in weights.values():
        assert str(tensor.dtype) == "bfloat16", (
            f"expected bf16, got {tensor.dtype} -- bf16 path lost dtype"
        )


def test_path_detection_tolerates_terminal_value_wrapper(tmp_path, monkeypatch):
    """NNX boxed params come back as ``..., 'w_lora_a', 'value'`` after a
    default orbax restore.  The walker has to look through the terminal
    ``value`` wrapper without losing the ``w_lora_a`` token.
    """
    _require_export_deps()
    from safetensors.numpy import load_file
    from scripts import export_tunix_lora_to_peft as exporter

    rng = np.random.default_rng(9)
    a = rng.standard_normal((EMBED_DIM, LORA_R), dtype=np.float32)
    b = rng.standard_normal((LORA_R, NUM_HEADS, HEAD_DIM), dtype=np.float32)
    tree = {
        "layers.0": {
            "attn": {
                "q_einsum": {
                    # NNX-style boxed param: actual array sits under "value".
                    "w_lora_a": {"value": a},
                    "w_lora_b": {"value": b},
                }
            }
        }
    }
    monkeypatch.setattr(exporter, "load_orbax_checkpoint", lambda _p: tree)

    output_dir = tmp_path / "out"
    exporter.main([
        "--orbax-dir", str(tmp_path / "fake"),
        "--output-dir", str(output_dir),
        "--base-model-name", "google/medgemma-27b-text-it",
        "--r", str(LORA_R),
    ])

    weights = load_file(str(output_dir / "adapter_model.safetensors"))
    assert (
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in weights
    )


# ---------------------------------------------------------------------------
# Step-dir resolution: --orbax-dir accepts manager root OR leaf step.
# ---------------------------------------------------------------------------


def test_resolve_step_dir_passthrough_for_leaf(tmp_path):
    """A path with no integer-named subdirs is returned unchanged.

    This covers the pre-fix contract (caller passes a leaf step dir
    directly) so nothing regresses for users / tests that already
    construct the leaf path themselves.
    """
    from scripts.export_tunix_lora_to_peft import _resolve_step_dir

    leaf = tmp_path / "step_2_payload"
    leaf.mkdir()
    # Drop a non-int-named child so the function explicitly chooses the
    # "no int subdirs -> treat as leaf" branch rather than trivially
    # passing because the dir is empty.
    (leaf / "metadata.json").write_text("{}")
    assert _resolve_step_dir(leaf) == leaf.resolve()


def test_resolve_step_dir_picks_latest_int_subdir(tmp_path):
    """Manager root with int-named subdirs resolves to the highest one.

    This is the bug Codex H2 flagged: tunix's CheckpointManager writes
    ``<root>/<step>/`` and the launcher passes ``<root>``; without this
    resolution the underlying PyTreeCheckpointer.restore fails on the
    parent dir.
    """
    from scripts.export_tunix_lora_to_peft import _resolve_step_dir

    root = tmp_path / "orbax"
    root.mkdir()
    for step in (0, 1, 7, 2):
        (root / str(step)).mkdir()
    # Plus an orbax-internal non-int child that should be ignored, not
    # crash the int parse.
    (root / "metadata").mkdir()
    assert _resolve_step_dir(root) == (root / "7").resolve()


def test_resolve_step_dir_missing_path_raises(tmp_path):
    """A non-existent --orbax-dir should fail loudly, not as an opaque
    Orbax restore error 30 lines deep into PyTree machinery.
    """
    from scripts.export_tunix_lora_to_peft import _resolve_step_dir

    with pytest.raises(FileNotFoundError, match="does not exist"):
        _resolve_step_dir(tmp_path / "no_such_dir")
