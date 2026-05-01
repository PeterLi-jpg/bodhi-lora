"""Unit tests for scripts/maxtext_lora/injector.inject_lora.

Builds a tiny synthetic Flax model whose attribute layout matches what we
expect from a Gemma-3 setup-style decoder:

    model
    └── layers (tuple of N blocks)
        └── layers[i]
            └── self_attn
                ├── q_proj   <-- target
                ├── k_proj   <-- NOT a target, must remain a plain Dense
                ├── v_proj   <-- target
                └── o_proj   <-- NOT a target

After ``inject_lora``, only ``q_proj`` and ``v_proj`` should be wrapped,
the partition spec should only carry entries for those, and ``k_proj`` /
``o_proj`` should be untouched.

Tests skip-with-reason if jax/flax aren't installed, or if Unit 2's
``LoraDense`` isn't on disk yet, so a CPU-only dev box still gets a clean
``N passed, M skipped`` and the cluster CI fills in the rest.
"""

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Optional-dep gate. importorskip raises pytest.skip if missing, so the
# rest of the module never imports flax-internal symbols on bare envs.
pytest.importorskip("jax", reason="jax not installed (CPU dev box)")
pytest.importorskip("flax.linen", reason="flax not installed")

from scripts.maxtext_lora import injector  # noqa: E402

# Unit 2 (LoraDense) is a separate PR; if it hasn't landed, skip the
# whole module rather than fail noisily. The user's integration step will
# catch a real import bug between Unit 2 and Unit 3.
LoraDense = pytest.importorskip(
    "scripts.maxtext_lora.layer",
    reason="scripts/maxtext_lora/layer.py (Unit 2's LoraDense) not on disk",
).LoraDense

from flax import linen as nn  # noqa: E402
from jax.sharding import PartitionSpec  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic Flax model: Gemma-3-style attribute layout.
#
# We use ``__post_init__`` rather than the more idiomatic ``setup()`` for
# one reason: ``setup()`` runs lazily, only when the module is bound (via
# ``init`` or ``apply``). The injector walks ``vars(module)``, which on a
# pre-bind setup-style module is empty. The MaxText fork's real Gemma-3
# module gets walked AFTER ``model.bind(params)`` makes its children
# visible. ``__post_init__`` shortcuts that for the unit test without
# allocating real parameters; the resulting attribute layout is identical
# to what a bound model exposes.
# ---------------------------------------------------------------------------

class _SelfAttn(nn.Module):
    """4-projection attention block, names match Gemma-3."""
    hidden: int = 8

    def __post_init__(self):
        # object.__setattr__ bypasses the frozen-dataclass guard. flax
        # uses the same trick internally when assigning children in setup.
        object.__setattr__(self, "q_proj", nn.Dense(self.hidden))
        object.__setattr__(self, "k_proj", nn.Dense(self.hidden))
        object.__setattr__(self, "v_proj", nn.Dense(self.hidden))
        object.__setattr__(self, "o_proj", nn.Dense(self.hidden))
        super().__post_init__()


class _Block(nn.Module):
    hidden: int = 8

    def __post_init__(self):
        object.__setattr__(self, "self_attn", _SelfAttn(hidden=self.hidden))
        super().__post_init__()


class _TinyGemma(nn.Module):
    """Tiny model with a tuple of blocks, mirrors Gemma-3's nested layout."""
    n_layers: int = 2
    hidden: int = 8

    def __post_init__(self):
        object.__setattr__(
            self,
            "layers",
            tuple(_Block(hidden=self.hidden) for _ in range(self.n_layers)),
        )
        super().__post_init__()


# Build a fresh model per test. Flax modules are frozen dataclasses and
# the injector mutates them in-place; sharing a model between tests would
# let the first test's mutations leak into the second.

def _build_model():
    return _TinyGemma(n_layers=2, hidden=8)


def _inject(model, target_modules=("q_proj", "v_proj"), rank=8, alpha=16.0, dropout=0.05):
    """Inject with the project's default LoRA hyperparams (r=8, alpha=16)."""
    return injector.inject_lora(
        model,
        target_modules=list(target_modules),
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_inject_replaces_only_target_modules():
    model = _build_model()
    injected, _ = _inject(model)

    for i, block in enumerate(injected.layers):
        attn = block.self_attn
        assert isinstance(attn.q_proj, LoraDense), (
            f"layers[{i}].self_attn.q_proj should be LoraDense, "
            f"got {type(attn.q_proj).__name__}"
        )
        assert isinstance(attn.v_proj, LoraDense), (
            f"layers[{i}].self_attn.v_proj should be LoraDense, "
            f"got {type(attn.v_proj).__name__}"
        )
        assert isinstance(attn.k_proj, nn.Dense)
        assert not isinstance(attn.k_proj, LoraDense)
        assert isinstance(attn.o_proj, nn.Dense)
        assert not isinstance(attn.o_proj, LoraDense)


def test_inject_returns_self_for_chaining():
    model = _build_model()
    injected, _ = _inject(model)
    # The injector mutates in-place AND returns the same object so callers
    # can write ``model, pspec = inject_lora(model, ...)`` without confusion.
    assert injected is model


def test_partition_spec_covers_only_lora_params():
    model = _build_model()
    _, pspec = _inject(model)

    assert "params" in pspec, f"partition spec missing 'params' wrapper: {pspec!r}"
    params = pspec["params"]

    # Walk down to every leaf. Leaves must be PartitionSpec, intermediates
    # must be dicts, and the only leaves at all should be lora_a / lora_b.
    leaves: list[tuple[tuple[str, ...], PartitionSpec]] = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + (k,))
        else:
            leaves.append((path, node))

    walk(params, ())

    assert leaves, "partition spec contains no leaves; injection produced nothing"
    for path, leaf in leaves:
        assert isinstance(leaf, PartitionSpec), (
            f"leaf at {path} is {type(leaf).__name__}, expected PartitionSpec"
        )
        assert path[-1] in ("lora_a", "lora_b"), (
            f"unexpected leaf at {path}; only lora_a / lora_b should be in "
            f"the optimizer partition spec"
        )

    # 2 layers x 2 targets x 2 LoRA factors = 8 leaves.
    assert len(leaves) == 2 * 2 * 2, (
        f"expected 8 LoRA leaves (2 layers x {{q_proj, v_proj}} x "
        f"{{lora_a, lora_b}}), got {len(leaves)}"
    )

    for path, _ in leaves:
        assert "k_proj" not in path, f"k_proj leaked into pspec at {path}"
        assert "o_proj" not in path, f"o_proj leaked into pspec at {path}"


def test_partition_spec_paths_match_module_paths():
    """The pspec tree should mirror the actual module nesting.

    For a 2-layer model, each q_proj lives at
    ``layers/<i>/self_attn/q_proj``; the spec tree must have the same shape.
    """
    model = _build_model()
    _, pspec = _inject(model)

    params = pspec["params"]
    assert "layers" in params, f"pspec missing 'layers': {list(params.keys())}"
    layers = params["layers"]
    # Tuple submodules are keyed by string indices in Flax's pytree layout.
    assert set(layers.keys()) == {"0", "1"}, (
        f"expected layer indices {{0,1}}, got {set(layers.keys())}"
    )
    for idx in ("0", "1"):
        layer = layers[idx]
        assert "self_attn" in layer, (
            f"pspec missing self_attn at layer {idx}: {list(layer.keys())}"
        )
        attn = layer["self_attn"]
        assert set(attn.keys()) == {"q_proj", "v_proj"}, (
            f"layer {idx}.self_attn pspec should have exactly q_proj, v_proj; "
            f"got {set(attn.keys())}"
        )
        for tgt in ("q_proj", "v_proj"):
            assert set(attn[tgt].keys()) == {"lora_a", "lora_b"}, (
                f"layer {idx}.self_attn.{tgt} should have lora_a, lora_b; "
                f"got {set(attn[tgt].keys())}"
            )


def test_inject_lora_rejects_empty_targets():
    model = _build_model()
    with pytest.raises(ValueError, match="target_modules is empty"):
        _inject(model, target_modules=())


def test_inject_lora_rejects_invalid_rank():
    model = _build_model()
    with pytest.raises(ValueError, match="rank must be positive"):
        _inject(model, rank=0)


def test_inject_lora_raises_when_no_targets_match():
    model = _build_model()
    with pytest.raises(ValueError, match="found no targets matching"):
        _inject(model, target_modules=("this_attr_does_not_exist",))


def test_inject_lora_rejects_non_module():
    with pytest.raises(TypeError, match="flax.linen.Module"):
        injector.inject_lora(
            object(),
            target_modules=["q_proj"],
            rank=8,
            alpha=16.0,
            dropout=0.05,
        )
