"""Unit tests for scripts.maxtext_lora.layer.LoraDense.

JAX/Flax may not be installed on local dev boxes (Unit 1 adds them to
setup_tpu.sh). When unavailable, every test in this file is skipped with a
visible reason — the production code itself still py_compiles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


try:
    import jax  # noqa: F401
    import jax.numpy as jnp
    import flax.linen as nn  # noqa: F401
    from jax.sharding import Mesh, NamedSharding, PartitionSpec  # noqa: F401

    from scripts.maxtext_lora.layer import LoraDense

    _JAX_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on dev boxes without JAX
    _JAX_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _JAX_AVAILABLE,
    reason="jax/flax not installed; layer code still py_compiles. Unit 1 adds them.",
)


# Match the hyperparameters in configs/lora_medgemma27b_tpu.yaml: r=8, alpha=16,
# dropout=0.05. Tests use a tiny base dim (8 -> 16) so they're fast on CPU.
RANK = 8
ALPHA = 16.0
DROPOUT = 0.05
IN_DIM = 8
OUT_DIM = 16
BATCH = 4


def _init_module(seed: int = 0, rank: int = RANK, dropout: float = DROPOUT):
    """Helper: build a LoraDense and return (module, params, x)."""
    module = LoraDense(features=OUT_DIM, rank=rank, alpha=ALPHA, dropout=dropout)
    x = jnp.ones((BATCH, IN_DIM), dtype=jnp.float32)
    params = module.init(jax.random.PRNGKey(seed), x, deterministic=True)
    return module, params, x


def test_forward_compiles_and_runs():
    """Forward of an 8 -> 16 LoraDense with rank=4 produces the right shape."""
    module = LoraDense(features=OUT_DIM, rank=4, alpha=8.0, dropout=0.0)
    x = jnp.ones((BATCH, IN_DIM), dtype=jnp.float32)
    params = module.init(jax.random.PRNGKey(0), x, deterministic=True)
    y = module.apply(params, x, deterministic=True)
    assert y.shape == (BATCH, OUT_DIM)


def test_lora_contribution_is_zero_at_init():
    """B is initialized to zeros, so the LoRA branch contributes 0 before any
    training. The wrapped module's output must equal the base Dense's output."""
    module, params, x = _init_module()

    # Run the wrapped module.
    y_wrapped = module.apply(params, x, deterministic=True)

    # Run just the base Dense by extracting its params.
    base_params = {"params": params["params"]["base"]}
    base = nn.Dense(features=OUT_DIM, use_bias=True, name="base")
    y_base = base.apply(base_params, x)

    # B = 0 -> LoRA term is exactly 0, so outputs match bit-for-bit.
    assert jnp.allclose(y_wrapped, y_base, atol=0.0, rtol=0.0), (
        "LoRA contribution should be exactly zero at init (B = 0)"
    )


def test_dropout_is_noop_in_eval_mode():
    """deterministic=True must produce identical output across calls even
    when dropout > 0."""
    module, params, x = _init_module(dropout=0.05)

    # Two calls with different dropout RNGs but deterministic=True should match.
    y1 = module.apply(
        params, x, deterministic=True, rngs={"dropout": jax.random.PRNGKey(1)}
    )
    y2 = module.apply(
        params, x, deterministic=True, rngs={"dropout": jax.random.PRNGKey(999)}
    )
    assert jnp.allclose(y1, y2)


def test_dropout_active_in_train_mode():
    """Sanity check: with deterministic=False and dropout > 0, two calls with
    different rngs should differ. (Catches a bug where deterministic flag is
    inverted.)

    Note: B = 0 at init means the LoRA branch is zero regardless of dropout, so
    we override B to a nonzero value before this check.
    """
    module, params, x = _init_module(dropout=0.5)
    # Force B to nonzero so the LoRA branch (and thus dropout) is observable.
    # `with_partitioning` boxes the param, so we replace the inner array via
    # `replace_boxed` while preserving the partition spec metadata.
    lora_b = params["params"]["lora_b"]
    params["params"]["lora_b"] = lora_b.replace_boxed(jnp.ones_like(lora_b.value) * 0.1)

    y1 = module.apply(
        params, x, deterministic=False, rngs={"dropout": jax.random.PRNGKey(1)}
    )
    y2 = module.apply(
        params, x, deterministic=False, rngs={"dropout": jax.random.PRNGKey(2)}
    )
    # At dropout=0.5 with different rngs, outputs almost certainly differ on
    # at least one element.
    assert not jnp.allclose(y1, y2)


def test_sharding_spec_compiles_under_mesh():
    """Init + forward inside a tiny CPU mesh must not raise. Uses
    jax.devices('cpu') so the test runs anywhere."""
    cpu_devices = jax.devices("cpu")
    # Single-device mesh is always available; that's sufficient to verify
    # the partition specs compile.
    mesh = Mesh(cpu_devices[:1], axis_names=("model",))

    module = LoraDense(
        features=OUT_DIM, rank=RANK, alpha=ALPHA, dropout=0.0,
        # Use the single mesh axis for the B-output dim; A stays replicated.
        a_axes=(None, None),
        b_axes=(None, "model"),
    )
    x = jnp.ones((BATCH, IN_DIM), dtype=jnp.float32)

    with mesh:
        @jax.jit
        def init_fn(rng, sample):
            return module.init(rng, sample, deterministic=True)

        params = init_fn(jax.random.PRNGKey(0), x)

        @jax.jit
        def fwd(p, sample):
            return module.apply(p, sample, deterministic=True)

        y = fwd(params, x)

    assert y.shape == (BATCH, OUT_DIM)


def test_param_shapes_match_rank_and_alpha():
    """A is (in, r); B is (r, out). Catches off-by-one or transposed init."""
    module, params, _ = _init_module()
    a = params["params"]["lora_a"]
    b = params["params"]["lora_b"]
    # flax's with_partitioning boxes params; .value gives the underlying array.
    a_arr = a.value if hasattr(a, "value") else a
    b_arr = b.value if hasattr(b, "value") else b
    assert a_arr.shape == (IN_DIM, RANK)
    assert b_arr.shape == (RANK, OUT_DIM)
