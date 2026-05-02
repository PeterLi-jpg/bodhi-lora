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
    # set_mesh: jax >= 0.5 thread-local context manager that pins a concrete
    # mesh, so flax.core.spmd.shard_value can resolve sharding annotations on
    # nnx.Params that are created outside model.init's mesh block. The
    # LoraDenseGeneral unit tests instantiate the layer eagerly (no init),
    # which trips flip/4844's eager-sharding rule without a global mesh.
    try:
        from jax.sharding import set_mesh  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover - older jax
        set_mesh = None  # type: ignore[assignment]

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


# ──────────────────────────────────────────────────────────────────────
# LoraDenseGeneral (NNX wrapper for MaxText DenseGeneral)
# ──────────────────────────────────────────────────────────────────────
# Skipped when nnx isn't installed. Exercises the multi-axis output
# reshape path that the Linen ``LoraDense`` doesn't cover. Uses a tiny
# duck-typed fake DenseGeneral instead of MaxText's real one (the real
# one needs Quant config + nnx_wrappers + the full vendored tree).

try:
    from flax import nnx  # type: ignore[import-not-found]
    from scripts.maxtext_lora.layer import LoraDenseGeneral
    _NNX_AVAILABLE = LoraDenseGeneral is not None
except Exception:  # pragma: no cover
    nnx = None  # type: ignore[assignment]
    LoraDenseGeneral = None  # type: ignore[assignment]
    _NNX_AVAILABLE = False


_skip_no_nnx = pytest.mark.skipif(
    not _NNX_AVAILABLE,
    reason="flax.nnx not installed; LoraDenseGeneral not exercisable here.",
)


if _NNX_AVAILABLE:

    class _FakeDenseGeneral(nnx.Module):
        """Tiny duck-typed stand-in for MaxText's DenseGeneral.

        Implements the surface ``LoraDenseGeneral`` reads:
        ``in_features_shape`` and ``out_features_shape``, plus a
        ``__call__`` matching DenseGeneral's signature.
        """

        def __init__(
            self,
            in_features_shape: tuple[int, ...],
            out_features_shape: tuple[int, ...],
            *,
            rngs,
        ) -> None:
            self.in_features_shape = in_features_shape
            self.out_features_shape = out_features_shape
            shape = tuple(in_features_shape) + tuple(out_features_shape)
            init = nnx.initializers.normal(stddev=0.02)
            self.kernel = nnx.Param(init(rngs.params(), shape, jnp.float32))

        def __call__(self, x, _initializing=False, out_sharding=None):
            in_total = 1
            for d in self.in_features_shape:
                in_total *= d
            out_shape = tuple(self.out_features_shape)
            kernel_flat = self.kernel[...].reshape(in_total, -1)
            out_flat = x @ kernel_flat
            return out_flat.reshape(out_flat.shape[:-1] + out_shape)


# LoraDenseGeneral creates nnx.Param(..., sharding=...) at __init__ time.
# Newer flax (>= 0.12) demands a mesh context whenever a sharding annotation
# is present on a variable (flip/4844 eager-sharding). Production hits this
# through apply_lora's `with mesh, nn_partitioning.axis_rules(...)` block;
# in unit tests we pin a single-device CPU mesh whose axis name matches the
# LoraDenseGeneral defaults (DEFAULT_NNX_B_SHARDING = (None, "model")).
def _single_device_lora_mesh():
    return Mesh(jax.devices("cpu")[:1], axis_names=("model",))


@_skip_no_nnx
def test_dense_general_lora_zero_at_init():
    """LoRA contribution is identically zero at init (B=0), so wrapper output
    matches the base DenseGeneral output."""
    rngs = nnx.Rngs(params=jax.random.PRNGKey(7))
    base = _FakeDenseGeneral((4,), (3, 5), rngs=rngs)
    wrap_rngs = nnx.Rngs(params=jax.random.PRNGKey(11))
    with set_mesh(_single_device_lora_mesh()):
        wrapped = LoraDenseGeneral(
            base=base, rank=2, alpha=4.0, dropout=0.0, rngs=wrap_rngs
        )
    x = jnp.ones((2, 4), dtype=jnp.float32)
    y_base = base(x)
    y_wrap = wrapped(x)
    assert y_base.shape == (2, 3, 5)
    assert y_wrap.shape == (2, 3, 5)
    assert jnp.allclose(y_base, y_wrap)


@_skip_no_nnx
def test_dense_general_lora_factor_shapes():
    """A is (in_total, r); B is (r, out_total). Confirm flat layout."""
    rngs = nnx.Rngs(params=jax.random.PRNGKey(7))
    base = _FakeDenseGeneral((4,), (3, 5), rngs=rngs)
    wrap_rngs = nnx.Rngs(params=jax.random.PRNGKey(11))
    with set_mesh(_single_device_lora_mesh()):
        wrapped = LoraDenseGeneral(
            base=base, rank=2, alpha=4.0, dropout=0.0, rngs=wrap_rngs
        )
    a = wrapped.lora_a[...]
    b = wrapped.lora_b[...]
    assert a.shape == (4, 2)
    assert b.shape == (2, 15)


@_skip_no_nnx
def test_dense_general_lora_nonzero_after_perturbing_b():
    """If B is non-zero, the LoRA delta equals scaling * reshape((x @ A) @ B)."""
    rngs = nnx.Rngs(params=jax.random.PRNGKey(7))
    base = _FakeDenseGeneral((4,), (3, 5), rngs=rngs)
    wrap_rngs = nnx.Rngs(params=jax.random.PRNGKey(11))
    with set_mesh(_single_device_lora_mesh()):
        wrapped = LoraDenseGeneral(
            base=base, rank=2, alpha=4.0, dropout=0.0, rngs=wrap_rngs
        )
    new_b = jnp.arange(2 * 15, dtype=jnp.float32).reshape(2, 15)
    wrapped.lora_b.value = new_b
    x = jnp.ones((2, 4), dtype=jnp.float32)
    delta = wrapped(x) - base(x)
    a = wrapped.lora_a[...]
    expected = ((x @ a) @ new_b).reshape(2, 3, 5) * (4.0 / 2)
    assert jnp.allclose(delta, expected, atol=1e-5)
