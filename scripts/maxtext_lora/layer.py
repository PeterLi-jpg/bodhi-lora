"""Flax LoRA wrapper for MaxText Stage 3.

`LoraDense` wraps a base `flax.linen.Dense` with two low-rank matrices A and B.
The forward pass returns `base(x) + (alpha/rank) * dropout(x @ A) @ B`.

Two construction modes are supported:
  - Standalone (``base=None``): ``LoraDense`` instantiates its own internal
    ``nn.Dense`` named ``"base"``. Used by the unit tests and any caller that
    just wants a drop-in low-rank Dense.
  - Wrapping (``base=<nn.Dense>``): ``LoraDense`` reuses a pre-existing
    ``nn.Dense`` instance for the full-rank projection. The injector path
    uses this so the wrapped LoRA module shares the original Dense's params
    instead of creating a fresh (uninitialised) one.

Init follows PEFT's default (LoraLayer.reset_lora_parameters):
  - A: kaiming_uniform with a=sqrt(5)  (== flax's lecun-style init for the rank-r fan-in)
  - B: zeros, so the LoRA contribution is exactly 0 at step 0 and the wrapped
    module behaves identically to the base Dense before any training.

Sharding annotations use `flax.linen.with_partitioning` so the params have
PartitionSpec metadata that propagates through `jax.jit` / `pjit`:
  - A: replicated across mesh axes ('replicated', None)
  - B: output dim sharded over the model-parallel axis ('replicated', 'model')

These names match the conventional MaxText mesh axes ('data', 'fsdp', 'model').
The actual mesh is supplied by the caller, so the spec names are configurable
via the `partition_axis_names` constructor arg if a fork uses different names.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

# NNX is a separate Flax flavour. MaxText's ``DenseGeneral`` is NNX, so the
# wrapper that lives next to it has to be NNX too — Linen modules can't
# directly own NNX submodules. We import lazily so the file stays
# importable on a CPU dev box where flax may be installed but nnx is not.
try:
    from flax import nnx  # type: ignore[import-not-found]
    _HAS_NNX = True
except ImportError:
    nnx = None  # type: ignore[assignment]
    _HAS_NNX = False


# (in-axes, out-axes) partition spec for A and B. A is replicated on both axes;
# B has its output dim sharded along the model-parallel axis.
DEFAULT_A_AXES: Tuple[str | None, str | None] = (None, None)
DEFAULT_B_AXES: Tuple[str | None, str | None] = (None, "model")


class LoraDense(nn.Module):
    """Low-rank adapter wrapping a base ``nn.Dense``.

    Args:
        features: output dim of the underlying Dense.
        rank: LoRA rank ``r``. Must be > 0.
        alpha: LoRA scaling factor. Effective scale is ``alpha / rank``.
        dropout: probability for the LoRA dropout applied to the input of A.
        use_bias: passed through to the base Dense (only when ``base`` is None).
        a_axes: partition-spec axes for the A matrix (rank-2: (in, r)).
        b_axes: partition-spec axes for the B matrix (rank-2: (r, out)).
        base: optional pre-built ``nn.Dense`` to use for the full-rank
            projection. If None, a fresh Dense named ``"base"`` is created
            inside this module. The injector passes the original Dense it's
            wrapping so the LoRA module shares (rather than re-allocates)
            the base params.
    """

    features: int
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    use_bias: bool = True
    a_axes: Sequence[str | None] = DEFAULT_A_AXES
    b_axes: Sequence[str | None] = DEFAULT_B_AXES
    # Keep ``base`` last so adding it doesn't reorder the existing fields —
    # Flax treats the dataclass field order as part of the module identity.
    base: Any = None

    @nn.compact
    def __call__(self, x: jax.Array, *, deterministic: bool = True) -> jax.Array:
        if self.rank <= 0:
            raise ValueError(f"LoraDense.rank must be > 0, got {self.rank}")

        in_features = x.shape[-1]
        # Base full-rank projection. Frozen during LoRA fine-tuning by the caller
        # (the injector will mark base params as non-trainable); LoraDense itself
        # does not enforce that — it just exposes the params.
        if self.base is None:
            base = nn.Dense(
                features=self.features,
                use_bias=self.use_bias,
                name="base",
            )
            base_out = base(x)
        else:
            # Reuse the caller-supplied Dense. We don't re-name it: Flax has
            # already assigned it a name in its parent scope, so calling it
            # here just re-applies the existing module.
            base_out = self.base(x)

        # PEFT default: A ~ kaiming_uniform(a=sqrt(5)), B = 0.
        # variance_scaling(scale=1/3, fan_in, uniform) reproduces torch's
        # kaiming_uniform(a=sqrt(5)) — the leaky_relu(sqrt(5)) gain is 1/sqrt(3),
        # and variance_scaling internally applies the gain^2 = 1/3 factor.
        a_init = nn.initializers.variance_scaling(
            scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
        )
        b_init = nn.initializers.zeros

        lora_a = self.param(
            "lora_a",
            nn.with_partitioning(a_init, self.a_axes),
            (in_features, self.rank),
            jnp.float32,
        )
        lora_b = self.param(
            "lora_b",
            nn.with_partitioning(b_init, self.b_axes),
            (self.rank, self.features),
            jnp.float32,
        )

        # Drop on the input of A — matches PEFT's LoraLayer where dropout is
        # applied before the down-projection.
        lora_in = x
        if self.dropout > 0.0:
            lora_in = nn.Dropout(rate=self.dropout, deterministic=deterministic)(lora_in)

        # Cast LoRA params to the input dtype so this composes with bf16 base
        # weights without forcing an upcast of the whole graph.
        a = lora_a.astype(x.dtype)
        b = lora_b.astype(x.dtype)

        scaling = self.alpha / self.rank
        lora_out = (lora_in @ a) @ b
        return base_out + scaling * lora_out


# ──────────────────────────────────────────────────────────────────────
# NNX wrapper: LoraDenseGeneral
# ──────────────────────────────────────────────────────────────────────
# MaxText's Gemma-3 attention/MLP projections are
# ``maxtext.layers.linears.DenseGeneral`` (an ``nnx.Module``). Their output
# can be multi-axis — e.g. ``query`` projects ``(B, T, hidden)`` to
# ``(B, T, num_query_heads, head_dim)``. The Linen ``LoraDense`` above
# only handles single-axis output and can't host an NNX child, so we add
# a sibling NNX wrapper that:
#
#   1. Holds the original DenseGeneral (so its frozen kernel is reused —
#      no doubled memory for the 27 B base).
#   2. Allocates ``lora_a: (in_dim, r)`` and ``lora_b: (r, out_dim_total)``
#      as ``nnx.Param``s, where ``out_dim_total = prod(out_features_shape)``.
#   3. Forward = base(x) + scaling * reshape((x @ A) @ B, (..., *out_features_shape)).
#
# The flat ``lora_b`` shape (rather than ``(r, *out_features_shape)``) is
# intentional: the export step (``scripts/export_maxtext_lora_to_peft``)
# already handles 2D LoRA matrices, and PEFT's HuggingFace adapter
# format also stores 2D ``lora_A`` / ``lora_B``. Reshaping at forward
# time keeps the on-disk layout flat and round-trippable.
#
# UNTESTED on TPU: this wrapper has not yet been exercised on a real v6e
# with MaxText. Local unit tests (``tests/test_maxtext_lora_layer.py``)
# verify the math against a numpy/Linen toy DenseGeneral; production
# behaviour around ``nnx.Param`` sharding on the v6e mesh and the
# interaction with MaxText's ``shard_mode`` field is the first
# smoke-debug surface to watch for.

# Sharding: replicate ``lora_a`` (small: in × r), shard ``lora_b`` output
# dim along the model axis to mirror MaxText's kernel sharding. The
# axis names match the conventional MaxText mesh ('data', 'fsdp',
# 'model'); override at the call site if a fork uses different names.
DEFAULT_NNX_A_SHARDING: tuple[str | None, ...] = (None, None)
DEFAULT_NNX_B_SHARDING: tuple[str | None, ...] = (None, "model")


def _flatten_features(shape: tuple[int, ...] | int) -> int:
    """Product of a (possibly multi-axis) features shape, returning a flat int."""
    if isinstance(shape, int):
        return int(shape)
    return int(np.prod(tuple(shape)))


if _HAS_NNX:

    class LoraDenseGeneral(nnx.Module):
        """NNX low-rank adapter wrapping a MaxText ``DenseGeneral``.

        Unlike ``LoraDense`` (Linen, single-axis output), this wrapper
        owns an NNX ``DenseGeneral`` whose output may be multi-axis. The
        LoRA factors are kept flat (``(in_dim, r)`` and ``(r, out_total)``);
        the multi-axis reshape happens at forward time.

        Args:
            base: the original ``DenseGeneral`` instance to wrap. Its
                ``in_features_shape`` and ``out_features_shape`` are
                read at init time to size the LoRA factors.
            rank: LoRA rank ``r``. Must be > 0.
            alpha: LoRA scaling. Effective scale = ``alpha / rank``.
            dropout: input-side dropout probability. The forward
                currently treats every call as deterministic
                (no dropout) until an NNX dropout RNG plumbing lands.
            a_sharding: per-axis logical mesh-axis names for ``lora_a``.
            b_sharding: per-axis logical mesh-axis names for ``lora_b``.
            rngs: NNX RNG state. Must include a ``params`` stream.
        """

        def __init__(
            self,
            base: Any,
            rank: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.0,
            a_sharding: tuple[str | None, ...] = DEFAULT_NNX_A_SHARDING,
            b_sharding: tuple[str | None, ...] = DEFAULT_NNX_B_SHARDING,
            *,
            rngs: Any,
        ) -> None:
            if rank <= 0:
                raise ValueError(f"LoraDenseGeneral.rank must be > 0, got {rank}")
            if rngs is None:
                raise ValueError(
                    "LoraDenseGeneral requires nnx.Rngs (the inject_lora call site "
                    "passes one explicitly so the LoRA factor init is reproducible)."
                )

            self.base = base
            self.rank = int(rank)
            self.alpha = float(alpha)
            # NOTE: dropout currently informational only — the forward path
            # threads no dropout RNG, matching MaxText's own DenseGeneral
            # behaviour (no dropout on attention projections). Plumbing a
            # dropout RNG through nnx is a follow-up if a config ever
            # requests non-zero LoRA dropout.
            self.dropout_rate = float(dropout)

            in_dim = _flatten_features(base.in_features_shape)
            out_total = _flatten_features(base.out_features_shape)
            self._out_features_shape: tuple[int, ...] = tuple(
                base.out_features_shape
                if not isinstance(base.out_features_shape, int)
                else (base.out_features_shape,)
            )

            # PEFT-style init: A ~ kaiming_uniform(a=sqrt(5)), B = 0. Same math
            # as LoraDense above, just routed through nnx.Param. We use
            # variance_scaling(1/3, fan_in, uniform) which matches PyTorch's
            # kaiming_uniform with the default leaky_relu(sqrt(5)) gain.
            a_init = nnx.initializers.variance_scaling(
                scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
            )
            b_init = nnx.initializers.zeros_init()

            self.lora_a = nnx.Param(
                a_init(rngs.params(), (in_dim, self.rank), jnp.float32),
                sharding=a_sharding,
            )
            self.lora_b = nnx.Param(
                b_init(rngs.params(), (self.rank, out_total), jnp.float32),
                sharding=b_sharding,
            )

        def __call__(
            self,
            inputs: jax.Array,
            _initializing: bool = False,
            out_sharding: Any = None,
        ) -> jax.Array:
            """Mirrors DenseGeneral's call signature so callers don't notice the swap."""
            base_out = self.base(
                inputs,
                _initializing=_initializing,
                out_sharding=out_sharding,
            )

            # x is typically (B, T, in_dim) for Gemma-3 q/k/v projections.
            # If DenseGeneral is contracting on multiple axes (rare on the
            # paths we target — q/k/v all contract a single hidden axis),
            # the flat (in_dim, r) factor still works because the input
            # contract dim is the trailing dim by default (axis=-1).
            x = jnp.asarray(inputs, base_out.dtype)

            a = jnp.asarray(self.lora_a[...], base_out.dtype)
            b = jnp.asarray(self.lora_b[...], base_out.dtype)

            lora_flat = (x @ a) @ b  # shape (..., out_total)
            # Reshape trailing axis to match base_out's multi-axis shape.
            lora_out = lora_flat.reshape(
                lora_flat.shape[:-1] + self._out_features_shape
            )
            scaling = self.alpha / max(1, self.rank)
            return base_out + scaling * lora_out

else:
    # NNX not installed — keep an obvious marker so importers can detect
    # the case without raising at module load.
    LoraDenseGeneral = None  # type: ignore[assignment]
