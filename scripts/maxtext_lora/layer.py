"""Flax LoRA wrapper for MaxText Stage 3.

`LoraDense` wraps a base `flax.linen.Dense` with two low-rank matrices A and B.
The forward pass returns `base(x) + (alpha/rank) * dropout(x @ A) @ B`.

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

from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


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
        use_bias: passed through to the base Dense.
        a_axes: partition-spec axes for the A matrix (rank-2: (in, r)).
        b_axes: partition-spec axes for the B matrix (rank-2: (r, out)).
    """

    features: int
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    use_bias: bool = True
    a_axes: Sequence[str | None] = DEFAULT_A_AXES
    b_axes: Sequence[str | None] = DEFAULT_B_AXES

    @nn.compact
    def __call__(self, x: jax.Array, *, deterministic: bool = True) -> jax.Array:
        if self.rank <= 0:
            raise ValueError(f"LoraDense.rank must be > 0, got {self.rank}")

        in_features = x.shape[-1]
        # Base full-rank projection. Frozen during LoRA fine-tuning by the caller
        # (the injector will mark base params as non-trainable); LoraDense itself
        # does not enforce that — it just exposes the params.
        base = nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
            name="base",
        )

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

        base_out = base(x)

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
