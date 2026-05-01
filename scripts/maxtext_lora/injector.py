"""LoRA injector for Gemma-3 (Flax / MaxText).

Walks a Flax module tree, finds named Linear/Dense children matching
``target_modules`` (e.g. ``q_proj``, ``v_proj``), and swaps each one for
a ``LoraDense`` wrapper that holds the original Dense as its frozen base
plus trainable LoRA-A / LoRA-B factors.

The frozen base weights stay shared (the wrapper points at the same
``nn.Dense`` instance the injector replaced), so memory cost is just the
LoRA factors: ``in_features * r + r * out_features`` parameters per
target. About 13 M for MedGemma-27B with r=8 and 2 targets.

Returns ``(injected_model, partition_spec_for_optimizer)``. The partition
spec is a pytree shaped like the LoRA parameter sub-tree only, so
optax (or any optimizer that walks it) allocates state exclusively for
LoRA factors. The frozen base never gets a 2x memory hit from Adam's
m / v buffers.

The walker handles two Flax submodule patterns:
  - direct attribute   (``self.q_proj = nn.Dense(...)``)
  - tuple/list field   (``self.layers = (Block(...), Block(...), ...)``)

Anything more exotic (nn.scan, dynamic dicts) needs a custom path; this
covers the Gemma-3 setup-style decoder used by the MaxText fork.
"""

from __future__ import annotations

from typing import Any, Iterator

# Flax / JAX import is deferred and wrapped: the injector module must be
# importable on a CPU-only dev box so unit tests can skip-with-reason
# instead of erroring at collection.
try:
    import jax  # noqa: F401
    from jax.sharding import PartitionSpec
    from flax import linen as nn
    _HAS_FLAX = True
except ImportError:
    PartitionSpec = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_FLAX = False

# Unit 2 supplies LoraDense. We import lazily so a missing Unit-2 file
# surfaces as an ImportError at inject_lora() call time, not at module
# import time. Keeps test collection clean.
try:
    from scripts.maxtext_lora.layer import LoraDense  # type: ignore[import-not-found]
    _HAS_LORA_DENSE = True
except ImportError:
    LoraDense = None  # type: ignore[assignment]
    _HAS_LORA_DENSE = False


def _is_flax_module(obj: Any) -> bool:
    """True if obj is a flax.linen.Module instance.

    Returns False on bare envs where flax isn't importable, so this file
    stays usable for static analysis even without the runtime deps.
    """
    return _HAS_FLAX and isinstance(obj, nn.Module)


def _iter_child_modules(parent: Any) -> Iterator[tuple[str, int | None, Any]]:
    """Yield ``(field, idx, child)`` for every submodule of ``parent``.

    ``idx`` is ``None`` for direct-attribute children and the integer index
    for sequence children (``self.layers = (Block(), Block(), ...)``). The
    caller composes whatever path or display string it needs from those
    two fields.

    Skips Flax-internal attrs (``_state``, ``_id``, ``_parent_ref``,
    ``name``) by ignoring leading-underscore names and the bare ``name``
    field; non-module fields (e.g. ``hidden: int = 8``) are filtered by
    the ``_is_flax_module`` check below.
    """
    for attr_name, value in vars(parent).items():
        if attr_name.startswith("_") or attr_name == "name":
            continue
        if _is_flax_module(value):
            yield (attr_name, None, value)
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                if _is_flax_module(item):
                    yield (attr_name, i, item)


def _replace_child(parent: Any, field: str, idx: int | None, new_child: Any) -> None:
    """Replace a child submodule on a (frozen) Flax module.

    Uses ``object.__setattr__`` to bypass the frozen-dataclass guard.
    Sequence fields (``idx`` not None) are rebuilt as the same kind
    (tuple stays tuple, list stays list).
    """
    if idx is None:
        object.__setattr__(parent, field, new_child)
        return
    seq = getattr(parent, field)
    if isinstance(seq, tuple):
        new_seq: tuple | list = seq[:idx] + (new_child,) + seq[idx + 1:]
    else:
        new_seq = list(seq)
        new_seq[idx] = new_child
    object.__setattr__(parent, field, new_seq)


def _wrap_in_lora(base_dense: Any, rank: int, alpha: float, dropout: float) -> Any:
    """Build a LoraDense wrapping ``base_dense`` with the given hyperparams.

    ``LoraDense`` (Unit 2) takes ``base`` (the frozen Dense), ``rank``,
    ``alpha``, and ``dropout`` as constructor args. We pass them by
    keyword so a future Unit-2 reorder of positional args doesn't break us.
    """
    if not _HAS_LORA_DENSE:
        raise ImportError(
            "scripts.maxtext_lora.layer.LoraDense not found. Unit 2 must be "
            "merged before the injector can run."
        )
    return LoraDense(base=base_dense, rank=rank, alpha=alpha, dropout=dropout)


def _walk_and_inject(
    module: Any,
    target_modules: tuple[str, ...],
    rank: int,
    alpha: float,
    dropout: float,
    path: tuple[str, ...] = (),
    injected_paths: list[tuple[str, ...]] | None = None,
) -> list[tuple[str, ...]]:
    """Recursively walk ``module``, swap matching children for LoraDense.

    Returns the list of fully-qualified paths (e.g. ``("layers", "0",
    "self_attn", "q_proj")``) where injection happened, so the caller can
    build the optimizer partition spec without a second walk.
    """
    if injected_paths is None:
        injected_paths = []

    for field, idx, child in _iter_child_modules(module):
        # Path component for sequence children is the index as a string,
        # matching Flax's parameter-tree naming convention.
        child_path = path + (field,) if idx is None else path + (field, str(idx))

        # Match by attribute name. PEFT does the same: the "is it really a
        # Dense?" duck-type check is weak (subclasses, custom Linears).
        if field in target_modules:
            new_child = _wrap_in_lora(child, rank, alpha, dropout)
            _replace_child(module, field, idx, new_child)
            injected_paths.append(child_path)
            # Don't recurse into a freshly-wrapped LoraDense: its base IS
            # the original q_proj/v_proj, so a recurse would re-match.
            continue

        _walk_and_inject(
            child,
            target_modules,
            rank,
            alpha,
            dropout,
            path=child_path,
            injected_paths=injected_paths,
        )

    return injected_paths


def _build_partition_spec(
    injected_paths: list[tuple[str, ...]],
    lora_param_names: tuple[str, ...] = ("lora_a", "lora_b"),
) -> dict[str, Any]:
    """Build a pytree of PartitionSpecs for the LoRA params only.

    Tree shape:
        { "params": {
            "<path[0]>": { "<path[1]>": { ...
                { "<target>": { "lora_a": PartitionSpec(),
                                 "lora_b": PartitionSpec() } } } } } }

    Each LoRA factor is small (``[in, r]`` or ``[r, out]``), so replicating
    across devices is fine and keeps the optimizer-state bookkeeping
    trivial. To shard a huge ``in_features``, swap in
    ``PartitionSpec("fsdp")`` here.

    Returns a plain nested dict (not ``flax.core.FrozenDict``) so the
    caller can freeze it themselves with whatever flax version they have.
    """
    tree: dict[str, Any] = {}
    for path in injected_paths:
        node = tree
        for component in path:
            node = node.setdefault(component, {})
        for lora_name in lora_param_names:
            # Empty PartitionSpec means replicated across all devices.
            node[lora_name] = PartitionSpec()
    return {"params": tree}


def inject_lora(
    model: Any,
    target_modules: list[str],
    rank: int,
    alpha: float,
    dropout: float,
):
    """Inject LoRA adapters into a Gemma-3 Flax model.

    Walks ``model``'s submodule tree and replaces every direct child whose
    attribute name matches ``target_modules`` with a ``LoraDense`` wrapper.
    Common Gemma-3 targets: ``q_proj``, ``v_proj`` (per
    ``configs/lora_medgemma27b_tpu.yaml`` defaults: r=8, alpha=16,
    dropout=0.05).

    Args:
        model: A Flax ``nn.Module`` instance whose children are visible
            via ``vars(model)``, i.e. either eagerly assigned in
            ``__post_init__`` or post-bind via ``model.bind(variables)``.
            Pure setup-style modules (``setup()`` deferred until
            ``init``/``apply``) need to be bound first; the MaxText fork
            does this in its model loader. Compact-style ``@nn.compact``
            modules don't expose their children at any stage and must
            either be converted to setup form, or handle LoRA in the
            parent module.
        target_modules: Bare attribute names to wrap (e.g. ``["q_proj",
            "v_proj"]``). Names match exactly, no regex.
        rank: LoRA rank (config calls this ``r``).
        alpha: LoRA alpha (effective scale = alpha / rank).
        dropout: LoRA dropout, applied to the LoRA branch's input.

    Returns:
        ``(injected_model, partition_spec_for_optimizer)``.
        The model is mutated in place AND returned for convenience.
        ``partition_spec_for_optimizer`` is a ``{"params": ...}`` pytree
        of ``jax.sharding.PartitionSpec`` covering only the LoRA factors,
        so optax doesn't allocate Adam moments for the frozen 27 B base.
    """
    if not _HAS_FLAX:
        raise ImportError(
            "inject_lora requires jax + flax. Install them before calling "
            "this function."
        )
    if not _is_flax_module(model):
        raise TypeError(
            f"inject_lora expects a flax.linen.Module instance, got "
            f"{type(model).__name__}."
        )
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    if not target_modules:
        raise ValueError("target_modules is empty; nothing to inject.")

    targets = tuple(target_modules)
    injected_paths = _walk_and_inject(
        model,
        targets,
        rank=rank,
        alpha=float(alpha),
        dropout=float(dropout),
    )

    if not injected_paths:
        raise ValueError(
            f"inject_lora found no targets matching {targets!r} in the "
            f"model tree. Common cause: the model uses compact-style "
            f"@nn.compact submodules (children invisible until init), or "
            f"the layer names differ from the upstream Gemma-3 naming."
        )

    partition_spec = _build_partition_spec(injected_paths)
    return model, partition_spec
