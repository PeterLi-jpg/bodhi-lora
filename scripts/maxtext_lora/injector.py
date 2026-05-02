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

# NNX is a separate module from Linen. MaxText's Gemma-3 attention/MLP
# projections are ``DenseGeneral(nnx.Module)`` instances (see
# third_party/maxtext/src/maxtext/layers/linears.py), which the Linen-only
# walker would skip. We import nnx if available so the walker can recognise
# either flavour; production wrapping of NNX modules is a follow-up (see
# _wrap_in_lora below: we raise NotImplementedError until a real NNX-aware
# LoraDense lands).
try:
    from flax import nnx  # type: ignore[import-not-found]
    _HAS_NNX = True
except ImportError:
    nnx = None  # type: ignore[assignment]
    _HAS_NNX = False

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
    """True if obj is a Flax module: either Linen ``nn.Module`` or NNX ``nnx.Module``.

    The injector walks both kinds: MaxText's own layers are NNX (e.g.
    ``DenseGeneral``), while the Linen-bridge wrapper that ``model_creation_utils``
    sometimes returns is plain Linen. Returns False on bare envs where flax
    isn't importable, so this file stays usable for static analysis even
    without the runtime deps.
    """
    if _HAS_FLAX and isinstance(obj, nn.Module):
        return True
    if _HAS_NNX and isinstance(obj, nnx.Module):
        return True
    return False


def _is_dense_general(obj: Any) -> bool:
    """True if ``obj`` looks like MaxText's ``DenseGeneral`` (duck-typed).

    We can't ``isinstance``-check ``DenseGeneral`` here without importing
    MaxText (which pulls in its own JAX/Flax deps and sometimes runs setup
    side-effects), so we use the same duck-type the wrapper would need
    anyway: ``in_features_shape`` + ``out_features_shape`` tuples and a
    ``kernel`` attribute. Linen's ``nn.Dense`` has ``features`` (a single
    int) instead and is detected separately.
    """
    return (
        hasattr(obj, "in_features_shape")
        and hasattr(obj, "out_features_shape")
        and hasattr(obj, "kernel")
    )


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

    Two base shapes are recognised:

    * Linen ``nn.Dense``: has ``features: int``. The existing ``LoraDense``
      wraps it directly; output dim is ``base.features``.
    * MaxText NNX ``DenseGeneral``: has ``out_features_shape`` (a tuple) and
      ``in_features_shape`` (also a tuple). Its output may be multi-axis
      (e.g. attention's ``query`` projects to
      ``(num_query_heads, head_dim)``), which the current Linen-only
      ``LoraDense`` does not handle. We raise ``NotImplementedError`` with
      a clear message rather than silently producing a wrapper that builds
      mis-shaped LoRA factors.

    Everything is passed by keyword so a future field-order tweak doesn't
    break us.
    """
    if not _HAS_LORA_DENSE:
        raise ImportError(
            "scripts.maxtext_lora.layer.LoraDense not found. Unit 2 must be "
            "merged before the injector can run."
        )
    if _is_dense_general(base_dense):
        # NNX DenseGeneral wrapping is intentionally a follow-up: it needs
        # an NNX-flavoured LoraDense that handles multi-axis outputs (q/k/v
        # project to (num_heads, head_dim), MLP wi to (intermediate_dim,)).
        # Crash with context instead of producing a broken wrapper.
        raise NotImplementedError(
            "inject_lora encountered a DenseGeneral target "
            f"(in_features_shape={getattr(base_dense, 'in_features_shape', None)!r}, "
            f"out_features_shape={getattr(base_dense, 'out_features_shape', None)!r}). "
            "The current LoraDense only wraps flax.linen.Dense; an NNX-aware "
            "LoraDense for DenseGeneral is a follow-up. See "
            "scripts/maxtext_lora/layer.py and the PR description for #6."
        )
    return LoraDense(
        features=base_dense.features,
        base=base_dense,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )


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


def _build_lora_filter_mask(params: Any) -> Any:
    """Walk a params pytree and return a same-shape pytree of bools,
    True at every leaf whose flat path contains a ``lora_a`` or
    ``lora_b`` segment.

    This is the mask we hand to ``optax.masked(...)`` so the optimizer
    only allocates Adam moments + applies updates for the LoRA factors;
    everything in the frozen base is left untouched (and stays at zero
    grad-update so the 27B base never moves).

    Implementation note: we use ``jax.tree_util.tree_map_with_path``
    when available (jax >= 0.4.20) to avoid having to flatten and
    reconstruct manually. The path comes through as a tuple of
    ``DictKey``/``GetAttrKey`` etc.; we look at the str() form to
    decide whether ``lora_a`` / ``lora_b`` appears anywhere along
    the path. This is robust to NNX vs Linen and to the exact
    container types Flax uses.
    """
    if not _HAS_FLAX:
        raise ImportError(
            "_build_lora_filter_mask requires jax + flax."
        )
    import jax  # local import: matches the deferred-jax pattern at the top of this module

    def _is_lora_leaf(path: Any, _leaf: Any) -> bool:
        flat = "/".join(str(p) for p in path)
        return ("lora_a" in flat) or ("lora_b" in flat)

    return jax.tree_util.tree_map_with_path(_is_lora_leaf, params)


def apply_lora(
    mt_cfg: Any,
    *,
    target_modules: list[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    variant: str = "standard",  # noqa: ARG001  (placeholder for DoRA / rsLoRA)
    seed: int = 0,
) -> tuple[Any, Any, Any, Any]:
    """High-level helper for the trainer: build the MaxText Gemma-3 model,
    inject LoRA, init params, and return the boolean mask the optimizer
    needs, plus the mesh.

    Returns ``(model, params, lora_filter_mask, mesh)`` where:
      - ``model`` is the LoRA-wrapped Linen module
      - ``params`` is the full nested params pytree
        (``{"params": {...}}`` shape) including freshly-initialized
        ``lora_a`` (kaiming-uniform) and ``lora_b`` (zero) factors AND
        the base weights from MaxText's model init. Replace the base
        weights with the orbax checkpoint via your training-state setup
        (see ``train_lora_maxtext.py``), or use the helper here as the
        first step before orbax restore.
      - ``lora_filter_mask`` is a same-shape pytree of bool, True at
        every LoRA factor leaf, suitable for ``optax.masked(...)``.
      - ``mesh`` is the JAX device mesh derived from ``mt_cfg``;
        returned explicitly so the trainer doesn't have to access
        ``model.mesh`` (the Linen wrapper variants don't all expose
        the attribute consistently).

    Why we don't do the orbax restore here: it requires a
    ``CheckpointManager`` and a ``mesh`` from MaxText's lifecycle
    helpers (``setup_initial_state``). Keeping the LoRA-specific work
    in this helper and the orbax-restore step in the trainer keeps
    each piece testable and matches MaxText's own pre_train factoring.

    Args:
        mt_cfg: a ``pyconfig.HyperParameters`` already initialized with
            the right ``model_name`` / ``per_device_batch_size`` /
            ``max_target_length`` / ``weight_dtype`` keys.
        target_modules: list of attribute names to wrap, e.g.
            ``["q_proj", "v_proj"]``.
        rank: LoRA rank ``r``.
        alpha: LoRA alpha (effective scale = ``alpha / rank``).
        dropout: applied to the input of the LoRA branch.
        variant: ``"standard"`` (the only supported value today). DoRA
            / rsLoRA reserved for a later PR.
        seed: RNG seed for the model init pass. The same seed should be
            used for ``mt_cfg.init_weights_seed`` so a re-init reproduces
            the same LoRA factors.

    UNTESTED: this helper has not yet run end-to-end on a TPU. The
    pieces (``inject_lora``, ``model_creation_utils.from_config``,
    ``model.init``) are individually tested, but the composition
    (init-after-inject) is new code that will need verification on
    a real v6e VM.
    """
    if not _HAS_FLAX:
        raise ImportError("apply_lora requires jax + flax.")
    if variant != "standard":
        raise ValueError(
            f"variant={variant!r} not supported yet. Only 'standard' "
            "(LoRA) ships today; DoRA / rsLoRA tracked separately."
        )

    import jax  # noqa: F401  matches deferred-jax pattern

    # MaxText is on sys.path (added by the trainer's _train()).
    from maxtext.utils import model_creation_utils
    from maxtext.utils import maxtext_utils

    # Build the base model (Linen, not NNX — rngs=None below).
    # MaxText derives the mesh from the config + currently-visible
    # devices, which is what the rest of the SFT loop expects too.
    mesh = maxtext_utils.get_mesh_from_config(mt_cfg)
    model = model_creation_utils.from_config(mt_cfg, mesh=mesh)

    # Inject LoRA wrappers into q_proj / v_proj / etc. inject_lora
    # mutates ``model`` in place AND returns it. The partition_spec it
    # also returns is the per-LoRA-factor PartitionSpec covering only
    # the new params; we keep it for future sharding tweaks but the
    # filter mask below is what optax.masked() needs.
    inject_lora(
        model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )

    # Init the full params pytree (base + LoRA) once. The base weights
    # will be overwritten by the orbax restore in the trainer; the
    # LoRA factors stay as initialized (lora_a kaiming, lora_b zeros).
    #
    # Three RNG collections like MaxText's init_initial_state (see
    # third_party/maxtext/src/maxtext/utils/maxtext_utils.py:1247):
    # ``params`` for weight init, ``dropout`` for dropout layers,
    # ``aqt`` for AQT (quantization noise) sampling. Missing ``aqt``
    # raises a ``KeyError`` from inside the Gemma-3 quantization
    # decorator on the first init pass.
    params_key, dropout_key, aqt_key = jax.random.split(
        jax.random.PRNGKey(seed), 3
    )
    init_rngs = {
        "params": params_key,
        "dropout": dropout_key,
        "aqt": aqt_key,
    }
    import jax.numpy as jnp
    from flax.linen import partitioning as nn_partitioning
    bsz = int(mt_cfg.per_device_batch_size)
    seqlen = int(mt_cfg.max_target_length)
    dummy_inputs = jnp.zeros((bsz, seqlen), dtype=jnp.int32)
    dummy_positions = jnp.broadcast_to(
        jnp.arange(seqlen, dtype=jnp.int32), (bsz, seqlen)
    )
    dummy_segmentation = jnp.ones((bsz, seqlen), dtype=jnp.int32)

    # Init under mesh + logical_axis_rules so the model's
    # ``with_partitioning`` annotations on its Dense kernels resolve to
    # real PartitionSpecs (and the resulting params are sharded across
    # the v6e fsdp axis instead of replicated). This mirrors
    # ``maxtext_utils.get_abstract_param`` (lines 1280+) which uses the
    # same context for jax.eval_shape.
    with mesh, nn_partitioning.axis_rules(mt_cfg.logical_axis_rules):
        variables = model.init(
            init_rngs,
            dummy_inputs,
            dummy_positions,
            decoder_segment_ids=dummy_segmentation,
            encoder_images=None,
            encoder_image_masks=None,
            enable_dropout=False,  # init pass; no dropout
            decoder_target_tokens=dummy_inputs,
            decoder_target_mask=dummy_segmentation,
        )

    lora_filter_mask = _build_lora_filter_mask(variables)
    return model, variables, lora_filter_mask, mesh
