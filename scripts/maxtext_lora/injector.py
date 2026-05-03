"""LoRA injector for Gemma-3 (Flax / MaxText).

Two entry points, one per kind of model:

* ``inject_lora(model, target_modules, ...)``
  Walks a Linen module tree (``vars(parent)``-visible children), finds
  every direct attribute or sequence-element whose name matches
  ``target_modules``, and swaps it for a ``LoraDense`` wrapper. Used by
  the synthetic Linen tests and any caller whose model is a plain Linen
  ``nn.Module`` whose children are eagerly assigned (post-bind, or via
  ``__post_init__``).

* ``apply_lora(mt_cfg, target_modules, ...)``
  The high-level helper for MaxText's Gemma-3. Builds the model with a
  LoRA-aware decoder layer baked in, init's the variables tree (base +
  LoRA), and returns the optimizer mask plus the device mesh. Necessary
  because MaxText's decoder layers live behind
  ``flax.nnx.bridge.ToLinen``, which recreates the underlying NNX module
  on every forward call from ``nnx_class(*args, **kwargs)`` — there is
  no persistent NNX instance to mutate. ``apply_lora`` instead
  monkey-patches ``gemma3.Gemma3DecoderLayerToLinen`` with a fresh
  ``ToLinen`` over a LoRA-aware ``Gemma3DecoderLayer`` subclass, so the
  wrapping is baked into every ToLinen rematerialisation. The patch is
  scoped to the call (``finally:`` restores the original symbol).

LoRA cost: only ``in_features * r + r * out_features`` parameters per
target. The base ``DenseGeneral`` kernel is reused (the wrapper holds a
reference to the same instance the original layer constructed), so
memory is dominated by the LoRA factors — about 13M for MedGemma-27B at
r=8 and 2 targets.

Returns from ``apply_lora`` are shaped to feed the trainer directly:
``(model, params, lora_filter_mask, mesh)``. The mask is a same-shape
bool tree, ``True`` only at ``lora_a`` / ``lora_b`` leaves, suitable
for ``optax.masked(...)`` so the optimizer never allocates Adam moments
for the frozen base.
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

# LoraDenseGeneral is the NNX sibling of LoraDense, used to wrap MaxText's
# multi-axis ``DenseGeneral`` projections (q/k/v on Gemma-3). Imported
# lazily for the same CPU-dev-box reasons.
try:
    from scripts.maxtext_lora.layer import LoraDenseGeneral  # type: ignore[import-not-found]
    _HAS_LORA_DG = LoraDenseGeneral is not None
except ImportError:
    LoraDenseGeneral = None  # type: ignore[assignment]
    _HAS_LORA_DG = False


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


def _wrap_in_lora(
    base_dense: Any,
    rank: int,
    alpha: float,
    dropout: float,
    *,
    rngs: Any = None,
) -> Any:
    """Build a LoRA wrapper for ``base_dense`` with the given hyperparams.

    Two base shapes are recognised:

    * Linen ``nn.Dense``: has ``features: int``. ``LoraDense`` wraps it
      directly; output dim is ``base.features``.
    * MaxText NNX ``DenseGeneral``: has ``in_features_shape`` and
      ``out_features_shape`` tuples. ``LoraDenseGeneral`` wraps it; the
      LoRA factors are flat (``(in_dim, r)`` and ``(r, prod(out_shape))``)
      and the multi-axis reshape happens at forward time. Requires NNX
      ``rngs`` for parameter init — the injector passes one through from
      ``inject_lora``'s seed.

    Everything is passed by keyword so a future field-order tweak doesn't
    break us.
    """
    if not _HAS_LORA_DENSE:
        raise ImportError(
            "scripts.maxtext_lora.layer.LoraDense not found. The layer "
            "module must be importable before the injector can run."
        )
    if _is_dense_general(base_dense):
        # The Linen walker can REACH a DenseGeneral when a synthetic
        # parent exposes one as a direct child, but inject_lora's path
        # cannot SAFELY wrap one inside a real MaxText model. Reason:
        # MaxText's DenseGeneral lives behind a flax.nnx.bridge.ToLinen
        # wrapper that recreates the underlying NNX module on every
        # forward call from ``nnx_class(*args, **kwargs)``. Mutating
        # the NNX instance after the fact does not survive — the next
        # call rebuilds a fresh, un-wrapped DenseGeneral.
        #
        # Use ``apply_lora`` for production: it monkey-patches the
        # decoder layer's nnx_class with a LoRA-aware subclass before
        # model creation, so the wrapping is baked into every ToLinen
        # rematerialisation. Raise loudly here so a future caller that
        # tries the inject_lora path on a maxtext model gets a clear
        # signal instead of silent loss-of-LoRA.
        raise NotImplementedError(
            "inject_lora cannot wrap a DenseGeneral child via the Linen "
            "walker — the wrapping won't survive ToLinen recreation. Use "
            "apply_lora(mt_cfg, ...) which subclasses the decoder layer's "
            "nnx_class instead."
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
    *,
    rngs: Any = None,
) -> list[tuple[str, ...]]:
    """Recursively walk ``module``, swap matching children for LoraDense.

    Returns the list of fully-qualified paths (e.g. ``("layers", "0",
    "self_attn", "q_proj")``) where injection happened, so the caller can
    build the optimizer partition spec without a second walk.

    ``rngs`` is forwarded to ``_wrap_in_lora`` for any DenseGeneral target
    that needs an NNX RNG for parameter init. Linen targets ignore it.
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
            new_child = _wrap_in_lora(child, rank, alpha, dropout, rngs=rngs)
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
            rngs=rngs,
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
    *,
    rngs: Any = None,
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
        rngs=rngs,
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


def _make_lora_decoder_subclass(
    base_decoder_cls: Any,
    target_modules: tuple[str, ...],
    rank: int,
    alpha: float,
    dropout: float,
    lora_seed: int,
) -> Any:
    """Build a LoRA-aware subclass of a MaxText decoder layer NNX class.

    The subclass overrides ``__init__`` to call ``super().__init__()`` first
    (which constructs the full attention block, including the
    ``DenseGeneral`` projections at ``self.self_attention.<name>``), then
    replaces each target projection with a ``LoraDenseGeneral`` that wraps
    the original DenseGeneral. The wrapped DenseGeneral keeps its kernel,
    so MaxText's base weights flow through Linen variables untouched; the
    new ``lora_a`` / ``lora_b`` ``nnx.Param``s show up alongside.

    Why this beats walking ``vars(model)``: MaxText's decoder layer is
    behind ``flax.nnx.bridge.ToLinen``, which recreates the NNX module on
    every forward call from ``nnx_class(*args, **kwargs)``. There is no
    persistent NNX instance to mutate. Subclassing ``nnx_class`` itself
    bakes the LoRA wrapping into every rematerialisation.

    Args:
        base_decoder_cls: e.g. ``maxtext.models.gemma3.Gemma3DecoderLayer``.
        target_modules: tuple of attribute names on
            ``self.self_attention`` to wrap (typically ``("query", "value")``).
        rank, alpha, dropout: LoRA hyperparameters forwarded to
            ``LoraDenseGeneral``.
        lora_seed: integer seed for the ``nnx.Rngs`` that initialises
            ``lora_a``. ``lora_b`` is zero-init so its seed doesn't matter,
            but we keep the API symmetric.
    """

    class LoraDecoderLayer(base_decoder_cls):
        """LoRA-wrapped decoder layer.

        Init-time replacement of the targeted DenseGeneral attributes is
        intentional: the MaxText layer's ``__init__`` does the
        ``self.self_attention = Attention(...)`` work, which constructs
        the q/k/v/o projections as DenseGeneral instances. After
        ``super().__init__()`` returns, those attributes are NNX modules
        we can swap out in place (NNX is eager, not lazy).
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Each fresh layer instance gets its own Rngs derived from
            # ``lora_seed``. Since ToLinen recreates the NNX module on
            # every call, this seed must produce the same factors every
            # time the subclass is instantiated — it does, because the
            # PRNGKey is deterministic and the LoRA factor shape is
            # fixed by the base DenseGeneral's in/out features. The Linen
            # variable tree carries the trained values forward through
            # ``nnx.update``, so the per-instance init is only used at
            # ``model.init`` time.
            inject_rngs = nnx.Rngs(params=jax.random.PRNGKey(lora_seed))

            attn = getattr(self, "self_attention", None)
            if attn is None:
                raise AttributeError(
                    f"{type(self).__name__}: no 'self_attention' attribute "
                    "found after super().__init__(); cannot inject LoRA."
                )
            for name in target_modules:
                if not hasattr(attn, name):
                    raise AttributeError(
                        f"{type(self).__name__}.self_attention has no "
                        f"attribute {name!r}; check target_modules against "
                        "the actual MaxText attention class."
                    )
                base_dg = getattr(attn, name)
                wrapped = LoraDenseGeneral(
                    base=base_dg,
                    rank=rank,
                    alpha=float(alpha),
                    dropout=float(dropout),
                    rngs=inject_rngs,
                )
                # NNX modules are mutable — direct setattr is fine and
                # registers the new submodule in the graph.
                setattr(attn, name, wrapped)

    LoraDecoderLayer.__name__ = f"Lora{base_decoder_cls.__name__}"
    LoraDecoderLayer.__qualname__ = LoraDecoderLayer.__name__
    return LoraDecoderLayer


def _patch_gemma3_to_linen(
    target_modules: tuple[str, ...],
    rank: int,
    alpha: float,
    dropout: float,
    lora_seed: int,
):
    """Monkey-patch ``gemma3.Gemma3DecoderLayerToLinen`` with a LoRA-aware
    ToLinen wrapper around a LoRA-aware ``Gemma3DecoderLayer`` subclass.

    Returns a ``(restore_fn, info_dict)`` pair. The caller decides when
    (or whether) to invoke ``restore_fn()``. ``apply_lora`` keeps the
    patch in place for the model's lifetime: ``decoders.py:469`` does a
    runtime ``gemma3.Gemma3DecoderLayerToLinen`` attribute lookup inside
    ``Decoder.setup()``, and Linen calls ``setup()`` again on every
    ``model.apply(...)``. Restoring the symbol after init would mean
    later apply calls re-build the decoder layers WITHOUT LoRA wrappers
    (variables present in the params tree, but not wired into the
    forward), which silently zeros the LoRA contribution.

    The patch DOES survive ToLinen recreation because ToLinen stores
    the class reference directly in ``self.nnx_class`` at
    instantiation. The dynamic part is the *outer* ``decoder.setup()``
    re-execution, which re-reads the module attribute. Replace the
    module-level symbol and every subsequent setup re-fetches the
    LoRA-aware class.
    """
    from maxtext.models import gemma3
    from maxtext.layers import nnx_wrappers
    from maxtext.layers import initializers

    base_cls = gemma3.Gemma3DecoderLayer
    LoraDecoderCls = _make_lora_decoder_subclass(
        base_decoder_cls=base_cls,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        lora_seed=lora_seed,
    )
    LoraToLinen = nnx_wrappers.to_linen_class(
        LoraDecoderCls,
        base_metadata_fn=initializers.variable_to_logically_partitioned,
    )

    original = gemma3.Gemma3DecoderLayerToLinen
    gemma3.Gemma3DecoderLayerToLinen = LoraToLinen

    def _restore() -> None:
        gemma3.Gemma3DecoderLayerToLinen = original

    return _restore, {
        "base_cls": base_cls,
        "lora_cls": LoraDecoderCls,
        "lora_to_linen": LoraToLinen,
    }


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
    """Build the MaxText Gemma-3 model with LoRA wrappers baked in,
    init params (base random + LoRA factors), and return the optimizer
    mask plus the device mesh.

    Approach: subclass the gemma3 decoder NNX class with a LoRA-aware
    version that, at construction time, replaces
    ``self.self_attention.<target>`` for each target name with a
    ``LoraDenseGeneral`` wrapping the original ``DenseGeneral``. We then
    monkey-patch ``gemma3.Gemma3DecoderLayerToLinen`` with a fresh
    ``ToLinen`` over the subclass before calling
    ``model_creation_utils.from_config(...)``. The patch is undone in a
    ``finally:`` so concurrent or later code paths that import gemma3
    aren't disturbed.

    This is the only injection path that survives ToLinen's
    re-instantiation behaviour: ``ToLinen.__call__`` rebuilds the NNX
    module from ``self.nnx_class(*args, **kwargs)`` on every forward,
    so a per-instance mutation is silently lost. The subclass approach
    bakes the wrapping into the class constructor itself.

    Returns ``(model, params, lora_filter_mask, mesh)``:
      - ``model``: the Linen ``TransformerLinenPure`` whose decoder
        layers are now LoRA-aware (via the patched ToLinen wrapper).
      - ``params``: full variables tree
        (``{"params": {...}}``) including base weights AND
        freshly-initialised ``lora_a`` (kaiming) and ``lora_b`` (zero)
        leaves at ``params/decoder/layers_<i>/self_attention/<target>/``.
        The base weights will be overwritten by the orbax restore in
        the trainer.
      - ``lora_filter_mask``: same-shape bool tree, ``True`` at every
        ``lora_a`` / ``lora_b`` leaf, suitable for ``optax.masked``.
      - ``mesh``: the JAX device mesh derived from ``mt_cfg``.

    Args:
        mt_cfg: a ``pyconfig.HyperParameters`` initialised with the
            right ``model_name`` / ``per_device_batch_size`` /
            ``max_target_length`` / ``weight_dtype`` keys. Must select
            a gemma3 decoder block.
        target_modules: attribute names on
            ``Gemma3DecoderLayer.self_attention`` to wrap. The MaxText
            attention exposes ``query``, ``key``, ``value``, ``out`` —
            the typical LoRA setup is ``["query", "value"]``. The
            trainer config currently passes the HuggingFace-style names
            ``["q_proj", "v_proj"]``; those are auto-translated to
            ``["query", "value"]`` here so callers don't have to re-map.
        rank: LoRA rank ``r``.
        alpha: LoRA alpha (effective scale = ``alpha / rank``).
        dropout: forwarded to ``LoraDenseGeneral`` (currently
            informational; LoraDenseGeneral's forward is dropout-free
            until an NNX dropout RNG is plumbed).
        variant: must be ``"standard"`` for now.
        seed: RNG seed shared between the model init pass and the LoRA
            factor init.
    """
    if not _HAS_FLAX:
        raise ImportError("apply_lora requires jax + flax.")
    if not _HAS_NNX:
        raise ImportError(
            "apply_lora requires flax.nnx (the NNX-side LoraDenseGeneral "
            "wraps MaxText's NNX DenseGeneral). Install a flax version "
            "that ships nnx, or run on the same py3.11 venv used on TPU."
        )
    if not _HAS_LORA_DG:
        raise ImportError(
            "scripts.maxtext_lora.layer.LoraDenseGeneral is not "
            "available — the layer module must be importable here."
        )
    if variant != "standard":
        raise ValueError(
            f"variant={variant!r} not supported yet. Only 'standard' "
            "(LoRA) ships today; DoRA / rsLoRA tracked separately."
        )
    if not target_modules:
        raise ValueError("target_modules is empty; nothing to inject.")
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")

    # We patch ``Gemma3DecoderLayerToLinen`` only. The scannable path
    # (``Gemma3ScannableBlockToLinen``) wraps a scan-built block whose
    # NNX class is ``Gemma3ScannableBlock`` — that class instantiates
    # ``Gemma3DecoderLayer`` inside a loop, NOT through the patched
    # symbol, so our subclass would be bypassed. Bail loudly so the
    # caller flips ``scan_layers=False`` (the converter uses unrolled
    # checkpoints anyway).
    if bool(getattr(mt_cfg, "scan_layers", False)):
        raise NotImplementedError(
            "apply_lora currently requires scan_layers=False. The "
            "scannable Gemma3ScannableBlock builds decoder layers "
            "directly from Gemma3DecoderLayer (not through the patched "
            "ToLinen wrapper), so the LoRA wrapping would be silently "
            "skipped on the scan path."
        )

    import jax  # noqa: F401  matches deferred-jax pattern
    import jax.numpy as jnp
    from flax.linen import partitioning as nn_partitioning

    # MaxText is on sys.path (added by the trainer's _train()).
    from maxtext.utils import model_creation_utils
    from maxtext.utils import maxtext_utils

    # Translate HuggingFace-style target names to MaxText attention attrs.
    # The trainer config historically uses ``q_proj`` / ``v_proj`` /
    # ``k_proj`` / ``o_proj`` (PEFT convention). MaxText's attention
    # class exposes ``query`` / ``key`` / ``value`` / ``out``. We map the
    # common names so existing configs keep working without surgery.
    _HF_TO_MAXTEXT = {
        "q_proj": "query",
        "k_proj": "key",
        "v_proj": "value",
        "o_proj": "out",
    }
    translated = tuple(_HF_TO_MAXTEXT.get(name, name) for name in target_modules)

    mesh = maxtext_utils.get_mesh_from_config(mt_cfg)

    # Patch the gemma3 ToLinen symbol BEFORE building the model. The
    # decoder picks up the LoRA-aware subclass for every layer.
    #
    # NOTE: the patch is intentionally LEFT IN PLACE for the lifetime of
    # the returned ``model``. ``decoders.py:469`` does a *runtime*
    # attribute lookup (``gemma3.Gemma3DecoderLayerToLinen``) inside
    # ``Decoder.setup()``, and Linen calls ``setup()`` again on every
    # ``model.apply(...)``. Restoring the symbol after init would mean
    # subsequent apply calls re-build decoder layers WITHOUT LoRA
    # wrappers — variables present, but not wired into the forward.
    # The trainer process only constructs one model per run, so the
    # global patch is harmless. If the caller really needs to undo
    # the patch later (e.g. test isolation), they can call the
    # returned ``restore_patch`` callback at the end of the model's
    # life — but DO NOT call it before training is done.
    restore_patch, _info = _patch_gemma3_to_linen(
        target_modules=translated,
        rank=rank,
        alpha=float(alpha),
        dropout=float(dropout),
        # Offset from the model-init seed so the LoRA factors are
        # decorrelated from the base init RNG stream. Same offset every
        # time so a re-init reproduces identical LoRA factors.
        lora_seed=int(seed) + 5000,
    )
    try:
        # Build the base model (Linen, not NNX — rngs=None below).
        # MaxText derives the mesh from the config + currently-visible
        # devices, which is what the rest of the SFT loop expects too.
        model = model_creation_utils.from_config(mt_cfg, mesh=mesh)

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
    except Exception:
        # If init failed, restore the symbol so the next attempt or
        # subsequent unrelated code path doesn't see a stale LoRA
        # subclass. On success, leave it patched (see comment above).
        restore_patch()
        raise

    lora_filter_mask = _build_lora_filter_mask(variables)
    # Stash the restore callback on the returned model so a caller who
    # wants to undo the patch (e.g. a unit test that builds another
    # model after) has a hook. Public, but undocumented in the type
    # signature; intended for advanced use only.
    try:
        object.__setattr__(model, "_lora_restore_patch", restore_patch)
    except Exception:
        # Some Linen versions reject setattr on the bare module pre-init;
        # not fatal — the trainer doesn't need this hook.
        pass
    return model, variables, lora_filter_mask, mesh
