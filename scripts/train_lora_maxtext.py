"""LoRA SFT on MedGemma-27B using a forked MaxText (Stage 3).

This is the JAX/Flax counterpart of ``scripts/train_lora.py``. The base
model (MedGemma-27B) is built via ``maxtext.utils.model_creation_utils``
from the orbax checkpoint produced by ``scripts/convert_medgemma_to_maxtext.py``;
LoRA factors are injected by ``scripts.maxtext_lora.injector.apply_lora``;
the training loop runs with optax + a deferred ``optax.masked`` optimizer
so only the LoRA factors get Adam moments + grad updates (the 27 B base
stays frozen). After training, ``scripts.maxtext_lora.export_peft.write_adapter``
converts the orbax LoRA save into a HF PEFT adapter directory that
Stage 4 eval can load.

Pipeline:
    1. Argparse + YAML load (no JAX needed — keeps ``--help`` light).
    2. Lazy-import JAX / orbax / MaxText.
    3. Build ``mt_cfg`` via ``pyconfig.initialize(...)`` from the YAML.
    4. Build the Gemma-3 Flax model (Linen, not NNX).
    5. Inject LoRA via the existing injector.
    6. Initialize the full params tree (base + LoRA).
    7. Restore the base orbax checkpoint into the params tree
       (the LoRA factors stay at their kaiming/zero init).
    8. Build optax with a deferred ``mask`` callable so optimizer state
       and updates are confined to LoRA params.
    9. Custom JAX/Optax training loop:
         - jit'd train_step computing forward + masked CE loss + grad +
           update.
         - Periodic logging.
         - End-of-epoch eval.
         - End-of-epoch orbax save of the LoRA factors only.
    10. Final orbax save → call ``export_peft.write_adapter`` so
        ``<output-dir>/best/`` ends up in the same PEFT layout the
        torch_xla path produced.

UNTESTED notes (each tagged at the call site below):
    - The orbax restore call shape (``PyTreeCheckpointer().restore``).
      MaxText's own loader uses a ``CheckpointManager`` with restore
      args; raw PyTreeCheckpointer may need those args set.
    - ``model.apply`` signature for Gemma-3 — may need explicit
      ``inputs_position`` / ``decoder_segment_ids`` args.
    - Whether the Linen model exposes children visible to ``inject_lora``
      after ``from_config`` (depends on MaxText's setup-vs-compact mode).

These are the smoke-debug points; first TPU iteration will surface
the actual API shapes we need to match.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


def _set_global_seeds(seed: int) -> None:
    """Seed Python + NumPy. JAX PRNG keys are derived inside _train()."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def _load_config(path: str) -> dict:
    """Read the YAML config and validate the keys this trainer expects."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for section in ("model", "lora", "training", "data", "paths"):
        if section not in cfg:
            raise ValueError(
                f"config {path!r}: missing required section {section!r}. "
                f"See configs/lora_medgemma27b_maxtext.yaml for the expected shape."
            )
    for key in ("maxtext_orbax_checkpoint", "maxtext_dataset_dir"):
        if key not in cfg["paths"]:
            raise ValueError(
                f"config {path!r}: paths.{key} is required. "
                "It points to the output of the HF→MaxText converter or "
                "the JSONL→tokenized converter, respectively."
            )
    return cfg


def _expand(path: str) -> str:
    """``~/...`` and ``$VAR`` expansion so configs are portable between VMs."""
    return os.path.expanduser(os.path.expandvars(path))


def _maxtext_base_config_path() -> str:
    """Resolve the absolute path to MaxText's base.yml under third_party/.

    MaxText's pyconfig.initialize expects argv[1] to be the base config
    YAML path; the rest are key=value overrides. Mirrors
    ``scripts/convert_medgemma_to_maxtext.py:_maxtext_base_config_path``.
    Falls back to the relative form if vendoring isn't in place yet so
    --help stays unit-testable without the vendored tree.
    """
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "third_party" / "maxtext" / "src" / "maxtext" / "configs" / "base.yml"
    if candidate.is_file():
        return str(candidate)
    return "src/maxtext/configs/base.yml"


def _unwrap_orbax_state(restored):
    """Some MaxText orbax saves wrap the param tree under ``state.params``
    or ``params``; some don't. Walk down known wrapper keys until we
    find a leaf shaped like the model's param tree (i.e. has nested
    decoder/embedder/etc. children). Returns the unwrapped tree."""
    if not isinstance(restored, dict):
        return restored
    # Heuristic: if the restored dict has exactly one of these wrappers,
    # peel it. Stop when the next level no longer has a single-key
    # wrapper structure.
    for wrapper in ("state", "params"):
        if (
            isinstance(restored, dict)
            and len(restored) == 1
            and wrapper in restored
            and isinstance(restored[wrapper], dict)
        ):
            restored = restored[wrapper]
    return restored


def _merge_base_into_params(params, restored, lora_filter_mask=None):
    """Overwrite ``params`` leaves with values from ``restored`` wherever
    paths match. Leaves only present in ``params`` (the LoRA factors)
    are preserved untouched.

    Defensive against orbax-wrapper differences: ``_unwrap_orbax_state``
    peels common wrappers (``state.params``, ``params``) before
    matching. Whatever remains is matched by string-form path against
    the LoRA-injected init tree.

    Aborts hard if the merge clearly did not work: zero matches, or
    fewer than half of the expected base leaves matched. ``lora_count``
    is computed from the LoRA filter mask if provided (preferred), else
    inferred by walking ``params`` for ``lora_a`` / ``lora_b`` paths.
    """
    import jax

    restored_inner = _unwrap_orbax_state(restored)
    flat_params = jax.tree_util.tree_flatten_with_path(params)[0]
    flat_restored = jax.tree_util.tree_flatten_with_path(restored_inner)[0]
    restored_lookup = {tuple(str(k) for k in path): leaf for path, leaf in flat_restored}

    matched = 0

    def _replace(path, leaf):
        nonlocal matched
        key = tuple(str(k) for k in path)
        if key in restored_lookup:
            matched += 1
            return restored_lookup[key]
        return leaf

    merged = jax.tree_util.tree_map_with_path(_replace, params)
    total_params = len(flat_params)
    total_restored = len(flat_restored)

    if lora_filter_mask is not None:
        lora_count = sum(
            1 for x in jax.tree_util.tree_leaves(lora_filter_mask) if bool(x)
        )
    else:
        lora_count = sum(
            1 for path, _ in flat_params if _is_lora_path(path)
        )

    print(
        f"[orbax merge] matched {matched}/{total_params} param leaves "
        f"(restored tree had {total_restored} leaves). LoRA factors "
        f"in init = {lora_count}. If matched is much lower than "
        "(total - LoRA factor count), the orbax pytree layout differs "
        "from the init tree; adjust _unwrap_orbax_state.",
        flush=True,
    )

    expected_base = max(1, total_params - lora_count)
    match_ratio = matched / expected_base
    # Tightened from "match_ratio < 0.5" — a partial merge that clears
    # 50-99% still trains on a partially-random base. Require the merge
    # to find every expected base leaf. ``MERGE_TOLERANCE`` is the
    # number of unmatched leaves we'll forgive (default 0). Override
    # via ``BOHDI_MERGE_TOLERANCE`` env var on a per-run basis if a
    # known-quirky orbax layout legitimately drops a small number of
    # auxiliary leaves (e.g. step counters that aren't in our init).
    import os
    tolerance = int(os.environ.get("BOHDI_MERGE_TOLERANCE", "0"))
    missing = expected_base - matched
    if matched == 0 or missing > tolerance:
        raise RuntimeError(
            f"orbax merge: only {matched}/{expected_base} base params matched "
            f"(missing {missing}; ratio={match_ratio:.2f}; "
            f"tolerance={tolerance}). The orbax pytree layout disagrees with "
            "the init tree; fix _unwrap_orbax_state or the converter output "
            "rather than training on a partially-random base. Set "
            "BOHDI_MERGE_TOLERANCE=<n> to permit up to n unmatched leaves "
            "if you know a small number are legitimate aux state."
        )
    return merged


def _build_lr_schedule(train_cfg: dict, total_steps: int):
    """Cosine decay with linear warmup. Matches the torch_xla trainer's
    schedule so smoke comparisons stay apples-to-apples."""
    import optax

    lr = float(train_cfg["learning_rate"])
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.03))
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0,
    )


def _is_lora_path(path) -> bool:
    """True if a tree-path tuple corresponds to a LoRA factor leaf."""
    return any(("lora_a" in str(p)) or ("lora_b" in str(p)) for p in path)


def _train(cfg: dict, seed: int, output_dir: str) -> None:
    """Run the LoRA SFT training loop end-to-end."""
    # --- Lazy heavy imports ----------------------------------------------------
    # JAX/orbax/MaxText come from the third_party/maxtext fork.
    # scripts.maxtext_lora.* are sibling modules.
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp

    repo_root = Path(__file__).resolve().parent.parent
    # CORRECTED (was pointing at third_party/maxtext, missing /src). The
    # vendored package layout is third_party/maxtext/src/maxtext/...
    # so the importable parent is the /src dir.
    maxtext_src = repo_root / "third_party" / "maxtext" / "src"
    if str(maxtext_src) not in sys.path:
        sys.path.insert(0, str(maxtext_src))

    # CORRECTED imports — package is lowercase ``maxtext`` and pyconfig
    # lives under ``maxtext.configs``. The previous ``MaxText.experimental.sft``
    # path doesn't exist in the vendored copy; we don't use Tunix's SFT
    # trainer (it requires Python 3.11+ which v6e doesn't have), so we
    # don't import it at all.
    from maxtext.configs import pyconfig
    from maxtext.common import checkpointing as mt_checkpointing
    from flax.linen import partitioning as nn_partitioning

    from scripts.maxtext_lora import lora_inject
    from scripts.maxtext_lora import dataset_loader
    from scripts.maxtext_lora import export_peft

    # --- Resolve config sections + paths --------------------------------------
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    paths_cfg = cfg["paths"]

    orbax_ckpt = _expand(paths_cfg["maxtext_orbax_checkpoint"])
    dataset_dir = _expand(paths_cfg["maxtext_dataset_dir"])
    train_file = _expand(data_cfg["train_file"])
    val_file = _expand(data_cfg["val_file"])

    if not Path(orbax_ckpt).exists():
        raise FileNotFoundError(
            f"orbax base-model checkpoint not found at {orbax_ckpt}. "
            "Run scripts/convert_medgemma_to_maxtext.py first."
        )
    if not Path(dataset_dir).exists():
        raise FileNotFoundError(
            f"MaxText dataset dir not found at {dataset_dir}. "
            "Run scripts/convert_traces_to_maxtext.py first."
        )
    # The trainer reads <dataset_dir>/<split>.tokenized.jsonl (the
    # pre-tokenized sidecars). The .jsonl files (raw messages) are kept
    # for re-tokenization but not consumed here.
    _expected = ["train.tokenized.jsonl", "val.tokenized.jsonl"]
    _missing = [f for f in _expected if not (Path(dataset_dir) / f).is_file()]
    if _missing:
        raise FileNotFoundError(
            f"MaxText dataset dir {dataset_dir} is missing tokenized "
            f"sidecars: {_missing}. Re-run scripts/convert_traces_to_maxtext.py."
        )

    output_dir = _expand(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    orbax_out = Path(output_dir) / "orbax"
    best_out = Path(output_dir) / "best"
    Path(orbax_out).mkdir(parents=True, exist_ok=True)
    Path(best_out).mkdir(parents=True, exist_ok=True)

    # --- Build MaxText config ------------------------------------------------
    mt_argv = [
        "model_name=gemma3-27b",
        f"load_parameters_path={orbax_ckpt}",
        f"dataset_path={dataset_dir}",
        f"per_device_batch_size={train_cfg['per_device_train_batch_size']}",
        f"gradient_accumulation_steps={train_cfg['gradient_accumulation_steps']}",
        f"max_target_length={train_cfg['max_seq_length']}",
        f"weight_dtype={'bfloat16' if train_cfg.get('bf16', True) else 'float32'}",
        f"dtype={'bfloat16' if train_cfg.get('bf16', True) else 'float32'}",
        f"base_output_directory={orbax_out}",
        f"run_name=lora_seed_{seed}",
        f"data_seed={seed}",
        f"init_weights_seed={seed}",
    ]
    # MaxText's pyconfig expects argv[0]=script name and argv[1]=base YAML
    # path; everything after is key=value overrides. Mirror the converter
    # script's pattern so the trainer doesn't crash on initialize().
    _base_yml_path = _maxtext_base_config_path()
    mt_cfg = pyconfig.initialize(["train_lora_maxtext.py", _base_yml_path, *mt_argv])
    print(f"MaxText config initialized: model={model_cfg['name']!r} "
          f"seed={seed} output_dir={output_dir!r}", flush=True)

    # --- LoRA injection ------------------------------------------------------
    # apply_lora: builds the Gemma-3 Linen model from mt_cfg, calls
    # inject_lora to wrap the target_modules with LoraDense, runs
    # model.init to materialize a full params tree (base random init +
    # LoRA factors at PEFT init: kaiming A, zero B), and returns a
    # boolean filter mask for optax.masked.
    model, params, lora_filter_mask, mesh = lora_inject.apply_lora(
        mt_cfg,
        target_modules=lora_cfg["target_modules"],
        rank=lora_cfg["r"],
        alpha=lora_cfg["lora_alpha"],
        dropout=lora_cfg.get("lora_dropout", 0.05),
        variant=lora_cfg.get("variant", "standard"),
        seed=seed,
    )
    n_lora = sum(int(x) for x in jax.tree_util.tree_leaves(lora_filter_mask) if x)
    print(f"LoRA injection done: {n_lora:,} trainable param leaves "
          f"(target_modules={lora_cfg['target_modules']}, r={lora_cfg['r']})",
          flush=True)

    # --- Restore the base orbax checkpoint into params -----------------------
    # Use MaxText's ``load_params_from_path`` rather than raw
    # PyTreeCheckpointer().restore() — the helper handles OCDBT / Zarr3
    # format flags and constructs the right ``restore_args`` from an
    # abstract-shaped params tree. The abstract tree is just our
    # already-initialized ``params`` mapped to ShapeDtypeStruct.
    print(f"Restoring base weights from {orbax_ckpt}...", flush=True)
    abstract_params = jax.tree_util.tree_map(
        lambda p: jax.ShapeDtypeStruct(
            shape=p.shape, dtype=p.dtype,
            sharding=getattr(p, "sharding", None),
        ),
        params,
    )
    # Strip the ``params`` outer key for load_params_from_path —
    # MaxText saves the inner ``params`` dict under the ``params`` key
    # of the on-disk checkpoint (see save_params_to_path). The helper
    # peels that wrapper itself; we hand it the inner abstract.
    abstract_inner = abstract_params.get("params", abstract_params) \
        if isinstance(abstract_params, dict) else abstract_params
    # Wrap restore in mesh + axis_rules — MaxText's setup_decode_state
    # does the same (line 1289 of maxtext_utils.py: ``with
    # nn_partitioning.axis_rules(config.logical_axis_rules):
    # checkpointing.load_params_from_path(...)``). Without the context
    # the restored arrays may end up replicated, blowing memory on
    # the 27 B base. ``mesh`` comes from apply_lora (returned
    # explicitly so we don't depend on ``model.mesh``, which isn't
    # exposed on every Linen wrapper variant).
    with mesh, nn_partitioning.axis_rules(mt_cfg.logical_axis_rules):
        try:
            restored_inner = mt_checkpointing.load_params_from_path(
                str(Path(orbax_ckpt).resolve()),
                abstract_inner,
                checkpoint_storage_concurrent_gb=96,
            )
            restored = {"params": restored_inner} \
                if isinstance(params, dict) and "params" in params \
                else restored_inner
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            # Narrow exception list so KeyboardInterrupt / SystemExit and
            # other genuinely fatal errors propagate; we only want to
            # fall back on storage / shape / key-mismatch failures.
            print(
                f"WARNING: load_params_from_path failed ({e!r}). Falling "
                "back to raw PyTreeCheckpointer().restore(); sharding "
                "may be wrong but this gives a clearer error to iterate on.",
                flush=True,
            )
            restored = ocp.PyTreeCheckpointer().restore(
                str(Path(orbax_ckpt).resolve())
            )
        params = _merge_base_into_params(params, restored, lora_filter_mask)
    print("Base weights restored.", flush=True)

    # --- Dataset iterators ---------------------------------------------------
    train_iter, eval_iter, steps_per_epoch = dataset_loader.build_iterators(
        dataset_dir=dataset_dir,
        train_file=train_file,
        val_file=val_file,
        per_device_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        max_seq_length=train_cfg["max_seq_length"],
        seed=seed,
    )
    num_epochs = int(train_cfg["num_epochs"])
    total_steps = num_epochs * steps_per_epoch
    print(f"Dataset built: {steps_per_epoch} steps/epoch x "
          f"{num_epochs} epochs = {total_steps} total steps", flush=True)

    # --- Optimizer (LoRA-only via deferred mask) -----------------------------
    lr_schedule = _build_lr_schedule(train_cfg, total_steps)
    base_tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        ),
    )

    def _mask_fn(params_tree):
        return jax.tree_util.tree_map_with_path(
            lambda path, _: _is_lora_path(path), params_tree
        )

    tx = optax.masked(base_tx, mask=_mask_fn)
    # Init the optimizer state inside mesh + axis_rules so the Adam
    # moments inherit the LoRA params' sharding (LoRA params are tiny
    # so this is cheap, but doing it under context keeps the jit'd
    # train_step's sharding inference consistent).
    with mesh, nn_partitioning.axis_rules(mt_cfg.logical_axis_rules):
        opt_state = tx.init(params)

    # --- jit'd train_step + eval_step ----------------------------------------
    def _ce_loss(logits, labels, loss_mask):
        """Per-token shifted cross-entropy with -100/loss_mask masking."""
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        # Standard next-token shift: predict labels[:, 1:] from logits[:, :-1, :].
        # Our converter wrote labels parallel to input_ids (-100 on prompt
        # tokens, real id on response). The shift therefore lines logits at
        # position t up against labels at position t+1, which is the
        # next-token prediction contract.
        shift_logits = log_probs[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_mask = loss_mask[:, 1:]
        # Replace -100 with 0 to avoid OOB index — these positions are
        # zero-weighted by shift_mask anyway.
        safe_labels = jnp.where(shift_labels < 0, 0, shift_labels)
        per_tok = -jnp.take_along_axis(
            shift_logits, safe_labels[..., None], axis=-1
        ).squeeze(-1)
        masked = per_tok * shift_mask
        return masked.sum() / (shift_mask.sum() + 1e-8)

    def _forward(params_tree, inputs, dropout_rng, params_rng, is_train):
        # Match MaxText pre_train.loss_fn (third_party/maxtext/.../pre_train/train.py
        # lines 136-148) — model.apply takes positional ``inputs`` and
        # ``inputs_position`` followed by a fan of kwargs. The model
        # returns ``(logits, intermediate_outputs)``; we discard the
        # second element since our loss is computed externally on the
        # logits + our explicit loss_mask.
        bsz, seqlen = inputs.shape
        positions = jnp.broadcast_to(
            jnp.arange(seqlen, dtype=jnp.int32), (bsz, seqlen)
        )
        segmentation = jnp.ones((bsz, seqlen), dtype=jnp.int32)
        logits, _intermediates = model.apply(
            params_tree,
            inputs,
            positions,
            decoder_segment_ids=segmentation,
            encoder_images=None,
            encoder_image_masks=None,
            enable_dropout=is_train,
            rngs={"dropout": dropout_rng, "params": params_rng},
            mutable=["intermediates"],
            decoder_target_tokens=inputs,  # not used for loss; placeholder for in-model paths
            decoder_target_mask=segmentation,
        )
        return logits

    def _train_step(params_tree, opt_state_tree, batch, rng):
        # Three RNGs: dropout, params (used by MaxText's AQT quantization
        # noise sampling), and the next-step seed.
        rng, dropout_rng, params_rng = jax.random.split(rng, 3)

        def loss_fn(p):
            logits = _forward(p, batch["input_ids"], dropout_rng, params_rng, True)
            return _ce_loss(logits, batch["labels"], batch["loss_mask"])

        loss, grads = jax.value_and_grad(loss_fn)(params_tree)
        updates, new_opt_state = tx.update(grads, opt_state_tree, params_tree)
        new_params = optax.apply_updates(params_tree, updates)
        return new_params, new_opt_state, loss, rng

    def _eval_step(params_tree, batch, rng):
        rng, dropout_rng, params_rng = jax.random.split(rng, 3)
        logits = _forward(params_tree, batch["input_ids"], dropout_rng, params_rng, False)
        return _ce_loss(logits, batch["labels"], batch["loss_mask"]), rng

    # JIT under mesh + axis_rules so the train_step's input/output
    # shardings propagate from the model's PartitionSpec annotations.
    # Without this context the optimizer state may be replicated
    # across all chips (LoRA-sized, so OK) but the model params would
    # also be replicated (27 B × 8 chips = OOM). Keeping the context
    # active for both jit and the loop body ensures consistency.
    with mesh, nn_partitioning.axis_rules(mt_cfg.logical_axis_rules):
        train_step = jax.jit(_train_step)
        eval_step = jax.jit(_eval_step)

    # --- Training loop -------------------------------------------------------
    print(f"--- training: {num_epochs} epoch(s), {total_steps} total step(s) ---",
          flush=True)
    rng = jax.random.PRNGKey(seed + 1000)
    t_start = time.monotonic()
    first_step_logged = False
    global_step = 0
    logging_steps = int(train_cfg.get("logging_steps", 5))

    # Run the loop inside the same mesh + axis_rules context as the jit
    # so cross-device collectives use the correct partition spec.
    with mesh, nn_partitioning.axis_rules(mt_cfg.logical_axis_rules):
        for epoch in range(num_epochs):
            for _ in range(steps_per_epoch):
                np_batch = next(train_iter)
                batch = {k: jnp.asarray(v) for k, v in np_batch.items()}
                params, opt_state, loss, rng = train_step(params, opt_state, batch, rng)
                global_step += 1

                if not first_step_logged:
                    first_step_logged = True
                    elapsed_min = (time.monotonic() - t_start) / 60.0
                    # The launcher's compile-done detector greps for this
                    # exact phrase to know XLA compile finished.
                    print(f"first step landed in {elapsed_min:.1f} min", flush=True)

                if global_step % logging_steps == 0 or global_step == total_steps:
                    jax.block_until_ready(loss)
                    print(f"step {global_step}/{total_steps} loss={float(loss):.4f}",
                          flush=True)

            # End of epoch: eval + per-epoch orbax save
            eval_iter_for_epoch = dataset_loader.build_iterators(
                dataset_dir=dataset_dir,
                train_file=train_file,
                val_file=val_file,
                per_device_batch_size=train_cfg["per_device_train_batch_size"],
                gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
                max_seq_length=train_cfg["max_seq_length"],
                seed=seed,
            )[1]  # eval_iter
            eval_losses = []
            for np_batch in eval_iter_for_epoch:
                batch = {k: jnp.asarray(v) for k, v in np_batch.items()}
                loss, rng = eval_step(params, batch, rng)
                eval_losses.append(float(loss))
            if eval_losses:
                mean_eval = sum(eval_losses) / len(eval_losses)
                print(f"epoch {epoch + 1}/{num_epochs} eval_loss={mean_eval:.4f}",
                      flush=True)

            # Save the full params tree (base + LoRA). The downstream
            # exporter (export_maxtext_lora_to_peft.export) walks the
            # restored pytree and only keeps leaves matching ``lora_a`` /
            # ``lora_b`` patterns, so saving the full tree is correct.
            # We tried saving only LoRA leaves (replacing base with None)
            # but PyTreeCheckpointer doesn't accept None leaves; falling
            # back to full save costs ~50 GB per epoch on disk but matches
            # MaxText's own checkpoint pattern and avoids a custom
            # save/restore protocol.
            epoch_dir = orbax_out / f"step-{global_step}"
            ocp.PyTreeCheckpointer().save(str(epoch_dir.resolve()), params)
            print(f"saved orbax checkpoint to {epoch_dir}", flush=True)

    # --- Final export to PEFT ------------------------------------------------
    final_ckpt = orbax_out / f"step-{global_step}"
    export_peft.write_adapter(
        orbax_checkpoint=final_ckpt,
        output_dir=best_out,
        base_model_name=model_cfg["name"],
        target_modules=lora_cfg["target_modules"],
        rank=lora_cfg["r"],
        alpha=lora_cfg["lora_alpha"],
        dropout=lora_cfg.get("lora_dropout", 0.05),
        variant=lora_cfg.get("variant", "standard"),
    )

    # Stage 4's _xla_lora_inference does AutoTokenizer.from_pretrained on
    # this directory; without the tokenizer files alongside the adapter
    # it crashes. Mirrors torch_xla's trainer (scripts/train_lora.py).
    # Drop this in BEFORE trainer_state.json is written so operators that
    # treat trainer_state.json as the "done" sentinel only see a complete
    # adapter dir.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    tokenizer.save_pretrained(str(best_out))

    # Mirror torch_xla's trainer_state.json so Stage 4 can confirm the
    # run finished without parsing orbax internals.
    summary = {
        "seed": seed,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "final_orbax_checkpoint": str(final_ckpt),
        "model_name": model_cfg["name"],
        "lora": {
            "r": lora_cfg["r"],
            "alpha": lora_cfg["lora_alpha"],
            "dropout": lora_cfg.get("lora_dropout", 0.05),
            "target_modules": list(lora_cfg["target_modules"]),
            "variant": lora_cfg.get("variant", "standard"),
        },
    }
    (Path(best_out) / "trainer_state.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"saved adapter to {best_out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA SFT of MedGemma-27B via MaxText (Stage 3)."
    )
    parser.add_argument(
        "--config", required=True,
        help="path to a MaxText LoRA YAML config "
             "(see configs/lora_medgemma27b_maxtext.yaml).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="override the seed in the YAML config.",
    )
    parser.add_argument(
        "--output-dir", default="checkpoints",
        help="<output-dir>/orbax/  -> periodic LoRA checkpoints; "
             "<output-dir>/best/   -> final HF/PEFT adapter for Stage 4 eval.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    seed = args.seed if args.seed is not None else int(
        cfg.get("seed", cfg.get("training", {}).get("seed", 42))
    )
    _set_global_seeds(seed)

    _train(cfg, seed=seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
