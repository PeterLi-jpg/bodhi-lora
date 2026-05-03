"""CPU smoke test for the tunix Stage 3 LoRA SFT path.

This is Unit 7 of the bohdi-lora MaxText to tunix+qwix migration. The
goal is to validate end-to-end wiring (model build, qwix LoRA apply,
TrainingInput dataset, PeftTrainer forward+backward+optimizer+
checkpoint) on a Mac CPU dev box, *without* a TPU and without real
gemma3 weights. Everything is shrunk: a 2-layer toy gemma3 with
embed_dim=256, random init, a 4-row in-memory dataset, max_steps=1.

What this test does NOT cover:
  - Real gemma3 safetensors loading (out of scope; covered by U1/U2).
  - TPU mesh + sharding semantics (covered by U8).
  - The dataset_loader.build_iterators tunix adapter (U4); we inline
    a minimal TrainingInput wrapper here so this file doesn't depend
    on U4 being merged first.

Skip behavior: if any of tunix / qwix / jax / flax / optax aren't
installed, the whole module skips with a clear reason. That keeps the
rest of the test suite green on dev boxes that haven't installed the
full Stage-3 stack yet.
"""

from __future__ import annotations

# IMPORTANT: set the XLA host-device count BEFORE jax is imported
# anywhere in the process. We force 8 simulated CPU devices so the mesh
# (fsdp=2, tp=4) lines up with what tunix's default ShardingConfig
# expects.
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
# Force CPU-only. Mac dev boxes can't initialize a TPU/GPU runtime and
# we want the test deterministic.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import pytest


_REQUIRED = ("tunix", "qwix", "jax", "flax", "optax", "numpy")


def _try_import_stack():
    """Actually import the stack and return the module bag, or return
    None + the offending error.

    `importlib.util.find_spec` is not enough: tunix imports kagglehub
    which imports `tqdm.contrib.concurrent`, and if any transitive
    dep is partially installed the spec lookup succeeds but the import
    blows up. We catch any ImportError or ModuleNotFoundError here so
    the rest of the suite still runs cleanly on a dev box that doesn't
    have the full preflight stack.
    """
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
        import numpy as np  # noqa: F401
        import optax  # noqa: F401
        import qwix  # noqa: F401
        from flax import nnx  # noqa: F401
        from tunix.models.gemma3 import model as gemma3_model  # noqa: F401
        from tunix.sft import peft_trainer  # noqa: F401
        from tunix.sft import utils as tunix_utils  # noqa: F401
    except ImportError as e:
        return None, e
    return {
        "jax": jax,
        "jnp": jnp,
        "np": np,
        "optax": optax,
        "qwix": qwix,
        "nnx": nnx,
        "gemma3_model": gemma3_model,
        "peft_trainer": peft_trainer,
        "tunix_utils": tunix_utils,
    }, None


_stack, _import_error = _try_import_stack()

pytestmark = pytest.mark.skipif(
    _stack is None,
    reason=(
        "tunix/qwix/jax/flax/optax not importable; the Stage-3 smoke runs "
        "only on the preflight-py311 venv that has the full Google JAX/tunix "
        f"stack. Required: {', '.join(_REQUIRED)}. Import error: "
        f"{_import_error!r}"
    ),
)


# Bind the imported modules at module scope only when present, so test
# functions can reference them by name without re-importing. On a dev
# box without the stack, the names below are never evaluated because
# pytestmark skips the whole module.
if _stack is not None:
    jax = _stack["jax"]
    jnp = _stack["jnp"]
    np = _stack["np"]
    optax = _stack["optax"]
    qwix = _stack["qwix"]
    nnx = _stack["nnx"]
    gemma3_model = _stack["gemma3_model"]
    peft_trainer = _stack["peft_trainer"]
    tunix_utils = _stack["tunix_utils"]


# ----------------------------------------------------------------------
# Tiny gemma3 config + helpers
# ----------------------------------------------------------------------


# Constants picked so the model is small enough to forward+backward on
# a Mac CPU in a few seconds. The shrink ratios from a real gemma3-1B:
# embed_dim 1152 -> 256, num_layers 26 -> 2, num_heads 4 -> 4 (kept),
# num_kv_heads 1 -> 2 (bumped so num_heads != num_kv_heads, which puts
# us on the q_einsum/kv_einsum branch, the same path qwix's LoraRule
# will match). vocab is tiny.
SMOKE_VOCAB = 128
SMOKE_SEQ_LEN = 8
SMOKE_BATCH = 2
SMOKE_EMBED_DIM = 256
SMOKE_HIDDEN_DIM = 256
SMOKE_NUM_LAYERS = 2
SMOKE_NUM_HEADS = 4
SMOKE_NUM_KV_HEADS = 2
SMOKE_HEAD_DIM = 32
LORA_RANK = 2
LORA_ALPHA = 4

# qwix module-path regex. qwix builds the module path by joining the
# nnx module attribute names with '/', e.g.  "layers/0/attn/q_einsum".
# We target the q_einsum and kv_einsum attention modules (the
# attention path the gemma3 model takes when num_heads != num_kv_heads,
# i.e. grouped-query attention, the production gemma3 case). qwix
# intercepts the einsum dot_general inside those modules and inserts
# the LoRA delta there, so the rule matches the module itself, not a
# specific param leaf.
LORA_MODULE_PATH = r".*/attn/(q_einsum|kv_einsum)"


def _build_unsharded_config() -> "gemma3_model.ShardingConfig":
    """All-None ShardingConfig.

    The default ShardingConfig.get_default_sharding() shards weights
    across `tp` and `fsdp` axes that don't divide our toy dims (e.g.
    kv_einsum's num_kv_heads=2 against tp=4). For a single-host CPU
    smoke we don't need real sharding, we just need the model to
    construct, forward, and backprop. Setting every axis to None makes
    every weight fully replicated.
    """
    none1 = (None,)
    none2 = (None, None)
    none3 = (None, None, None)
    none4 = (None, None, None, None)
    return gemma3_model.ShardingConfig(
        emb_vd=none2,
        q_weight_ndh=none3,
        kv_weight_cndh=none4,
        qkv_weight_cndh=none4,
        o_weight_nhd=none3,
        ffw_weight_df=none2,
        ffw_weight_fd=none2,
        rms_norm_weight=none1,
        act_btd=none3,
        act_btf=none3,
        act_btnh=none4,
        # vision_* + siglip are unused when vision_config=None on
        # ModelConfig. Set them to safe defaults anyway.
        vision_proj=none2,
        vision_soft_emb_norm_weight=none1,
        siglip=None,
    )


def _build_smoke_config() -> "gemma3_model.ModelConfig":
    """Build a shrunk gemma3 ModelConfig that fits on Mac CPU."""
    return gemma3_model.ModelConfig(
        num_layers=SMOKE_NUM_LAYERS,
        num_embed=SMOKE_VOCAB,
        embed_dim=SMOKE_EMBED_DIM,
        hidden_dim=SMOKE_HIDDEN_DIM,
        num_heads=SMOKE_NUM_HEADS,
        head_dim=SMOKE_HEAD_DIM,
        num_kv_heads=SMOKE_NUM_KV_HEADS,
        # No sliding window: single global-attention layer pattern keeps
        # the wiring simple and avoids the rope/sliding cache codepath
        # that's irrelevant for the smoke.
        sliding_window_size=SMOKE_SEQ_LEN,
        local_base_frequency=10_000,
        global_base_frequency=10_000,
        # Replace the default fsdp/tp sharding (those axes don't
        # divide our shrunk weight shapes).
        shd_config=_build_unsharded_config(),
        # Use float32 on CPU. bfloat16 (the gemma3 default) is
        # supported on CPU but slower and noisier for finite-checks.
        param_dtype=jnp.float32,
    )


def _dummy_model_inputs():
    """Inputs of the shape Transformer.__call__ expects, used by qwix
    to trace the model when applying LoRA."""
    tokens = jnp.zeros((SMOKE_BATCH, SMOKE_SEQ_LEN), dtype=jnp.int32)
    positions = jnp.broadcast_to(
        jnp.arange(SMOKE_SEQ_LEN, dtype=jnp.int32),
        (SMOKE_BATCH, SMOKE_SEQ_LEN),
    )
    # Causal mask: shape (B, L, L), bool, lower-triangular.
    causal = jnp.tril(jnp.ones((SMOKE_SEQ_LEN, SMOKE_SEQ_LEN), dtype=jnp.bool_))
    attention_mask = jnp.broadcast_to(
        causal, (SMOKE_BATCH, SMOKE_SEQ_LEN, SMOKE_SEQ_LEN)
    )
    return {
        "last_tokens": tokens,
        "positions": positions,
        "cache": None,
        "attention_mask": attention_mask,
    }


def _build_lora_model() -> "nnx.Module":
    """Random-init the toy gemma3, then apply qwix LoRA."""
    cfg = _build_smoke_config()
    # The model class lives at tunix.models.gemma3.model.Gemma3 (the
    # task spec called it `Transformer`; same module, just a rename).
    model = gemma3_model.Gemma3(cfg, rngs=nnx.Rngs(0))
    provider = qwix.LoraProvider(
        [
            qwix.LoraRule(
                module_path=LORA_MODULE_PATH,
                rank=LORA_RANK,
                alpha=LORA_ALPHA,
            )
        ]
    )
    # apply_lora_to_model traces the model with these inputs, so the
    # shapes must match the real call graph the trainer will hit.
    return qwix.apply_lora_to_model(
        model, provider, **_dummy_model_inputs()
    )


def _build_dataset(n_rows: int = 4):
    """Build a list of TrainingInput rows with random tokens.

    Returns a plain list so it can be passed as both train_ds and
    eval_ds (the trainer calls iter() fresh each pass, so a list is
    re-iterable without us caching anything).
    """
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n_rows):
        tokens = rng.integers(
            low=1, high=SMOKE_VOCAB, size=(SMOKE_BATCH, SMOKE_SEQ_LEN), dtype=np.int32
        )
        # input_mask: every position is "real" (no padding). Keeps the
        # smoke's loss arithmetic non-zero.
        mask = np.ones((SMOKE_BATCH, SMOKE_SEQ_LEN), dtype=np.int32)
        rows.append(
            peft_trainer.TrainingInput(input_tokens=tokens, input_mask=mask)
        )
    return rows


def _gen_model_input_fn(x):
    """Mirror tunix.cli.peft_main's gen_model_input_fn: derive
    positions and a causal attention mask from the input mask."""
    pad_mask = x.input_tokens != 0
    positions = tunix_utils.build_positions_from_mask(pad_mask)
    attention_mask = tunix_utils.make_causal_attn_mask(pad_mask)
    return {
        "input_tokens": x.input_tokens,
        "input_mask": x.input_mask,
        "positions": positions,
        "attention_mask": attention_mask,
    }


def _build_mesh() -> "jax.sharding.Mesh":
    """8-device CPU mesh matching tunix's default ShardingConfig
    ('fsdp', 'tp')."""
    devices = np.asarray(jax.devices()).reshape(2, 4)
    return jax.sharding.Mesh(devices, ("fsdp", "tp"))


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_simulated_devices_available():
    """XLA_FLAGS=--xla_force_host_platform_device_count=8 must take
    effect; without it the mesh build below would fail with a wrong
    device count and the rest of the suite gives a confusing error."""
    assert len(jax.devices()) == 8, (
        f"expected 8 simulated CPU devices, got {len(jax.devices())}. "
        "XLA_FLAGS must be set before jax is first imported anywhere "
        "in the process."
    )


def test_apply_qwix_lora_yields_lora_params():
    """After qwix.apply_lora_to_model, the model's nnx state contains
    LoRAParam entries with the expected lora_a / lora_b leaves at
    every q_einsum and kv_einsum the rule matched."""
    model = _build_lora_model()

    # Confirm the trainer's helper sees lora. It's how the optimizer
    # decides to update only LoRAParam vs every Param.
    assert tunix_utils.is_lora_enabled(model), (
        "qwix.apply_lora_to_model returned a model with no LoRAParam "
        "registered; tunix's PeftTrainer would silently train the full "
        "model instead of just the LoRA adapters."
    )

    # Walk the nnx state and assert lora_a + lora_b appear under at
    # least one .attn.q_einsum and one .attn.kv_einsum path.
    lora_state = nnx.state(model, nnx.LoRAParam)
    flat = jax.tree_util.tree_flatten_with_path(lora_state)[0]
    paths = [jax.tree_util.keystr(p) for p, _ in flat]

    has_q_lora_a = any(
        "attn" in p and "q_einsum" in p and "lora_a" in p for p in paths
    )
    has_q_lora_b = any(
        "attn" in p and "q_einsum" in p and "lora_b" in p for p in paths
    )
    has_kv_lora_a = any(
        "attn" in p and "kv_einsum" in p and "lora_a" in p for p in paths
    )
    has_kv_lora_b = any(
        "attn" in p and "kv_einsum" in p and "lora_b" in p for p in paths
    )

    assert has_q_lora_a and has_q_lora_b, (
        "expected lora_a/lora_b leaves under attn.q_einsum, got paths:\n"
        + "\n".join(paths)
    )
    assert has_kv_lora_a and has_kv_lora_b, (
        "expected lora_a/lora_b leaves under attn.kv_einsum, got paths:\n"
        + "\n".join(paths)
    )


def test_dataset_loader_tunix_format_yields_TrainingInput():
    """The dataset rows we hand to the trainer must be tunix
    TrainingInput instances with the expected (B, T) input_tokens /
    input_mask shapes. This stands in for U4's adapter until U4 lands.
    """
    rows = _build_dataset(n_rows=4)
    assert len(rows) == 4
    for row in rows:
        assert isinstance(row, peft_trainer.TrainingInput)
        assert row.input_tokens.shape == (SMOKE_BATCH, SMOKE_SEQ_LEN)
        assert row.input_mask.shape == (SMOKE_BATCH, SMOKE_SEQ_LEN)
        # Sanity: tokens are within vocab. Out-of-range tokens would
        # nan the loss and silently pass the "finite loss" check below.
        assert int(row.input_tokens.max()) < SMOKE_VOCAB
        assert int(row.input_tokens.min()) >= 0


def test_peft_trainer_runs_one_step_on_cpu():
    """The full path: build model + LoRA, build optimizer, build
    PeftTrainer, run one train step under a CPU mesh. Asserts no
    exception, train_steps advances to 1, and the LoRA params after
    training are finite."""
    model = _build_lora_model()
    optimizer = optax.adamw(learning_rate=1e-3)
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=1,
        max_steps=1,
    )
    trainer = peft_trainer.PeftTrainer(model, optimizer, config)
    trainer = trainer.with_gen_model_input_fn(_gen_model_input_fn)

    train_ds = _build_dataset(n_rows=4)
    eval_ds = _build_dataset(n_rows=4)

    mesh = _build_mesh()
    with mesh:
        trainer.train(train_ds, eval_ds)

    assert trainer.train_steps == 1, (
        f"trainer should have completed exactly 1 step, got {trainer.train_steps}"
    )

    # After training, the LoRA params must be finite. NaN/Inf here
    # would silently pass an "exception-free" check but mean the
    # forward+backward arithmetic has a real bug.
    lora_state = nnx.state(trainer.model, nnx.LoRAParam)
    leaves = jax.tree_util.tree_leaves(lora_state)
    assert leaves, "no LoRA leaves after training; LoRA was not applied"
    for leaf in leaves:
        arr = np.asarray(leaf.value if hasattr(leaf, "value") else leaf)
        assert np.all(np.isfinite(arr)), (
            f"non-finite LoRA param after one step: shape={arr.shape}"
        )
        # Shape sanity: all LoRA tensors must have rank>=1 and one of
        # the dims must be the LoRA rank we configured.
        assert arr.ndim >= 1
        assert LORA_RANK in arr.shape, (
            f"LoRA leaf shape {arr.shape} does not contain rank={LORA_RANK}; "
            "qwix may have built the adapter against the wrong axis."
        )


def test_peft_trainer_saves_checkpoint(tmp_path: Path):
    """When checkpoint_root_directory is set, train() must produce a
    non-empty checkpoint directory by the time it returns. This
    catches the orbax wiring breaking silently, a regression we'd
    only spot at the end of a multi-hour TPU run otherwise."""
    ckpt_dir = tmp_path / "ckpt"
    # Orbax requires an absolute path.
    ckpt_dir = ckpt_dir.resolve()

    model = _build_lora_model()
    optimizer = optax.adamw(learning_rate=1e-3)
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=1,
        max_steps=1,
        checkpoint_root_directory=str(ckpt_dir),
    )
    trainer = peft_trainer.PeftTrainer(model, optimizer, config)
    trainer = trainer.with_gen_model_input_fn(_gen_model_input_fn)

    train_ds = _build_dataset(n_rows=4)
    eval_ds = _build_dataset(n_rows=4)

    mesh = _build_mesh()
    with mesh:
        trainer.train(train_ds, eval_ds)

    # close() runs at the end of train() and force-saves the last
    # checkpoint via _save_last_checkpoint(force=True). After that the
    # root directory should exist with at least one step subdir.
    assert ckpt_dir.exists(), f"checkpoint root {ckpt_dir} was not created"
    entries = list(ckpt_dir.iterdir())
    assert entries, (
        f"checkpoint root {ckpt_dir} is empty; orbax CheckpointManager "
        "did not flush a step. The default policy is 180s minimum "
        "interval; force-save in close() should bypass that."
    )
