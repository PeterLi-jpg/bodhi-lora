"""LoRA SFT on filtered BOHDI traces.

The training knobs live in a YAML config (see ``configs/*.yaml``). The two
families worth noting:

- ``model.quantization``: ``null`` (default, full-precision LoRA) or ``"4bit"``
  for QLoRA. 4-bit mode loads the base model via bitsandbytes NF4 with
  bf16 compute dtype, then wires PEFT's ``prepare_model_for_kbit_training``.
  8-bit mode (``"8bit"``) is also supported for large-batch speed runs.
- ``lora.variant``: ``"standard"`` (default), ``"dora"``, or ``"rslora"``. DoRA
  requires full-precision weights and will error out if combined with
  quantization.

Anything unset keeps pre-existing behavior so current configs still work.
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model

# Detect whether we're running under PyTorch/XLA (Google Cloud TPU).
# When True:  device_map="auto" must NOT be used — accelerate owns placement.
# When False: device_map="auto" is used as before (multi-GPU or single GPU).
try:
    import torch_xla  # noqa: F401
    import torch_xla.core.xla_model as _xm
    _ON_TPU = True
except ImportError:
    _xm = None
    _ON_TPU = False

def _needs_spmd(model_name: str) -> bool:
    """Return True only for models too large to fit on one v6e chip (32 GB).

    Models ≤8B in bf16 need ~16 GB — well under the per-chip limit — so SPMD
    column-parallel sharding is unnecessary and triggers an XLA fusion-emitter
    crash on Gemma-3 (shape_indices RET_CHECK in fusion_emitter.cc).  The 27B
    model (54 GB bf16) does need sharding, so it still gets SPMD.

    Uses the same regex as _auto_tp() in _vllm_engine.py to stay consistent.
    """
    return not bool(re.search(r"(?<!\d)[1-8]b(?!\w)", model_name.lower()))


# XLA persistent compile cache — first run on a fresh VM compiles the
# 27B-with-LoRA training graph (~40-60 min), subsequent runs (each seed,
# resumes after preemption) load it from disk and skip compile entirely.
# Cache lives on the boot disk under ~/.xla_cache.  Safe to set before
# use_spmd() — initialize_cache only configures a path, doesn't intercept.
if _ON_TPU:
    try:
        import os as _os_cache
        _xla_cache = _os_cache.path.expanduser("~/.xla_cache")
        _os_cache.makedirs(_xla_cache, exist_ok=True)
        from torch_xla import runtime as _xr_cache
        if hasattr(_xr_cache, "initialize_cache"):
            _xr_cache.initialize_cache(_xla_cache, readonly=False)
            print(f"XLA persistent compile cache: {_xla_cache}")
    except Exception as _e:
        print(f"XLA persistent cache unavailable ({_e!r}); compiles will not be saved.")

# Why this matters:
#   v6e-8 has 8 chips × 32 GB = 256 GB HBM.  MedGemma-27B in bfloat16 is 54 GB.
#   accelerate's default TPU path (distributed_type=TPU + num_processes=8) gives
#   each chip a FULL replica of the model (xmp.MpModelWrapper(model).to(device)
#   in accelerator.py) — 54 GB doesn't fit on a 32 GB chip, so training OOMs at
#   model load.
#
# Fix: single-process SPMD.  ONE Python process drives all 8 chips, and we shard
# the model parameters across them with torch_xla.distributed.spmd.mark_sharding
# (same pattern Stage 1 uses in scripts/generate_traces.py).  This matches our
# accelerate config tpu/accelerate_config_v6e8.yaml which sets num_processes: 1.
#
# IMPORTANT — DO NOT call xr.use_spmd() at module level.  use_spmd() globally
# intercepts every set_data() call; if it is active when from_pretrained()
# runs, every internal weight assignment raises "incompatible tensor type"
# (this is documented in scripts/generate_traces.py / eval_healthbench.py).
# The correct order is:
#   1) from_pretrained on CPU
#   2) xr.use_spmd()
#   3) move params to XLA + mark_sharding
# All of which now happen inside main() in that order.

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# Known LoRA variants we support via PEFT's LoraConfig flags.
LORA_VARIANTS = ("standard", "dora", "rslora")

_tokenizer = None


def load_sft_jsonl(path):
    """Load graded SFT JSONL, keeping only what SFTTrainer needs.

    The graded files also include a ``grade`` field with per-example variable
    rubric keys (tag_scores / criteria_results differ per prompt). HF datasets'
    schema inference picks up the first file's keys and fails casting the second
    when rubric keys differ. Stripping to the fields we actually use avoids that.
    """
    rows = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            rows.append({"messages": obj["messages"], "response": obj["response"]})
    return Dataset.from_list(rows)


def format_example(batch):
    """Format messages+response into a single training string.

    TRL probes ``formatting_func`` on a single example first to determine
    whether it returns str or list[str], then calls it in either mode.
    On a single example, ``batch["response"]`` is a str (not list[str])
    and ``batch["messages"]`` is a list of dicts (one conversation, not
    list of conversations). We must handle both shapes or zip will
    silently iterate characters of the response string -> malformed
    training text. See issue #2.
    """
    def _render(msgs, resp):
        msgs = list(msgs)
        msgs.append({"role": "assistant", "content": resp})
        return _tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )

    if isinstance(batch["response"], str):
        # single-example shape
        return _render(batch["messages"], batch["response"])
    # batched shape: dict of lists
    return [_render(msgs, resp)
            for msgs, resp in zip(batch["messages"], batch["response"])]


def find_response_template(tokenizer):
    """Detect the assistant turn header by comparing templates with/without generation prompt.

    The difference between add_generation_prompt=True and False is exactly
    the assistant turn header (e.g. "<start_of_turn>model\\n" for Gemma,
    "<|start_header_id|>assistant<|end_header_id|>\\n\\n" for Llama 3).
    """
    dummy = [{"role": "user", "content": "hi"}]
    without_gen = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=False)
    with_gen = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=True)

    if with_gen.startswith(without_gen):
        template = with_gen[len(without_gen):]
        if template.strip():
            return template

    # Print full before/after templates so the fix — usually a small pattern
    # extension in this function — is obvious instead of requiring a debug rerun.
    raise ValueError(
        "Could not auto-detect response template. The tokenizer's chat template\n"
        "does not append a clean assistant-turn header to add_generation_prompt=True.\n"
        f"without_gen = {without_gen!r}\n"
        f"with_gen    = {with_gen!r}\n"
        f"diff suffix = {with_gen[-50:]!r}\n"
        "Extend find_response_template() to handle this template family."
    )


def main():
    global _tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="override the seed in the YAML config (useful for "
                             "multi-seed runs where one YAML is reused with "
                             "different seeds per invocation)")
    parser.add_argument("--output-dir", default="checkpoints",
                        help="directory to save checkpoints + best model. "
                             "The final adapter is always written to "
                             "<output-dir>/best. Override per seed in "
                             "multi-seed runs, e.g. checkpoints/seed_42")
    parser.add_argument("--train-file", default=None,
                        help="override data.train_file from the YAML config")
    parser.add_argument("--val-file", default=None,
                        help="override data.val_file from the YAML config")
    parser.add_argument("--quantization", default=None,
                        choices=["4bit", "8bit", "null"],
                        help="override model.quantization from the YAML config. "
                             "'null' means full-precision (same as leaving it unset). "
                             "4bit = QLoRA (NF4 + bf16 compute, GPU only). "
                             "8bit = bitsandbytes 8-bit (GPU only).")
    parser.add_argument("--lora-variant", default=None,
                        choices=list(LORA_VARIANTS),
                        help="override lora.variant from the YAML config. "
                             "standard = classic LoRA (default). "
                             "dora = direction+magnitude decomposition (full-precision only). "
                             "rslora = alpha/sqrt(r) scaling, better at higher ranks.")
    parser.add_argument("--lora-r", type=int, default=None,
                        help="override lora.r (rank) from the YAML config.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides take precedence over YAML values.
    if args.quantization is not None:
        cfg["model"]["quantization"] = None if args.quantization == "null" else args.quantization
    if args.lora_variant is not None:
        cfg["lora"]["variant"] = args.lora_variant
    if args.lora_r is not None:
        cfg["lora"]["r"] = args.lora_r

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    # CLI > YAML precedence for seed, so one YAML can be reused across seeds.
    seed = args.seed if args.seed is not None else int(
        cfg.get("seed", train_cfg.get("seed", 42))
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    _tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    # TRL half-precision training overflows when padding_side != "right"; the
    # SFTTrainer warns about this. Set it explicitly so we don't depend on the
    # tokenizer's upstream default (which flips between model families).
    _tokenizer.padding_side = "right"
    # Gemma-family footgun: the chat template emits {{ bos_token }} at its
    # start AND tokenizer_config has add_bos_token=True.  When SFTTrainer
    # tokenizes the rendered chat template, the tokenizer would prepend a
    # SECOND <bos>, giving every example "<bos><bos>...".  The model never
    # saw that distribution during pretraining → measurably worse training.
    # The chat template already handles BOS, so disable auto-prepend here.
    _tokenizer.add_bos_token = False

    # HF's get_constant_schedule ignores warmup_ratio silently — warmup only
    # takes effect on scheduler types that support it (constant_with_warmup,
    # linear, cosine, polynomial, etc.). Fail loudly instead of letting the
    # "I set warmup, why is the LR still full on step 1?" bug happen on cluster.
    if train_cfg.get("warmup_ratio", 0) > 0 and train_cfg.get("lr_scheduler_type") == "constant":
        raise ValueError(
            "lr_scheduler_type='constant' does not apply warmup_ratio. "
            "Use 'constant_with_warmup', 'linear', 'cosine', or 'polynomial' "
            "if you want warmup, or set warmup_ratio: 0.0 to be explicit."
        )

    dtype_str = model_cfg.get("torch_dtype") or "bfloat16"
    dtype = DTYPE_MAP.get(dtype_str, torch.bfloat16)

    # -------- Optional quantization (QLoRA) -----------------------------------
    # model.quantization in YAML is null (default, full-precision) or one of
    # "4bit" / "8bit". The 4bit path matches the original QLoRA paper: NF4
    # weights with bf16 compute dtype, double-quantized.
    quant = model_cfg.get("quantization")
    # bitsandbytes is CUDA-only; it will hard-fail on TPU at import time.
    if quant in ("4bit", "8bit") and _ON_TPU:
        raise ValueError(
            f"model.quantization={quant!r} uses bitsandbytes which is CUDA-only "
            "and cannot run on TPU. Set model.quantization: null in your config "
            "(not needed on TPU — you have plenty of HBM)."
        )
    quant_config = None
    if quant in ("4bit", "8bit"):
        # Import lazily so CPU-only / Mac dev boxes without bitsandbytes still
        # import this file fine when running in full-precision mode.
        from transformers import BitsAndBytesConfig
        if quant == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:  # 8bit
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        print(f"Loading {model_cfg['name']} ({quant} quantized, compute {dtype})...")
    elif quant is not None:
        raise ValueError(
            f"model.quantization={quant!r} is not recognised. "
            f"Use null (default), '4bit', or '8bit'."
        )
    else:
        print(f"Loading {model_cfg['name']} ({dtype})...")

    # On TPU we keep device_map=None — accelerate's TPU path doesn't shard
    # the model on its own (only does xmp.MpModelWrapper.to(device) per
    # process).  We do the sharding manually below via SPMD.
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=dtype,
        device_map=None if _ON_TPU else "auto",
        quantization_config=quant_config,
    )

    # Quantized weights need gradient checkpointing + input-grad rewiring before
    # LoRA adapters are attached. PEFT's helper handles both.
    if quant_config is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    # ── SPMD setup + optional sharding ──────────────────────────────────────
    # use_spmd() MUST be called on v6e-8 regardless of model size — it
    # initialises the multi-chip XLA runtime.  Skipping it causes the main
    # Python thread to deadlock on a futex while waiting for chip-0's
    # communication buffers that never get set up.
    #
    # mark_sharding is only applied for large models (>8B) where the weights
    # genuinely don't fit on one chip.  For ≤8B we call use_spmd() for the
    # runtime init but skip sharding: all parameters are replicated across
    # the 8-chip virtual device.  Applying mark_sharding to Gemma-3-4B
    # triggers an XLA fusion-emitter RET_CHECK during the backward pass
    # (shape_indices.size() == 1 with 5 indices, fusion_emitter.cc:9554).
    if _ON_TPU:
        # Try the SPMD module names across torch_xla versions; experimental
        # was renamed to distributed.spmd in 2.5.
        _xs = None
        for _spmd_mod in ("torch_xla.distributed.spmd",
                          "torch_xla.experimental.xla_sharding",
                          "torch_xla.distributed.xla_sharding"):
            try:
                import importlib as _il
                _xs = _il.import_module(_spmd_mod)
                print(f"SPMD module: {_spmd_mod}")
                break
            except ModuleNotFoundError:
                continue

        if _xs is not None:
            from torch_xla import runtime as _xr
            import torch.nn as _nn
            # NOW that from_pretrained is done (model is on CPU), it's safe to
            # enable SPMD.  Doing this earlier breaks from_pretrained because
            # use_spmd() globally intercepts set_data() and the loader does
            # many such assignments while building the model.
            _xr.use_spmd()
            # On v6e-8: addressable_device_count() == 8 (physical chips).
            # global_device_count() returns 1 in SPMD mode (single virtual dev).
            _n_dev = getattr(_xr, "addressable_device_count",
                             _xr.global_device_count)()
            if _n_dev < 2:
                # Last resort — count the physical chips off /dev/vfio.
                import os as _os
                _vfio = ([d for d in _os.listdir("/dev/vfio") if d.isdigit()]
                         if _os.path.exists("/dev/vfio") else [])
                _n_dev = len(_vfio) or _n_dev
            _device_ids = np.arange(_n_dev)
            _mesh = _xs.Mesh(_device_ids, (_n_dev,), ("tp",))
            _dev = _xm.xla_device()
            do_shard = _needs_spmd(model_cfg["name"])
            print(f"SPMD: {'sharding' if do_shard else 'replicating'} "
                  f"base model across {_n_dev} chips")
            # Iterate every parameter, move to XLA, and (for large models)
            # shard 2-D params with an output dim > 1024 along axis 0.
            # Direct dict assignment avoids set_data() which use_spmd() blocks.
            for _mod in model.modules():
                for _pname, _p in list(_mod._parameters.items()):
                    if _p is not None:
                        _xp = _nn.Parameter(
                            _p.data.to(_dev),
                            requires_grad=_p.requires_grad,
                        )
                        if do_shard and _xp.dim() == 2 and _xp.shape[0] > 1024:
                            _xs.mark_sharding(_xp, _mesh, (0, None))
                        _mod._parameters[_pname] = _xp
                for _bname, _b in list(_mod._buffers.items()):
                    if _b is not None:
                        _mod._buffers[_bname] = _b.to(_dev)
            _xm.mark_step()
        else:
            print("WARNING: no SPMD module found in torch_xla; 27B will OOM")

    # -------- LoRA variant selection ------------------------------------------
    variant = lora_cfg.get("variant", "standard").lower()
    if variant not in LORA_VARIANTS:
        raise ValueError(
            f"lora.variant={variant!r} not recognised. "
            f"Use one of {LORA_VARIANTS}."
        )
    # DoRA requires non-quantized linear layers (it reads the base weights to
    # compute direction vs. magnitude). Combining with QLoRA fails silently
    # or gives meaningless results, so gate it upfront.
    if variant == "dora" and quant_config is not None:
        raise ValueError(
            "DoRA requires full-precision weights but model.quantization is set. "
            "Either set lora.variant='standard' or model.quantization=null."
        )

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
        # use_dora / use_rslora are PEFT's opt-in flags for the variants.
        # Default both False so variant='standard' matches prior behavior.
        use_dora=(variant == "dora"),
        use_rslora=(variant == "rslora"),
    )
    print(f"LoRA variant: {variant}")

    # On TPU we apply PEFT manually here, AFTER the base model is SPMD-sharded.
    # Reason: SFTTrainer would otherwise call get_peft_model() inside __init__,
    # but that runs after accelerate.prepare() which on the XLA path does
    # nothing useful in single-process mode.  Pre-wrapping ourselves means the
    # LoRA adapters are constructed against already-sharded base linears, so
    # they land on the XLA device and reference the sharded weights correctly.
    # SFTTrainer detects an existing PeftModel and skips its own wrap.
    if _ON_TPU:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        # Tell HF Trainer "model placement is already handled" so it does NOT
        # call model.to(args.device) at __init__ time (transformers 4.57
        # trainer.py L612).  Under active SPMD, .to() invokes set_data() on
        # every parameter — that's intercepted and may raise "incompatible
        # tensor type" because the params are already sharded XLA tensors.
        # Setting is_parallelizable + model_parallel makes Trainer's
        # `is_model_parallel` property True, which flips
        # `place_model_on_device` off (see trainer.py L587-594).
        model.is_parallelizable = True
        model.model_parallel = True
        # peft_config is now baked into the model — don't pass it to SFTTrainer
        # again (would no-op but also clutter the config snapshot).
        _peft_for_trainer = None
        if _xm is not None:
            _xm.mark_step()
    else:
        # GPU path: keep the original behavior of letting SFTTrainer apply PEFT
        # so quantization-prepare hooks / kbit handling stay in one place.
        _peft_for_trainer = lora_config

    train_file = args.train_file or data_cfg["train_file"]
    val_file = args.val_file or data_cfg["val_file"]
    ds = {
        "train": load_sft_jsonl(train_file),
        "validation": load_sft_jsonl(val_file),
    }
    train_size = len(ds["train"])
    val_size = len(ds["validation"])
    print(f"Train ({train_file}): {train_size}  Val ({val_file}): {val_size}")

    eval_dataset = ds["validation"] if val_size > 0 else None
    eval_strategy = train_cfg["eval_strategy"]
    # Disable best-model selection on TPU: the PEFT adapter reload at end of
    # training calls model.load_state_dict() which under active SPMD invokes
    # set_data on every adapter param.  Even though adapters are unsharded,
    # the interception path is fragile — and the failure happens AFTER the
    # last save, inside trainer.train(), so we'd lose the whole multi-hour
    # run.  Use the LAST checkpoint instead (with cosine LR + 3 epochs the
    # last is typically the best anyway).
    load_best_model_at_end = not _ON_TPU
    metric_for_best_model = "eval_loss" if load_best_model_at_end else None
    if eval_dataset is None:
        print("Validation split is empty; disabling eval_strategy and best-model selection.")
        eval_strategy = "no"
        load_best_model_at_end = False
        metric_for_best_model = None

    # only compute loss on the assistant response, not on the prompt tokens
    response_template = find_response_template(_tokenizer)
    print(f"Response template for masking: {response_template!r}")
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=_tokenizer,
    )

    # derive bf16 from torch_dtype so the two flags can't diverge
    use_bf16 = train_cfg.get("bf16", dtype == torch.bfloat16)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        eval_strategy=eval_strategy,
        bf16=use_bf16,
        seed=seed,
        data_seed=seed,
        report_to="none",
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        save_total_limit=3,
        # torch.utils.checkpoint doesn't handle XLA device type ("xla") —
        # getattr(torch, "xla") raises AttributeError on TPU.  Default to
        # False; GPU runs can opt in via gradient_checkpointing: true in yaml.
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        max_seq_length=train_cfg.get("max_seq_length", 4096),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=_peft_for_trainer,
        train_dataset=ds["train"],
        eval_dataset=eval_dataset,
        tokenizer=_tokenizer,
        data_collator=collator,
        formatting_func=format_example,
    )

    trainer.train()
    trainer.save_state()
    best_path = f"{args.output_dir.rstrip('/')}/best"
    trainer.save_model(best_path)
    _tokenizer.save_pretrained(best_path)
    trainer.state.save_to_json(str(Path(best_path) / "trainer_state.json"))
    print(f"saved to {best_path}")


if __name__ == "__main__":
    main()
