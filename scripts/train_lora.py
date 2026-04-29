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


class _RequiresGradOnlyOptimizerSFTTrainer(SFTTrainer):
    """SFTTrainer that builds the optimizer from requires_grad=True params only.

    Workaround for transformers#39795 — under FSDP + LoRA on transformers
    >= 4.50, HF's default ``create_optimizer`` ends up allocating optimizer
    state for ALL parameters (not just trainable ones).  For MedGemma-27B
    that is 27B × 8 bytes (Adam m + v in fp32) ≈ 216 GB of host RAM, and
    the XLA compile thread starts page-thrashing.  Filtering on
    ``p.requires_grad`` reduces the optimizer state to the actual ~13 M
    LoRA adapter params (≈ 100 MB) and lets the compile finish without
    exhausting host memory.

    We mirror HF Trainer's two-group structure (decay vs. no-decay on
    bias / LayerNorm) so weight-decay behaviour matches the upstream
    defaults — just gated on requires_grad.
    """

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        no_decay_substrings = ("bias", "LayerNorm.weight", "layernorm.weight")
        decay_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad
            and not any(s in n for s in no_decay_substrings)
        ]
        no_decay_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad
            and any(s in n for s in no_decay_substrings)
        ]
        n_decay = sum(p.numel() for p in decay_params)
        n_no_decay = sum(p.numel() for p in no_decay_params)
        print(
            f"create_optimizer: {n_decay + n_no_decay:,} trainable params "
            f"({n_decay:,} with weight_decay, {n_no_decay:,} without). "
            f"requires_grad-filter saves ~{(27_000_000_000 - n_decay - n_no_decay) * 8 / 1e9:.0f} GB "
            f"of optimizer state vs. the buggy all-param path."
        )
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        # Reuse HF's optimizer-class + kwarg resolution so any optim_type
        # config setting (adamw_torch, adafactor, etc.) is honoured.
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters, **optimizer_kwargs
        )
        return self.optimizer

# Detect whether we're running under PyTorch/XLA (Google Cloud TPU).
# When True:  device_map="auto" must NOT be used — accelerate owns placement.
# When False: device_map="auto" is used as before (multi-GPU or single GPU).
try:
    import torch_xla  # noqa: F401
    import torch_xla.core.xla_model as _xm
    _ON_TPU = True
    # The use_reentrant=False gradient-checkpoint path (used by MedGemma-27B)
    # calls getattr(torch, 'xla') internally.  torch_xla is a separate package
    # so torch.xla is not set by default; alias it here to prevent AttributeError
    # during the backward pass when gradient_checkpointing_kwargs.use_reentrant=false.
    if not hasattr(torch, 'xla'):
        torch.xla = torch_xla  # type: ignore[attr-defined]
except ImportError:
    _xm = None
    _ON_TPU = False

# FSDPv2 setup — replaces the manual mark_sharding loop we used previously.
# This must run BEFORE any model load: setting XLA_USE_SPMD=1 and calling
# xr.use_spmd() activates the SPMD runtime; calling it AFTER an XLA tensor
# has been allocated raises "SPMD must be enabled before any device alloc".
#
# Manual mark_sharding on Gemma-3 27B was hanging the first-step XLA compile
# for 30+ min with the cache frozen at ~4 MB — torch_xla 2.5's fusion-
# emitter regression on Gemma-3 (#8591) plus a manually-partitioned graph
# the compiler had to legalize on every recompile.  FSDPv2 (xla_fsdp_v2:
# True in fsdp_config) is the HF/Google blessed path on v6e and unblocks
# scan_layers in 2.6+.  See https://huggingface.co/docs/optimum-tpu .
#
# We do the init ourselves rather than relying on optimum-tpu's
# use_fsdp_v2() because that helper is essentially three lines and pinning
# optimum-tpu's transitive deps is fragile (it tugs at transformers /
# accelerate / peft).  optimum-tpu IS still useful for its
# get_fsdp_training_args() helper which maps model class -> decoder layer
# class — we try it later but fall back to a hand-rolled config if it
# doesn't recognise the model (e.g., Gemma-3).
_fsdp_v2 = None
if _ON_TPU:
    import os as _os_spmd
    _os_spmd.environ.setdefault("PJRT_DEVICE", "TPU")
    _os_spmd.environ["XLA_USE_SPMD"] = "1"
    try:
        from torch_xla import runtime as _xr_init
        if hasattr(_xr_init, "use_spmd"):
            _xr_init.use_spmd()
            print("FSDPv2: XLA_USE_SPMD=1 set, xr.use_spmd() called.")
    except Exception as _e:
        print(f"WARNING: xr.use_spmd() init failed ({_e!r}); "
              "FSDPv2 may not work — training will likely OOM on Gemma-3 27B.")
    try:
        from optimum.tpu import fsdp_v2 as _fsdp_v2  # used for get_fsdp_training_args
        print("optimum-tpu fsdp_v2 helper available")
    except ImportError:
        # not fatal — we have a manual fallback for get_fsdp_training_args
        _fsdp_v2 = None
        print("optimum-tpu not installed; using manual FSDPv2 config")

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
# Fix: single-process SPMD via FSDPv2.  ONE Python process drives all 8 chips
# and the model parameters get sharded across them by HF Trainer's FSDP
# plugin (xla_fsdp_v2: True), wrapping every Gemma3DecoderLayer in an XLA FSDP
# unit.  This matches our accelerate config tpu/accelerate_config_v6e8.yaml
# (num_processes: 1) and replaces the previous manual mark_sharding loop that
# was hanging the first-step XLA compile on Gemma-3 27B for 30+ min.
#
# Note about ordering: with optimum.tpu.fsdp_v2.use_fsdp_v2() (called above
# at import time), it IS safe to enable SPMD before from_pretrained — the
# v2 backend uses XLA_USE_SPMD=1 which doesn't intercept set_data() the way
# the manual xr.use_spmd() + mark_sharding pattern did.  This is also why
# scripts/generate_traces.py and eval_healthbench.py still use the old
# "use_spmd after model load" order: they don't go through optimum-tpu.

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

    # Disable KV cache for training.  With use_cache=True (the model default),
    # Gemma-3's HybridCache.update() tries to slice the last (sliding_window - 1)
    # elements from the key states.  For Gemma-3 sliding_window=1024, so it
    # attempts index -1023 on a sequence of length max_seq_length (e.g. 512),
    # which is out of range and XLA raises "Value out of range [-512, 511]".
    # Disabling the cache avoids this path entirely — KV caching is only needed
    # for autoregressive inference, not for training on full sequences.
    # (When gradient_checkpointing=True the Trainer sets this automatically, but
    # we set it unconditionally here so it's always safe.)
    model.config.use_cache = False

    # Quantized weights need gradient checkpointing + input-grad rewiring before
    # LoRA adapters are attached. PEFT's helper handles both.
    if quant_config is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    # ── FSDPv2 sharding ──────────────────────────────────────────────────────
    # Sharding is now handled by FSDPv2 inside the HF Trainer (configured via
    # SFTConfig fsdp + fsdp_config below).  use_fsdp_v2() was already called
    # at module import so XLA_USE_SPMD=1 and the runtime is initialized.
    # The Trainer's FSDP plugin walks the model graph and wraps every module
    # whose class name matches transformer_layer_cls_to_wrap (Gemma3DecoderLayer
    # for medgemma) — sharding the LARGE per-layer weights across all 8 chips
    # while replicating embeddings + lm_head.  This avoids the manual
    # mark_sharding loop's pitfalls (vocab-dim gathers, tied-weight reconcile,
    # and the manually-partitioned graph that the compiler had to legalize
    # on every recompile).  Per HF/optimum-tpu blog posts, this is the
    # blessed path for Gemma + LoRA on v6e and reduces "hours of compile"
    # to minutes.

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

    # IMPORTANT — do NOT pre-apply PEFT here on TPU.  SFTTrainer (TRL 0.11+)
    # has a known bug (trl#3926) where, if you pass a PeftModel as `model`,
    # it calls prepare_model_for_kbit_training() on the already-PEFT'd
    # model and freezes 100 % of params (trainable% drops to 0 — training
    # silently does nothing for hours).  The supported pattern is:
    #
    #     SFTTrainer(model=base_model, peft_config=lora_config, ...)
    #
    # Under that pattern SFTTrainer calls get_peft_model() itself inside
    # __init__, BEFORE accelerator.prepare(), so when the FSDPv2 plugin
    # wraps each Gemma3DecoderLayer the LoRA adapter Linears are already
    # bundled inside their parent layer and end up in the same FSDP unit.
    #
    # Old comment about "needing to pre-wrap before SPMD-sharding so set_data
    # interception doesn't fire" no longer applies under FSDPv2 — the v2
    # backend uses XLA_USE_SPMD=1 (not the manual xr.use_spmd() set_data
    # interception path) and accelerate handles XLA device placement via
    # the FSDP plugin during prepare().
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
    # pad_to_multiple_of=max_seq_length forces every batch to be padded to
    # exactly max_seq_length (since SFTTrainer truncates to max_seq_length
    # during tokenization, all examples are <= max_seq_length, so the next
    # multiple is always exactly max_seq_length).
    #
    # Without this, the default collator pads each batch to the longest
    # example IN THAT BATCH — so a dataset with varied sequence lengths
    # produces dozens of distinct batch shapes, and XLA compiles a fresh
    # HLO graph for each.  On a 27B SPMD model with each compile taking
    # 30+ minutes, that's an effectively-infinite loop.  Saw a 7+ hour
    # hang in this exact configuration.
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=_tokenizer,
        pad_to_multiple_of=train_cfg["max_seq_length"],
    )

    # derive bf16 from torch_dtype so the two flags can't diverge
    use_bf16 = train_cfg.get("bf16", dtype == torch.bfloat16)

    # ── FSDPv2 trainer args ─────────────────────────────────────────────────
    # Build the fsdp / fsdp_config kwargs that tell HF Trainer to wrap each
    # decoder-layer module (here Gemma3DecoderLayer for medgemma-27b) in an
    # XLA FSDP-v2 unit.  We try optimum-tpu's get_fsdp_training_args() first
    # — it auto-detects the right transformer_layer_cls_to_wrap from the
    # model's architecture string.  If that helper isn't installed or
    # doesn't recognize the model class (older optimum-tpu versions don't
    # know Gemma-3), fall back to a manual config keyed on
    # Gemma3DecoderLayer (the per-layer module on medgemma-27b-text-it).
    _fsdp_kwargs = {}
    if _ON_TPU:
        _fsdp_kwargs = {
            "fsdp": "full_shard",
            "fsdp_config": {
                # Gemma3DecoderLayer is the per-layer module on
                # google/medgemma-27b-text-it.  Update if base model changes.
                "transformer_layer_cls_to_wrap": ["Gemma3DecoderLayer"],
                "xla": True,
                "xla_fsdp_v2": True,
                "xla_fsdp_grad_ckpt": train_cfg.get(
                    "gradient_checkpointing", False
                ),
            },
        }
        if _fsdp_v2 is not None:
            try:
                _detected = _fsdp_v2.get_fsdp_training_args(model)
                # Only adopt optimum-tpu's recommendation if it returned a
                # valid config — older versions raise on unknown model types.
                if _detected and "fsdp_config" in _detected:
                    _fsdp_kwargs = _detected
                    print("FSDPv2 args from optimum-tpu.get_fsdp_training_args()")
            except Exception as _e:
                print(f"get_fsdp_training_args failed ({_e!r}); using manual config")
        print(f"FSDPv2 trainer args: {_fsdp_kwargs}")

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
        # Gradient checkpointing: default off.  On TPU the legacy reentrant
        # path raises AttributeError ("torch has no attribute 'xla'"); always
        # use use_reentrant=False (non-reentrant, saved_tensors_hooks) which
        # works on XLA.  Configs should set:
        #   gradient_checkpointing: true
        #   gradient_checkpointing_kwargs: {use_reentrant: false}
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs=train_cfg.get("gradient_checkpointing_kwargs", None),
        max_seq_length=train_cfg.get("max_seq_length", 4096),
        **_fsdp_kwargs,
    )

    # On TPU we use _RequiresGradOnlyOptimizerSFTTrainer so the optimizer
    # state is sized to the ~13 M LoRA params instead of all 27 B base params
    # (transformers#39795 workaround — drops 216 GB of host RAM and stops
    # the XLA compile thread from page-thrashing).  The override is a no-op
    # on GPU since the bug is FSDP-specific, but using one class for both
    # paths keeps the call site simple.
    _trainer_cls = _RequiresGradOnlyOptimizerSFTTrainer if _ON_TPU else SFTTrainer
    trainer = _trainer_cls(
        model=model,
        args=training_args,
        peft_config=_peft_for_trainer,
        train_dataset=ds["train"],
        eval_dataset=eval_dataset,
        tokenizer=_tokenizer,
        data_collator=collator,
        formatting_func=format_example,
    )

    # Resume from the latest checkpoint if one exists in output_dir.  This
    # makes preemption recovery cheap: rescued checkpoints are SCP'd back to
    # the new VM, training resumes from the last save instead of step 0.
    # `resume_from_checkpoint=True` is a no-op (starts fresh) if no checkpoint
    # is present, so safe on first run.
    from pathlib import Path as _Path
    _ckpt_dir = _Path(args.output_dir)
    _has_ckpt = _ckpt_dir.exists() and any(
        p.name.startswith("checkpoint-") for p in _ckpt_dir.iterdir()
    )
    if _has_ckpt:
        print(f"Found existing checkpoint(s) in {args.output_dir}, resuming.")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # On TPU/SPMD the computation graph is asynchronous.  Without an explicit
    # barrier, trainer.save_model() can race against in-flight sharded tensor
    # copies across chips and either deadlock or save partially-written weights.
    # mark_step() flushes and waits for all pending XLA ops to complete before
    # we touch the filesystem.
    if _ON_TPU and _xm is not None:
        _xm.mark_step()

    trainer.save_state()
    best_path = f"{args.output_dir.rstrip('/')}/best"
    Path(best_path).mkdir(parents=True, exist_ok=True)

    # transformers#36004: under xla_fsdp_v2 the Trainer's save_model path
    # writes a near-base-model state_dict (~18 GB on Gemma-3 27B) — this
    # is NOT a valid PEFT adapter (no adapter_model.safetensors, no
    # adapter_config.json) AND it fills the boot disk on a stock Cloud TPU
    # VM mid-write, taking the rest of save_pretrained down with it.
    # On TPU we therefore skip trainer.save_model() entirely and write
    # ONLY the LoRA adapter via PeftModel.save_pretrained.  Stage 4 eval
    # loads the base model from HF and the adapter from this directory,
    # so the base state_dict isn't needed here.
    # On GPU (HF/DDP/FSDP, not xla_fsdp_v2) save_model emits a valid
    # adapter via PEFT integration, so keep the existing path.
    if _ON_TPU:
        try:
            _peft_model = trainer.accelerator.unwrap_model(trainer.model)
            # PEFT's safetensors backend calls tensor.data_ptr() under the
            # hood, which raises "invalid python storage" on xla_fsdp_v2-
            # sharded params (they are XLA-virtual; no CPU data pointer).
            # Gather LoRA params to CPU first and pass them in via the
            # state_dict kwarg so PEFT serializes a real CPU state dict.
            # Only requires_grad params are LoRA deltas — base weights stay
            # out of the file (8.4M params, ~16 MB on disk).
            _adapter_cpu = {
                name: param.detach().cpu()
                for name, param in _peft_model.named_parameters()
                if param.requires_grad
            }
            _peft_model.save_pretrained(
                best_path,
                safe_serialization=True,
                state_dict=_adapter_cpu,
            )
            print(f"LoRA adapter saved via PeftModel.save_pretrained -> {best_path}")
        except Exception as _e:
            print(f"WARNING: explicit adapter save failed ({_e!r}); "
                  f"the checkpoint at {best_path} may be missing adapter "
                  "weights (transformers#36004) and Stage 4 eval will fail.")
    else:
        trainer.save_model(best_path)
    _tokenizer.save_pretrained(best_path)
    # trainer.state may contain NaN eval_loss (e.g. when all val labels are
    # masked).  Python's json module raises ValueError on NaN by default, so
    # guard the save.
    try:
        trainer.state.save_to_json(str(Path(best_path) / "trainer_state.json"))
    except (ValueError, TypeError) as _e:
        print(f"trainer_state.json skipped ({_e!r}) — NaN in eval metrics")
    print(f"saved to {best_path}")


if __name__ == "__main__":
    main()
