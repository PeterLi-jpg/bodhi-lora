"""LoRA SFT on filtered BODHI traces — GPU-only, tuned for throughput.

Rebuttal branch: this is the single, clean GPU training path. All TPU / MaxText /
tunix / FSDPv2 / XLA code has been removed (that lives on `main`). One GPU per
process: pin it with CUDA_VISIBLE_DEVICES and the run driver parallelizes seeds
across idle GPUs. QLoRA (model.quantization: 4bit) fits a 24-27B base on one
80GB H100 with room to spare; 7-8B bases train full-precision.

Config knobs (see rebuttal/configs/*.yaml):
- model.quantization: null (full-precision LoRA) or "4bit"/"8bit" (QLoRA, bitsandbytes).
- lora.variant: "standard" (default), "dora" (full-precision only), or "rslora".
- data.response_template: optional override for completion-only masking (e.g. "[/INST]"
  for Mistral, whose [INST]...[/INST] template has no auto-detectable assistant header).

GPU throughput: FlashAttention-2 when available (else SDPA), TF32 matmuls, fused
AdamW, and padding to the longest sequence in each batch (not a fixed 4096 — that
was a TPU XLA-recompile workaround and wastes GPU compute).
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

# TF32 matmuls on Ampere+ (H100): large SFT throughput win, negligible precision
# cost for LoRA fine-tuning. Safe to enable unconditionally on CUDA.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
LORA_VARIANTS = ("standard", "dora", "rslora")

_tokenizer = None


def _pick_attn_impl():
    """FlashAttention-2 if the package is present, else PyTorch SDPA.

    FA2 is a big throughput/memory win for long-sequence SFT; SDPA is the safe
    fallback that ships with torch. We never hard-require FA2 so a box without
    the flash-attn wheel still runs (just slower).
    """
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_sft_jsonl(path):
    """Load graded SFT JSONL, keeping only what SFTTrainer needs.

    The graded files also include a ``grade`` field with per-example variable
    rubric keys; HF datasets' schema inference trips over that when files differ.
    Stripping to messages+response avoids the cast error.
    """
    rows = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            rows.append({"messages": obj["messages"], "response": obj["response"]})
    return Dataset.from_list(rows)


def format_example(batch):
    """Format messages+response into a single training string via the tokenizer's
    chat template (model-agnostic).

    TRL probes ``formatting_func`` on a single example first to decide str vs
    list[str], so we handle both shapes explicitly (issue #2): on a single
    example ``response`` is a str and ``messages`` is one conversation.
    """
    def _render(msgs, resp):
        msgs = list(msgs)
        msgs.append({"role": "assistant", "content": resp})
        return _tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )

    if isinstance(batch["response"], str):
        return _render(batch["messages"], batch["response"])
    return [_render(msgs, resp)
            for msgs, resp in zip(batch["messages"], batch["response"])]


def find_response_template(tokenizer):
    """Detect the assistant-turn header by diffing chat templates with/without the
    generation prompt (e.g. "<start_of_turn>model\\n" for Gemma,
    "<|start_header_id|>assistant<|end_header_id|>\\n\\n" for Llama 3).

    Some templates (Mistral [INST]...[/INST]) expose no such header; for those,
    set data.response_template in the config instead of relying on this.
    """
    dummy = [{"role": "user", "content": "hi"}]
    without_gen = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=False)
    with_gen = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=True)
    if with_gen.startswith(without_gen):
        template = with_gen[len(without_gen):]
        if template.strip():
            return template
    raise ValueError(
        "Could not auto-detect the response template. Set data.response_template "
        "in the config (e.g. \"[/INST]\" for Mistral).\n"
        f"without_gen = {without_gen!r}\nwith_gen = {with_gen!r}"
    )


def latest_checkpoint(output_dir):
    root = Path(output_dir)
    checkpoints = []
    for path in root.glob("checkpoint-*"):
        try:
            step = int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    checkpoints.sort()
    return str(checkpoints[-1][1])


def main():
    global _tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="override the seed in the YAML (for multi-seed runs)")
    parser.add_argument("--output-dir", default="checkpoints",
                        help="checkpoints + final adapter go here; adapter -> <output-dir>/best")
    parser.add_argument("--train-file", default=None, help="override data.train_file")
    parser.add_argument("--val-file", default=None, help="override data.val_file")
    parser.add_argument("--quantization", default=None, choices=["4bit", "8bit", "null"],
                        help="override model.quantization ('null' = full precision)")
    parser.add_argument("--lora-variant", default=None, choices=list(LORA_VARIANTS),
                        help="override lora.variant")
    parser.add_argument("--lora-r", type=int, default=None, help="override lora.r")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.quantization is not None:
        cfg["model"]["quantization"] = None if args.quantization == "null" else args.quantization
    if args.lora_variant is not None:
        cfg["lora"]["variant"] = args.lora_variant
    if args.lora_r is not None:
        cfg["lora"]["r"] = args.lora_r

    model_cfg, lora_cfg, train_cfg, data_cfg = cfg["model"], cfg["lora"], cfg["training"], cfg["data"]

    seed = args.seed if args.seed is not None else int(cfg.get("seed", train_cfg.get("seed", 42)))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); set_seed(seed)

    # model.tokenizer_kwargs lets a config pass loader flags, e.g. Mistral-Small's
    # fix_mistral_regex=True (its tekken tokenizer otherwise warns of incorrect
    # tokenization). Default {} preserves prior behavior for every other model.
    _tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], **model_cfg.get("tokenizer_kwargs", {}))
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    # Half-precision SFT overflows with left padding; force right.
    _tokenizer.padding_side = "right"
    # The chat template already emits BOS; disable auto-prepend so we don't get
    # "<bos><bos>..." (a distribution the model never trained on).
    _tokenizer.add_bos_token = False

    if train_cfg.get("warmup_ratio", 0) > 0 and train_cfg.get("lr_scheduler_type") == "constant":
        raise ValueError("lr_scheduler_type='constant' ignores warmup_ratio; use "
                         "'cosine'/'linear'/'constant_with_warmup' or set warmup_ratio: 0.")

    dtype = DTYPE_MAP.get(model_cfg.get("torch_dtype") or "bfloat16", torch.bfloat16)

    # -------- optional QLoRA quantization --------
    quant = model_cfg.get("quantization")
    quant_config = None
    if quant in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        if quant == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            )
        else:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        print(f"Loading {model_cfg['name']} ({quant} quantized, compute {dtype})...")
    elif quant is not None:
        raise ValueError(f"model.quantization={quant!r} not recognised (null / '4bit' / '8bit').")
    else:
        print(f"Loading {model_cfg['name']} ({dtype})...")

    attn_impl = _pick_attn_impl()
    print(f"Attention implementation: {attn_impl}")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=dtype,
        device_map="auto",              # single visible GPU -> whole model on it
        quantization_config=quant_config,
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False      # incompatible with gradient checkpointing / training

    if quant_config is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True)
        )

    # -------- LoRA config --------
    variant = lora_cfg.get("variant", "standard").lower()
    if variant not in LORA_VARIANTS:
        raise ValueError(f"lora.variant={variant!r} not in {LORA_VARIANTS}.")
    if variant == "dora" and quant_config is not None:
        raise ValueError("DoRA needs full-precision weights; set model.quantization: null.")
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
        use_dora=(variant == "dora"),
        use_rslora=(variant == "rslora"),
    )
    print(f"LoRA variant: {variant}")

    # -------- data --------
    train_file = args.train_file or data_cfg["train_file"]
    val_file = args.val_file or data_cfg["val_file"]
    ds = {"train": load_sft_jsonl(train_file), "validation": load_sft_jsonl(val_file)}
    train_size, val_size = len(ds["train"]), len(ds["validation"])
    print(f"Train ({train_file}): {train_size}  Val ({val_file}): {val_size}")

    eval_dataset = ds["validation"] if val_size > 0 else None
    eval_strategy = train_cfg["eval_strategy"]
    load_best = True
    metric_for_best = "eval_loss"
    if eval_dataset is None:
        print("Empty validation split; disabling eval + best-model selection.")
        eval_strategy, load_best, metric_for_best = "no", False, None

    # Completion-only loss masking: allow an explicit response_template override
    # (Mistral etc.), else auto-detect from the chat template.
    response_template = data_cfg.get("response_template") or find_response_template(_tokenizer)
    print(f"Response template for masking: {response_template!r}")
    # Pad to the longest sequence IN EACH BATCH (GPU handles dynamic shapes fine).
    # The old code padded every batch to max_seq_length for TPU XLA compile
    # stability, which wasted 2-3x compute here since most traces are far shorter.
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=_tokenizer
    )

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
        save_steps=train_cfg.get("save_steps"),
        eval_strategy=eval_strategy,
        eval_steps=train_cfg.get("eval_steps"),
        bf16=use_bf16,
        tf32=True,
        seed=seed,
        data_seed=seed,
        report_to="none",
        load_best_model_at_end=load_best,
        metric_for_best_model=metric_for_best,
        save_total_limit=2,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=train_cfg.get(
            "gradient_checkpointing_kwargs", {"use_reentrant": False}),
        max_seq_length=train_cfg.get("max_seq_length", 4096),
        # GPU throughput: group similar-length sequences to minimise padding,
        # fused AdamW, and a couple of dataloader workers.
        group_by_length=train_cfg.get("group_by_length", True),
        optim=train_cfg.get("optim", "adamw_torch_fused"),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=lora_config,        # SFTTrainer applies PEFT itself (trl#3926-safe)
        train_dataset=ds["train"],
        eval_dataset=eval_dataset,
        tokenizer=_tokenizer,
        data_collator=collator,
        formatting_func=format_example,
    )

    resume_path = latest_checkpoint(args.output_dir)
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        trainer.train()

    trainer.save_state()
    best_path = f"{args.output_dir.rstrip('/')}/best"
    Path(best_path).mkdir(parents=True, exist_ok=True)
    trainer.save_model(best_path)       # emits a valid PEFT adapter on GPU
    _tokenizer.save_pretrained(best_path)
    try:
        trainer.state.save_to_json(str(Path(best_path) / "trainer_state.json"))
    except (ValueError, TypeError) as e:
        print(f"trainer_state.json skipped ({e!r}) — NaN in eval metrics")
    print(f"saved adapter to {best_path}")


if __name__ == "__main__":
    main()
