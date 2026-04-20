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
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quant_config,
    )

    # Quantized weights need gradient checkpointing + input-grad rewiring before
    # LoRA adapters are attached. PEFT's helper handles both.
    if quant_config is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

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

    train_file = args.train_file or data_cfg["train_file"]
    val_file = args.val_file or data_cfg["val_file"]
    ds = {
        "train": load_sft_jsonl(train_file),
        "validation": load_sft_jsonl(val_file),
    }
    print(f"Train ({train_file}): {len(ds['train'])}  Val ({val_file}): {len(ds['validation'])}")

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
        eval_strategy=train_cfg["eval_strategy"],
        bf16=use_bf16,
        seed=seed,
        data_seed=seed,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=3,
        gradient_checkpointing=True,
        max_seq_length=train_cfg.get("max_seq_length", 4096),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=lora_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
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
