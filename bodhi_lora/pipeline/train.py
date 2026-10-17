"""Stage train: LoRA or QLoRA on the seed's filtered traces, with PEFT and TRL.

    python -m bodhi_lora.pipeline.train --cell <cell id> --seed N

Runs in the training environment (the train extra of pyproject.toml), on one GPU, or on the CPU for the
smoke cell. The recipe is the cell's train block and the model record's chat-template
facts; the resolved recipe goes into the manifest. The seed's adapter is saved in
seed_<n>/train/ (adapter_config.json, the weights and trainer_state.json), with the
intermediate checkpoints in seed_<n>/train/checkpoint-<k>/; a rerun resumes from the
newest checkpoint.

The loss covers the reply only (completion-only masking): the response template marks
where the reply starts, from the model record or detected from the chat template. The
loss must also cover the token that ends the reply, or the adapter never learns to stop:
the pad token may not be EOS (set_pad_token), and check_replies_in_loss runs the real
collator on the first training rows before any step.

API used: TRL 0.11.4 (SFTConfig, SFTTrainer with formatting_func and peft_config,
DataCollatorForCompletionOnlyLM), per the TRL v0.11.4 SFT Trainer docs
(huggingface.co/docs/trl/v0.11.4/en/sft_trainer); transformers 4.57.6
(BitsAndBytesConfig, from_pretrained with revision); PEFT 0.19.1 (LoraConfig with
use_dora and use_rslora, prepare_model_for_kbit_training).
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from bodhi_lora import manifest as mf
from bodhi_lora.pipeline.context import StageContext, context_from_args, stage_parser
from bodhi_lora.pipeline.filter import split_files

# TF32 matmuls on Ampere and newer GPUs: a large SFT throughput gain at negligible
# precision cost for LoRA. It has no effect on the CPU.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

_tokenizer = None


def _pick_attn_impl():
    """FlashAttention-2 if the package is present, else PyTorch SDPA.

    FA2 is a large throughput and memory gain for long-sequence SFT; SDPA ships with
    torch, so a box without the flash-attn wheel still trains.
    """
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_sft_jsonl(path):
    """Load filtered traces, keeping only what SFTTrainer needs.

    The graded files also hold a ``grade`` field with per-example rubric keys; HF
    datasets' schema inference trips over it when files differ, so it is dropped.
    """
    rows = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            rows.append({"messages": obj["messages"], "response": obj["response"]})
    return Dataset.from_list(rows)


def format_example(batch):
    """Render messages and reply into one training string with the chat template.

    TRL probes ``formatting_func`` on a single example first to decide str or
    list[str], so both shapes are handled (issue #2).
    """
    def _render(msgs, resp):
        msgs = list(msgs)
        msgs.append({"role": "assistant", "content": resp})
        return _tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    if isinstance(batch["response"], str):
        return _render(batch["messages"], batch["response"])
    return [_render(msgs, resp) for msgs, resp in zip(batch["messages"], batch["response"])]


def find_response_template(tokenizer):
    """Detect the assistant-turn header by diffing the chat template with and without
    the generation prompt.

    Some templates ([INST] ... [/INST]) expose no such header; their model record sets
    chat_template.response_template instead.
    """
    dummy = [{"role": "user", "content": "hi"}]
    without_gen = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=False)
    with_gen = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=True)
    if with_gen.startswith(without_gen):
        template = with_gen[len(without_gen):]
        if template.strip():
            return template
    raise ValueError(
        "Could not detect the response template. Set chat_template.response_template in the "
        f"model record.\nwithout_gen = {without_gen!r}\nwith_gen = {with_gen!r}")


def set_pad_token(tokenizer, pad_token):
    """Pad with the model record's pad token, and never with EOS.

    DataCollatorForCompletionOnlyLM (TRL 0.11.4) builds its labels with transformers'
    DataCollatorForLanguageModeling, which sets the label of every token equal to
    pad_token_id to -100 (transformers/data/data_collator.py at v4.57.6). With pad = EOS
    that takes the EOS that ends each reply out of the loss, and the adapter never learns
    to stop. Until 2026-10-17 this code set pad = EOS when the tokenizer had no pad token,
    and every Mistral and Llama adapter was trained that way (KNOWN_ISSUES.md).
    """
    if pad_token is not None:
        pad_id = tokenizer.convert_tokens_to_ids(pad_token)
        if pad_id is None or (pad_id == tokenizer.unk_token_id and pad_token != tokenizer.unk_token):
            raise SystemExit(f"the model record's pad token {pad_token!r} is not in the vocabulary")
        tokenizer.pad_token = pad_token
    if tokenizer.pad_token is None:
        raise SystemExit("the tokenizer has no pad token: name one in the model record "
                         "(chat_template.pad_token), a token the data never holds and not EOS")
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise SystemExit(f"the tokenizer pads with its EOS {tokenizer.eos_token!r}, which takes the "
                         "end of each reply out of the loss: name another pad token in the model "
                         "record (chat_template.pad_token)")


def check_replies_in_loss(dataset, collator, tokenizer, n=16):
    """Stop before training when a reply token is out of the loss.

    The collator masks the prompt up to the last response template and every token with
    the pad id. On the first n tokenized training rows, run through the real collator,
    every label from the first trained token to the end of the row must equal its input
    id, the end-of-turn token included. Returns the number of rows checked.
    """
    rows = [{"input_ids": dataset[i]["input_ids"], "attention_mask": dataset[i]["attention_mask"]}
            for i in range(min(n, len(dataset)))]
    batch = collator(rows)
    for i, row in enumerate(rows):
        ids = list(row["input_ids"])
        labels = batch["labels"][i][:len(ids)].tolist()
        trained = [p for p, label in enumerate(labels) if label != -100]
        if not trained:
            raise SystemExit(f"training row {i}: no reply token is in the loss, so the response "
                             "template was not found (chat_template.response_template)")
        dropped = [p for p in range(trained[0], len(ids)) if labels[p] != ids[p]]
        if dropped:
            raise SystemExit(f"training row {i}: {len(dropped)} reply tokens are out of the loss, the "
                             f"first {tokenizer.decode([ids[dropped[0]]])!r}; the pad token "
                             f"{tokenizer.pad_token!r} must be one the replies never use")
    return len(rows)


def latest_checkpoint(output_dir):
    checkpoints = []
    for path in Path(output_dir).glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.split("-", 1)[1]), path))
        except (IndexError, ValueError):
            continue
    return str(sorted(checkpoints)[-1][1]) if checkpoints else None


def effective_batch(train_cfg, world_size=None):
    """Examples per optimizer step: per-device batch x accumulation x processes."""
    world_size = world_size or int(os.environ.get("WORLD_SIZE", "1"))
    return (int(train_cfg["per_device_train_batch_size"])
            * int(train_cfg["gradient_accumulation_steps"]) * world_size)


def trainer_record(state) -> dict:
    """Step, epoch and loss summary from a trainer_state dict."""
    history = state.get("log_history") or []
    train_losses = [h["loss"] for h in history if "loss" in h]
    eval_losses = [(h.get("step"), h["eval_loss"]) for h in history if "eval_loss" in h]
    return {
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "max_steps": state.get("max_steps"),
        "num_train_epochs": state.get("num_train_epochs"),
        "first_train_loss": train_losses[0] if train_losses else None,
        "last_train_loss": train_losses[-1] if train_losses else None,
        "eval_loss_by_step": eval_losses,
        "best_metric": state.get("best_metric"),
        "best_model_checkpoint": state.get("best_model_checkpoint"),
    }


def recipe(ctx: StageContext) -> dict:
    """The resolved recipe: the cell's train block and the model record's template facts."""
    t, m = ctx.cell.train, ctx.model
    return {
        "model": {"id": m.id, "hf_id": m.hf_id, "revision": m.revision, "dtype": m.dtype,
                  "quantization": t.quantization,
                  "tokenizer_kwargs": m.chat_template.tokenizer_kwargs,
                  "response_template": m.chat_template.response_template,
                  "pad_token": m.chat_template.pad_token},
        "lora": {"method": t.method, "r": t.r, "alpha": t.alpha, "dropout": t.dropout,
                 "target_modules": t.target_modules},
        "training": {"epochs": t.epochs, "max_steps": t.max_steps,
                     "per_device_train_batch_size": t.per_device_batch,
                     "gradient_accumulation_steps": t.grad_accum, "learning_rate": t.lr,
                     "lr_scheduler_type": t.scheduler, "warmup_ratio": t.warmup_ratio,
                     "max_seq_length": t.max_seq_length,
                     "gradient_checkpointing": t.gradient_checkpointing,
                     "checkpoint": t.checkpoint, "eval_every": t.eval_every,
                     "eval_steps": t.eval_steps, "logging_steps": t.logging_steps},
    }


def build_train_manifest(ctx: StageContext, seed: int, cfg: dict, train_file, val_file,
                         adapter_dir, state, timer) -> dict:
    """Manifest for one training run. Kept apart from run() for testing."""
    return mf.build_manifest(
        "train", cell=ctx.cell.id, seed=seed, config=cfg, models={"base": ctx.model},
        adapter_dir=adapter_dir, inputs=[train_file, val_file], timer=timer,
        counts={"seed": seed, "effective_batch": effective_batch(cfg["training"]),
                "world_size": int(os.environ.get("WORLD_SIZE", "1")), **trainer_record(state)},
        extra={"status": "complete", "quantization": cfg["model"]["quantization"]})


def run(ctx: StageContext, seed: int) -> dict:
    global _tokenizer
    if ctx.cell.train is None or ctx.cell.train.engine != "peft":
        raise SystemExit(f"{ctx.cell.id} has no train block this package can run")
    cfg = recipe(ctx)
    model_cfg, lora_cfg, train_cfg = cfg["model"], cfg["lora"], cfg["training"]
    out_dir = ctx.paths.stage(seed, "train")
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = mf.Timer()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    on_gpu = torch.cuda.is_available()
    if model_cfg["quantization"] and not on_gpu:
        raise SystemExit("4-bit and 8-bit training need a GPU (bitsandbytes)")

    _tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_id"], revision=model_cfg["revision"],
                                               **model_cfg["tokenizer_kwargs"])
    set_pad_token(_tokenizer, model_cfg["pad_token"])
    # Half-precision SFT overflows with left padding.
    _tokenizer.padding_side = "right"
    # The chat template already emits BOS; a second one would be a sequence the model
    # never saw.
    _tokenizer.add_bos_token = False

    if train_cfg["warmup_ratio"] > 0 and train_cfg["lr_scheduler_type"] == "constant":
        raise ValueError("a constant schedule ignores warmup_ratio; set warmup_ratio: 0")
    dtype = DTYPE_MAP[model_cfg["dtype"]]
    quant_config = None
    if model_cfg["quantization"] == "4bit":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    elif model_cfg["quantization"] == "8bit":
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    print(f"Loading {model_cfg['hf_id']} ({model_cfg['quantization'] or model_cfg['dtype']}, "
          f"{'GPU' if on_gpu else 'CPU'})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["hf_id"], revision=model_cfg["revision"], torch_dtype=dtype,
        device_map="auto", quantization_config=quant_config, attn_implementation=_pick_attn_impl())
    model.config.use_cache = False      # incompatible with gradient checkpointing
    if quant_config is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=train_cfg["gradient_checkpointing"])
    lora_config = LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"], task_type="CAUSAL_LM",
        use_dora=lora_cfg["method"] == "dora", use_rslora=lora_cfg["method"] == "rslora")

    train_file, val_file = split_files(ctx, seed)
    ds_train, ds_val = load_sft_jsonl(train_file), load_sft_jsonl(val_file)
    print(f"Train ({train_file}): {len(ds_train)}  Val ({val_file}): {len(ds_val)}", flush=True)
    has_val = len(ds_val) > 0
    response_template = model_cfg["response_template"] or find_response_template(_tokenizer)
    print(f"Response template for masking: {response_template!r}", flush=True)
    # Pad to the longest sequence in each batch, not to max_seq_length.
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=_tokenizer)
    every = train_cfg["eval_every"]
    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=train_cfg["epochs"] or 1,
        max_steps=train_cfg["max_steps"] or -1,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=every,
        # Read only when saving every eval_steps; 500 is the transformers default.
        save_steps=train_cfg["eval_steps"] or 500,
        eval_strategy=every if has_val else "no",
        eval_steps=train_cfg["eval_steps"],
        bf16=on_gpu and dtype == torch.bfloat16,
        tf32=on_gpu,
        seed=seed,
        data_seed=seed,
        report_to="none",
        load_best_model_at_end=has_val and train_cfg["checkpoint"] == "best_val_loss",
        metric_for_best_model="eval_loss" if has_val else None,
        save_total_limit=2,
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=train_cfg["max_seq_length"],
        # Group similar lengths to cut padding.
        group_by_length=True,
        optim="adamw_torch_fused" if on_gpu else "adamw_torch",
        dataloader_num_workers=4 if on_gpu else 0,
        dataloader_pin_memory=on_gpu,
    )
    trainer = SFTTrainer(model=model, args=args, peft_config=lora_config, train_dataset=ds_train,
                         eval_dataset=ds_val if has_val else None, tokenizer=_tokenizer,
                         data_collator=collator, formatting_func=format_example)
    n_checked = check_replies_in_loss(trainer.train_dataset, collator, _tokenizer)
    print(f"Reply tokens in the loss: checked on {n_checked} training rows "
          f"(pad {_tokenizer.pad_token!r}, EOS {_tokenizer.eos_token!r})", flush=True)
    resume = latest_checkpoint(out_dir)
    if resume:
        print(f"Resuming from checkpoint: {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out_dir))
    try:
        trainer.state.save_to_json(str(out_dir / "trainer_state.json"))
    except (ValueError, TypeError) as e:
        print(f"trainer_state.json skipped ({e!r}): NaN in eval metrics", flush=True)
    state = {k: getattr(trainer.state, k, None) for k in (
        "global_step", "epoch", "max_steps", "num_train_epochs", "log_history",
        "best_metric", "best_model_checkpoint")}
    manifest = build_train_manifest(ctx, seed, cfg, train_file, val_file, out_dir, state, timer)
    if trainer.is_world_process_zero():
        mf.write_manifest(out_dir / "manifest.json", manifest)
    print(f"adapter -> {out_dir}", flush=True)
    return manifest["counts"]


def main(argv=None) -> int:
    args = stage_parser(__doc__.splitlines()[0], seed="required").parse_args(argv)
    run(context_from_args(args), args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
