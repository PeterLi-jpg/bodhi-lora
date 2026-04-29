"""Talk to the BOHDI LoRA model from a shell.

Useful for spot-checking a trained checkpoint without running the full eval
harness. Type a prompt, get a response, toggle the BODHI wrapper mid-session
to compare. Uses greedy decoding (same as eval_healthbench) so repeated
prompts produce the same answer.

Examples:
    python scripts/chat.py --model google/gemma-3n-E4B-it --lora-path checkpoints/best
    python scripts/chat.py --model google/medgemma-27b-text-it --lora-path checkpoints/best --use-bodhi
    python scripts/chat.py --model google/gemma-3n-E4B-it   # base only, no LoRA
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# Same pattern as train_lora.py:85-100. On a TPU host, device_map="auto"
# triggers accelerate's CUDA placement (it inspects nvidia-smi/cuda
# available); we want torch_xla to own placement instead.
try:
    import torch_xla  # noqa: F401
    _ON_TPU = True
except ImportError:
    _ON_TPU = False


def _device_map_for_host(on_tpu: bool):
    """device_map gating; mirrors train_lora.py:400 convention."""
    return None if on_tpu else "auto"


def load_model(base_name, lora_path, dtype=torch.bfloat16):
    # Use the checkpoint's tokenizer when available; SFTTrainer may have
    # updated pad/eos from training and we want the exact same tokenization
    # at chat time as at eval time.
    tokenizer = AutoTokenizer.from_pretrained(lora_path or base_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=dtype,
        device_map=_device_map_for_host(_ON_TPU),
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages, max_new_tokens):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def make_wrapper(model, tokenizer, max_new_tokens):
    # Same adapter pattern as eval_healthbench.make_bodhi_wrapper: BODHI's
    # two-pass reasoning calls chat_fn under the hood for each pass.
    from bodhi import BODHI, BODHIConfig

    def chat_fn(msgs):
        return generate(model, tokenizer, msgs, max_new_tokens)

    return BODHI(chat_function=chat_fn, config=BODHIConfig(domain="medical"))


HELP = """Commands (prefix with ':'):
  :bodhi     toggle the BODHI wrapper on/off
  :reset     clear conversation history
  :tokens N  change max_new_tokens to N
  :help      show this message
  :q         quit
Everything else is sent as a user message."""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-3n-E4B-it",
                   help="Base model. Must match what the LoRA was trained on.")
    p.add_argument("--lora-path", default=None,
                   help="LoRA adapter path. Omit to chat with the base model only.")
    p.add_argument("--use-bodhi", action="store_true",
                   help="Start the session with the BODHI wrapper active.")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    args = p.parse_args()

    banner = f"Loading {args.model}"
    if args.lora_path:
        banner += f" + LoRA from {args.lora_path}"
    print(banner)
    model, tokenizer = load_model(args.model, args.lora_path)

    wrapper = make_wrapper(model, tokenizer, args.max_new_tokens) if args.use_bodhi else None
    use_bodhi = args.use_bodhi
    max_new_tokens = args.max_new_tokens
    history = []

    print(HELP)
    print()

    tag_kind = "lora" if args.lora_path else "base"

    while True:
        prompt_tag = f"[{tag_kind}, bodhi:{'on' if use_bodhi else 'off'}]"
        try:
            user_input = input(f"{prompt_tag} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue

        if user_input in (":q", ":quit", ":exit"):
            break
        if user_input in (":help", ":h", ":?"):
            print(HELP)
            continue
        if user_input == ":reset":
            history = []
            print("(history cleared)")
            continue
        if user_input == ":bodhi":
            use_bodhi = not use_bodhi
            # lazy-init: only build the wrapper the first time it's turned on
            if use_bodhi and wrapper is None:
                wrapper = make_wrapper(model, tokenizer, max_new_tokens)
            print(f"(bodhi wrapper: {'on' if use_bodhi else 'off'})")
            continue
        if user_input.startswith(":tokens "):
            try:
                max_new_tokens = int(user_input.split()[1])
                # rebuild the wrapper so its internal chat_fn uses the new limit
                if wrapper is not None:
                    wrapper = make_wrapper(model, tokenizer, max_new_tokens)
                print(f"(max_new_tokens: {max_new_tokens})")
            except (ValueError, IndexError):
                print("usage: :tokens N")
            continue

        history.append({"role": "user", "content": user_input})

        if use_bodhi:
            # BODHI accepts either a string or a messages list; pass the full
            # history so multi-turn context reaches the two-pass reasoning.
            # Print the analysis separately so it's obvious what the wrapper
            # thought about before the final response.
            resp = wrapper.complete(history)
            if resp.analysis:
                preview = resp.analysis if len(resp.analysis) < 600 else resp.analysis[:600] + "..."
                print(f"\n[analysis] {preview}\n")
            response_text = resp.content
        else:
            response_text = generate(model, tokenizer, history, max_new_tokens)

        print(response_text)
        print()
        history.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
