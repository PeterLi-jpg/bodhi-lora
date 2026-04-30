"""Direct transformers + PEFT inference on TPU XLA.

Drop-in replacement for ``VLLMEngine`` when running on TPU with a LoRA
adapter — vllm/vllm-tpu raises ``NotImplementedError`` from ``add_lora``
(see scripts/_vllm_engine.py:142-160 fail-fast guard). This backend
loads the base model + adapter directly via transformers + PEFT and
runs ``model.generate()`` on the XLA device.

Trade-offs vs VLLMEngine docker mode:

- No paged KV cache or continuous batching — concurrent calls serialize
  on the single XLA device. ``EVAL_CONCURRENCY=1`` is enforced
  internally; callers do NOT need to change the ThreadPoolExecutor max
  workers value.
- Per-prompt latency is dominated by XLA graph compile on the first
  call; subsequent calls hit the persistent compile cache
  (``~/.xla_cache``) and are much faster.
- ``chat_with_logprobs`` returns an empty logprob list — output_scores
  on Gemma-3 is impractical (vocab=262144, materializes ~1 GB per
  prompt). Downstream code already treats logprobs=None as "metric
  unavailable" (see ``score_response_confidence``).

Used only by ``scripts/eval_healthbench.py`` when
``_detect_accelerator() == "tpu" and lora_path is not None``. The
``VLLMEngine`` fail-fast guard prevents accidental Docker fallback.
"""

import os
import sys
import threading
from typing import List, Optional, Tuple


_LOAD_LOCK = threading.Lock()


class XLALoRAEngine:
    """API-compatible subset of ``VLLMEngine`` for the TPU+LoRA path.

    Implements:
      - ``__enter__`` / ``__exit__`` (context manager)
      - ``chat(messages, max_new_tokens=, temperature=, top_p=)`` -> str
      - ``chat_with_logprobs(messages, max_new_tokens=, temperature=)``
        -> (str, []) — returns empty logprob list (see module docstring).

    All concurrent ``chat`` calls serialize on a single internal lock
    because the XLA device is shared and ``model.generate()`` is not
    safe to call from multiple threads simultaneously.
    """

    def __init__(
        self,
        model: str,
        tp_size: Optional[int] = None,        # ignored, accepted for parity
        max_model_len: int = 4096,            # ignored, accepted for parity
        lora_path: Optional[str] = None,
        port: int = 8000,                     # ignored, accepted for parity
        hf_token: Optional[str] = None,
        enforce_eager: bool = True,           # ignored, accepted for parity
    ):
        if not lora_path:
            # XLA backend exists specifically to bridge the LoRA gap on
            # vllm-tpu. Without an adapter, the regular VLLMEngine docker
            # path is faster — refuse to pretend to be a base-model engine.
            raise ValueError(
                "XLALoRAEngine requires lora_path. Use VLLMEngine for base "
                "models — it serves them on TPU without issue."
            )
        self.model_name = model
        self.lora_path = os.path.realpath(lora_path)
        self.max_model_len = max_model_len
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self._tokenizer = None
        self._model = None
        self._device = None

    def __enter__(self) -> "XLALoRAEngine":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self) -> "XLALoRAEngine":
        # Lazy imports — torch/transformers/peft are heavy, and on a
        # non-TPU dev box this module may be imported just to spell
        # out the API. Defer the real load to start().
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch_xla.core.xla_model as xm

        # Match _vllm_engine.py auth: HF_TOKEN env var is used by the HF
        # download path; explicit token kwarg overrides if provided.
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token

        print(
            f"Starting XLA LoRA engine: {self.model_name} + adapter at "
            f"{self.lora_path}"
        )

        # Load tokenizer from the adapter dir if it has one (training writes
        # a tokenizer there); fall back to the base. Mirrors chat.py:24.
        self._tokenizer = AutoTokenizer.from_pretrained(self.lora_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        with _LOAD_LOCK:
            base = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=torch.bfloat16,
            )
            self._model = PeftModel.from_pretrained(base, self.lora_path)
            self._device = xm.xla_device()
            self._model = self._model.to(self._device)
            self._model.eval()
            xm.mark_step()
        print("  XLA LoRA engine ready (first chat() will trigger compile)")
        return self

    def stop(self) -> None:
        # Drop references; let GC reclaim. There is no XLA equivalent of
        # torch.cuda.empty_cache that we need to call here — the next
        # tenant on this TPU just gets a fresh xla_device handle.
        self._model = None
        self._tokenizer = None
        self._device = None
        import gc
        gc.collect()
        print("  XLA LoRA engine stopped")

    # ── public API ──────────────────────────────────────────────────────

    def chat(
        self,
        messages: list,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """Generate a response; return text only.

        Greedy when ``temperature == 0`` (matches VLLMEngine.chat default
        and the rest of the eval pipeline). Sampling enabled otherwise.
        """
        import torch
        import torch_xla.core.xla_model as xm

        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._device)

        # Single-thread the actual XLA op — concurrent generate() calls on
        # one XLA device deadlock or produce wrong results.
        with _LOAD_LOCK:
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0.0,
                    temperature=max(temperature, 1e-5),
                    top_p=top_p,
                )
            xm.mark_step()

        n_prompt = inputs["input_ids"].shape[1]
        new_tokens = out[0, n_prompt:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def chat_with_logprobs(
        self,
        messages: list,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Tuple[str, List[float]]:
        """Generate a response. Returns (text, []) — see module docstring.

        Downstream confidence metrics that take a logprob list will
        compute ``None`` from an empty list (see
        ``eval_healthbench.score_response_confidence``).
        """
        return self.chat(messages, max_new_tokens, temperature), []
