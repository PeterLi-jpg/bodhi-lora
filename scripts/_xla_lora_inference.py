"""TPU-LoRA Stage-4 backend: merge LoRA into base on CPU, then serve via vllm-tpu.

vllm/vllm-tpu raises ``NotImplementedError`` from ``add_lora`` (see
scripts/_vllm_engine.py:142-160 fail-fast guard). The previous version
of this module worked around that by running ``model.generate()`` directly
on XLA, but ``model.to(xla_device)`` REPLICATES the 27B base on every chip
— v6e-8 has 32 GB HBM per chip, so the 54 GB model OOMs immediately
("RESOURCE_EXHAUSTED: Attempting to allocate 220.50M. There are 23.10M free").

This version takes the merge-then-serve route instead:

  1. Load base + LoRA on **CPU** (TPU VMs have ~250-400 GB host RAM).
  2. ``merge_and_unload`` fuses the LoRA delta into base weights.
  3. ``save_pretrained`` to a temp dir on disk (~54 GB safetensors).
  4. Hand that path to a regular ``VLLMEngine`` — no ``lora_path``, so
     vllm-tpu's ``add_lora`` is never called and the engine shards the
     merged model across the 8 chips like any base-model run.
  5. On ``stop()``, tear down the engine and ``rmtree`` the temp dir.

Trade-offs:

- Disk: ~54 GB merged checkpoint per ``XLALoRAEngine`` instance. The
  bohdi-cache-eur4a / bohdi-cache-use1d persistent SSDs (300 GB) host
  these comfortably; on a stock 100 GB boot disk it fits but is tight.
- CPU peak RAM: ~110 GB during ``merge_and_unload`` (base + adapter +
  merged copy held simultaneously). v6e-8 host RAM is well above this.
- Merge wall: ~30 s compute + 100-200 s disk write = ~3 min one-time.
- Inference wall: vllm speed (continuous batching, paged KV cache) —
  same as base-model serving, not a slow direct-XLA path.

Used by ``scripts/eval_healthbench.py`` when
``_detect_accelerator() == "tpu" and lora_path is not None``.
"""

import gc
import os
import shutil
import sys
import tempfile
from typing import List, Optional, Tuple


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)
from _vllm_engine import VLLMEngine  # noqa: E402


class XLALoRAEngine:
    """Merge LoRA into base on CPU, serve the merged checkpoint via VLLMEngine.

    API matches the subset of VLLMEngine that scripts/eval_healthbench.py
    uses: ``__enter__`` / ``__exit__`` / ``chat`` / ``chat_with_logprobs``.
    Concurrency works the same as the underlying VLLMEngine (vllm-tpu
    handles batching), so callers do not need to special-case
    ``EVAL_CONCURRENCY``.
    """

    def __init__(
        self,
        model: str,
        tp_size: Optional[int] = None,
        max_model_len: int = 4096,
        lora_path: Optional[str] = None,
        port: int = 8000,
        hf_token: Optional[str] = None,
        enforce_eager: bool = True,
        merged_dir: Optional[str] = None,
    ):
        if not lora_path:
            raise ValueError(
                "XLALoRAEngine requires lora_path. Use VLLMEngine for base "
                "models — vllm-tpu serves them on TPU without issue."
            )
        self.base_model = model
        self.lora_path = os.path.realpath(lora_path)
        # Forwarded to the inner VLLMEngine on the merged model.
        self.tp_size = tp_size
        self.max_model_len = max_model_len
        self.port = port
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.enforce_eager = enforce_eager
        # Where to write the merged checkpoint. Default = $TMPDIR/bodhi_merged_*
        # which is usually on the boot disk; pass ``merged_dir=/mnt/cache/...``
        # to put it on the persistent SSD when one is mounted (saves the boot
        # disk and survives preemption-resume).
        self._merged_dir_root = merged_dir
        self._merged_path: Optional[str] = None
        self._inner: Optional[VLLMEngine] = None

    # ── lifecycle ───────────────────────────────────────────────────────────

    def __enter__(self) -> "XLALoRAEngine":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def start(self) -> "XLALoRAEngine":
        self._merged_path = self._merge_to_disk()
        # vllm-tpu serves the merged path as a plain HF checkpoint. No
        # lora_path means add_lora is never called → no NotImplementedError,
        # full FSDP/SPMD sharding kicks in normally.
        self._inner = VLLMEngine(
            self._merged_path,
            tp_size=self.tp_size,
            max_model_len=self.max_model_len,
            lora_path=None,
            port=self.port,
            hf_token=self.hf_token,
            enforce_eager=self.enforce_eager,
        )
        self._inner.start()
        return self

    def stop(self) -> None:
        if self._inner is not None:
            try:
                self._inner.stop()
            finally:
                self._inner = None
        # Free the ~54 GB merged checkpoint. We re-merge on every start()
        # because the LoRA + base inputs may have changed between calls.
        if self._merged_path and os.path.isdir(self._merged_path):
            shutil.rmtree(self._merged_path, ignore_errors=True)
            print(f"  XLALoRA merged checkpoint cleaned up: {self._merged_path}",
                  flush=True)
        self._merged_path = None

    # ── merge step ──────────────────────────────────────────────────────────

    def _merge_to_disk(self) -> str:
        """Load base + adapter on CPU, merge, save. Returns the merged dir."""
        # Lazy imports — torch/transformers/peft are heavy and we only want
        # to pay that cost on the TPU host, not at module import time.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token

        # tempfile.mkdtemp returns an absolute path.
        merged_dir = tempfile.mkdtemp(
            prefix="bodhi_merged_",
            dir=self._merged_dir_root or os.path.expanduser("~"),
        )
        print(
            f"XLALoRA: merging adapter at {self.lora_path} into "
            f"{self.base_model} on CPU (peak ~110 GB host RAM, "
            f"writes ~54 GB to {merged_dir})...",
            flush=True,
        )

        # CPU load. dtype=bf16 keeps memory usage close to model size.
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        peft_model = PeftModel.from_pretrained(base, self.lora_path)
        merged = peft_model.merge_and_unload()

        # safe_serialization writes safetensors shards (vllm-tpu loads them).
        merged.save_pretrained(merged_dir, safe_serialization=True)

        # The training pipeline writes the tokenizer alongside the adapter
        # (see train_lora.py's _tokenizer.save_pretrained call). vllm-tpu
        # expects tokenizer files to live with the model directory, so copy
        # them through. AutoTokenizer.from_pretrained reads the lora_path
        # version which is the one trained against.
        tokenizer = AutoTokenizer.from_pretrained(self.lora_path)
        tokenizer.save_pretrained(merged_dir)

        # Free CPU memory before vLLM brings up its own copy on TPU.
        del base, peft_model, merged, tokenizer
        gc.collect()

        print(f"XLALoRA: merged checkpoint ready at {merged_dir}",
              flush=True)
        return merged_dir

    # ── public API (forwarded to inner VLLMEngine) ──────────────────────────

    def chat(
        self,
        messages: list,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        return self._inner.chat(
            messages, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p,
        )

    def chat_with_logprobs(
        self,
        messages: list,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> Tuple[str, List[float]]:
        return self._inner.chat_with_logprobs(
            messages, max_new_tokens=max_new_tokens, temperature=temperature,
        )
