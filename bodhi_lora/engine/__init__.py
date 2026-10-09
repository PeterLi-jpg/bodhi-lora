"""Model serving: vLLM's OpenAI-compatible server as a subprocess, and a CPU stub.

make_engine() is the one way the stages build an engine: from a model record, the
serving settings of the stage, and the adapter and precision of the condition.
"""
from pathlib import Path
from typing import Optional, Union

from bodhi_lora.engine.stub import Reply, StubEngine
from bodhi_lora.engine.vllm import Completion, ContextLengthError, VLLMEngine, placement_from_env
from bodhi_lora.registry.schema import ModelConfig, Serving

Engine = Union[VLLMEngine, StubEngine]

__all__ = ["Completion", "ContextLengthError", "Engine", "StubEngine", "VLLMEngine", "make_engine",
           "placement_from_env"]


def make_engine(model: ModelConfig, *, serving: Serving, adapter: Optional[Path] = None,
                quantize: bool = False, stub: bool = False, stub_reply: Reply = "STUB RESPONSE",
                enforce_eager: bool = True, log_path: Optional[Path] = None) -> Engine:
    """An engine for one model record.

    quantize serves the base in 4-bit NF4, the precision a QLoRA adapter was trained on.
    stub returns a StubEngine with the same settings, for CPU runs.
    """
    settings = dict(revision=model.revision, max_model_len=serving.max_model_len,
                    max_num_seqs=serving.max_num_seqs, request_timeout=serving.request_timeout_s,
                    dtype=model.dtype, quantization="bitsandbytes" if quantize else None,
                    lora_path=str(adapter) if adapter else None, enforce_eager=enforce_eager)
    if stub:
        return StubEngine(model.hf_id, reply=stub_reply, **settings)
    return VLLMEngine(model.hf_id, log_path=str(log_path) if log_path else None, **settings)
