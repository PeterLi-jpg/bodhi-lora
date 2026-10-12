"""Model serving: vLLM's OpenAI-compatible server as a subprocess, and a CPU stub.

make_engine() is the one way the stages build an engine: from a model record, the
serving settings of the stage, and the adapter and precision of the condition.
"""
from pathlib import Path
from typing import Dict, Optional, Union

from bodhi_lora.engine.stub import Reply, StubEngine
from bodhi_lora.engine.vllm import Completion, ContextLengthError, Route, VLLMEngine, placement_from_env
from bodhi_lora.registry.schema import ModelConfig, Serving

Engine = Union[VLLMEngine, StubEngine]

__all__ = ["Completion", "ContextLengthError", "Engine", "Route", "StubEngine", "VLLMEngine", "make_engine",
           "placement_from_env"]


def make_engine(model: ModelConfig, *, serving: Serving, adapter: Optional[Path] = None,
                adapters: Optional[Dict[str, Path]] = None, lora_rank: Optional[int] = None,
                quantize: bool = False, stub: bool = False, stub_reply: Reply = "STUB RESPONSE",
                enforce_eager: Optional[bool] = None, log_path: Optional[Path] = None) -> Engine:
    """An engine for one model record.

    adapter: one adapter, which every request uses. adapters: name -> adapter folder,
    served next to the base; a request picks one with engine.route(name). lora_rank: the
    adapters' rank (the train block's r), so the server holds it.
    quantize serves the base in 4-bit NF4, the precision a QLoRA adapter was trained on.
    Eager mode (no CUDA graphs): enforce_eager when given (the bench's own setting), else
    serving.enforce_eager when the record sets it, else only when an adapter is loaded.
    stub returns a StubEngine with the same settings, for CPU runs.
    """
    if enforce_eager is None:
        enforce_eager = serving.enforce_eager
    if enforce_eager is None:
        enforce_eager = bool(adapter or adapters)
    settings = dict(revision=model.revision, max_model_len=serving.max_model_len,
                    max_num_seqs=serving.max_num_seqs, request_timeout=serving.request_timeout_s,
                    dtype=model.dtype, quantization="bitsandbytes" if quantize else None,
                    lora_path=str(adapter) if adapter else None,
                    lora_modules={name: str(path) for name, path in (adapters or {}).items()},
                    lora_rank=lora_rank, enforce_eager=enforce_eager, env=dict(model.serving_env))
    engine = VLLMEngine(model.hf_id, log_path=str(log_path) if log_path else None, **settings)
    if stub:
        # The stub reports the settings this server would have (nothing starts), so the
        # manifests of a stub run carry the fields of a GPU run's.
        return StubEngine(model.hf_id, reply=stub_reply, served=engine.settings(), **settings)
    return engine
