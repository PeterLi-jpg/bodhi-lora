"""MaxText LoRA: JAX/Flax LoRA helpers for the MaxText fork of Stage 3 (Gemma-3 fine-tuning on TPU).

Module aliases re-exported for ``train_lora_maxtext.py``:

  - ``lora_inject``    -> ``scripts.maxtext_lora.injector``
  - ``dataset_loader`` -> ``scripts.maxtext_lora.dataset_loader``
  - ``export_peft``    -> ``scripts.export_maxtext_lora_to_peft``

The PEFT exporter lives as a top-level script; the injector and the
dataset loader live under this package. Re-binding them here keeps
the trainer's import block short and lets us move things later
without touching callers.

Aliases resolve lazily via PEP 562 ``__getattr__`` so importing this package
doesn't drag in transformers / peft / etc. on a CPU dev box that just wants
the LoRA layer or injector. The heavy modules only get imported when a
caller actually does ``from scripts.maxtext_lora import dataset_loader`` or
similar.
"""

from __future__ import annotations

import importlib
from types import ModuleType

_ALIASES = {
    "lora_inject": "scripts.maxtext_lora.injector",
    # Was scripts.convert_traces_to_maxtext (the offline JSONL writer);
    # the trainer needs an iterator builder, which lives in its own
    # module now. Stage 2's offline converter still produces the
    # *.tokenized.jsonl files this loader reads.
    "dataset_loader": "scripts.maxtext_lora.dataset_loader",
    "export_peft": "scripts.export_maxtext_lora_to_peft",
}


def __getattr__(name: str) -> ModuleType:
    """Lazy alias resolver. Raises AttributeError for unknown names so the
    rest of Python's attribute machinery (hasattr, dir, etc.) behaves
    normally."""
    target = _ALIASES.get(name)
    if target is None:
        raise AttributeError(f"module 'scripts.maxtext_lora' has no attribute {name!r}")
    module = importlib.import_module(target)
    globals()[name] = module  # cache so subsequent lookups skip __getattr__
    return module


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *_ALIASES.keys()})
