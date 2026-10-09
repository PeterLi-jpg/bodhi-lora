"""Distill a teaching protocol's epistemic behaviors into LoRA adapters, and measure them.

One experiment unit, the cell (benchmark x model x protocol, over seeds), one
registry of configs, one results root. docs/ARCHITECTURE.md describes the design.

The package is named bodhi_lora, not bodhi, because the installed bodhi-llm
dependency already imports as ``bodhi`` and a repo-root ``bodhi/`` would hide it.
"""
