"""Pytest configuration shared by every test module under tests/.

Several existing tests (test_filter_traces, test_pure_logic,
test_bodhi_ablation) install ``MagicMock`` stand-ins for ``torch``,
``transformers``, and ``peft`` so they can run on dev boxes without
those packages installed. The mocks are gated on ``if mod not in
sys.modules``, which means once ``torch`` has been imported for real
the mock is *not* installed.

That gating is necessary for tests that DO need real torch (e.g. the
LoRA export round-trip in test_lora_export.py) to coexist with the
mock-using tests. Pre-importing the real packages here populates
``sys.modules`` with the real implementations and the mock-installers
silently skip.

On CI (no ML deps installed), the imports gracefully fail and the
mock-using tests proceed unaffected; tests that genuinely need torch
will skip themselves.
"""

# Best-effort import of the real packages; any ImportError just means
# the mock-installers in other test files will run unimpeded.
try:
    import torch  # noqa: F401
    import safetensors.torch  # noqa: F401
except ImportError:
    pass

try:
    import peft  # noqa: F401
    import transformers  # noqa: F401
except ImportError:
    pass
