"""Pytest configuration shared by every test module under tests/.

Several existing tests (test_filter_traces, test_pure_logic,
test_bodhi_ablation) install ``MagicMock`` stand-ins for ``torch``,
``transformers``, and ``peft`` so they can run on dev boxes without
those packages installed.  The mocks are gated on ``if mod not in
sys.modules``, which means once ``torch`` has been imported for real
the mock is *not* installed.

That gating is necessary for tests that DO need real torch (e.g. the
LoRA export round-trip in test_lora_export.py) to coexist with the
mock-using tests.  Pre-importing the real packages here — conftest.py
loads before the test files in the same directory — populates
``sys.modules`` with the real implementations and the mock-installers
silently skip.

If torch / safetensors / peft are genuinely missing from the
environment (CI image without ML deps), the imports fail fast here
and pytest reports a clear collection error instead of the cryptic
``MagicMock has no __version__`` we hit otherwise.
"""

# Imported only for their side effect: populating sys.modules so the
# conditional mock installers in other test files don't override the
# real implementations.
import torch  # noqa: F401
import safetensors.torch  # noqa: F401

try:
    # peft / transformers are heavier and not strictly required for
    # every test; allow them to be missing on minimal CI images.
    import peft  # noqa: F401
    import transformers  # noqa: F401
except ImportError:
    pass
