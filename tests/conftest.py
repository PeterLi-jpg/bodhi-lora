"""Pytest configuration shared by every test module under tests/.

The real ML packages are imported once here, before any test module runs. A test that
stubs a module with ``monkeypatch.setitem(sys.modules, ...)`` then cannot leave a stub
behind that a later test mistakes for the real package. On a machine without the ML
packages the imports fail quietly, and the tests that need them skip themselves with
``pytest.importorskip``.
"""

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
