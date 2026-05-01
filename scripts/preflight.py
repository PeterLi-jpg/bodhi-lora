"""Pre-flight check: validate env before the expensive pipeline stages run.

Run this at the top of every slurm script (and smoke.sh) so any
environment problem — missing package, bad HF_TOKEN, missing GPU, gated
model terms not accepted — surfaces in seconds instead of after model
downloads or 48 hours of generation.

Usage:
    python scripts/preflight.py [--models MODEL [MODEL ...]]

The model list defaults to the production targets (medgemma-27b-text-it
and Qwen2.5-14B-Instruct-AWQ); smoke.sh passes its own smaller models.
"""

import argparse
import importlib
import os
import sys
from typing import List


# TPU detection mirrors scripts/_vllm_engine.py:_detect_accelerator and
# scripts/train_lora.py:_ON_TPU. setup_tpu.sh deliberately does NOT install
# autoawq / bitsandbytes on TPU hosts (they are CUDA-only), so the GPU-flavored
# preflight checks below need to skip them when we are on a TPU.
def _on_tpu() -> bool:
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        return True
    if os.path.isdir("/dev/vfio") and any(
        d.isdigit() for d in os.listdir("/dev/vfio")
    ):
        return True
    try:
        import torch_xla  # noqa: F401
        return True
    except ImportError:
        return False


_ON_TPU = _on_tpu()


# Deps that must import for the main scripts to even start.
# vllm is required on every host that runs inference (CUDA build on GPU,
# vLLM-TPU Docker on TPU); without it generate_traces / eval_healthbench
# crash 5+ minutes in. Keep it unconditional so preflight catches the
# missing wheel up-front.
REQUIRED_IMPORTS = [
    "torch", "transformers", "peft", "trl", "datasets",
    "bodhi", "timm", "PIL", "rich", "yaml", "numpy", "tqdm",
    "accelerate", "huggingface_hub", "vllm",
]

# autoawq is CUDA-built and platform-gated in requirements.txt to
# `sys_platform == "linux"`. On TPU hosts setup_tpu.sh skips it
# intentionally, so don't require it there. Module name is `awq`
# (pip name is autoawq).
if sys.platform == "linux" and not _ON_TPU:
    REQUIRED_IMPORTS.append("awq")


def check_imports() -> List[str]:
    failed = []
    for mod in REQUIRED_IMPORTS:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            failed.append(f"  {mod}: {e}")
    return failed


def check_hf_token() -> List[str]:
    if not os.environ.get("HF_TOKEN"):
        return ["  HF_TOKEN env var is not set (gated models will 401)"]
    return []


def check_hf_access(models: List[str]) -> List[str]:
    """Validate (a) HF_TOKEN works, (b) every model listed is accessible.

    model_info() is a metadata-only API call — no weights are downloaded.
    """
    failed = []
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        return [f"  huggingface_hub import failed: {e}"]

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        who = api.whoami()
    except Exception as e:
        return [f"  whoami failed (bad HF_TOKEN?): {e}"]

    for m in models:
        try:
            api.model_info(m)
        except Exception as e:
            failed.append(f"  {m}: {e} (accept terms at https://huggingface.co/{m} as {who['name']})")
    return failed


def check_accelerator() -> List[str]:
    """Validate the accelerator the host actually has.

    Linux + TPU: torch_xla must import and an XLA device must be visible.
    Linux + GPU: CUDA must be available and >=1 device visible.
    macOS: warn-only (MPS).
    """
    try:
        import torch
    except ImportError as e:
        return [f"  torch import failed: {e}"]

    if sys.platform == "darwin":
        # local dev; MPS is fine for smoke
        if torch.backends.mps.is_available():
            return []
        return ["  (warning) MPS not available on Mac; smoke will be CPU-only and slow"]

    if _ON_TPU:
        try:
            import torch_xla  # noqa: F401
            import torch_xla.core.xla_model as xm
        except ImportError as e:
            return [f"  TPU host detected but torch_xla import failed: {e}"]
        try:
            _ = xm.xla_device()
        except Exception as e:
            return [f"  torch_xla imported but xla_device() failed (libtpu missing?): {e}"]
        return []

    # Linux GPU expected.
    if not torch.cuda.is_available():
        return ["  CUDA is not available (but we're on Linux — likely GPU allocation missing)"]
    if torch.cuda.device_count() == 0:
        return ["  CUDA present but 0 devices visible"]
    return []


def print_env_summary():
    """Emit info useful in slurm/TPU logs for debugging later."""
    try:
        import torch
        accel_info = f"cuda={torch.cuda.is_available()}"
        if _ON_TPU:
            try:
                import torch_xla
                accel_info += f"  xla={torch_xla.__version__}"
            except ImportError:
                accel_info += "  xla=missing"
        print(f"  torch        {torch.__version__}  {accel_info}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"    gpu{i}: {p.name}  {p.total_memory / 1e9:.1f} GB")
        import transformers, peft, trl, accelerate
        print(f"  transformers {transformers.__version__}")
        print(f"  peft         {peft.__version__}")
        print(f"  trl          {trl.__version__}")
        print(f"  accelerate   {accelerate.__version__}")
    except Exception as e:
        print(f"  (could not print env summary: {e})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--models", nargs="+",
        default=["google/medgemma-27b-text-it", "Qwen/Qwen2.5-14B-Instruct-AWQ"],
        help="HF model ids to verify access to. Default: production targets.",
    )
    parser.add_argument(
        "--skip-hf-access", action="store_true",
        help="Skip the HF model-info check (offline / air-gapped runs).",
    )
    args = parser.parse_args()

    print("=== preflight ===")
    errors = []

    missing = check_imports()
    if missing:
        errors.append("import failures:\n" + "\n".join(missing))

    if not args.skip_hf_access:
        token_errs = check_hf_token()
        if token_errs:
            errors.append("HF_TOKEN:\n" + "\n".join(token_errs))
        else:
            access_errs = check_hf_access(args.models)
            if access_errs:
                errors.append("HF access:\n" + "\n".join(access_errs))
    else:
        print("  skipping HF_TOKEN and model-access checks")

    accel_errs = check_accelerator()
    if accel_errs:
        # treat Mac MPS/warnings separately from hard accelerator failures
        is_warn_only = all("(warning)" in e for e in accel_errs)
        if is_warn_only:
            print("\n".join(accel_errs))
        else:
            label = "TPU" if _ON_TPU else "GPU"
            errors.append(f"{label}:\n" + "\n".join(accel_errs))

    if errors:
        print("\n=== PREFLIGHT FAILED ===")
        for e in errors:
            print(e)
            print()
        sys.exit(1)

    print("\n=== env summary ===")
    print_env_summary()
    print("\n=== preflight OK ===")


if __name__ == "__main__":
    main()
