"""Generate BOHDI wrapper traces over HealthBench for SFT training data."""

import argparse
import json
import os
import random
import traceback
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# TPU detection — same pattern as train_lora.py and filter_traces.py.
# MedGemma-27B in bfloat16 = 54 GB. A single v4 chip has 32 GB, so the model
# must be spread across 2+ chips. On TPU we load without device_map and let
# the XLA runtime place layers; use device_map="auto" on GPU as before.
try:
    import torch_xla.core.xla_model as _xm
    _ON_TPU = True
except ImportError:
    _xm = None
    _ON_TPU = False

# Fix for transformers DynamicCache bug on XLA/TPU.
#
# In transformers 4.57+, DynamicLayer.lazy_initialization() creates
# self.keys = torch.tensor([]) — a 1D tensor with shape (0,).  The first
# real update() call then does torch.cat([shape_(0,), shape_(B,H,S,D)], dim=-2)
# which fails with a dimension mismatch.  On XLA the error is deferred until
# after graph compilation (~15 min), then raises for every example — all
# traces fail silently and the run produces 0 output.
#
# Fix: replace lazy_initialization to create properly-shaped zero tensors
# (shape [B, H, 0, D]) that torch.cat along dim=-2 can handle correctly.
if _ON_TPU:
    try:
        from transformers.cache_utils import DynamicLayer

        def _patched_lazy_init(self, key_states):
            self.dtype = key_states.dtype
            self.device = key_states.device
            shape = list(key_states.shape)
            shape[-2] = 0  # zero seq length, matching number of dims
            self.keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self.values = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self.is_initialized = True

        DynamicLayer.lazy_initialization = _patched_lazy_init
        print("Applied DynamicLayer.lazy_initialization patch (XLA cache fix)")
    except (ImportError, AttributeError):
        pass  # transformers version without DynamicLayer — no patch needed

DATA_DIR = Path("data/raw")

DATASET_URLS = {
    "healthbench": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl",
    "healthbench_hard": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl",
    "healthbench_consensus": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl",
}


def ensure_downloaded(name):
    """Download dataset to data/raw/ if not already there."""
    import urllib.request
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        print(f"Downloading {name}...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATASET_URLS[name], path)
    return path


def load_healthbench(name):
    path = ensure_downloaded(name)
    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            ex["_source"] = name
            examples.append(ex)
    print(f"  {name}: {len(examples)} examples")
    return examples


def load_multiple_datasets(names):
    all_ex = []
    seen = set()
    for name in names:
        for ex in load_healthbench(name):
            if ex["prompt_id"] not in seen:
                seen.add(ex["prompt_id"])
                all_ex.append(ex)
    print(f"Total unique: {len(all_ex)}")
    return all_ex


def load_exclude_ids(paths):
    """Collect prompt_ids to exclude, from one or more files.

    Accepts:
      - .json  containing a list of ids, or {"prompt_ids": [...]}
      - .jsonl containing one HealthBench example per line (reads
        ``prompt_id`` from each). Useful for excluding an entire dataset,
        e.g. ``data/raw/healthbench_hard.jsonl`` to drop all 1000 Hard
        prompts for the HealthBench-only generalization experiment.

    Accepts a string (one path) or a list of paths.
    """
    if isinstance(paths, str):
        paths = [paths]
    ids = set()
    for path in paths:
        if path.endswith(".jsonl"):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ex = json.loads(line)
                    ids.add(ex["prompt_id"])
        else:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data["prompt_ids"]
            ids.update(data)
    return ids


class LocalModel:
    def __init__(self, model_name, device="auto"):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if _ON_TPU:
            # MedGemma-27B (54 GB bfloat16) exceeds one v6e chip (32 GB).
            # Try SPMD sharding across all chips; fall back to host CPU if
            # the installed torch_xla build doesn't include the sharding API.
            _xs = None
            for _spmd_mod in ("torch_xla.experimental.xla_sharding",
                              "torch_xla.distributed.xla_sharding",
                              "torch_xla.distributed.spmd"):
                try:
                    import importlib as _il
                    _xs = _il.import_module(_spmd_mod)
                    print(f"SPMD module found: {_spmd_mod}")
                    break
                except ModuleNotFoundError:
                    continue

            if _xs is not None:
                from torch_xla import runtime as _xr
                import numpy as _np
                # CRITICAL: load model on CPU BEFORE calling use_spmd().
                # use_spmd() globally intercepts all set_data() calls; if it is
                # active when from_pretrained() runs, every internal weight
                # assignment raises "incompatible tensor type".  Safe order:
                # 1) load to CPU  2) enable SPMD  3) move params to XLA.
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16
                )
                _xr.use_spmd()
                # global_device_count() returns 1 in SPMD mode (1 virtual device).
                # addressable_device_count() returns the physical chip count (8 on v6e-8).
                # mark_sharding() internally uses addressable_device_count, so the mesh
                # must match that number or we get "not mappable over N devices".
                _n_dev = getattr(_xr, 'addressable_device_count', _xr.global_device_count)()
                if _n_dev < 2:
                    # Last resort: count VFIO groups = physical chips
                    import os as _os
                    _vfio = [d for d in _os.listdir('/dev/vfio') if d.isdigit()] if _os.path.exists('/dev/vfio') else []
                    _n_dev = len(_vfio) or _n_dev
                _device_ids = _np.arange(_n_dev)
                _mesh = _xs.Mesh(_device_ids, (_n_dev,), ("tp",))
                _dev = _xm.xla_device()
                print(f"SPMD: sharding model across {_n_dev} chips")
                # Replace module._parameters entries with new nn.Parameter objects
                # wrapping XLA tensors.  Direct dict assignment avoids set_data()
                # entirely (which SPMD blocks even for the post-use_spmd path).
                import torch.nn as _nn
                for _mod in self.model.modules():
                    for _pname, _p in list(_mod._parameters.items()):
                        if _p is not None:
                            _xp = _nn.Parameter(
                                _p.data.to(_dev), requires_grad=_p.requires_grad
                            )
                            if _xp.dim() == 2 and _xp.shape[0] > 1024:
                                _xs.mark_sharding(_xp, _mesh, (0, None))
                            _mod._parameters[_pname] = _xp
                    for _bname, _b in list(_mod._buffers.items()):
                        if _b is not None:
                            _mod._buffers[_bname] = _b.to(_dev)
                _xm.mark_step()
                self._device = _dev
            else:
                # CPU fallback -- v6e-8 VM has 384 GB RAM so 27B fits
                import os
                print("SPMD unavailable in this torch_xla build; using host CPU")
                os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, device_map="cpu"
                )
                self._device = torch.device("cpu")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map=device,
            )
            self._device = next(self.model.parameters()).device

        self.model.eval()

    def generate(self, messages, max_new_tokens=1024):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Tokenize on CPU so we can pad short prompts before handing to XLA.
        inputs = self.tokenizer(text, return_tensors="pt")
        if _ON_TPU:
            # XLA compiles a new graph for every distinct input shape, and each
            # compile takes 20-30 min for a 27B SPMD model.  Pad ALL prompts to
            # one fixed length so the entire run shares a single prefill graph.
            # 2048 covers all HealthBench prompts (typically 500-1500 tokens) and
            # compiles ~4x faster than 4096 (attention is O(n^2) in sequence len).
            _fixed_len = 2048
            _seq_len = inputs["input_ids"].shape[1]
            if _seq_len < _fixed_len:
                _pad = _fixed_len - _seq_len
                _pad_id = self.tokenizer.pad_token_id or 0
                _pad_ids = torch.full((1, _pad), _pad_id, dtype=inputs["input_ids"].dtype)
                _pad_mask = torch.zeros((1, _pad), dtype=inputs["attention_mask"].dtype)
                inputs["input_ids"] = torch.cat([_pad_ids, inputs["input_ids"]], dim=1)
                inputs["attention_mask"] = torch.cat([_pad_mask, inputs["attention_mask"]], dim=1)
        prompt_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if _ON_TPU:
            _xm.mark_step()
        return self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def make_bodhi_wrapper(model):
    """Set up BODHI wrapper once, reuse across examples."""
    from bodhi import BODHI, BODHIConfig
    chat_fn = lambda msgs: model.generate(msgs)
    return BODHI(chat_function=chat_fn, config=BODHIConfig(domain="medical"))


def generate_response(model, messages, use_bodhi, bodhi_wrapper=None):
    """Return {content, analysis, metadata}. analysis/metadata are None for
    non-BODHI runs so callers get a stable schema.

    Per Sebastian: saving analysis + metadata lets us audit *why* the model
    decided what it did, not just what it said — critical for finding where
    the humility wrapper went wrong on specific examples.
    """
    if not use_bodhi:
        return {"content": model.generate(messages), "analysis": None, "metadata": None}
    resp = bodhi_wrapper.complete(messages)
    return {
        "content": resp.content,
        "analysis": resp.analysis,
        "metadata": resp.metadata,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/medgemma-27b-text-it")
    parser.add_argument("--datasets", nargs="+", default=["healthbench_hard", "healthbench"],
                        choices=list(DATASET_URLS.keys()))
    parser.add_argument("--exclude-ids", nargs="+", default=None,
                        help="one or more files listing prompt_ids to skip. "
                             "Accepts .json (list or {prompt_ids: [...]}) and "
                             ".jsonl (reads prompt_id from each row). For the "
                             "HealthBench-only generalization experiment, pass "
                             "data/raw/healthbench_hard.jsonl here to drop all "
                             "1000 Hard prompts from training.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-bodhi", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-resume", action="store_true",
                        help="skip the model/bodhi metadata consistency check on resume "
                             "(issue #6/7). Only use when you intentionally want to "
                             "append rows generated with different settings.")
    args = parser.parse_args()

    # Greedy decoding is deterministic without a seed, but the BODHI wrapper
    # may use sampling internally (prompt shuffling, tie-breaking) — seed so
    # reruns on identical hardware produce identical raw_traces.jsonl.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    examples = load_multiple_datasets(args.datasets)

    if args.exclude_ids:
        exclude = load_exclude_ids(args.exclude_ids)
        before = len(examples)
        examples = [ex for ex in examples if ex["prompt_id"] not in exclude]
        print(f"Excluded {before - len(examples)} eval examples, {len(examples)} left")

    if args.max_examples:
        examples = examples[:args.max_examples]

    done_ids = set()
    if args.resume_from and Path(args.resume_from).exists():
        # Issue #6/7: validate that existing rows were generated with the
        # same model + bodhi setting. Mixing settings silently corrupts the
        # training corpus. Refuse to resume on mismatch unless --force-resume.
        prev_models = set()
        prev_bodhi = set()
        with open(args.resume_from) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done_ids.add(row["prompt_id"])
                    if "model" in row:
                        prev_models.add(row["model"])
                    if "bodhi" in row:
                        prev_bodhi.add(row["bodhi"])
                except (json.JSONDecodeError, KeyError):
                    pass  # skip corrupt lines from interrupted runs

        model_mismatch = prev_models and prev_models != {args.model}
        bodhi_mismatch = prev_bodhi and prev_bodhi != {args.use_bodhi}
        if model_mismatch or bodhi_mismatch:
            msg = (
                f"resume config mismatch — refusing to append to {args.resume_from}\n"
                f"  existing rows: model={prev_models or '{unknown}'} "
                f"bodhi={prev_bodhi or '{unknown}'}\n"
                f"  this run:      model={{{args.model!r}}} bodhi={{{args.use_bodhi}}}\n"
                f"  re-running with different settings would silently mix "
                f"outputs (issue #6/7)\n"
                f"  if you truly want to append across settings, pass --force-resume"
            )
            if not args.force_resume:
                raise SystemExit(msg)
            print(f"WARNING: {msg}\n(continuing because --force-resume was passed)")

        examples = [ex for ex in examples if ex["prompt_id"] not in done_ids]
        print(f"Resuming, skipping {len(done_ids)} already done "
              f"(prev model={prev_models or '?'} bodhi={prev_bodhi or '?'})")

    print(f"\nGenerating {len(examples)} traces, bodhi={args.use_bodhi}\n")
    model = LocalModel(args.model)

    bodhi_wrapper = make_bodhi_wrapper(model) if args.use_bodhi else None

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if done_ids else "w"
    ok, fail = 0, 0

    with open(out_path, mode) as f:
        for ex in tqdm(examples):
            try:
                out = generate_response(model, ex["prompt"], args.use_bodhi, bodhi_wrapper)
                trace = {
                    "prompt_id": ex["prompt_id"],
                    "messages": ex["prompt"],
                    "response": out["content"],
                    "bodhi_analysis": out["analysis"],
                    "bodhi_metadata": out["metadata"],
                    "tags": ex.get("example_tags", []),
                    "source_dataset": ex.get("_source", "unknown"),
                    "model": args.model,
                    "bodhi": args.use_bodhi,
                }
                f.write(json.dumps(trace) + "\n")
                f.flush()
                ok += 1
            except Exception as e:
                # Full traceback helps distinguish OOM from tokenizer/BODHI bugs
                # when a 48h run has a few failures we want to diagnose later.
                traceback.print_exc()
                print(f"  Error on {ex['prompt_id']}: {e}")
                fail += 1

    print(f"\nDone: {ok} ok, {fail} failed -> {out_path}")


if __name__ == "__main__":
    main()
