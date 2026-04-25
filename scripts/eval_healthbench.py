"""Evaluate model on HealthBench Hard (base vs lora, with/without BOHDI wrapper)."""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
import random
import subprocess
import urllib.request

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel

# make sure we can import from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.filter_traces import GRADER_TEMPLATE, LocalGrader, parse_json_response, grade_trace

# TPU detection — same pattern used across the pipeline.
try:
    import torch_xla.core.xla_model as _xm
    _ON_TPU = True
except ImportError:
    _xm = None
    _ON_TPU = False

# XLA persistent compile cache — same pattern as generate_traces.py.
# Stage 4 runs eval 4× per seed; persistent cache means each LoRA-vs-base
# graph variant is compiled at most once across all those runs.
if _ON_TPU:
    try:
        _xla_cache = os.path.expanduser("~/.xla_cache")
        os.makedirs(_xla_cache, exist_ok=True)
        from torch_xla import runtime as _xr_cache
        if hasattr(_xr_cache, "initialize_cache"):
            _xr_cache.initialize_cache(_xla_cache, readonly=False)
            print(f"XLA persistent compile cache: {_xla_cache}")
    except Exception as _e:
        print(f"XLA persistent cache unavailable ({_e!r}); compiles will not be saved.")

# Fix for transformers DynamicCache bug on XLA/TPU — same patch as
# generate_traces.py; see that file for the full explanation.
if _ON_TPU:
    try:
        from transformers.cache_utils import DynamicLayer

        def _patched_lazy_init(self, key_states):
            self.dtype = key_states.dtype
            self.device = key_states.device
            shape = list(key_states.shape)
            shape[-2] = 0
            self.keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self.values = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self.is_initialized = True

        DynamicLayer.lazy_initialization = _patched_lazy_init
        print("Applied DynamicLayer.lazy_initialization patch (XLA cache fix)")
    except (ImportError, AttributeError):
        pass

HEALTHBENCH_HARD_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
DATA_DIR = Path("data/raw")


def load_eval_data(sample_ids_path):
    path = DATA_DIR / "healthbench_hard.jsonl"
    if not path.exists():
        print("Downloading HealthBench Hard...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HEALTHBENCH_HARD_URL, path)

    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))

    with open(sample_ids_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data["prompt_ids"]
    eval_ids = set(data)

    filtered = [ex for ex in examples if ex["prompt_id"] in eval_ids]
    print(f"{len(filtered)} eval examples loaded")
    return filtered


def load_model(model_name, lora_path=None):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Gemma chat template emits {{ bos_token }} itself; default tokenize()
    # would prepend another <bos> on top, producing "<bos><bos>...".  Skip
    # auto-BOS; the template provides it.
    if getattr(tokenizer, "add_bos_token", False):
        tokenizer.add_bos_token = False

    if _ON_TPU:
        _xs = None
        for _spmd_mod in ("torch_xla.experimental.xla_sharding",
                          "torch_xla.distributed.xla_sharding",
                          "torch_xla.distributed.spmd"):
            try:
                import importlib as _il
                _xs = _il.import_module(_spmd_mod)
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
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16
            )
            _xr.use_spmd()
            # global_device_count() returns 1 in SPMD mode (1 virtual device).
            # addressable_device_count() returns the physical chip count (8 on v6e-8).
            _n_dev = getattr(_xr, 'addressable_device_count', _xr.global_device_count)()
            if _n_dev < 2:
                import os as _os
                _vfio = [d for d in _os.listdir('/dev/vfio') if d.isdigit()] if _os.path.exists('/dev/vfio') else []
                _n_dev = len(_vfio) or _n_dev
            _device_ids = _np.arange(_n_dev)
            _mesh = _xs.Mesh(_device_ids, (_n_dev,), ("tp",))
            _dev = _xm.xla_device()
            print(f"SPMD: sharding model across {_n_dev} chips")
            import torch.nn as _nn
            for _mod in model.modules():
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
            _device = _dev
        else:
            import os
            print("SPMD unavailable; using host CPU for eval")
            os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="cpu"
            )
            _device = torch.device("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        _device = next(model.parameters()).device

    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}")
        # NOTE on TPU: do NOT call merge_and_unload().  merge_and_unload
        # replaces LoRA-wrapped Linears with plain Linears holding merged
        # weights (base + scale * B @ A) — which destroys the SPMD sharding
        # annotations on the base weights and gives every chip a full 54 GB
        # replica → OOM.  The PEFT-wrapped model produces the same outputs as
        # a merged model anyway (just one extra small matmul per linear).
        #
        # Also: PEFT constructs the new lora_A / lora_B Linear modules with
        # default device = CPU, then loads the adapter weights into them.
        # Forward would then mix CPU adapter with XLA base → device error.
        # Move the adapter params onto the XLA device explicitly after load.
        model = PeftModel.from_pretrained(model, lora_path)
        if _ON_TPU and _device is not None:
            import torch.nn as _nn
            for _name, _p in list(model.named_parameters()):
                # Adapter param names contain 'lora_' (lora_A.default.weight,
                # lora_B.default.weight, lora_embedding_*, etc.).
                if "lora_" in _name and _p is not None and _p.device != _device:
                    # Replace the parameter on its parent module so the new
                    # XLA tensor is what forward() sees.
                    _parent_path, _, _attr = _name.rpartition(".")
                    _parent = model.get_submodule(_parent_path)
                    _xp = _nn.Parameter(_p.data.to(_device),
                                        requires_grad=_p.requires_grad)
                    setattr(_parent, _attr, _xp)
            _xm.mark_step()
        # Don't merge — we keep the PEFT wrapper so the SPMD-sharded base
        # weights stay sharded.

    model.eval()
    return model, tokenizer, _device


def _tpu_pad_short_inputs(inputs, tokenizer, fixed_len=4096):
    """Pad inputs to a single fixed length on CPU before moving to XLA.

    XLA compiles a new graph for every distinct input shape; for a 27B SPMD
    model each compile takes 20-30 min.  Padding everything to one fixed
    length means one prefill compile covers ALL eval examples.  4096 (vs the
    previous 2048) has the same redundancy budget the trace generator does:
    BOHDI two-pass response-prompts can hit ~3000 tokens, and a single
    overflow blows ~30 minutes on a fresh compile.
    """
    seq_len = inputs["input_ids"].shape[1]
    if seq_len > fixed_len:
        # Don't silently truncate; surface so the cap can be bumped.  The
        # fresh compile is bounded — only fires for the first overflow.
        print(f"WARN: eval input ({seq_len} tok) exceeds fixed pad ({fixed_len}); "
              f"this triggers a fresh XLA compile (~30 min).")
        return inputs
    if seq_len == fixed_len:
        return inputs
    pad = fixed_len - seq_len
    pad_id = tokenizer.pad_token_id or 0
    pad_ids = torch.full((1, pad), pad_id, dtype=inputs["input_ids"].dtype)
    pad_mask = torch.zeros((1, pad), dtype=inputs["attention_mask"].dtype)
    return {
        "input_ids": torch.cat([pad_ids, inputs["input_ids"]], dim=1),
        "attention_mask": torch.cat([pad_mask, inputs["attention_mask"]], dim=1),
    }


def make_bodhi_wrapper(model, tokenizer, device, max_new_tokens=1024):
    """Build a reusable BODHI wrapper around the given model."""
    from bodhi import BODHI, BODHIConfig

    def chat_fn(msgs):
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        if _ON_TPU:
            inputs = _tpu_pad_short_inputs(inputs, tokenizer)
        prompt_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
        if _ON_TPU:
            _gen_kwargs["cache_implementation"] = "static"
        with torch.no_grad():
            out = model.generate(**inputs, **_gen_kwargs)
        return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

    return BODHI(chat_function=chat_fn, config=BODHIConfig(domain="medical"))


def gen_response(model, tokenizer, device, messages, use_bodhi, bodhi_wrapper=None, max_new_tokens=1024):
    if not use_bodhi:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        if _ON_TPU:
            inputs = _tpu_pad_short_inputs(inputs, tokenizer)
        prompt_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
        if _ON_TPU:
            _gen_kwargs["cache_implementation"] = "static"
        with torch.no_grad():
            out = model.generate(**inputs, **_gen_kwargs)
        return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

    resp = bodhi_wrapper.complete(messages)
    return resp.content


def _longest_common_prefix_len(lhs, rhs):
    """Return the shared token-prefix length between two token-id sequences."""
    prefix_len = 0
    for left_tok, right_tok in zip(lhs, rhs):
        if left_tok != right_tok:
            break
        prefix_len += 1
    return prefix_len


def score_response_confidence(model, tokenizer, messages, response_text):
    """Score the emitted response under the model and derive a confidence proxy.

    We use the geometric mean token probability of the response tokens given the
    prompt. This is length-normalized, model-derived, and works for both plain
    generations and BOHDI-wrapped responses.

    Per the Hugging Face chat templating docs, text produced by
    ``apply_chat_template(tokenize=False)`` should later be tokenized with
    ``add_special_tokens=False`` to avoid duplicating special tokens.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_tokens = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]
    full_tokens = tokenizer(
        prompt_text + response_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    prompt_len = _longest_common_prefix_len(
        prompt_tokens[0].tolist(),
        full_tokens[0].tolist(),
    )
    if prompt_len < 1 or full_tokens.shape[1] <= prompt_len:
        return {
            "response_token_count": 0,
            "mean_token_logprob": None,
            "geomean_token_prob": None,
        }

    response_token_count = full_tokens.shape[1] - prompt_len

    # On TPU: pad inputs to a single fixed length (4096) so XLA compiles ONE
    # forward graph for the confidence pass and reuses it across all 200 eval
    # examples.  Without this each unique (prompt+response) length would
    # trigger a fresh compile (10-30 min each) — making confidence scoring
    # unusable in practice.  4096 budget: ~700-2000 token prompt (HealthBench
    # Hard worst case) + up to 1024 response tokens + slack = comfortable.
    if _ON_TPU:
        _fixed = 4096
        _seq = full_tokens.shape[1]
        if _seq > _fixed:
            # extremely rare for HealthBench prompts; truncate from the LEFT
            # to preserve the response tail (the part we score).
            full_tokens = full_tokens[:, _seq - _fixed:]
            prompt_len = max(0, prompt_len - (_seq - _fixed))
            _seq = _fixed
        if _seq < _fixed:
            _pad_n = _fixed - _seq
            _pad_id = tokenizer.pad_token_id or 0
            _pad_t = torch.full((1, _pad_n), _pad_id, dtype=full_tokens.dtype)
            # left-pad so the response tail stays at the right edge (where the
            # logits-shift below expects it).  prompt_len shifts by _pad_n.
            full_tokens = torch.cat([_pad_t, full_tokens], dim=1)
            prompt_len = prompt_len + _pad_n

    full_tokens = full_tokens.to(model.device if hasattr(model, "device") else _xm.xla_device())
    with torch.no_grad():
        outputs = model(full_tokens)

    target_ids = full_tokens[:, prompt_len:]
    response_logits = outputs.logits[:, prompt_len - 1:-1, :]
    token_log_probs = torch.log_softmax(response_logits, dim=-1).gather(
        -1, target_ids.unsqueeze(-1)
    ).squeeze(-1)
    # When we left-padded, the logprobs at pad positions are meaningless;
    # but with our left-pad scheme prompt_len already points past the pads,
    # so target_ids contains only the real response tokens. No mask needed.
    # response_token_count is the real (pre-pad) length saved above.

    mean_logprob = float(token_log_probs.mean().item())
    return {
        "response_token_count": int(response_token_count),
        "mean_token_logprob": mean_logprob,
        "geomean_token_prob": float(np.exp(mean_logprob)),
    }


# -- confidence / calibration-style metrics --

def _collect_binary_labels(results, confidence_key):
    """Expand per-example confidence into per-criterion binary labels."""
    y_true = []
    y_pred = []
    for result in results:
        confidence = result.get(confidence_key)
        if confidence is None:
            continue
        confidence = max(0.0, min(1.0, float(confidence)))
        for criterion in result["criteria_results"]:
            # Positive-point criteria are the only ones that behave like
            # correctness labels; negative-point criteria are penalties.
            if criterion["points"] > 0:
                y_true.append(1.0 if criterion["criteria_met"] else 0.0)
                y_pred.append(confidence)
    return y_true, y_pred


def compute_brier_score(results, confidence_key):
    """Brier score across positive-point rubric criteria."""
    y_true, y_pred = _collect_binary_labels(results, confidence_key)
    if not y_true:
        return None
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean((y_pred - y_true) ** 2))


def compute_ece(results, confidence_key, n_bins=10):
    """Expected Calibration Error with equal-width bins."""
    y_true, y_pred = _collect_binary_labels(results, confidence_key)
    if not y_true:
        return None

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_pred > lo) & (y_pred <= hi)
        if lo == 0.0:
            mask = mask | (y_pred == 0.0)
        n = mask.sum()
        if n == 0:
            continue
        avg_conf = y_pred[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (n / len(y_true)) * abs(avg_acc - avg_conf)
    return float(ece)


def _safe_package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _safe_git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def collect_run_metadata(seed, grader_model):
    gpu_names = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            gpu_names.append(torch.cuda.get_device_properties(idx).name)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _safe_git_sha(),
        "seed": seed,
        "grader_model": grader_model,
        "bodhi_version": _safe_package_version("bodhi-llm"),
        "transformers_version": _safe_package_version("transformers"),
        "peft_version": _safe_package_version("peft"),
        "torch_version": torch.__version__,
        "n_gpus": len(gpu_names),
        "gpu_names": gpu_names,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/medgemma-27b-text-it")
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--use-bodhi", action="store_true")
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--grader-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Eval uses greedy decoding so scores are deterministic given fixed inputs,
    # but BODHI internals + grader may sample — seed for stability across reruns.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    examples = load_eval_data(args.sample_ids)
    if args.max_examples:
        examples = examples[:args.max_examples]

    model, tokenizer, device = load_model(args.model, args.lora_path)
    grader = LocalGrader(args.grader_model)

    bodhi_wrapper = make_bodhi_wrapper(model, tokenizer, device) if args.use_bodhi else None

    tag = f"{'lora' if args.lora_path else 'base'}_{'bodhi' if args.use_bodhi else 'no_wrapper'}"
    print(f"\nEval: {tag}  ({len(examples)} examples)\n")

    all_results = []
    scores = []
    model_confidences = []
    total_parse_failures = 0
    total_rubric_items = 0
    for ex in tqdm(examples, desc=tag):
        resp = gen_response(model, tokenizer, device, ex["prompt"], args.use_bodhi, bodhi_wrapper)
        confidence = score_response_confidence(model, tokenizer, ex["prompt"], resp)
        grade = grade_trace(grader, ex["prompt"], resp, ex["rubrics"])
        all_results.append({
            "prompt_id": ex["prompt_id"], "response": resp,
            "score": grade["overall_score"], "tag_scores": grade["tag_scores"],
            "criteria_results": grade["criteria_results"],
            "parse_failures": grade["parse_failures"],
            "model_confidence_geomean_prob": confidence["geomean_token_prob"],
            "model_confidence_mean_token_logprob": confidence["mean_token_logprob"],
            "response_token_count": confidence["response_token_count"],
        })
        scores.append(grade["overall_score"])
        if confidence["geomean_token_prob"] is not None:
            model_confidences.append(confidence["geomean_token_prob"])
        total_parse_failures += grade["parse_failures"]
        total_rubric_items += len(grade["criteria_results"])

    model_brier = compute_brier_score(all_results, "model_confidence_geomean_prob")
    model_ece = compute_ece(all_results, "model_confidence_geomean_prob")
    grader_brier = compute_brier_score(all_results, "score")
    grader_ece = compute_ece(all_results, "score")
    parse_fail_rate = (total_parse_failures / total_rubric_items) if total_rubric_items else None

    summary = {
        "config": tag, "model": args.model,
        "lora_path": args.lora_path, "use_bodhi": args.use_bodhi,
        "n_examples": len(examples),
        "run_metadata": collect_run_metadata(args.seed, args.grader_model),
        "mean": float(np.mean(scores)) if scores else None,
        "std": float(np.std(scores)) if scores else None,
        "median": float(np.median(scores)) if scores else None,
        "model_confidence_method": (
            "geometric_mean_token_probability over the emitted response, "
            "computed from next-token logprobs conditioned on the prompt"
        ),
        "mean_model_confidence": (
            float(np.mean(model_confidences)) if model_confidences else None
        ),
        "brier_model_calibration": model_brier,
        "ece_model_calibration": model_ece,
        # Keep the legacy grader-derived fields so older comparisons do not
        # silently break, but label them explicitly as grader consistency.
        "brier_grader_consistency": grader_brier,
        "ece_grader_consistency": grader_ece,
        "grader_parse_failure_rate": parse_fail_rate,
        "grader_parse_failures_total": total_parse_failures,
        "grader_rubric_items_total": total_rubric_items,
        "results": all_results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    model_brier_str = f"{model_brier:.4f}" if model_brier is not None else "n/a"
    model_ece_str = f"{model_ece:.4f}" if model_ece is not None else "n/a"
    grader_brier_str = f"{grader_brier:.4f}" if grader_brier is not None else "n/a"
    grader_ece_str = f"{grader_ece:.4f}" if grader_ece is not None else "n/a"
    mean_str = f"{summary['mean']:.4f}" if summary['mean'] is not None else "n/a"
    std_str = f"{summary['std']:.4f}" if summary['std'] is not None else "n/a"
    fail_str = f"{parse_fail_rate*100:.2f}%" if parse_fail_rate is not None else "n/a"
    print(f"\n{tag}: score={mean_str} +/- {std_str}  "
          f"model_brier={model_brier_str}  model_ece={model_ece_str}  "
          f"grader_brier*={grader_brier_str}  grader_ece*={grader_ece_str}  "
          f"grader_parse_fail={fail_str}  -> {out}")
    print("  (* grader_brier / grader_ece are legacy grader-consistency proxies)")


if __name__ == "__main__":
    main()
