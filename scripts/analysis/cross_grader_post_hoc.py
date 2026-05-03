"""Post-hoc cross-grader pass on existing Stage 4 response JSONs.

Runs a SECOND grader model (default Qwen/Qwen2.5-14B-Instruct) over the
same generated responses that the primary Llama-3.1-8B grader scored,
without regenerating responses. Adds a ``secondary_grader_runs[]`` entry
to each per-config eval JSON in place. Then computes Spearman rho per
config between the primary and secondary grader's per-prompt scores —
that's the cross-grader correlation the methodology section claims.

This script is the post-hoc version of eval_healthbench.py's built-in
``--secondary-grader-model`` flag. We wrote it separately so we can run
it on a v37 VM after Stages 4+5 land WITHOUT killing v37 to add the flag
to the launcher's run_eval call.

Runs vLLM-TPU on the *current VM*, so:
  - This must execute on a v6e-8 TPU VM (or any vllm-tpu-capable host).
  - The launcher's EXIT trap must NOT have deleted the VM yet — the
    user manually SSHes in (or the launcher gets SIGSTOPped) before
    delete_vm fires. See tpu/launch_5seeds_tunix.sh:888 for the trap.
  - Loads the secondary grader fresh; the primary grader's container
    must be torn down first (cleanup_eval in the launcher does this).

Usage on a v37 VM:
    python -m scripts.analysis.cross_grader_post_hoc \
        --eval-dir eval/seed_42 \
        --secondary-grader-model Qwen/Qwen2.5-14B-Instruct \
        --output-correlation eval/seed_42/cross_grader_correlation.json

This will:
  1. Read base_no_wrapper.json, base_bodhi.json, lora_no_wrapper.json,
     lora_bodhi.json from --eval-dir.
  2. Spin up a single VLLMEngine for the secondary grader (one for the
     whole run, not per-config — saves the load cost).
  3. Re-grade each config's existing responses, append a new entry to
     ``secondary_grader_runs`` in that file (mutates in place).
  4. Compute Spearman rho per config between primary scores
     (results[*].score) and the secondary grader's scores. Writes
     cross_grader_correlation.json with {primary_grader, secondary_grader,
     per_config: {<name>: {n, rho, primary_mean, secondary_mean}}}.

Output is a small JSON the writeup can cite directly. The big eval JSONs
get the secondary_grader_runs entry appended in place so other tooling
(scripts/grader_correlation.py) keeps working.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

# Lazy/runtime imports for the parts that need vLLM. The Spearman code is
# kept dependency-free so we can validate I/O paths on a CPU dev box.


def _rank(xs):
    sorted_xs = sorted(enumerate(xs), key=lambda t: t[1])
    n = len(xs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_xs[j + 1][1] == sorted_xs[i][1]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_xs[k][0]] = avg
        i = j + 1
    return ranks


def _spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = (sum((a - mx) ** 2 for a in rx)) ** 0.5
    den_y = (sum((b - my) ** 2 for b in ry)) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


CONFIGS = ("base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi")


def _grade_one(grader, item, grade_trace):
    """Re-grade a single response item with the secondary grader."""
    try:
        grade = grade_trace(grader, item["messages"], item["response"], item["rubrics"])
        return {
            "prompt_id": item["prompt_id"],
            "score": grade["overall_score"],
            "tag_scores": grade["tag_scores"],
            "criteria_results": grade["criteria_results"],
            "parse_failures": grade["parse_failures"],
        }, None
    except Exception:
        tb = "\n".join(traceback.format_exc().strip().splitlines()[-30:])
        return None, (item.get("prompt_id", "?"), tb)


def _regrade_config(eval_path: Path, sec_engine, grader_cls, grade_trace,
                     concurrency: int) -> tuple:
    """Load a config JSON, regrade with the secondary engine, mutate file in place.

    Returns (primary_scores, secondary_scores) lists, ordered to align
    by prompt_id so we can Spearman them.
    """
    with open(eval_path) as f:
        data = json.load(f)
    grader = grader_cls(sec_engine)
    raw_items = data.get("results", [])
    # We need ``messages`` + ``rubrics`` per item to call grade_trace; the
    # eval_healthbench primary pass attaches both to the per-prompt result
    # alongside the response. If the schema differs, surface clearly.
    missing_fields = [f for f in ("messages", "rubrics", "response")
                      if not all(f in r for r in raw_items)]
    if missing_fields:
        raise SystemExit(
            f"{eval_path.name}: results[*] missing required fields "
            f"{missing_fields!r} for re-grading. Was this eval run with the "
            "current eval_healthbench.py? If results[*] only has 'response' "
            "(no rubrics), regenerate with the latest eval_healthbench."
        )

    sec_results = []
    sec_scores: dict = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_grade_one, grader, item, grade_trace)
                   for item in raw_items]
        for fut in as_completed(futures):
            r, err = fut.result()
            if r is not None:
                sec_results.append(r)
                sec_scores[r["prompt_id"]] = r["score"]

    # Aligned vectors for Spearman.
    primary_aligned = []
    secondary_aligned = []
    for r in raw_items:
        pid = r["prompt_id"]
        if pid in sec_scores:
            primary_aligned.append(float(r["score"]))
            secondary_aligned.append(float(sec_scores[pid]))

    # Append to secondary_grader_runs[] without disturbing the primary.
    runs = data.setdefault("secondary_grader_runs", [])
    runs.append({
        "grader_model": getattr(sec_engine, "model_name", "unknown"),
        "n": len(sec_results),
        "results": sec_results,
        "mean_score": mean([r["score"] for r in sec_results]) if sec_results else None,
    })
    with open(eval_path, "w") as f:
        json.dump(data, f, indent=2)

    return primary_aligned, secondary_aligned


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--eval-dir", required=True, type=Path,
                   help="Directory containing the 4 Stage-4 config JSONs.")
    p.add_argument("--secondary-grader-model",
                   default="Qwen/Qwen2.5-14B-Instruct",
                   help="HF model id for the secondary grader (different "
                        "family from primary Llama).")
    p.add_argument("--output-correlation", required=True, type=Path,
                   help="Where to write cross_grader_correlation.json.")
    p.add_argument("--concurrency", type=int, default=16,
                   help="Concurrent grade requests to vLLM (matches "
                        "EVAL_CONCURRENCY default).")
    args = p.parse_args()

    eval_dir = args.eval_dir.expanduser()
    if not eval_dir.is_dir():
        raise SystemExit(f"--eval-dir {eval_dir} does not exist")

    # Lazy imports — keep --help fast and runnable on CPU dev box.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _vllm_engine import VLLMEngine  # type: ignore[import-not-found]
    from eval_healthbench import LocalGrader, grade_trace  # type: ignore[import-not-found]

    print(f"loading secondary grader: {args.secondary_grader_model}", flush=True)
    print(f"  this takes ~5 min for first vLLM startup; gemma3-4B container "
          f"must already be torn down (cleanup_eval).", flush=True)
    time.sleep(5)

    per_config: dict = {}
    with VLLMEngine(args.secondary_grader_model) as sec_engine:
        # Stash model_name on the engine so _regrade_config can record it.
        try:
            sec_engine.model_name = args.secondary_grader_model
        except Exception:
            pass
        for cfg in CONFIGS:
            cfg_path = eval_dir / f"{cfg}.json"
            if not cfg_path.is_file():
                per_config[cfg] = {"error": f"missing {cfg_path.name}"}
                continue
            print(f"\n  re-grading {cfg}.json ...", flush=True)
            primary, secondary = _regrade_config(
                cfg_path, sec_engine, LocalGrader, grade_trace,
                concurrency=args.concurrency,
            )
            rho = _spearman(primary, secondary)
            per_config[cfg] = {
                "n": len(primary),
                "spearman_rho_primary_vs_secondary": rho,
                "primary_mean": mean(primary) if primary else None,
                "secondary_mean": mean(secondary) if secondary else None,
            }
            print(f"    n={len(primary)} rho={rho}")

    out = {
        "eval_dir": str(eval_dir),
        "primary_grader": "meta-llama/Llama-3.1-8B-Instruct",
        "secondary_grader": args.secondary_grader_model,
        "per_config": per_config,
        "interpretation": (
            "spearman_rho_primary_vs_secondary close to 1.0 means the two "
            "graders rank the responses similarly per config — the primary "
            "grader's signal is not an artifact of one grader's idiosyncrasy. "
            "Workshop appendix can cite the lowest per-config rho across the "
            "4 configs as a worst-case grader-agreement bound."
        ),
    }
    args.output_correlation.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_correlation, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.output_correlation}")


if __name__ == "__main__":
    main()
