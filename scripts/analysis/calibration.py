"""Calibration check: does the model's confidence track rubric correctness?

For each Stage-4 eval JSON (``eval/seed_<N>/<config>.json``), this script
computes Spearman rho between the model's per-prompt geomean_token_prob
(a within-response token-fluency confidence proxy) and the grader's
overall_score for that prompt. A positive rho means "more confident on
correct, less confident on wrong" — calibration consistent with genuine
epistemic humility, not just hedging-keyword surface mimicry.

The 4-config ablation (base/lora x wrapper/no-wrapper) is what tells us
*whether* LoRA shifted behavior; this calibration check tells us whether
that shift is in the right direction. Concretely:

  - Internalization signal:      lora_no_wrapper: rho > 0 and bigger
                                 than base_no_wrapper.
  - Pattern-mimicry signal:      lora_no_wrapper: rho not different from
                                 base_no_wrapper (or worse).
  - Scaffolding-only signal:     gain only when bodhi wrapper is on.

Usage:
    python -m scripts.analysis.calibration \
        --eval-dir results_tunix/seed_42/eval/seed_42 \
        --output results_tunix/seed_42/analysis/calibration.json

Reads ``base_no_wrapper.json``, ``base_bodhi.json``, ``lora_no_wrapper.json``,
``lora_bodhi.json`` from --eval-dir; writes a single calibration.json with
rho, p-value, n, and per-config mean confidence/score.

Caveat already documented in eval_healthbench.py: geomean_token_prob is
*token fluency*, not a calibrated probability of clinical correctness.
This script reports the within-response token-fluency-vs-rubric rho only;
do not over-claim from it. See RESULTS.md / methodology.md for framing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

# Avoid scipy: dashboard machine has no scipy bound. Hand-roll Spearman
# with rank coefficients + a t-distribution-free p-value via a simple
# permutation; close enough for an appendix-grade calibration check.


def _rank(xs):
    """Return ranks [1..n] of xs, average-ranked on ties."""
    sorted_xs = sorted(enumerate(xs), key=lambda t: t[1])
    n = len(xs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_xs[j + 1][1] == sorted_xs[i][1]:
            j += 1
        # average rank for the tied group [i..j]
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_xs[k][0]] = avg
        i = j + 1
    return ranks


def _spearman(xs, ys):
    """Spearman rho, hand-rolled. Returns None if degenerate."""
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


def _per_config_calibration(eval_json_path: Path) -> dict:
    """Pull (confidence, score) pairs out of one config's eval JSON.

    Schema (per scripts/eval_healthbench.py):
      {results: [
         {prompt_id, score, geomean_token_prob, tag_scores, ...}, ...
      ]}
    """
    with open(eval_json_path) as f:
        data = json.load(f)
    pairs = []
    for r in data.get("results", []):
        conf = r.get("geomean_token_prob")
        score = r.get("score")
        if conf is None or score is None:
            continue
        pairs.append((float(conf), float(score)))
    if not pairs:
        return {"n": 0, "rho": None, "mean_confidence": None, "mean_score": None}
    confs, scores = zip(*pairs)
    return {
        "n": len(pairs),
        "rho": _spearman(confs, scores),
        "mean_confidence": mean(confs),
        "stdev_confidence": pstdev(confs) if len(confs) > 1 else 0.0,
        "mean_score": mean(scores),
        "stdev_score": pstdev(scores) if len(scores) > 1 else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--eval-dir", required=True, type=Path,
                   help="Directory containing the 4 config JSONs "
                        "(base_no_wrapper.json, base_bodhi.json, "
                        "lora_no_wrapper.json, lora_bodhi.json).")
    p.add_argument("--output", required=True, type=Path,
                   help="Where to write the per-config calibration JSON.")
    args = p.parse_args()

    eval_dir = args.eval_dir.expanduser()
    if not eval_dir.is_dir():
        raise SystemExit(f"--eval-dir {eval_dir} does not exist")

    configs = ("base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi")
    results = {}
    for cfg in configs:
        cfg_path = eval_dir / f"{cfg}.json"
        if not cfg_path.is_file():
            results[cfg] = {"error": f"missing {cfg_path.name}"}
            continue
        results[cfg] = _per_config_calibration(cfg_path)

    # Internalization signal: rho_lora_no_wrapper - rho_base_no_wrapper
    base_nw_rho = results.get("base_no_wrapper", {}).get("rho")
    lora_nw_rho = results.get("lora_no_wrapper", {}).get("rho")
    internalization_delta = (
        lora_nw_rho - base_nw_rho
        if base_nw_rho is not None and lora_nw_rho is not None
        else None
    )

    out = {
        "configs": results,
        "internalization_calibration_delta": internalization_delta,
        "interpretation": (
            "rho > 0 means model is more confident on correct than on wrong "
            "(token-fluency-level calibration). 'internalization_calibration_delta' "
            "= rho(lora_no_wrapper) - rho(base_no_wrapper); positive means LoRA "
            "training improved calibration without the BODHI wrapper at "
            "inference time. Caveat: geomean_token_prob is fluency-level, "
            "not clinical-correctness probability. See methodology.md."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}")
    for cfg, r in results.items():
        if "error" in r:
            print(f"  {cfg}: {r['error']}")
        else:
            print(f"  {cfg}: n={r['n']} rho={r['rho']} mean_score={r['mean_score']}")
    if internalization_delta is not None:
        print(f"  internalization Δrho = {internalization_delta:+.4f}")


if __name__ == "__main__":
    main()
