"""U-shape evaluation: post-hoc stratification of eval scores by difficulty.

Inspired by the Nature Medicine 2026 "ChatGPT triage" paper
(s41591-026-04297-7) which showed LLM failure rates follow an inverted-U
across clinical acuity, worst at the extremes (nonurgent and emergency),
best in the middle.

BOHDI's central claim is epistemic humility. If LoRA-BOHDI works, it should
flatten this U: lift the tails, especially the high-acuity failures where
overconfident wrong answers are most harmful.

This script does NO model inference. It re-aggregates existing eval
outputs from scripts/eval_healthbench.py against HealthBench metadata,
producing per-tier and per-theme stats for plotting.

Two difficulty axes:
  1. Rubric complexity tertiles (pos-point sum: easy / medium / hard)
  2. Theme (emergency_referrals / hedging / context_seeking / ...)

Reported per tier: n, mean score, fail rate (score < --fail-threshold).
With --bootstrap, 95% CIs are attached to every metric via
nonparametric resampling of the per-example results.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_healthbench_meta(paths):
    """Map prompt_id -> {themes, pos_points, n_rubrics}."""
    meta = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                ex = json.loads(line)
                themes = [t.split(":", 1)[1] for t in ex.get("example_tags", [])
                          if t.startswith("theme:")]
                pos = sum(r["points"] for r in ex["rubrics"] if r["points"] > 0)
                meta[ex["prompt_id"]] = {
                    "themes": themes,
                    "pos_points": pos,
                    "n_rubrics": len(ex["rubrics"]),
                }
    return meta


def compute_tertile_cutoffs(meta, restrict_to=None):
    """Return (q1, q2) cutoffs for pos_points tertiles.

    If restrict_to is given (an iterable of prompt_ids), compute cutoffs on
    just that subset — useful when the eval only covers a holdout.
    """
    if restrict_to is not None:
        restrict_to = set(restrict_to)
        pts = [m["pos_points"] for pid, m in meta.items() if pid in restrict_to]
    else:
        pts = [m["pos_points"] for m in meta.values()]
    if len(pts) < 3:
        raise ValueError(f"need at least 3 examples to compute tertiles, got {len(pts)}")
    qs = statistics.quantiles(pts, n=3)
    return qs[0], qs[1]


def tier_of(pos_points, q1, q2):
    if pos_points <= q1:
        return "easy"
    if pos_points <= q2:
        return "medium"
    return "hard"


def summarize(scores, fail_threshold, bootstrap=0, rng=None, ci=0.95):
    """Standard stats for a list of per-example scores.

    If bootstrap > 0, also return 95% (or --ci) CIs for mean and fail_rate
    computed by resampling scores with replacement bootstrap times.
    We resample the score list only, i.e. non-parametric bootstrap over
    the test examples. This captures test-set variance, not model-seed
    variance; for the latter, use the run_multi_seed wrapper instead.
    """
    if not scores:
        return {"n": 0}
    out = {
        "n": len(scores),
        "mean": float(sum(scores) / len(scores)),
        "median": float(statistics.median(scores)),
        "min": float(min(scores)),
        "max": float(max(scores)),
        "fail_rate": sum(1 for s in scores if s < fail_threshold) / len(scores),
    }
    if bootstrap > 0:
        rng = rng if rng is not None else np.random.default_rng()
        arr = np.asarray(scores, dtype=float)
        n = len(arr)
        # resample indices with replacement, vectorised over bootstrap draws
        idx = rng.integers(0, n, size=(bootstrap, n))
        resampled = arr[idx]                              # shape (bootstrap, n)
        means = resampled.mean(axis=1)
        fails = (resampled < fail_threshold).mean(axis=1)
        lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100   # e.g. 2.5, 97.5 for 95%
        out["ci"] = float(ci)
        out["mean_ci"] = [float(np.percentile(means, lo)),
                          float(np.percentile(means, hi))]
        out["fail_rate_ci"] = [float(np.percentile(fails, lo)),
                               float(np.percentile(fails, hi))]
    return out


def aggregate_by_tier(results, meta, q1, q2, fail_threshold, bootstrap=0, rng=None):
    tier_scores = defaultdict(list)
    missing = 0
    for r in results:
        pid = r["prompt_id"]
        if pid not in meta:
            missing += 1
            continue
        tier_scores[tier_of(meta[pid]["pos_points"], q1, q2)].append(r["score"])
    out = {
        t: summarize(s, fail_threshold, bootstrap=bootstrap, rng=rng)
        for t, s in tier_scores.items()
    }
    # ensure all three tiers present even if empty
    for t in ("easy", "medium", "hard"):
        out.setdefault(t, {"n": 0})
    out["_missing_prompt_ids"] = missing
    return out


def aggregate_by_theme(results, meta, fail_threshold, bootstrap=0, rng=None):
    theme_scores = defaultdict(list)
    for r in results:
        pid = r["prompt_id"]
        if pid not in meta:
            continue
        for theme in meta[pid]["themes"]:
            theme_scores[theme].append(r["score"])
    return {
        t: summarize(s, fail_threshold, bootstrap=bootstrap, rng=rng)
        for t, s in theme_scores.items()
    }


def summarize_overall(results, fail_threshold, bootstrap=0, rng=None):
    """Top-level (non-stratified) stats over all per-example scores."""
    scores = [r["score"] for r in results]
    return summarize(scores, fail_threshold, bootstrap=bootstrap, rng=rng)


def _fmt_mean_ci(s):
    """Render a summary cell: '0.73 [0.68,0.78]' if CI present, else '0.73'."""
    if not s or s.get("mean") is None:
        return "-"
    if "mean_ci" in s:
        lo, hi = s["mean_ci"]
        return f"{s['mean']:.2f} [{lo:.2f},{hi:.2f}]"
    return f"{s['mean']:.3f}"


def _fmt_fail_ci(s):
    if not s or s.get("fail_rate") is None:
        return "-"
    if "fail_rate_ci" in s:
        lo, hi = s["fail_rate_ci"]
        return f"{s['fail_rate']:.2f} [{lo:.2f},{hi:.2f}]"
    return f"{s['fail_rate']:.2f}"


def print_table(summary, fail_threshold):
    configs = list(summary["configs"].keys())
    if not configs:
        return

    has_ci = any(
        "mean_ci" in summary["configs"][c].get("by_tier", {}).get("easy", {})
        for c in configs
    )
    ci_note = " (95% CI in brackets)" if has_ci else ""
    col_w = 24 if has_ci else 12  # wider cells when CIs are printed

    print(f"\n=== U-shape by rubric complexity{ci_note} ===")
    print(f"(tier cutoffs: pos_points q1={summary['thresholds']['q1']:.1f}, "
          f"q2={summary['thresholds']['q2']:.1f})")
    print(f"{'config':<28} "
          f"{'easy mean':>{col_w}} {'med mean':>{col_w}} {'hard mean':>{col_w}} "
          f"{'easy fail':>{col_w}} {'hard fail':>{col_w}}")
    for name in configs:
        t = summary["configs"][name]["by_tier"]
        print(f"{name:<28} "
              f"{_fmt_mean_ci(t.get('easy')):>{col_w}} "
              f"{_fmt_mean_ci(t.get('medium')):>{col_w}} "
              f"{_fmt_mean_ci(t.get('hard')):>{col_w}} "
              f"{_fmt_fail_ci(t.get('easy')):>{col_w}} "
              f"{_fmt_fail_ci(t.get('hard')):>{col_w}}")

    print(f"\n=== By theme (mean score; fail rate = score < {fail_threshold}) ===")
    all_themes = set()
    for c in summary["configs"].values():
        all_themes.update(c["by_theme"].keys())
    header = f"{'theme':<24} " + " ".join(f"{n:<24}" for n in configs)
    print(header)
    for theme in sorted(all_themes):
        row = [f"{theme:<24}"]
        for n in configs:
            s = summary["configs"][n]["by_theme"].get(theme, {})
            if s.get("n"):
                # compact form: mean/fail (n=N)
                row.append(f"{s['mean']:.2f}/{s['fail_rate']:.2f} (n={s['n']})".ljust(24))
            else:
                row.append("-".ljust(24))
        print(" ".join(row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsons", nargs="+", required=True,
                        help="eval outputs from scripts/eval_healthbench.py")
    parser.add_argument("--healthbench", nargs="+", required=True,
                        help="HealthBench JSONL files with metadata")
    parser.add_argument("--output", required=True, help="where to write aggregated JSON")
    parser.add_argument("--fail-threshold", type=float, default=0.4,
                        help="score below this counts as a failure (default 0.4, "
                             "matches filter_traces default)")
    parser.add_argument("--tertile-on-holdout-only", action="store_true",
                        help="compute tier cutoffs on eval subset rather than full dataset "
                             "(default: use full HealthBench)")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="nonparametric bootstrap resamples for 95%% CIs "
                             "(default 0 = disabled; 1000 is a reasonable paper number)")
    parser.add_argument("--bootstrap-seed", type=int, default=42,
                        help="seed for the bootstrap RNG so CIs are reproducible")
    args = parser.parse_args()

    rng = np.random.default_rng(args.bootstrap_seed)

    meta = load_healthbench_meta(args.healthbench)
    print(f"Loaded metadata for {len(meta)} HealthBench prompts")

    # gather prompt_ids referenced by eval files, for the restrict-to option
    all_eval_pids = set()
    evals = []
    for ep in args.eval_jsons:
        with open(ep) as f:
            ev = json.load(f)
        evals.append((ep, ev))
        for r in ev.get("results", []):
            all_eval_pids.add(r["prompt_id"])

    restrict = all_eval_pids if args.tertile_on_holdout_only else None
    q1, q2 = compute_tertile_cutoffs(meta, restrict_to=restrict)
    print(f"Tier cutoffs (pos_points): q1={q1:.1f} q2={q2:.1f}")

    summary = {
        "thresholds": {"q1": q1, "q2": q2, "fail_below": args.fail_threshold},
        "bootstrap": {
            "n_resamples": args.bootstrap,
            "seed": args.bootstrap_seed,
        } if args.bootstrap else None,
        "configs": {},
    }
    for ep, ev in evals:
        name = ev.get("config") or Path(ep).stem
        results = ev.get("results", [])
        # bootstrap the overall too so the abstract gets a headline CI
        overall = summarize_overall(results, args.fail_threshold,
                                    bootstrap=args.bootstrap, rng=rng)
        summary["configs"][name] = {
            "source": ep,
            "n_examples": len(results),
            "overall": overall,
            # legacy fields kept for backward compatibility with older consumers
            "overall_mean": ev.get("mean"),
            "overall_brier_model_calibration": ev.get("brier_model_calibration"),
            "overall_ece_model_calibration": ev.get("ece_model_calibration"),
            "model_confidence_method": ev.get("model_confidence_method"),
            # renamed in eval_healthbench.py to reflect that these are NOT
            # model-calibration measures; see issue #1. fall back to legacy names.
            "overall_brier_grader_consistency": (
                ev.get("brier_grader_consistency", ev.get("brier"))
            ),
            "overall_ece_grader_consistency": (
                ev.get("ece_grader_consistency", ev.get("ece"))
            ),
            "grader_parse_failure_rate": ev.get("grader_parse_failure_rate"),
            "by_tier": aggregate_by_tier(
                results, meta, q1, q2, args.fail_threshold,
                bootstrap=args.bootstrap, rng=rng,
            ),
            "by_theme": aggregate_by_theme(
                results, meta, args.fail_threshold,
                bootstrap=args.bootstrap, rng=rng,
            ),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print_table(summary, args.fail_threshold)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
