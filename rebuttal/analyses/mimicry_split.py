"""Rebuttal analysis: calibration vs. template mimicry (meta-review W4, Reviewer LxMF W2).

LxMF's exact proposal: "isolate samples that genuinely have missing information and
cannot be answered, and compare with those that can be answered without additional
information, and observe if CoT or LoRA triggers more inquiry in the former group than
the latter."

We operationalize the split with an INDEPENDENT label (not derived from the grader's
own context_seeking score, which would be circular):

  context_dependent  = HealthBench theme == 'context_seeking'
                       OR the prompt's rubric rewards asking for clarification/context
                       (criterion text matches a clarification keyword).
  self_contained     = everything else.

Real calibration predicts the adapter raises active inquiry MORE on context_dependent
prompts and does NOT ask indiscriminately on self_contained prompts. We summarize this
with a discrimination index:

  discrimination(config) = P(ask | context_dependent) - P(ask | self_contained)

Template mimicry (asking everywhere) collapses discrimination toward Base. Internalized
calibration keeps LoRA's discrimination close to the wrapper's and well above Base.

Runs against the committed results in results_modal/. No inference, no H100.
"""

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HARD = REPO / "data" / "raw" / "healthbench_hard.jsonl"
SEED_DIRS = ["seed_7", "seed_13", "seed_42", "seed_99", "seed_101"]

CONFIG_DISPLAY = {
    "base_no_wrapper": "Base",
    "base_bodhi": "Wrapper (Base+CoT)",
    "lora_no_wrapper": "LoRA",
    "lora_bodhi": "LoRA+Wrapper",
}
CONFIG_ORDER = ["base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi"]

# Keywords in a rubric CRITERION that indicate the ideal answer asks for more
# information / context before committing. Independent of the model output.
CLARIFY_RE = re.compile(
    r"\b(ask(s|ing)?|clarif\w*|follow[- ]?up|more (information|context|detail)"
    r"|additional (information|context|detail)|missing (information|context)"
    r"|request(s|ing)? (more|additional)|gather\w* (more|additional|context)"
    r"|what (information|else)|need(s|ed)? (more|additional|to know))\b",
    re.IGNORECASE,
)

# Heuristic persona detector (bonus: quantify clinician- vs patient-posed prompts,
# rebutting the premise that HealthBench is 'only patients googling').
CLINICIAN_RE = re.compile(
    r"\b(i'?m|i am|as)\s+(an?\s+)?(emergency\s+medicine\s+)?"
    r"(physician|doctor|clinician|resident|attending|nurse|np\b|physician\s+assistant"
    r"|\bpa\b|pharmacist|md\b|surgeon|cardiologist|paramedic|medical\s+student)",
    re.IGNORECASE,
)


def load_prompt_meta():
    """prompt_id -> {theme, clarify_rubric, clinician_posed}."""
    meta = {}
    with open(HARD) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            pid = ex["prompt_id"]
            theme = None
            for t in ex.get("example_tags", []):
                if t.startswith("theme:"):
                    theme = t.split(":", 1)[1]
                    break
            clarify = any(
                CLARIFY_RE.search(r.get("criterion", "")) for r in ex.get("rubrics", [])
            )
            first_user = ""
            for m in ex.get("prompt", []):
                if m.get("role") == "user":
                    first_user = m.get("content", "")
                    break
            meta[pid] = {
                "theme": theme,
                "clarify_rubric": clarify,
                "clinician_posed": bool(CLINICIAN_RE.search(first_user)),
            }
    return meta


def group_of(m):
    """Primary label. context_dependent if HealthBench flags it as context-seeking
    OR the rubric rewards asking; else self_contained."""
    if m is None:
        return None
    if m["theme"] == "context_seeking" or m["clarify_rubric"]:
        return "context_dependent"
    return "self_contained"


def group_theme_only(m):
    """Robustness label: HealthBench's own context_seeking theme tag ONLY (does not
    depend on our clarification-keyword regex over rubric text)."""
    if m is None:
        return None
    return "context_dependent" if m["theme"] == "context_seeking" else "self_contained"


def load_examples():
    """Pool every graded response across the 5 seeds -> list of per-response rows."""
    rows = []
    for sd in SEED_DIRS:
        path = REPO / "results_modal" / sd / "epistemic_scores.json"
        if not path.exists():
            continue
        data = json.load(open(path))
        for cfg in data["configs"]:
            name = cfg["name"]
            for ex in cfg["examples"]:
                sc = ex.get("scores") or {}
                if ex.get("parse_failure"):
                    continue
                ai = sc.get("active_inquiry")
                if ai is None:
                    continue
                rows.append(
                    {
                        "seed": sd,
                        "config": name,
                        "prompt_id": ex["prompt_id"],
                        "active_inquiry": bool(ai),
                        "n_questions": sc.get("n_questions", 0) or 0,
                        "red_flag": sc.get("red_flag_identification"),
                    }
                )
    return rows


def _rate(bools):
    return (sum(1.0 for b in bools if b) / len(bools)) if bools else float("nan")


def build_pid_index(rows, labelfn, meta):
    """config -> pid -> {"group":..., "ai":[bool,...]} pooling responses across seeds."""
    idx = {cfg: {} for cfg in CONFIG_ORDER}
    for r in rows:
        cfg = r["config"]
        if cfg not in idx:
            continue
        pid = r["prompt_id"]
        slot = idx[cfg].setdefault(pid, {"group": labelfn(meta.get(pid)), "ai": []})
        slot["ai"].append(r["active_inquiry"])
    return idx


def discrimination(pid_map, pids):
    """P(ask|context_dependent) - P(ask|self_contained) over the given pid list,
    pooling each pid's responses. Returns (disc, cd_rate, sc_rate)."""
    cd, sc = [], []
    for pid in pids:
        slot = pid_map.get(pid)
        if not slot:
            continue
        if slot["group"] == "context_dependent":
            cd.extend(slot["ai"])
        elif slot["group"] == "self_contained":
            sc.extend(slot["ai"])
    return _rate(cd) - _rate(sc), _rate(cd), _rate(sc)


def bootstrap_disc(idx, eval_pids, B=5000, seed=0):
    """Prompt-level bootstrap. Resample prompt_ids (paired across configs) B times;
    return per-config discrimination CI and the LoRA-Base discrimination-diff CI."""
    import numpy as np
    rng = np.random.default_rng(seed)
    pids = list(eval_pids)
    per_cfg = {cfg: [] for cfg in CONFIG_ORDER}
    diff_lora_base = []
    for _ in range(B):
        samp = [pids[i] for i in rng.integers(0, len(pids), len(pids))]
        ds = {}
        for cfg in CONFIG_ORDER:
            d, _, _ = discrimination(idx[cfg], samp)
            per_cfg[cfg].append(d)
            ds[cfg] = d
        diff_lora_base.append(ds["lora_no_wrapper"] - ds["base_no_wrapper"])

    def ci(a):
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    return {cfg: ci(per_cfg[cfg]) for cfg in CONFIG_ORDER}, ci(diff_lora_base)


def report(idx, eval_pids, meta, labelname, out):
    from collections import Counter
    # group sizes over unique pids present in the Base config's map
    gcount = Counter(idx["base_no_wrapper"].get(pid, {}).get("group") for pid in eval_pids)
    print("=" * 78)
    print(f"Active-inquiry rate (%) by config x group   [label: {labelname}]")
    print(f"  context_dependent={gcount.get('context_dependent',0)}  "
          f"self_contained={gcount.get('self_contained',0)}   (pooled 5 seeds)")
    print("=" * 78)
    header = f"{'Config':22s} {'context_dep':>12s} {'self_cont':>12s} {'discrim':>10s} {'95% CI (pp)':>18s}"
    print(header)
    print("-" * len(header))
    ci_cfg, diff_ci = bootstrap_disc(idx, eval_pids)
    disc = {}
    label_out = {"groups": dict(gcount), "configs": {}}
    for cfg in CONFIG_ORDER:
        d, cd, sc = discrimination(idx[cfg], eval_pids)
        disc[cfg] = d
        lo, hi = ci_cfg[cfg]
        print(f"{CONFIG_DISPLAY[cfg]:22s} {100*cd:11.1f}% {100*sc:11.1f}% "
              f"{100*d:8.1f}pp   [{100*lo:+5.1f},{100*hi:+6.1f}]")
        label_out["configs"][cfg] = {
            "context_dependent_rate": cd, "self_contained_rate": sc,
            "discrimination": d, "discrimination_ci95": [lo, hi],
        }
    dd = disc["lora_no_wrapper"] - disc["base_no_wrapper"]
    print("-" * len(header))
    print(f"  LoRA - Base discrimination difference: {100*dd:+.1f}pp  "
          f"95% CI [{100*diff_ci[0]:+.1f}, {100*diff_ci[1]:+.1f}]")
    verdict = "EXCLUDES zero" if (diff_ci[0] > 0 or diff_ci[1] < 0) else "contains zero"
    print(f"  -> {verdict}: ", end="")
    if diff_ci[0] > 0:
        print("adapter targets missing-info prompts MORE than base. Calibration, not mimicry.")
    else:
        print("targeting gain not distinguishable from base on this cut.")
    print()
    label_out["lora_minus_base_disc"] = dd
    label_out["lora_minus_base_disc_ci95"] = list(diff_ci)
    out[labelname] = label_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "rebuttal" / "results" / "mimicry_split.json"))
    args = ap.parse_args()

    meta = load_prompt_meta()
    rows = load_examples()
    eval_pids = sorted({r["prompt_id"] for r in rows})

    # ---- persona composition of the evaluated prompts (context for FzpA W2) ----
    clin = sum(1 for pid in eval_pids if meta.get(pid, {}).get("clinician_posed"))
    print("=" * 78)
    print("HealthBench-Hard evaluated prompts: persona (heuristic self-identification)")
    print("=" * 78)
    print(f"  unique evaluated prompt_ids: {len(eval_pids)}")
    print(f"  explicit clinician self-ID : {clin} ({100*clin/max(len(eval_pids),1):.1f}%)")
    print("  -> predominantly patient/consumer-framed as posed; a clinician-facing")
    print("     benchmark (e.g. MedQA open-ended) is the right generality complement.")
    print()

    out = {"n_eval_prompts": len(eval_pids), "clinician_posed_heuristic": clin}

    # Primary label (theme OR clarification-rubric) + theme-only robustness label.
    idx_primary = build_pid_index(rows, group_of, meta)
    idx_theme = build_pid_index(rows, group_theme_only, meta)
    report(idx_primary, eval_pids, meta, "primary_theme_or_rubric", out)
    report(idx_theme, eval_pids, meta, "robustness_theme_only", out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
