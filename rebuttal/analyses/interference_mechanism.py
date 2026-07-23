"""Rebuttal analysis: what drives the LoRA+BODHI red-flag drop? (Reviewer LxMF W4)

LxMF: "The claim that CoT and LoRA compete for the same behavioral capacity is
speculative. The drop in red-flag identification could be caused by context window
truncation or attention dilution from long outputs, because this combination
generates long outputs."

We test LxMF's alternative directly on the committed eval outputs. The paper's
mechanism (Figure interference) is that under LoRA+BODHI the model emits the BODHI
Pass-1 INTERNAL ANALYSIS format ("RED FLAGS: None", "KEY UNCERTAINTIES", ...) instead
of a patient-facing answer. If that is the cause, then:

  (a) red-flag identification should be depressed specifically on the responses that
      LEAK the Pass-1 analysis format, and
  (b) LoRA+BODHI responses that do NOT leak (clean patient-facing) should retain a
      red-flag rate close to LoRA-alone, and
  (c) the drop should NOT be explained by length/truncation: if it were attention
      dilution from long outputs, red-flag would fall with length regardless of format.

Runs against results_modal/. No inference, no H100.
"""

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEED_DIRS = ["seed_7", "seed_13", "seed_42", "seed_99", "seed_101"]
CONFIG_DISPLAY = {
    "base_no_wrapper": "Base",
    "base_bodhi": "Wrapper",
    "lora_no_wrapper": "LoRA",
    "lora_bodhi": "LoRA+Wrapper",
}
CONFIG_ORDER = ["base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi"]

# Canonical BODHI Pass-1 analysis-format section headers. The internal analysis
# pass emits these numbered/structured sections; a patient-facing answer does not.
LEAK_MARKERS = [
    r"TASK TYPE",
    r"KEY UNCERTAINT",
    r"SAFE RECOMMENDATION",
    r"RED FLAGS?\s*:",
    r"WHAT I THINK",
    r"QUESTIONS TO ASK",
    r"AUDIENCE\s*:",
    r"DIFFERENTIAL",
    r"SCOPE (OF|BOUNDAR)",
]
LEAK_RE = [re.compile(m, re.IGNORECASE) for m in LEAK_MARKERS]


def count_leak_markers(text):
    return sum(1 for r in LEAK_RE if r.search(text or ""))


def is_leaked(text, threshold=2):
    """A response 'leaks' the Pass-1 analysis format if >= threshold canonical
    analysis-section headers appear (a patient-facing answer has ~0)."""
    return count_leak_markers(text) >= threshold


def load(config_name):
    """Pool a config's per-response rows across the 5 seeds."""
    rows = []
    for sd in SEED_DIRS:
        path = REPO / "results_modal" / sd / "epistemic_scores.json"
        if not path.exists():
            continue
        data = json.load(open(path))
        for cfg in data["configs"]:
            if cfg["name"] != config_name:
                continue
            for ex in cfg["examples"]:
                if ex.get("parse_failure"):
                    continue
                sc = ex.get("scores") or {}
                rf = sc.get("red_flag_identification")
                if rf is None:
                    continue
                resp = ex.get("response") or ""
                rows.append({
                    "prompt_id": ex["prompt_id"],
                    "red_flag": rf,
                    "len": len(resp),
                    "leaked": is_leaked(resp),
                    "n_markers": count_leak_markers(resp),
                })
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "rebuttal" / "results" / "interference_mechanism.json"))
    args = ap.parse_args()

    data = {c: load(c) for c in CONFIG_ORDER}
    out = {"configs": {}}

    print("=" * 82)
    print("Pass-1 analysis-format LEAKAGE and red-flag (0-2), by config (pooled 5 seeds)")
    print("=" * 82)
    hdr = f"{'Config':14s} {'n':>5s} {'leaked%':>8s} {'redflag_all':>12s} {'len_med':>8s}"
    print(hdr); print("-" * len(hdr))
    for c in CONFIG_ORDER:
        rows = data[c]
        n = len(rows)
        leak = mean([1.0 if r["leaked"] else 0.0 for r in rows])
        rf = mean([r["red_flag"] for r in rows])
        lens = sorted(r["len"] for r in rows)
        med = lens[len(lens)//2] if lens else 0
        print(f"{CONFIG_DISPLAY[c]:14s} {n:5d} {100*leak:7.1f}% {rf:12.3f} {med:8d}")
        out["configs"][c] = {"n": n, "leaked_rate": leak, "red_flag_mean": rf, "len_median": med}
    print()

    # --- The decisive cut: within LoRA+Wrapper, split leaked vs non-leaked ---
    lb = data["lora_bodhi"]
    leaked = [r for r in lb if r["leaked"]]
    clean = [r for r in lb if not r["leaked"]]
    lora = data["lora_no_wrapper"]
    print("=" * 82)
    print("Decisive test: LoRA+Wrapper split by whether the response leaked the analysis format")
    print("=" * 82)
    print(f"  LoRA (no wrapper) red-flag:                 {mean([r['red_flag'] for r in lora]):.3f}  (n={len(lora)})")
    print(f"  LoRA+Wrapper, NON-leaked (patient-facing):  {mean([r['red_flag'] for r in clean]):.3f}  (n={len(clean)})")
    print(f"  LoRA+Wrapper, LEAKED (Pass-1 analysis):     {mean([r['red_flag'] for r in leaked]):.3f}  (n={len(leaked)})")
    print()
    print("  Response length (median chars):")
    def med_len(rs):
        L = sorted(r["len"] for r in rs); return L[len(L)//2] if L else 0
    print(f"    non-leaked = {med_len(clean)}   leaked = {med_len(leaked)}")
    out["decisive"] = {
        "lora_red_flag": mean([r["red_flag"] for r in lora]),
        "lora_bodhi_clean_red_flag": mean([r["red_flag"] for r in clean]),
        "lora_bodhi_leaked_red_flag": mean([r["red_flag"] for r in leaked]),
        "n_clean": len(clean), "n_leaked": len(leaked),
        "len_median_clean": med_len(clean), "len_median_leaked": med_len(leaked),
    }
    print()

    # --- Length control: does red-flag fall with length WITHIN non-leaked? ---
    # If LxMF's attention-dilution story held, red-flag should drop with length even
    # among clean responses. Bin clean LoRA+Wrapper responses by length quartile.
    print("=" * 82)
    print("Length control: red-flag by length quartile, NON-leaked LoRA+Wrapper only")
    print("=" * 82)
    if clean:
        cl = sorted(clean, key=lambda r: r["len"])
        q = len(cl) // 4 or 1
        quarts = [cl[:q], cl[q:2*q], cl[2*q:3*q], cl[3*q:]]
        qout = []
        for i, grp in enumerate(quarts, 1):
            rf = mean([r["red_flag"] for r in grp])
            print(f"  Q{i} (len~{med_len(grp)}): red-flag {rf:.3f}  (n={len(grp)})")
            qout.append({"quartile": i, "len_median": med_len(grp), "red_flag": rf, "n": len(grp)})
        out["length_control_clean"] = qout
    print()

    print("=" * 82)
    print("Interpretation")
    print("=" * 82)
    c_rf = mean([r['red_flag'] for r in clean])
    l_rf = mean([r['red_flag'] for r in leaked])
    lora_rf = mean([r['red_flag'] for r in lora])
    if leaked and clean and l_rf < c_rf - 0.2 and abs(c_rf - lora_rf) < 0.25:
        print("  Red-flag collapses on the LEAKED (Pass-1-format) responses and stays ~LoRA-level")
        print("  on the NON-leaked ones. The drop tracks FORMAT LEAKAGE, not length/truncation.")
        print("  -> supports the paper's competition mechanism; refutes attention-dilution as the cause.")
    else:
        print("  Pattern is weaker than expected on this cut; report the numbers as-is.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
