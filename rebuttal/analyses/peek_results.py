"""Peek at a rebuttal cell's Base-vs-LoRA epistemic aggregates across whatever
seeds have finished so far. Usage: python peek_results.py <cell_work_dir>"""
import glob, json, statistics, sys

work = sys.argv[1].rstrip("/")
KEYS = [
    ("active_inquiry_rate", "active inquiry %", 100),
    ("context_seeking_mean", "context-seek 0-2", 1),
    ("red_flag_identification_mean", "red-flag 0-2", 1),
    ("scope_bounding_mean", "scope-bound 0-2", 1),
    ("hedging_mean", "hedging 0-2", 1),
]
CFGS = [("base_no_wrapper", "Base"), ("base_bodhi", "Wrapper"),
        ("lora_no_wrapper", "LoRA"), ("lora_bodhi", "LoRA+CoT")]

rows = {}
files = sorted(glob.glob(f"{work}/seed_*/eval/epistemic_scores.json"))
for f in files:
    d = json.load(open(f))
    for c in d["configs"]:
        for k, _, _ in KEYS:
            rows.setdefault(c["name"], {}).setdefault(k, []).append(c["aggregates"].get(k))


def mean(cfg, k, mult):
    v = [x for x in rows.get(cfg, {}).get(k, []) if isinstance(x, (int, float))]
    return statistics.mean(v) * mult if v else float("nan")


print(f"{work.split('/')[-1]}   ({len(files)} of 5 seeds done)")
print(f"  {'dimension':16s} " + "  ".join(f"{lbl:>9s}" for _, lbl in CFGS))
for k, label, mult in KEYS:
    cells = "  ".join(f"{mean(name, k, mult):9.2f}" for name, _ in CFGS)
    print(f"  {label:16s} {cells}")
