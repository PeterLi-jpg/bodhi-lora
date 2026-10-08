"""Write the small derived inputs that paper_stats.py reads, from the large ignored files.

Two tables in the paper come from files that git ignores because of their size:

  data/sft_qwen_graded/seed_7_train.jsonl  (49 MB: the seed-7 training traces)
  results/seed_42/eval/<condition>.json     (8 MB: the seed-42 run graded by Qwen2.5-14B)

paper_stats.py needs only the scores in them. This script copies those scores, with
the prompt ids, into data/derived/, which is tracked. So `make numbers` rebuilds every
number in a clean checkout. The output keeps the source row order, so float sums in
paper_stats.py match a run on the source files bit for bit.

The derived files hold no response text. Each records the sha256 of its source, so a
reader with the source file can rerun this script and diff the output.

Usage:
    python3 scripts/export_derived_inputs.py [--filter-train PATH] [--qwen-grader-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "derived"
FILTER_SCORES = OUT_DIR / "seed_7_filter_scores.json"
QWEN_SCORES = OUT_DIR / "seed_42_qwen_grader_scores.json"
# The side run graded these three conditions only.
QWEN_CONDITIONS = ("base_no_wrapper", "base_bodhi", "lora_no_wrapper")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path: Path, obj: dict) -> None:
    # indent=1 puts every number on its own line; fold each [prompt_id, value] pair
    # back onto one line so the file stays short and diffs read row by row.
    text = re.sub(r"\[\n\s+(\"[^\"]*\"),\n\s+([^\n]+)\n\s+\]", r"[\1, \2]", json.dumps(obj, indent=1))
    if json.loads(text) != obj:
        raise SystemExit(f"export_derived_inputs: folding changed the content of {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def export_filter_scores(src: Path) -> dict:
    rows = []
    with open(src) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append([r["prompt_id"], r["grade"]["normalized_score"]])
    return {
        "what": "Qwen2.5-14B filter score of each trace in the seed-7 training split",
        "source": "data/sft_qwen_graded/seed_7_train.jsonl",
        "source_sha256": _sha256(src),
        "columns": ["prompt_id", "normalized_score"],
        "rows": rows,
    }


def export_qwen_scores(qdir: Path) -> dict:
    conditions, sources, graders = {}, {}, set()
    for cfg in QWEN_CONDITIONS:
        p = qdir / f"{cfg}.json"
        with open(p) as f:
            q = json.load(f)
        graders.add(q.get("run_metadata", {}).get("grader_model"))
        conditions[cfg] = [[r["prompt_id"], r["score"]] for r in q["results"]]
        sources[cfg] = {"source": f"results/seed_42/eval/{cfg}.json", "source_sha256": _sha256(p)}
    if len(graders) != 1:
        raise SystemExit(f"export_derived_inputs: grader differs across conditions: {graders}")
    return {
        "what": "seed-42 side run: aggregate HealthBench score per response, graded by Qwen2.5-14B",
        "grader_model": graders.pop(),
        "columns": ["prompt_id", "score"],
        "sources": sources,
        "conditions": conditions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--filter-train", type=Path,
                    default=ROOT / "data" / "sft_qwen_graded" / "seed_7_train.jsonl")
    ap.add_argument("--qwen-grader-dir", type=Path, default=ROOT / "results" / "seed_42" / "eval")
    a = ap.parse_args()
    _write(FILTER_SCORES, export_filter_scores(a.filter_train))
    _write(QWEN_SCORES, export_qwen_scores(a.qwen_grader_dir))


if __name__ == "__main__":
    main()
