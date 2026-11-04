#!/usr/bin/env python3
"""One-off: move every result into the single results root (docs/ARCHITECTURE.md section 10).

    python scripts/migrate_results.py --plan     # list every move; change nothing
    python scripts/migrate_results.py --apply    # move, verify, write the provenance

Before the move, results lived in seven places: results_modal/ (the 2026-05 cell),
results_rebuttal/ (the 2026-07 cells), results/seed_42 and results/_rescue (the 2026-04
pilot, ignored), rebuttal/results/ and data/rebuttal_cells/ (analysis outputs),
data/derived/ and data/irr_instructions/, plus the host logs in logs/host/, the pilot
adapter in checkpoints/ and a filter split in data/sft_qwen_graded/. After it, every
file is in results/<cell id>/ (or results/cross_cell/ for the outputs that span cells),
or in the archive folder next to the repository. Nothing is deleted.

Tracked files move with git mv, so git log --follow keeps their history; ignored files
move on disk. Every file is hashed before and after, and the run stops if a hash or a
count differs. results/provenance.json gets one entry per old file:
    {"old", "new" (a list: one file can go to several cells), "kind", "sha256",
     "bytes", "tracked_before", "tracked_after", "note"}
kind is moved (byte-identical), archived (moved next to the repo), or merged (text
joined into another file; its sha256 is of the old file). results/PROVENANCE.md says the
same in prose. A stage folder of a recorded cell gets a manifest that lists its files.

The 2026-05 cell's raw eval and grade JSON stayed tracked, as they were, until one
tracked/ignored rule covered every cell on 2026-10-11 (open decision O2); their entries in
results/provenance.json say so in an "untracked" field.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT.parent / "_archive" / "foundation_2026-10-08"
sys.path.insert(0, str(ROOT))

from bodhi_lora import manifest as mf  # noqa: E402
from bodhi_lora.registry import condition_from_legacy, load_registry  # noqa: E402

M = "healthbench__medgemma-27b-text__bodhi__2026-05"
PILOT = "healthbench__medgemma-27b-text__bodhi__2026-04"
CROSS = "cross_cell"
# results_rebuttal/<folder> -> cell id. The folder name is <bench>__<repo with / . : as _>.
REBUTTAL_CELLS = {
    "healthbench__mistralai_Mistral-Small-24B-Instruct-2501": "healthbench__mistral-small-24b__bodhi__2026-07",
    "healthbench__m42-health_Llama3-Med42-8B": "healthbench__med42-8b__bodhi__2026-07",
    "healthbench__BioMistral_BioMistral-7B": "healthbench__biomistral-7b__bodhi__2026-07",
    "chatdoctor__mistralai_Mistral-Small-24B-Instruct-2501": "chatdoctor__mistral-small-24b__bodhi__2026-07",
    "chatdoctor__m42-health_Llama3-Med42-8B": "chatdoctor__med42-8b__bodhi__2026-07",
    "medquad__mistralai_Mistral-Small-24B-Instruct-2501": "medquad__mistral-small-24b__bodhi__2026-07",
    "medqa__mistralai_Mistral-Small-24B-Instruct-2501": "medqa-open-preamble__mistral-small-24b__bodhi__2026-07",
    "medqa__BioMistral_BioMistral-7B": "medqa-open-preamble__biomistral-7b__bodhi__2026-07",
    "medqa__google_medgemma-27b-text-it": "medqa-open-preamble__medgemma-27b-text__bodhi__2026-07",
    "medquad__google_medgemma-27b-text-it": "medquad__medgemma-27b-text__bodhi__2026-07",
    # A run that left logs and no folder.
    "healthbench__google_gemma-2-9b-it": "healthbench__gemma-2-9b__bodhi__2026-07",
}
# The model part of the host's cell_<short>_<bench>.log and grid_<short>_<bench>.log names.
SHORT = {"mistral": "mistralai_Mistral-Small-24B-Instruct-2501", "med42": "m42-health_Llama3-Med42-8B",
         "biomistral": "BioMistral_BioMistral-7B", "medgemma": "google_medgemma-27b-text-it"}
BENCH_SHORT = {"hb": "healthbench", "healthbench": "healthbench", "chatdoctor": "chatdoctor",
               "medqa": "medqa", "medquad": "medquad"}
# Old condition file name -> condition (docs/ARCHITECTURE.md section 4.1). These four are
# the only conditions the old runs wrote.
CONDITIONS = {old: condition_from_legacy(old)
              for old in ("base_no_wrapper", "base_bodhi", "lora_no_wrapper", "lora_bodhi")}
EVALUATOR_2026_05 = "llama-3.1-8b"
EVALUATOR_2026_07 = "llama-3.1-8b-nous"
# Host logs that span several cells or none; they go to the archive.
HOST_ONLY_LOGS = {"grid.log", "install.log", "install_infer.log", "regrade_chatdoctor_driver.log"}


@dataclass
class Entry:
    old: str
    new: list = field(default_factory=list)
    kind: str = "moved"
    note: str = ""
    sha256: str = ""
    bytes: int = 0
    tracked_before: bool = False
    tracked_after: bool = False


def git(*args, check=True) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check).stdout


def sha256(path: Path) -> str:
    return mf.sha256_file(path)


def condition_file(name: str) -> str:
    return CONDITIONS[name] + ".json"


def rebuttal_map(rel: str) -> tuple[str, str] | None:
    """(new path, note) for one file under results_rebuttal/, or None to archive it."""
    parts = rel.split("/")
    folder, rest = parts[1], parts[2:]
    cell = REBUTTAL_CELLS[folder]
    base = f"results/{cell}"
    if rest == ["eval_ids.json"]:
        return f"{base}/shared/inputs/eval_ids.json", "the eval pool, one fixed set for every seed"
    if rest == ["multi_seed_summary.json"]:
        return None
    seed = rest[0]
    if rest[1] == "eval":
        name = rest[2]
        if name == "epistemic_scores.json":
            return f"{base}/{seed}/grade/epistemic__{EVALUATOR_2026_07}.json", "the evaluator's dimension grades"
        stem = name.removesuffix(".json")
        return f"{base}/{seed}/eval/{condition_file(stem)}", f"condition {stem} renamed {CONDITIONS[stem]}"
    if rest[1] == "ckpt":
        if rest[2] == "best":
            return f"{base}/{seed}/train/{rest[3]}", "the evaluated adapter (best validation loss)"
        return f"{base}/{seed}/train/{rest[2]}/{rest[3]}", "an epoch checkpoint"
    raise SystemExit(f"no rule for {rel}")


def host_log_cell(name: str) -> str | None:
    """The cell a host log belongs to, from its name, or None for host-only logs."""
    if name in HOST_ONLY_LOGS:
        return None
    m = re.match(r"^(gen|filter|eval|train|probe|regrade)_([a-z]+)_{1,2}(.+?)(_s\d+)?\.log$", name)
    if m and m.group(2) in BENCH_SHORT and f"{m.group(2)}__{m.group(3)}" in REBUTTAL_CELLS:
        return REBUTTAL_CELLS[f"{m.group(2)}__{m.group(3)}"]
    m = re.match(r"^(cell|grid)_([a-z0-9]+)_([a-z]+)\.log$", name)
    if m and m.group(2) in SHORT:
        return REBUTTAL_CELLS[f"{BENCH_SHORT[m.group(3)]}__{SHORT[m.group(2)]}"]
    fixed = {"gemma2_9b_hb.log": "healthbench__google_gemma-2-9b-it",
             "med42_hb.log": "healthbench__m42-health_Llama3-Med42-8B",
             "regrade2_chatdoctor.log": "chatdoctor__m42-health_Llama3-Med42-8B",
             "regrade2_driver.log": "chatdoctor__m42-health_Llama3-Med42-8B",
             "regrade_chatdoctor_med42.log": "chatdoctor__m42-health_Llama3-Med42-8B",
             "regrade_chatdoctor_mis_driver.log": "chatdoctor__mistralai_Mistral-Small-24B-Instruct-2501",
             "regrade_medquad_driver.log": "medquad__mistralai_Mistral-Small-24B-Instruct-2501"}
    if name in fixed:
        return REBUTTAL_CELLS[fixed[name]]
    raise SystemExit(f"no cell for host log {name}")


# The folders the move empties. Their files are found on disk, not through git, because
# which of them git ignores changes with the new .gitignore.
OLD_ROOTS = ("results_modal", "results_rebuttal", "results/seed_42", "results/_rescue", "rebuttal/results",
             "data/derived", "data/rebuttal_cells", "data/irr_instructions", "data/sft_qwen_graded",
             "data/sft", "logs/host", "checkpoints", "eval")


def files_on_disk() -> set[str]:
    out = set()
    for root in OLD_ROOTS:
        for dirpath, _, names in os.walk(ROOT / root):
            out |= {str((Path(dirpath) / n).relative_to(ROOT)) for n in names if n != ".DS_Store"}
    return out


def plan() -> list[Entry]:
    tracked = set(git("ls-files").splitlines())
    on_disk = files_on_disk()
    ignored = on_disk - tracked
    entries: list[Entry] = []

    def add(old, new, note="", kind="moved"):
        entries.append(Entry(old=old, new=[new] if isinstance(new, str) else list(new or []),
                             kind=kind if new else "archived", note=note))

    # The 2026-05 cell.
    for path in sorted(p for p in tracked if p.startswith("results_modal/")):
        rest = path.split("/")[1:]
        if rest[0].startswith("seed_"):
            seed, name = rest
            if name == "epistemic_scores.json":
                add(path, f"results/{M}/{seed}/grade/epistemic__{EVALUATOR_2026_05}.json", "the evaluator's dimension grades")
            else:
                stem = name.removesuffix(".json")
                add(path, f"results/{M}/{seed}/eval/{condition_file(stem)}", f"condition {stem} renamed {CONDITIONS[stem]}")
        elif rest[0] == "irr":
            add(path, f"results/{M}/validation/{rest[1]}", "physician grading sheets and agreement")
        else:
            add(path, f"results/{M}/analysis/{rest[0]}", "analysis output; make analyze supersedes it (ARCHITECTURE 10.3)")
    for path in sorted(p for p in tracked if p.startswith("data/irr_instructions/")):
        add(path, f"results/{M}/validation/instructions/{Path(path).name}", "rater instructions")
    add("data/derived/seed_7_filter_scores.json", f"results/{M}/seed_7/filter/filter_scores.json",
        "filter scores of the seed-7 train split, copied from the ignored train file")
    add("data/sft_qwen_graded/seed_7_train.jsonl", f"results/{M}/seed_7/filter/train.jsonl",
        "the seed-7 train split (ignored)")
    for name in ("mimicry_split.json", "interference_mechanism.json", "second_person_questions.json"):
        add(f"rebuttal/results/{name}", f"results/{M}/analysis/{name}",
            "analysis output of the 2026-05 cell; make analyze supersedes it (ARCHITECTURE 10.3)")
    for name in ("quality_drop.json", "cell_recipe.json"):
        add(f"rebuttal/results/{name}", f"results/{CROSS}/{name}",
            "analysis output over the 2026-07 cells; make analyze supersedes it (ARCHITECTURE 10.3)")
    add("data/rebuttal_cells/generated_values.json", f"results/{CROSS}/generated_values.json",
        "added-cell values from the raw outputs; make analyze supersedes it (ARCHITECTURE 10.3)")
    add("data/rebuttal_cells/posted_values.json", "rebuttal/posted_values.json",
        "the values as posted, transcribed; part of the review record")
    entries.append(Entry(old="data/rebuttal_cells/README.md", new=["rebuttal/README.md"], kind="merged",
                         note="appended to rebuttal/README.md as its section on the added-cell values"))

    # The eval ids of the 2026-07 HealthBench cells and the pilot: one file, several cells.
    hb = [c for c in REBUTTAL_CELLS.values() if c.startswith("healthbench__")]
    add("data/raw/hard_200_sample_ids.json",
        [f"results/{c}/shared/inputs/eval_ids.json" for c in sorted(hb + [PILOT])],
        "the fixed eval set of the 2026-07 HealthBench cells and the pilot; copied to each")

    # The 2026-07 cells.
    for path in sorted(p for p in (tracked | ignored) if p.startswith("results_rebuttal/")):
        target = rebuttal_map(path)
        if target is None:
            add(path, None, "the host's summary lists its seeds as [\"eval\"] (a wrong glob)")
        else:
            add(path, *target)
    for path in sorted(p for p in ignored if p.startswith("logs/host/")):
        cell = host_log_cell(Path(path).name)
        add(path, f"results/{cell}/logs/{Path(path).name}" if cell else None,
            "host log" if cell else "host log of several cells or none")

    # The pilot.
    for path in sorted(p for p in ignored if p.startswith("results/seed_42/") or p.startswith("results/_rescue/")):
        name = Path(path).name
        if path.startswith("results/seed_42/eval/") and name.removesuffix(".json") in CONDITIONS:
            add(path, f"results/{PILOT}/seed_42/eval/{condition_file(name.removesuffix('.json'))}",
                f"condition {name.removesuffix('.json')} renamed {CONDITIONS[name.removesuffix('.json')]}")
        elif path == "results/_rescue/raw_traces.jsonl":
            add(path, f"results/{PILOT}/shared/traces/traces.jsonl", "the pilot's teacher traces")
        elif path.startswith("results/_rescue/sft/"):
            add(path, f"results/{PILOT}/seed_42/filter/{name}", "the pilot's train/val split")
        else:
            add(path, None, "pilot figures and analysis outputs that no table uses")
    add("data/derived/seed_42_qwen_grader_scores.json", f"results/{PILOT}/analysis/qwen_grader_scores.json",
        "the pilot's seed-42 scores under the filter grader, copied from its eval files")
    for path in sorted(p for p in ignored if p.startswith("checkpoints/")):
        rel = path.split("/", 1)[1]
        if rel.startswith("seed_42/best/"):
            add(path, f"results/{PILOT}/seed_42/train/{rel.split('/', 2)[2]}", "the pilot adapter (best validation loss)")
        elif re.match(r"^seed_42/checkpoint-\d+/", rel):
            add(path, f"results/{PILOT}/seed_42/train/{rel.split('/', 1)[1]}", "a pilot epoch checkpoint")
        elif rel == "seed_42/trainer_state.json":
            add(path, None, "identical to the pilot adapter's trainer_state.json")
        else:
            add(path, None, "adapters of smoke runs")

    # Leftovers of removed tools.
    for path in sorted(p for p in ignored if p.startswith(("data/sft/", "eval/smoke/"))):
        add(path, None, "outputs of the removed smoke script")
    add("data/sft/.gitkeep", None, "the folder is no longer used")
    return entries


def check_plan(entries: list[Entry]) -> None:
    seen, problems = {}, []
    for e in entries:
        if not (ROOT / e.old).is_file():
            problems.append(f"missing source {e.old}")
        for n in e.new:
            if e.kind != "merged" and n in seen:
                problems.append(f"{n} is the target of {seen[n]} and {e.old}")
            seen[n] = e.old
            if e.kind != "merged" and (ROOT / n).exists():
                problems.append(f"target exists: {n}")
    if problems:
        raise SystemExit("plan check failed:\n  " + "\n  ".join(problems[:40]))


def apply(entries: list[Entry]) -> None:
    tracked = set(git("ls-files").splitlines())
    for e in entries:
        src = ROOT / e.old
        e.sha256, e.bytes, e.tracked_before = sha256(src), src.stat().st_size, e.old in tracked
    for e in entries:
        src = ROOT / e.old
        if e.kind == "archived":
            dst = ARCHIVE / e.old
            dst.parent.mkdir(parents=True, exist_ok=True)
            if e.tracked_before:
                shutil.copy2(src, dst)
                git("rm", "-q", e.old)
            else:
                shutil.move(str(src), str(dst))
            e.new = [str(dst.relative_to(ROOT.parent))]
            continue
        if e.kind == "merged":
            continue
        first, *rest = [ROOT / n for n in e.new]
        first.parent.mkdir(parents=True, exist_ok=True)
        if e.tracked_before:
            git("mv", e.old, e.new[0])
        else:
            shutil.move(str(src), str(first))
        for other in rest:
            other.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(first, other)
    # Verify: every moved file is byte-identical where it landed.
    bad = [f"{e.old} -> {n}" for e in entries if e.kind == "moved" for n in e.new if sha256(ROOT / n) != e.sha256]
    bad += [e.old for e in entries if e.kind == "archived" and sha256(ROOT.parent / e.new[0]) != e.sha256]
    if bad:
        raise SystemExit("hash check failed:\n  " + "\n  ".join(bad))


STAGE_FOLDERS = ("traces", "filter", "train", "eval", "grade")


def _files(folder: Path) -> dict:
    """{relative path: sha256} of the files in a folder, its manifests left out."""
    if not folder.is_dir():
        return {}
    return {str(f.relative_to(folder)): sha256(f) for f in sorted(folder.rglob("*"))
            if f.is_file() and not f.name.endswith("manifest.json") and f.name != ".DS_Store"}


def record_manifest(stage: str, **kwargs) -> dict:
    """A manifest for the outputs of a run made before this package.

    build_manifest records the environment of the process that writes the manifest. Here
    that process is this script, not the run, so the package versions, hardware and host
    are left out, and written_by names the script and the commit it ran at.
    """
    m = mf.build_manifest(stage, **kwargs)
    for key in ("packages", "hardware", "host_sha256"):
        del m[key]
    m["written_by"] = {"script": "scripts/migrate_results.py --manifests", "git": m.pop("git")}
    return m


def write_record_manifests() -> list[Path]:
    """cell.yaml and a manifest per stage folder for every recorded cell.

    The stages of a recorded cell ran before this package, so its manifests list what the
    move kept (file and sha256) and the facts the old records give, with status recorded,
    or not_kept when the files stayed on the GPU host or were lost. The runner never
    reruns a recorded cell, so these manifests are not resume markers.
    """
    import yaml
    reg = load_registry(ROOT / "configs")
    recipe = json.loads((ROOT / "results" / CROSS / "cell_recipe.json").read_text())["cells"]
    by_cell = {cell: recipe.get(folder) for folder, cell in REBUTTAL_CELLS.items()}
    written = []
    for cid, cell in sorted(reg.cells.items()):
        if cell.report.status == "planned":
            continue
        root = ROOT / "results" / cid
        root.mkdir(parents=True, exist_ok=True)
        (root / "cell.yaml").write_text(yaml.safe_dump(cell.model_dump(mode="json"), sort_keys=False))
        written.append(root / "cell.yaml")
        rec = by_cell.get(cid) or {}
        flt = rec.get("filter") or {}
        if flt:
            # Extracted from the host filter log through cell_recipe.json (PROV-3).
            extract = {k: flt[k] for k in ("tau", "raw", "graded", "kept", "train", "val", "n_filter_passes",
                                           "rubric_parse_failures", "score_field") if k in flt}
            shared = root / "shared" / "filter"
            shared.mkdir(parents=True, exist_ok=True)
            m = record_manifest("filter", cell=cid, extra={
                "status": "not_kept", "recorded": extract, "source": "results/cross_cell/cell_recipe.json",
                "source_log_sha256": flt.get("sha256"),
                "note": "graded, kept and split traces stayed on the GPU host; counts and tau from the filter log"})
            mf.write_manifest(shared / "manifest.json", m)
            written.append(shared / "manifest.json")
        for seed in cell.seeds:
            sd = root / f"seed_{seed}"
            if not sd.is_dir():
                continue
            for stage in STAGE_FOLDERS:
                folder = sd / stage
                files = _files(folder)
                status = "recorded" if files else "not_kept"
                note = {"traces": "teacher traces are shared by every seed",
                        "filter": "the train/val split",
                        "train": "adapter files; weights are ignored by git or were not retained",
                        "eval": "replies and rubric grades per condition",
                        "grade": "the evaluator's dimension grades"}[stage]
                extra = {"status": status, "files": files, "note": note}
                if stage == "traces" and (root / "shared" / "traces").is_dir():
                    extra["shared"] = f"results/{cid}/shared/traces"
                if stage == "filter" and flt:
                    extra["shared"] = f"results/{cid}/shared/filter"
                if stage == "train" and cell.train is not None and cell.train.engine == "external":
                    extra["note"] = f"trained outside this package ({cell.train.record}); adapters not retained"
                folder.mkdir(parents=True, exist_ok=True)
                mf.write_manifest(folder / "manifest.json", record_manifest(stage, cell=cid, seed=seed, extra=extra))
                written.append(folder / "manifest.json")
        models = {"base": reg.models[cell.model], "filter": reg.models[cell.grading.filter],
                  "evaluator": reg.models[cell.grading.evaluator],
                  **{f"judge:{j}": reg.models[j] for j in cell.grading.judges}}
        index = {str(m.relative_to(root)): sha256(m) for m in sorted(root.rglob("manifest.json")) if m.parent != root}
        mf.write_manifest(root / "manifest.json", record_manifest("cell", cell=cid, models=models, extra={
            "status": "recorded", "record_status": cell.report.status,
            "family": {"waiver": cell.grading.same_family_waiver},
            "note": "outputs of a run made before this package; results/PROVENANCE.md maps every old path",
            "stage_manifests": index}))
        written.append(root / "manifest.json")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--manifests", action="store_true",
                      help="write cell.yaml and the stage manifests of every recorded cell")
    args = ap.parse_args()
    load_registry(ROOT / "configs")
    if args.manifests:
        written = write_record_manifests()
        print(f"wrote {len(written)} cell records and manifests")
        return 0
    prov_path = ROOT / "results" / "provenance.json"
    done = {e["old"]: Entry(**e) for e in json.loads(prov_path.read_text())} if prov_path.is_file() else {}
    entries = [e for e in plan() if e.old not in done]
    check_plan(entries)
    counts = {}
    for e in entries:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    if args.plan:
        for e in entries:
            print(f"{e.kind:8s} {e.old} -> {', '.join(e.new) or 'archive'}")
        print(f"{len(entries)} files: {counts}")
        return 0
    apply(entries)
    entries = list(done.values()) + entries
    out = [e.__dict__ for e in sorted(entries, key=lambda e: e.old)]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "provenance.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"moved and verified {len(entries)} files: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
