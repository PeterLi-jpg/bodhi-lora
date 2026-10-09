"""Build and verify the figure bundle used by paper/main.tex.

This script consolidates the paper figure pipeline so the manuscript figure
directory can be regenerated from the checked-in results in one command.

Every step runs with SOURCE_DATE_EPOCH and PYTHONHASHSEED fixed (0 unless the
caller sets them). matplotlib and tectonic stamp PDFs with the build time
otherwise, and plot_postprocessing.py bins prompts by iterating a set of
prompt ids, whose order (and so the U-shape bootstrap band) changes with the
string hash seed. With both fixed, two builds of the same data are
byte-identical.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


FIGURE_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")

LEGACY_FIGURES = (
    "theme_fail.png",
    "training_loss.png",
    # The paper dropped the training-loss figure: the only local trainer state is
    # an early 838-trace run, not the submitted adapters (audit PROV-6, D14).
    "training_loss_seed42.pdf",
    "u_curve.png",
    "u_curve_smooth.png",
    "u_fail.png",
    "u_fail_smooth.png",
    "u_scatter.png",
)

SUPPORTING_ASSETS = (
    "pipeline.pdf",
    "pipeline.tex",
    "pipeline_standalone.tex",
    "score_distribution.pdf",
)


def parse_referenced_figures(tex_path: Path) -> list[str]:
    text = tex_path.read_text()
    figures = []
    for match in FIGURE_RE.finditer(text):
        figure_path = match.group(1)
        if not figure_path.startswith("figures/"):
            continue
        figures.append(figure_path.split("/", 1)[1])
    return sorted(set(figures))


def _build_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("SOURCE_DATE_EPOCH", "0")
    env.setdefault("PYTHONHASHSEED", "0")
    return env


def run_step(cmd: list[str], root: Path) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=root, env=_build_env())


def build_pipeline_figure(root: Path, out_dir: Path) -> None:
    """Compile Figure 1 from its TikZ source.

    Figure 1 is hand-drawn TikZ, not matplotlib, so it needs a LaTeX run. If no
    engine is installed we leave the checked-in pipeline.pdf alone: overwriting a
    correct figure with nothing is worse than skipping the step.
    """
    source = root / "paper" / "figures" / "pipeline_standalone.tex"
    engine = shutil.which("tectonic")
    if engine is None:
        print(f"SKIP {source.relative_to(root)}: tectonic not installed")
        return

    with tempfile.TemporaryDirectory() as tmp:
        run_step([engine, "-X", "compile", str(source), "--outdir", tmp], root)
        target = out_dir / "pipeline.pdf"
        shutil.copyfile(Path(tmp) / "pipeline_standalone.pdf", target)
    print(f"Wrote {target}")


def maybe_remove(path: Path, root: Path) -> None:
    if path.exists():
        path.unlink()
        print(f"Removed legacy figure -> {path.relative_to(root)}")


# The submitted cell, whose results the paper's figures show.
MAIN_CELL = "results/healthbench__medgemma-27b-text__bodhi__2026-05"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", default="paper/main.tex")
    parser.add_argument("--out-dir", default="paper/figures")
    parser.add_argument(
        "--aggregate",
        default=f"{MAIN_CELL}/analysis/cross_seed_aggregate.json",
    )
    parser.add_argument("--data-dir", default=MAIN_CELL)
    parser.add_argument("--hb-hard", default="data/raw/healthbench/hard.jsonl")
    parser.add_argument(
        "--clean-legacy",
        action="store_true",
        help="remove legacy PNG outputs from paper/figures before rebuilding",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    tex_path = (root / args.tex).resolve()
    out_dir = (root / args.out_dir).resolve()
    aggregate = (root / args.aggregate).resolve()
    data_dir = (root / args.data_dir).resolve()
    hb_hard = (root / args.hb_hard).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_legacy:
        for name in LEGACY_FIGURES:
            maybe_remove(out_dir / name, root)

    python = sys.executable
    build_pipeline_figure(root, out_dir)
    run_step(
        [
            python,
            str(root / "scripts" / "plot_epistemic_figures.py"),
            "--aggregate",
            str(aggregate),
            "--out-dir",
            str(out_dir),
        ],
        root,
    )
    run_step(
        [
            python,
            str(root / "scripts" / "plot_postprocessing.py"),
            "--data-dir",
            str(data_dir),
            "--hb-hard",
            str(hb_hard),
            "--out-dir",
            str(out_dir),
        ],
        root,
    )

    referenced = parse_referenced_figures(tex_path)
    missing = [name for name in referenced if not (out_dir / name).exists()]
    if missing:
        print("Missing figure(s) referenced by the manuscript:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        return 1

    print("\nReferenced manuscript figures:")
    for name in referenced:
        print(f"  - paper/figures/{name}")

    supporting = [name for name in SUPPORTING_ASSETS if (out_dir / name).exists()]
    if supporting:
        print("\nSupporting figure assets retained:")
        for name in supporting:
            print(f"  - paper/figures/{name}")

    extras = sorted(
        path.name
        for path in out_dir.iterdir()
        if path.is_file()
        and path.name not in set(referenced)
        and path.name not in set(supporting)
    )
    if extras:
        print("\nAdditional generated assets present:")
        for name in extras:
            print(f"  - paper/figures/{name}")

    print("\nFigure bundle is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
