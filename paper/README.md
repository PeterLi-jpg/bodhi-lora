# Paper

`paper/main.tex` is the canonical source of the paper. It is the only copy we edit.
`checklist.tex` and `references.bib` are its inputs.

## Build

```bash
python3 scripts/build_paper_figures.py --clean-legacy   # figures, first build only
tectonic -X compile --outdir paper/build paper/main.tex
```

`make paper` runs the same build where the top-level Makefile has that target.
The output is `paper/build/main.pdf`. Git ignores `paper/build/`.

The figure PDFs in `paper/figures/` are build outputs and git ignores them. The one
exception is `figures/pipeline.pdf`, which is tracked. Its source is
`figures/pipeline_standalone.tex`.

## Numbers

Numbers in the paper come from `paper/generated/`. The stats scripts write that
folder; see its README. Do not type a result into `main.tex` by hand. A table that
still holds typed numbers must be changed to `\input` the generated file.

## Frozen submission

`paper/submitted/neurips2026/` holds the anonymous source of the NeurIPS 2026
submission, rebuilt and checked against the official PDF. It is a record. Do not edit
it. Its README gives the provenance and the known differences.

## No other copies

On 2026-10-07 we moved every other copy out of the repo, to
`BODHI/_archive/paper_copies_2026-10-07/` on the first author's machine. These were
`arxiv/`, `arxiv_upload/`, `overleaf/`, `preprint/`, `neurips_submission/`,
`BODHI_overleaf.zip`, `main_cam.*`, `preview.*` (the old workshop draft) and old build
outputs. Do not make a new copy of the paper in the repo. Make a venue or arXiv
version from `main.tex` at build time, and mark the commit with a tag.

## Anonymity

`main.tex` keeps the anonymous author block. Real names go only in the gitignored
`paper/AUTHORS.md`. Code links go to the anonymous mirror. Before you share a build,
check it for names, emails, account names, bucket names and local paths.
