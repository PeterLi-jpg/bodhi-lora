# SD4H 2026 workshop submission

Target: **Structured Data for Health (SD4H)** workshop at ICML 2026, Seoul.

## Key constraints

- **Deadline**: April 28, 2026 AoE
- **Format**: ICML 2026 LaTeX style, double-blind, fully anonymized
- **Length**: 4 pages main body (excluding references and appendix)
- **Archival**: No — we can still submit a full version elsewhere later

## Files

- `main.tex` — the paper itself
- `references.bib` — citations (seeded with core refs, expand as needed)
- `outline.md` — plain-prose version of the paper for quick review without LaTeX
- `figures/` — generated paper figures plus the source-controlled pipeline diagram in `figures/pipeline.tex`
- `TODO.md` — what's still a placeholder and blocks a real submission

## Figure inventory

`main.tex` currently references:

- `figures/epistemic_bar.pdf`
- `figures/epistemic_seeds.pdf`
- `figures/effect_size_forest.pdf`
- `figures/training_loss_seed42.pdf`
- `figures/ushape_difficulty.pdf`
- `figures/axis_breakdown.pdf`

Supporting assets retained in `figures/` but not currently included in the
main draft:

- `figures/pipeline.tex` and `figures/pipeline.pdf`
- `figures/score_distribution.pdf`

Legacy PNGs such as `theme_fail.png`, `training_loss.png`, and
`u_fail_smooth.png` are obsolete and can be removed by the build script.

## Before compiling

`main.tex` now uses the checked-in `neurips_2026.sty`, so no external style
download is required for this workspace copy.

Regenerate the figure bundle first:

```bash
python3 scripts/build_paper_figures.py --clean-legacy
```

## Build

```bash
cd paper/
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
# or: latexmk -pdf main.tex
```

Generated LaTeX build products in `paper/` are intentionally ignored.

## Anonymization

- `main.tex` currently has no author list (blind). DO NOT add names before
  submission.
- Any code link must go to an anonymized mirror, e.g.
  https://anonymous.4open.science/r/bohdi-lora
- Do not reference `REDACTED-FOR-ANONYMOUS-REVIEW/bohdi-lora` or MIT by name in acknowledgments.
