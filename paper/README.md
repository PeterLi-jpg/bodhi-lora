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
- `figures/` — paper figures, including the source-controlled pipeline diagram in `figures/pipeline.tex`
- `TODO.md` — what's still a placeholder and blocks a real submission

## Before compiling

Download the official ICML 2026 style file from
https://icml.cc/Conferences/2026/StyleAuthorInstructions (or the 2025 one as a
near-equivalent). Place `icml2026.sty` in this directory. Main.tex uses it.

## Build

```bash
cd paper/
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
# or: latexmk -pdf main.tex
```

## Anonymization

- `main.tex` currently has no author list (blind). DO NOT add names before
  submission.
- Any code link must go to an anonymized mirror, e.g.
  https://anonymous.4open.science/r/bohdi-lora
- Do not reference `PeterLi-jpg/bohdi-lora` or MIT by name in acknowledgments.
