# NeurIPS 2026 submission 32818: frozen source

This folder holds the anonymous LaTeX source of the paper as it was submitted to
NeurIPS 2026. It is a record. Do not edit it. The live paper is `paper/main.tex`.

## Version of record

The version of record is the official PDF from OpenReview:

- file: `neurips2026_submission_OFFICIAL.pdf`, 23 pages, 2,710,773 bytes
- sha256: `44bafafa4e426693a2d3ec4f6339166dde7629c24f50725ec171553fc63ea607`

The PDF is not in the repo. A copy is outside the repo at
`BODHI/_archive/paper_copies_2026-10-07/paper/neurips_submission/`.

## Where this source comes from

No file in the repo was the exact submitted source. We checked two candidates
against the official PDF with `compare_to_official.py` on 2026-10-07:

- `paper/neurips_submission/main.tex` was labelled "the anonymous source". It is
  a later working copy. It has no Conclusion section, a Keywords line and three
  extra appendices (Robustness Analyses, Training Score Distribution, Qualitative
  Examples). It builds to 28 pages. 816 words of the official PDF are missing
  from its build, and its build has 3,855 extra words.
- `paper/arxiv_upload/main.tex` is an export of the Overleaf project used for the
  submission, edited for arXiv between 2026-07-18 and 2026-07-22. It has the same
  sections as the official PDF, but it has a named author block and some text
  edits made after the submission.

This source is `paper/arxiv_upload/` with the changes below. We took each
replacement text from `paper/neurips_submission/`, which still had the wording
that the official PDF prints. We did not write any new text.

1. Anonymity: the author block is now `\author{Anonymous Author(s)}`. The
   `[preprint]` style option, the "Preprint. Under review" stamp and the
   `\@noticestring` override are removed.
2. Introduction, RLHF sentence: back to "Human raters in preference labeling
   favor decisive responses over qualified ones". The export said that models
   "remain poorly calibrated".
3. Introduction, epistemic virtues sentence: the citations to Singhal et al.
   (2023) and Nori et al. (2023) are back.
4. BODHI expansion: back to "Bridging, Open, Discerning, Humble, Inquiring".
   The export said "Balanced, Open-minded, Diagnostic, Humble, Inquisitive".
5. Related work, HealthBench sentence: back to "This structure penalizes both
   overconfidence (meeting negative criteria) and excessive hedging (failing
   positive criteria ...), making it suitable for evaluating calibrated behavior."
6. `references.bib`: the `tian2023just` entry is back to the arXiv preprint with
   five authors. The export had the EMNLP entry.
7. Layout: four figures are back at `\linewidth` (pipeline, epistemic bar
   chart, axis breakdown, forest plot), and the two appendix tables use `[t]`
   instead of `[H]`. We measured the figure widths from the official PDF: the
   text inside these figures is 1.111 times (pipeline) and 1.25 times (the
   other three) larger there than in the export build. This matches export
   widths of 0.9 and 0.8 against `\linewidth`. With `[H]` tables, Tables 3 and
   4 and the headings of Appendices D and E are on other pages than in the
   official PDF. With `[t]` they are on the same pages.

`checklist.tex`, `neurips_2026.sty` and the seven figure PDFs come from
`paper/arxiv_upload/` without change. The figure PDFs have the same md5 as
`paper/neurips_submission/figures/` (dated 2026-07-17). In the build, the
words inside each figure have the same relative positions as in the official
PDF (scale ratio 1.000 for all seven figures).

## Check against the official PDF

```bash
tectonic -X compile --outdir paper/build/submitted paper/submitted/neurips2026/main.tex
python3 paper/submitted/neurips2026/compare_to_official.py \
    path/to/neurips2026_submission_OFFICIAL.pdf paper/build/submitted/main.pdf
```

Result on 2026-10-07 (tectonic 0.17.0, pdftotext 26.04.0):

- 23 pages in both PDFs.
- Every section heading, table and figure is on the same page in both PDFs.
- With word order ignored, the words are the same. The script still prints 8
  and 12 words. Each one is a word that a line break splits in one PDF and not
  in the other (for example "985–998" against "985–" and "998").

## Differences that remain

- The venue added text to the official PDF after compilation (PDF producer:
  pypdf). This is a reviewer-copy footer on every page and one hidden line on
  pages 2 and 23. This text is in no source file.
- Line breaks inside paragraphs differ a little. The official PDF was built with
  pdfTeX on Overleaf. tectonic uses XeTeX, where microtype font expansion is off.
  The line count of a page differs by at most 4 lines. The page breaks are the same.
- A comparison of the PDF text cannot see source lines that print nothing, for
  example comments, unused macros, and `\raggedbottom` or `\clearpage` in the
  appendix. These lines come from the export and can differ from the submitted
  project.

## Anonymity check

On 2026-10-07: no clinician name, author name, affiliation or email is in
`main.tex`, `checklist.tex` or the figure metadata. Author names occur only in
the `bodhi2026` entry of `references.bib`. That entry is the cited BODHI paper.
The official PDF prints it in its reference list, and the tracked
`paper/references.bib` has the same entry.

## Known errors in this version

These errors are part of what we submitted. They are corrected in `paper/main.tex`,
not here.

- The hyperparameter table gives LoRA r = 16, alpha = 32, seven target modules,
  3 epochs and an effective batch of 16. The submitted adapters came from the
  tunix path with r = 8 and alpha = 16 (audit finding HYG-4).
- The training-loss figure comes from an early 5-epoch run on 838 traces, not
  from the reported adapters (audit finding PROV-6).
