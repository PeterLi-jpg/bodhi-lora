# Submission guide (rebuttals are per reviewer)

OpenReview takes one rebuttal per reviewer thread. There is no separate box per meta-review
weakness, so the AC's six weaknesses are answered inside the three reviewer replies below. Post
three comments, one per thread. Nothing else needs posting.

## What to post

| Thread | Paste | Chars | Answers meta weaknesses |
|---|---|---|---|
| Reviewer FzpA (rating 3) | `reviewer_FzpA.md` | 7,663 | W1, W2, W5, W6 (as open) |
| Reviewer LxMF (rating 4) | `reviewer_LxMF.md` | 6,319 | W3, W4 |
| Reviewer HomM (rating 2) | `reviewer_HomM.md` | 7,713 | W2, W3, W4, W6 |

Every weakness the AC listed is answered in at least one thread:

- **W1** MedGemma/HealthBench misalignment: FzpA
- **W2** one pipeline vs general framework: FzpA, HomM
- **W3** automated grading limited: LxMF, HomM
- **W4** calibration vs mimicry: LxMF, HomM
- **W5** discussion/contextualization clarity: FzpA
- **W6** no comparison with prior work: HomM (FzpA names it as remaining open)

## Before posting: replace four stale drafts

Four "Authors comment" boxes already hold drafts written before these runs existed. They contradict
the replies above, so they must be **replaced**, not left alongside them.

- **FzpA W2** says commercial models cannot be LoRA-adapted and lists "extending to Llama, Mistral"
  as future work. We have now done exactly that, so as written it hides the strongest answer to that
  reviewer.
- **FzpA W1** promises to reword the abstract to "a demonstration in one concrete setting." Keep the
  narrowing of the claim, but it should no longer describe the evidence as a single setting.
- **HomM W1** frames the work as "proof-of-concept rather than comprehensive validation" and lists
  testing other models and domains as next steps. Two of three are now done, so it concedes more than
  warranted, to the reviewer who rated Reject.
- **HomM W3** describes physician validation as a plan; it should state the current partial result
  (1 of 3 raters, kappa = 0.35).

Still accurate and can stand: the prior-work draft and the Table 1 sample-size draft.

## Formatting notes

- Paste as markdown. OpenReview renders the tables and `**bold**` as written.
- Math uses `$...$` (MathJax renders math mode only, no text-mode LaTeX).
- No em dashes anywhere, and every doc is under the 10,000 character limit.

## One claim is still future tense

Figure 2's two-panel split is the only paper change described as forthcoming rather than done. It
needs the figure regenerated. Everything else the replies describe as done is in `paper/main.tex`.
