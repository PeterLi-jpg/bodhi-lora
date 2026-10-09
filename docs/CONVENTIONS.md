# Lab conventions for this repo

Adopted on 2026-10-07. These ten rules come from section 4 of the audit of that date,
which traced most of the review's complaints to missing conventions. Each rule says why
it exists and how this repo enforces it: a script, a hook, a make target, or a review
step that a person must do before merging or posting.

## 1. Claims ledger

Each claim in the abstract or the contributions maps to a number, a result file, a
script and a commit, with a status: live, restated or withdrawn.

- Why: withdrawals happened in OpenReview replies and never reached the paper source
  (CLAIM-1, CLAIM-2, CLAIM-3).
- Enforced by: [CLAIMS.md](CLAIMS.md). Review step: a pull request that changes the
  abstract, the contributions or a reply also updates CLAIMS.md. A claim marked
  withdrawn must not appear in `paper/main.tex`.

## 2. Generated numbers only

Every number in the paper or in a reply comes from a script output. Nobody types one by
hand.

- Why: transcription drift produced wrong sample sizes, confidence intervals and
  p-values (PROV-4, PROV-9, EVAL-11).
- Enforced by: `make numbers` writes every table and inline number from the result
  files, and `make paper` builds the paper from them. Review step: a literal number in
  `paper/main.tex` outside the generated files needs a reason in the pull request.

## 3. Dated analysis plan before results

Before a new run, commit a dated plan that fixes the metrics, the statistical unit, the
test, the correction family and any margin.

- Why: the evaluation and its reading changed during the rebuttal, and labels were
  defined together with their results (EVAL-10).
- Enforced by: review step. The plan's commit must be older than the first commit of
  the results it covers. `bodhi_lora/manifest.py` records the commit of each run, so
  the order can be checked.

## 4. Inferential language

"Preserved", "equivalent" and "non-inferior" need a stated margin and an equivalence
test. "Prevents" and "rules out" need a direct test. No significance word appears before
the test exists.

- Why: equivalence was inferred from p = 0.57, and significance words were posted
  before any test ran (EVAL-1, PROV-8, CLAIM-9).
- Enforced by: review step. Before a paper version or a reply goes out, run the command
  below and back each hit with a test, or remove it:

  ```bash
  grep -n -i -E "preserv|equivalen|non-?inferior|indistinguishable|essentially unchanged|prevents|rules? out|ruling out|no degradation" paper/main.tex
  ```

## 5. Baselines and grader families are fixed at design time

The baseline set and the grader families are chosen before any run. The filter grader
and the evaluator come from different families, and no base model may share a family
with the evaluator.

- Why: the prior-work baseline was promised and never built, and Med42-8B shares the
  evaluator's family (EVAL-8, EVAL-7).
- Enforced by: `bodhi_lora/protocols/` holds the baseline set. Cells that break the family
  rule are marked † and regraded with `bodhi_lora/pipeline/grade.py` (D6). Review
  step until the launcher checks this itself: a new base model's family is compared
  with Llama and Qwen before its cell runs (KNOWN_ISSUES).

## 6. Validation gates are blocking

A pre-registered gate either passes, or a dated note amends it before any claim that
depends on it.

- Why: the physician-agreement gate failed, and the protocol changed without a record
  (EVAL-5).
- Enforced by: review step. The amendment goes in [DECISIONS.md](DECISIONS.md) as a
  dated entry (D5 is the first).

## 7. One results root, a manifest per run, synced before posting

Every result lives under one results root in the repo, each run has a manifest, and no
number is posted from a file that exists only on a remote machine.

- Why: the added cells existed only on the GPU host, and the recorded settings did not
  describe what ran (PROV-1, HYG-4, REPRO-7).
- Enforced by: `bodhi_lora/manifest.py` writes the manifest. `.gitignore` decides by the
  kind of file under `results/`: manifests, cell records, id lists, adapter configs,
  trainer states, analysis and validation files are tracked; traces, raw eval and grade
  JSON, adapter weights and logs stay out (results/PROVENANCE.md). Review step: a reply or
  paper number cites a file that is in git.

## 8. One canonical paper source

`paper/main.tex` is the only live source. The submitted version is frozen in
`paper/submitted/neurips2026/`. Venue and arXiv versions are tags, not folders.

- Why: seven copies of `main.tex` in five versions, none of them the submitted source
  (HYG-9, CLAIM-4, CLAIM-15).
- Enforced by: `.gitignore` keeps the old copy folders (`paper/arxiv/`,
  `paper/arxiv_upload/`, `paper/preprint/`, `paper/neurips_submission/`) out of git.
  Review step: a new venue version is a tag on the mainline.

## 9. Branch policy

One mainline. Rebuttal and experiment work merges back into it. Scratch branches stay
local and are never pushed. No commit carries a coding-assistant trailer or mention.

- Why: the rebuttal ran on a branch planned never to merge, and scratch branches and a
  tool trailer reached the public remote (HYG-5, HYG-12, HYG-15).
- Enforced by: `hooks/pre-push` rejects `claude/*` and `worktree-agent*` pushes, pushed
  commits with tool traces, and machine-hostname identities. `hooks/commit-msg` rejects
  tool trailers and mentions. Activate both with `git config core.hooksPath hooks`.
  `scripts/maintenance/prune_branches.sh` lists stale branches and deletes them with
  `--apply`.

## 10. Anonymity by export

Review copies are a fresh export that a script checks. Rater identities never enter
tracked files.

- Why: anonymity by editing files in a public, author-named repo left names in history
  and in many files (HYG-1, EVAL-6, HYG-13).
- Enforced by: `scripts/export_anonymous.sh` builds a squashed copy, redacts identities
  and fails if any remain; pass rater and co-author names with `--deny-file`. Raters are
  AnonClinician1-4 in every tracked file. `.gitignore` keeps `IRR-clinicians/`,
  `paper/AUTHORS.md` and `CONTRIBUTORS.md` out of git. Review step: before sharing a
  review link, run the export and keep its gate output.

## Also checked on every pull request

- `python scripts/check_doc_paths.py`: every repo path a doc names exists (HYG-14).
- `bash scripts/check_no_secrets.sh`: no token shapes in tracked files.
- `pytest tests/`.

CI runs all three.
