# Lab conventions for this repo

Rules 1 to 10 were adopted on 2026-10-07. They come from section 4 of the audit of that
date, which traced most of the review's complaints to missing conventions. Rules 11 to 14
were adopted on 2026-10-08 with the architecture (D22): they keep the code general, so
that a new model, benchmark or protocol is a new record and not new code. Each rule says
why it exists and how this repo enforces it: a test, a hook, a make target, or a review
step that a person must do before merging or posting. Where a rule has no automatic
check yet, it says so.

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
test, the correction family, the decision rule and any margin. The plan is
[ANALYSIS_PLAN.md](ANALYSIS_PLAN.md). A change to it is a dated entry in its section 12,
never an edit.

- Why: the evaluation and its reading changed during the rebuttal, and labels were
  defined together with their results (EVAL-10).
- Enforced by: review step. The plan's commit must be older than the first commit of
  the results it covers. `bodhi_lora/manifest.py` records the commit of each run, so
  the order can be checked. While a cell runs, progress checks read its manifests only,
  never its scores (ANALYSIS_PLAN.md section 2.5). `tests/test_plan_coverage.py` fails
  when a cell record with status `planned` is not named in ANALYSIS_PLAN.md, or when
  ANALYSIS_PLAN.md, RUN_PLAN.md or EXPERIMENTS.md names a cell id with no record.

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
- Enforced by: the registry. Every model record names its family, and
  `validate_registry()` in `bodhi_lora/registry/schema.py` applies the family rule (R1 to
  R3, ARCHITECTURE.md section 3.1) to every cell whenever the registry loads. So
  `make check`, `tests/test_registry_configs.py` and every `python -m bodhi_lora` command
  fail on a cell that breaks the rule without a stated `same_family_waiver`, and the
  runner refuses it. A waived cell is marked † in every table and needs a judge from a
  fourth family (D6). The baseline set is the protocol records in `configs/protocols/`
  that ANALYSIS_PLAN.md names (H4); a new baseline after outputs exist is exploratory.

## 6. Validation gates are blocking

A pre-registered gate either passes, or a dated note amends it before any claim that
depends on it.

- Why: the physician-agreement gate failed, and the protocol changed without a record
  (EVAL-5).
- Enforced by: review step. The amendment goes in [DECISIONS.md](DECISIONS.md) as a
  dated entry (D5 is the first).

## 7. One results root, a manifest per run, synced before posting

Every result lives under one results root, `results/<cell_id>/seed_<n>/<stage>/`, set in
`configs/project.yaml`. Each stage has a manifest, and no number is posted from a file
that exists only on a remote machine.

- Why: the added cells existed only on the GPU host, the code wrote to six result
  locations, and the recorded settings did not describe what ran (PROV-1, HYG-4,
  REPRO-7).
- Enforced by: `bodhi_lora/paths.py` builds every path from `configs/project.yaml`, and
  `tests/test_no_hardcode.py` (rule 3) fails when package code names a results folder or
  the root itself. `bodhi_lora/manifest.py` writes the manifests.
  `tests/test_migration_provenance.py` checks that every file of the old locations has an
  entry in `results/provenance.json`. `.gitignore` decides by the kind of file under
  `results/`: manifests, cell records, id lists, adapter configs, trainer states, response
  tables, analysis and validation files are tracked; traces, raw eval and grade JSON,
  adapter weights and logs stay out (results/PROVENANCE.md).
  `scripts/check_doc_paths.py` (`make check`) fails when a doc names a path that does not
  exist. Review step: a reply or paper number cites a file that is in git.

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

## 11. One registry

Every model, benchmark, protocol and cell is one YAML record in `configs/`, in a file
named after its id. Project settings are in `configs/project.yaml`. No setting lives in a
script default, a shell profile or a second config folder.

- Why: settings lived in four places (`configs/`, a rebuttal config folder, the profile
  blocks of a launcher and script flag defaults), so the recorded settings did not match
  what ran (ARCHITECTURE.md section 1, REPRO-3).
- Enforced by: the pydantic schema in `bodhi_lora/registry/schema.py` forbids unknown
  keys and checks each record; `load_registry()` checks file names and references.
  `make check` loads the whole registry, and CI runs it. `tests/test_registry_configs.py`
  checks that every record loads, that each file is named after its id, and that the
  example records in ARCHITECTURE.md and EXPERIMENTS.md match the schema.
  `python -m bodhi_lora list cells --settings` prints every setting that differs between
  cells.

## 12. One runner

Every stage of every cell runs through `python -m bodhi_lora run <cell_id>`
(`bodhi_lora/pipeline/runner.py`). The Makefile calls the command line, the repository
tools in `scripts/` and the LaTeX build, and nothing else. There are no shell launchers
and no shims. A record of a past run is frozen: the runner refuses to run it again,
except to add judge re-grades, so a new run of the same design is a new cell.

- Why: the 2026-05 cell ran through one launcher and the 2026-07 cells through another,
  so neither path could reproduce both (HYG-5).
- Enforced by: `tests/test_pipeline_runner.py` (a stub run of the smoke cell through every
  stage, resume, refusal of recorded and external cells, the dry run). `make smoke` runs
  the smoke cell on the CPU, training included. Review step: a new file in `scripts/` is
  a repository tool, not a pipeline step, and `legacy/` stays frozen.

## 13. No names in code

The package names no model, benchmark, results path, run word or seed list. Each comes
from a record.

- Why: model names, benchmark case statements, seed lists and data paths were literals
  in code, so a new cell meant editing code (ARCHITECTURE.md section 1).
- Enforced by: `tests/test_no_hardcode.py` reads every string in `bodhi_lora/` with `ast`
  and fails on any model or benchmark id from the registry, a results or data path, a
  file or folder name with a run word (`modal`, `tpu`, `rebuttal`, `submitted` and
  others), or the five seeds as a literal (ARCHITECTURE.md section 9). It has no allow-list; an exception
  is a design change.

## 14. Uniform analysis

The same code computes every statistic for every cell, from the per-seed response tables.
A statistic is absent only when its input is absent, and then the output says "not
measured" and why. Contrasts and correction families come from the cell's conditions by
rule, not from a list in code.

- Why: the full statistics existed for one cell only, the added cells had a subset, and
  condition names were code, so a cell with a new protocol could not be analyzed
  (ARCHITECTURE.md section 1).
- Enforced by: `tests/test_no_hardcode.py` covers `bodhi_lora/analysis/` and
  `bodhi_lora/report/` like the rest of the package. `tests/test_analysis_cell.py` checks
  the contrast rule, the seed-level tests and "not measured" on synthetic tables, and
  `tests/test_analysis_equivalence.py` checks that the 2026-05 cell's statistics match
  the outputs they supersede (ARCHITECTURE.md section 10.3). Known gap: the paper's
  numbers still come from `scripts/paper_stats.py` (`make numbers`) until
  `python -m bodhi_lora report` is built.

## Generality rules at a glance

| Rule | Where | Automatic check |
|---|---|---|
| One registry | rule 11 | `make check`, `tests/test_registry_configs.py` |
| One runner | rule 12 | `tests/test_pipeline_runner.py`, `make smoke` |
| One results root | rule 7 | `tests/test_no_hardcode.py`, `tests/test_migration_provenance.py`, `make check` |
| No names in code | rule 13 | `tests/test_no_hardcode.py` |
| Uniform analysis | rule 14 | `tests/test_analysis_*.py`, `make analyze` |
| Analysis plan before runs | rule 3 | `tests/test_plan_coverage.py`; the order of commits is a review step |

## Also checked on every pull request

- `make check`: `scripts/check_no_secrets.sh` (no token shapes in tracked files),
  `scripts/check_doc_paths.py` (every repo path a doc names exists, HYG-14), and a load
  of the whole registry with the family rule.
- `pytest tests/` (`make test`).
- A rebuild of `paper/generated/` from the checkout, diffed against the committed files.

CI runs all of these.
