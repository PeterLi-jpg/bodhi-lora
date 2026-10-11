# Contributing

How to set up a clone, what to run before a pull request, and how review copies are
made. The lab rules behind these steps are in [docs/CONVENTIONS.md](docs/CONVENTIONS.md).

## Set up a clone

```bash
git clone https://github.com/PeterLi-jpg/bohdi-lora.git
cd bohdi-lora
git config core.hooksPath hooks
```

Then follow [docs/INSTALL.md](docs/INSTALL.md).

The last line activates the three hooks in `hooks/`. Git does not run hooks from a
tracked folder until you point `core.hooksPath` at it, so every clone needs it once. The
pre-commit framework is not used: it refuses to install while `core.hooksPath` is set.

- `hooks/pre-commit` runs the secrets scan and the doc path check (about 7 seconds).
- `hooks/commit-msg` rejects a commit message that names a coding assistant or carries
  its trailer (`Co-Authored-By` for a tool, "Generated with").
- `hooks/pre-push` rejects a push of a `claude/*` or `worktree-agent*` branch, a pushed
  commit whose message fails `commit-msg`, and a commit whose author or committer email
  is a machine hostname (`*.local`). That last one means git identity was never set.
  Deleting a scratch branch on the remote is allowed.

## Before a pull request

```bash
make check     # secrets scan, doc paths, and every record in configs/ loads and checks
make test      # pytest suite, no GPU needed
```

CI (`.github/workflows/ci.yml`) runs `make check` and the tests, compiles `bodhi_lora/`
and `scripts/`, and rebuilds `paper/generated/` from the checkout with `make numbers`,
failing on any difference. `.github/workflows/lint.yml` runs shellcheck on the shell
scripts and hooks.

## Pull requests

1. Branch from the mainline with a short descriptive name. Scratch branches stay local.
2. One logical change per commit. The subject line says what changed, for example
   `scripts: compute seed-level paired t for all dims`.
3. If you change code a doc describes, change the doc in the same pull request.
4. If you add a number to the paper, add the script that makes it. `make numbers` must
   regenerate it.
5. Open the pull request against the mainline. CI must pass.

## Results and data

- Commit what `.gitignore` lets through: the manifests, cell records, eval id lists,
  adapter configs, trainer states, response tables
  (`results/<cell_id>/seed_<n>/table.csv`) and each cell's analysis and validation folders. Git ignores traces and other JSONL, raw eval and
  grade JSON, adapter weights, checkpoint state and logs. The one exception is open
  decision O2: the 2026-05 cell's raw eval and grade JSON stay tracked
  (docs/ARCHITECTURE.md section 12).
- The runner writes a manifest into every stage folder (`bodhi_lora/manifest.py`).
- Never put a rater's name in a tracked file, a file name or a commit message. Raters
  are AnonClinician1-4. The key from number to person stays with the study coordinator.

## Review copies

Reviewers never get this repo or its history. Make a squashed, anonymized copy:

```bash
scripts/export_anonymous.sh /path/outside/repo/bodhi-review \
    --deny-file /path/outside/repo/names.txt
```

The script copies the tracked files of `HEAD`, redacts author names, handles, emails,
cloud project and bucket names, host names and home paths, and then scans the copy. It
exits 1 and makes no git commit if anything identifying is left. The deny file lists
extra tokens to block (rater and co-author names), one per line; keep it outside the
repo.

## Questions

Open a GitHub issue.
