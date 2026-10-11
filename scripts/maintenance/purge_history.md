# History purge: prepared, not run

Prepared on 2026-10-07. Nothing here has been run against the remote. Running it
rewrites public history, so it is the first author's decision (audit section 5,
decision 8; docs/DECISIONS.md D16). Ask the raters about consent first (EVAL-6).

## What it removes

| Item | Where | Count on 2026-10-07 |
|---|---|---|
| Rater-named files `results_modal/irr/Grading-<name>.xlsx` | `origin/main` and older commits | 3 paths |
| Rater, adjudicator and handle text inside files and xlsx sheets | IRR scripts, IRR JSON, docs, paper drafts | 39 blobs in 21 paths |
| Rater names in commit messages | 6 commits | 6 |
| Full author block (names, affiliations, correspondence email) | `paper/main.tex` from `fcdcd9e` until `aa0c505`, on `origin/rebuttal/generality-experiments` | 1 block |
| `Co-Authored-By` trailers for an AI tool | `cf72766` and its twin `72a428b` | 2 commits |

It does not remove:

- MedGemma weights and optimizer state on `origin/results-temp`. Delete that branch with
  `prune_branches.sh` (step 0).
- Local-only refs in the working repo (step 9).
- Clones, forks and caches that already exist. GitHub keeps pull-request refs and
  cached views until GitHub Support purges them (step 8).

Every commit id from the first rewritten commit onward changes. Commit links in
OpenReview replies stop working, and the ids that tracked files cite must be rewritten
before the push (step 7b).

## Step 0. Before you start

1. Tell collaborators to stop pushing until step 9 is done.
2. Prune branches, so scratch and temp branches are not rewritten and pushed back:

   ```bash
   scripts/maintenance/prune_branches.sh            # dry run, read the list
   scripts/maintenance/prune_branches.sh --apply    # asks before deleting
   ```

3. Install the tool. These steps follow the git-filter-repo 2.47.0 manual
   (`--file-info-callback`, `--sensitive-data-removal`).

   ```bash
   python3 -m pip install git-filter-repo==2.47.0
   git filter-repo --version
   ```

## Step 1. A private folder outside every repo

The files below hold real names. Keep them out of every repo.

```bash
SECURE="$HOME/bodhi-purge"            # outside every git repo
mkdir -p "$SECURE" && chmod 700 "$SECURE"
NAMES=/path/to/clinician_names.txt    # one rater token per line, never committed
cp scripts/maintenance/purge_callback.py "$SECURE/"
```

## Step 2. The name map

One `Token==>Replacement` per line. Raters first, in the order of
`Grading-reviewer1..3.xlsx`, which is the order of the names file:

```bash
awk 'NF{n++; print $0 "==>AnonClinician" n}' "$NAMES" > "$SECURE/purge_map.txt"
```

Then add lines by hand, with the replacement each person has in `paper/AUTHORS.md`
or `IRR-clinicians/`:

- each adjudicator first name, and the adjudicator GitHub handle, `==>AnonClinician4`
- each co-author name that appears outside the `\author{...}` block, for example in
  acknowledgments or comments, `==>AnonAuthorN`

The `\author{...}` block itself is replaced as a whole by the callback.

## Step 3. Test the rules on the working repo (reads only)

Run inside the working repo, which has every ref:

```bash
python3 "$SECURE/purge_callback.py" selftest "$SECURE/purge_map.txt"
```

Go on only if it ends with `residual hits after rules: 0`. On 2026-10-07, with the
three rater tokens, both adjudicator names and the handle, it changed 39 blobs in 21
paths and left 0 residual hits. On every `paper/main.tex` that held the author block,
the block became `\author{Anonymous Authors}` and no affiliation or email was left.

## Step 4. Path list, message rules and callback body

```bash
git log --all --format= --name-only | sort -u \
  | grep -i -E "^results_modal/irr/Grading-($(grep -v '^$' "$NAMES" | paste -sd'|' -))\.xlsx$" \
  > "$SECURE/purge_paths.txt"
wc -l < "$SECURE/purge_paths.txt"     # expect 3

python3 "$SECURE/purge_callback.py" messages "$SECURE/purge_map.txt" > "$SECURE/messages.txt"
printf '%s\n' 'regex:(?im)^co-authored-by:[^\n]*(claude|anthropic)[^\n]*\n?==>' >> "$SECURE/messages.txt"

cat > "$SECURE/callback_body.py" <<'PY'
import importlib.util, os
g = globals()
if "_purger" not in g:
    spec = importlib.util.spec_from_file_location("purge_callback", os.environ["PURGE_CALLBACK"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    g["_purger"] = mod.Purger(os.environ["PURGE_MAP"])
purger = g["_purger"]
if filename is None or not purger.wants(filename):
    return (filename, mode, blob_id)
data = value.get_contents_by_identifier(blob_id)
new = purger.transform(filename, data)
if new != data:
    blob_id = value.insert_file_with_contents(new)
return (filename, mode, blob_id)
PY
```

## Step 5. Backup and a fresh clone

```bash
cd "$SECURE"
git clone --mirror git@github.com:PeterLi-jpg/bohdi-lora.git backup.git
git -C backup.git fsck --no-dangling && git -C backup.git rev-list --all --count
git clone --bare git@github.com:PeterLi-jpg/bohdi-lora.git purge.git
```

Open `backup.git` and check the counts before going on. Keep it until step 9 is done
everywhere.

## Step 6. Rewrite

```bash
cd "$SECURE/purge.git"
PURGE_CALLBACK="$SECURE/purge_callback.py" PURGE_MAP="$SECURE/purge_map.txt" \
git filter-repo --sensitive-data-removal \
  --invert-paths --paths-from-file "$SECURE/purge_paths.txt" \
  --file-info-callback "$SECURE/callback_body.py" \
  --replace-message "$SECURE/messages.txt"
```

## Step 7. Check before any push

All of these must print 0 or nothing:

```bash
cd "$SECURE/purge.git"
python3 "$SECURE/purge_callback.py" selftest "$SECURE/purge_map.txt"   # expect: blobs changed: 0, residual 0
git log --all --format= --name-only | grep -c -i -f <(grep -v '^$' "$NAMES")
git log --all --format=%B | grep -c -i -w -f <(grep -v '^$' "$NAMES")
git log --all -i -E --grep='co-authored-by:.*(claude|anthropic)' --format=%h
git log --all -S'MIT Critical Data' --format=%h
git rev-list --all --count     # compare with backup.git; a few emptied commits may drop
```

The first `grep -c` counts path hits, so a short token can match inside a longer
word (`flash`). Read any hit before you decide it is a name.

## Step 7b. Rewrite the commit ids that tracked files cite

filter-repo rewrites the ids in commit messages, not in files. On 2026-10-11 the tracked
files cited 64 commits, 47 of them after `cf72766`, and 78 of the 83 `<commit>:<path>`
references point at those 47. `scripts/check_doc_paths.py` checks each of them with
`git cat-file`, and `make check`, CI and `hooks/pre-commit` run it, so without this step
every later commit fails the hook. The run manifests under `results/` also cite the
commit they ran at. `filter-repo/commit-map` in the rewritten repository maps old ids to
new ones:

```bash
git clone "$SECURE/purge.git" "$SECURE/work" && cd "$SECURE/work"
git checkout cleanup/resubmission        # then the same for main, if it cites ids
python3 scripts/maintenance/remap_commit_ids.py "$SECURE/purge.git/filter-repo/commit-map"           # read every line
python3 scripts/maintenance/remap_commit_ids.py "$SECURE/purge.git/filter-repo/commit-map" --apply
git config core.hooksPath hooks
make check && make test
git commit -am "docs: cite the commit ids of the rewritten history"
git push "$SECURE/purge.git" HEAD:cleanup/resubmission
```

`make check` must pass, and `make test` must pass with the same skips as before the
rewrite, before step 8.

## Step 8. Push and clean the server

Only after the first author says so, and only after step 7b:

```bash
cd "$SECURE/purge.git"
git remote add origin git@github.com:PeterLi-jpg/bohdi-lora.git   # filter-repo removes it
git push --force origin 'refs/heads/*:refs/heads/*' 'refs/tags/*:refs/tags/*'
```

The explicit refspecs leave `refs/pull/*` alone, which GitHub refuses anyway. Then:

1. Follow GitHub's "Removing sensitive data from a repository" guide. Send GitHub
   Support the first changed commits (filter-repo writes them under
   `.git/filter-repo/` with `--sensitive-data-removal`) so they purge pull-request
   refs and cached views.
2. List forks with `gh api repos/PeterLi-jpg/bohdi-lora/forks --jq '.[].full_name'`.
3. Delete or refresh the anonymous mirror of the old submission
   (`anonymous.4open.science/r/bodhi-lora-CEB5`).

## Step 9. Other clones, including this machine

Re-clone. Do not run filter-repo again in other clones; it gives different ids.

On this machine the working repo also has local-only refs that hold the old history
and must never be pushed:

```bash
git for-each-ref --format='%(refname)' refs/codex refs/original refs/tags/backup-before-email-rewrite*
git ls-remote --tags origin | grep backup-before-email-rewrite   # pushed? then delete remotely too
```

Before you delete an old clone, move out what git does not hold: untracked and ignored
files (`data/raw/`, `IRR-clinicians/`, `paper/AUTHORS.md`, `CONTRIBUTORS.md`), stashes,
and every worktree under `_wt/`. Then re-clone, activate the hooks with
`git config core.hooksPath hooks`, and copy those files back.
