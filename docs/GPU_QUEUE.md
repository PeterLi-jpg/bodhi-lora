# GPU queue

`scripts/gpu_queue.sh` runs a list of commands on one GPU of the shared host, one after
another, with no person at the keyboard. It holds the card in the group's lease file
from the first command to the last, so no other job takes the card between two
commands. Added on 2026-10-12 for R3 to R7 ([RUN_PLAN.md](RUN_PLAN.md) section 4.2),
which run on GPU 3 after R2. The script has no BODHI-specific part, so another project
on the host can use it.

## 1. What it does

In this order:

1. **Wait** (with `--after`). It checks every minute whether that tmux session still
   exists. When the session is gone, it reads the session's exit marker. It goes on
   only if the first line of the marker is `0`. If the marker is missing or says
   anything else, it stops and claims nothing. A session that this script did not start
   leaves no marker; for it, `--after-check <command>` replaces the marker: the queue
   runs the command in the checkout, before the claim and the switch, and goes on only
   if it exits 0.
2. **Claim.** First the card must use less than 2,000 MiB, after up to 60 seconds of
   waiting for the job before it to let go. The lease tool's `claim` gives out a card
   whose row is free whatever the card holds, and a vLLM server that outlived a killed
   job holds its card with no lease, because the engine starts the server in a session
   of its own. If the card is in use, the queue logs the `nvidia-smi` table, stops and
   claims nothing. The row must say `free`: the queue never claims a busy row, even one
   the lease tool would count stale (its PID gone, the card empty), because a launcher's
   PID often exits while its job is still starting (the note at the top of the node's
   lease file). A row of this owner held by its holder session (section 9) is used again
   with no claim. Then it runs `<lease tool> claim <gpu> <owner> <purpose>` with
   `CLAIM_PID` set to its own PID, or to the holder's with `--keep-lease`. That process
   lives as long as the queue, so the lease stays live between commands. The lease tool
   takes its own lock and refuses a card that another live job holds. If the tool
   refuses, or the lease file does not then show the card with this owner and PID, the
   queue stops and runs nothing. With `--gpu any` it does not stop: it waits for a card
   (section 9).
3. **Switch the checkout** (with `--bundle` and `--commit`). It stops if `git status`
   shows a change to a tracked file other than a `--keep-local` file, or an untracked
   file outside the `--allow-untracked` paths; ignored files are allowed. It makes this check at the start too, before the
   wait, so a checkout that cannot switch fails at once. It fetches the bundle's refs
   into `FETCH_HEAD` only, so no branch changes, and checks out the commit as a detached
   HEAD. After the switch the checkout must pass the same check, so every manifest
   records a clean commit. A `--keep-local` path (relative to the checkout, or absolute
   inside it) is a file or a folder, which means every file git tracks under it. Its
   files come through the switch by two rules:
   - The host's own copy (a file marked skip-worktree, one git does not track, or a
     tracked file the host changed) comes back whatever the commit holds. Git 2.37 and
     later show a file that a sparse-checkout pattern leaves out, but that is present,
     as a changed tracked file, so it counts as the host's copy too. If the commit tracks the file, it gets
     skip-worktree again; if not, it goes into the checkout's `.git/info/exclude`, unless
     an ignore rule covers it already.
   - A committed copy (tracked, with no local change) comes back only if the commit no
     longer tracks it, that is, if the switch deleted it. If the commit still tracks it,
     the commit's version stays.
4. **Source the env file** (with `--env`), with every variable exported. The script
   never logs its values.
5. **Run the commands**, each in the checkout and in its own process group, with
   `CUDA_VISIBLE_DEVICES` set to the claimed card and `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so
   CUDA numbers the cards as `nvidia-smi` and the lease file do. A line whose `--gpus`
   names another card stops the queue before any line runs, because the runner sets
   `CUDA_VISIBLE_DEVICES` from `--gpus`. `{gpu}` in a line becomes the claimed card's
   index, for a queue that does not know its card in advance. The queue stops at the
   first command that fails.
6. **Release and mark**, on every exit: success, a failed command, a refused setup step,
   or SIGINT, SIGTERM or SIGHUP. With `--keep-lease` it keeps the card and its holder
   (section 9). Otherwise it releases the card only if the lease file still names
   this queue, and only once the card uses less than 2,000 MiB (the lease tool's own
   limit), after up to 60 seconds of waiting. Otherwise it keeps the lease and logs the
   `nvidia-smi` table: a vLLM server can outlive a step that was killed, because the
   engine starts it in a session of its own. The lease tool counts such a lease stale
   once the queue's PID is gone and the card is empty, so it frees itself then. Last, it
   writes the exit marker.

On a signal, the running command gets SIGINT, which Python turns into
KeyboardInterrupt, so the stage stops its vLLM server. After 120 seconds the rest of its
process group gets SIGKILL. SIGKILL to the queue itself cannot be caught: then nothing
is released, and the lease frees itself as above.

## 2. The queue file

One shell command per line, run with `bash -o pipefail -c`. Blank lines and lines that
start with `#` are skipped. A relative `--queue` path is read from the checkout after
the switch, so the file can come with the commit. The queue copies the file into its
log folder (`<name>.queue.txt`) as the record of what it ran.

`configs/queues/tier_a.txt` is the queue of R3 to R7 (RUN_PLAN.md section 4.2 lists the
same lines; `tests/test_gpu_queue.py` checks that they agree).

## 3. Files

All in `--log-dir`, which should be outside the checkout. `<name>` is `--name`, or the
queue file's name without its extension (`tier_a`).

| File | What it holds |
|---|---|
| `<name>.log` | The queue log: start, wait, claim, switch, each command with its exit code and time, release. One timestamp (UTC) per line. No output of a command. |
| `<name>.stepNN.log` | Everything command NN printed. |
| `<name>.exit` | The exit marker: the exit code, the reason and the time, one per line. `running` while the queue runs. |
| `<name>.queue.txt` | The queue file as it was run. |
| `<name>.kept/` | The saved `--keep-local` files. |

The eval stage prints each condition's mean rubric score, so the step logs hold scores.
Read the queue log, the marker and the manifests while a cell runs, not the step logs
([ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) section 2.5).

Exit codes: 0, every command finished; 2, bad arguments or setup; 3, the queue it
waited for did not finish; 4, no claim; 5, the switch failed; 6, the env file failed; 7,
a bad queue file; 128 plus the signal number after a signal; any other code is the
failed command's own.

## 4. R3 to R7 after R2

R2 runs on GPU 3 from commit 854761f (the old id of 7e02348) in `/data/PROJECTS/bohdi`.
On this machine, make a bundle of the commit to run, with the host's commit as its base:

```bash
git bundle create ../bohdi-tier-a.bundle cleanup/resubmission
git rev-parse cleanup/resubmission      # the commit for --commit
```

The bundle holds the whole history. The history rewrite of 2026-10-11 (the AI-trailer
purge, scripts/maintenance/purge_history.md) gave 854761f the id 7e02348, so a bundle
with `^854761f` as its base no longer applies: the host's checkout has the old id and
this machine has the new one.

Copy the bundle and `scripts/gpu_queue.sh` to a folder outside the checkout on the host,
for example `/data/PROJECTS/bohdi-queue/`. The host's checkout at 854761f has no copy of
the script, and a copy inside it would show as an untracked file. On the host, write the
env file there, then start the queue in its own tmux session while R2 still runs:

```bash
cd /data/PROJECTS/bohdi-queue
printf '%s\n' '. /data/PROJECTS/bohdi/.env.run' '. /data/PROJECTS/bohdi/.venv/bin/activate' > tier_a.env
tmux new-session -d -s tier_a "bash gpu_queue.sh --gpu 3 --owner PeterLi \
    --purpose 'BODHI R3-R6' --queue configs/queues/tier_a.txt --name tier_a \
    --log-dir /data/PROJECTS/bohdi-queue --repo /data/PROJECTS/bohdi \
    --after bohdi-gpu3 \
    --after-check \"/data/PROJECTS/bohdi/.venv/bin/python -c 'import json, sys; m = json.load(open(sys.argv[1])); sys.exit(m[sys.argv[2]] != sys.argv[3])' /data/PROJECTS/bohdi/results/healthbench__mistral-small-24b__bodhi/manifest.json status finished\" \
    --bundle bohdi-tier-a.bundle --commit <commit> --allow-untracked results/ \
    --keep-local /data/PROJECTS/bohdi/scripts/gpu_lease.sh \
    --keep-local results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_42 \
    --keep-local results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_7 \
    --keep-local results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_13 \
    --keep-local results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_99 \
    --keep-local results/healthbench__medgemma-27b-text__bodhi__2026-05/seed_101 \
    --env /data/PROJECTS/bohdi-queue/tier_a.env --lease-tool /data/PROJECTS/bohdi/gpu_lease.sh"
```

- `bohdi-gpu3` is the tmux session that runs R1 and R2. This script did not start it, so
  it leaves no exit marker that the queue can read, and `--after-check` decides instead:
  once the session is gone, the queue goes on only if R2's cell manifest says
  `finished`. The runner writes that file when a run starts (`started`) and ends
  (`finished` or `failed`). Otherwise the queue stops with exit 3 and claims nothing.
- The env file reads the host's `.env.run` (the Hugging Face token and the caches on
  `/data`) and puts the Python of `.venv` first on PATH, for the `python` of the queue
  lines. The stages take their own interpreters from `configs/project.yaml`. Gemma 2's
  FlashAttention setting comes from its model record, not from the env file.
- `--allow-untracked results/` lets the switch pass the outputs of R1 and R2. Git does
  not ignore their manifests, `cell.yaml` files and tables, and no commit holds them yet.
  Any other untracked file stops the queue at the start: move it, or add it to
  `.git/info/exclude` on the host.
- The host has two copies of the group's lease tool, and other projects call both. The
  in-place deploy of 2026-10-11 kept both out of git: `.git/info/exclude` lists them,
  and a sparse-checkout pattern keeps `854761f:scripts/gpu_lease.sh` out of the
  checkout. `/data/PROJECTS/bohdi/gpu_lease.sh` is the node's tool, the one that
  claims and releases here (`--lease-tool`, also the default). `--keep-local
  /data/PROJECTS/bohdi/scripts/gpu_lease.sh` brings the other copy through the switch unchanged: the commit
  to run does not track it, and the exclude rule keeps it out of `git status`.
- The five seed folders of the 2026-05 cell hold its 20 eval files and 5 grade files,
  which 854761f tracks and the new commits do not. The switch would delete them, and
  R6's last line reads them (RUN_PLAN.md section 4.3, item 4). The new `.gitignore`
  covers them, so they need no exclude rule. Only those folders are kept: the cell's
  three old analysis files, which the new commits move to `analysis/withdrawn/`, and its
  answer key, which they no longer track, stay as the commit has them.
- RUN_PLAN.md section 4.3 lists the other checks before the queue starts: R2 complete,
  the eval files of the other recorded cells, the model downloads and the disk.
- `--lease-file` is `GPU_LEASE_FILE` or `/home/nvidia/GPUS.txt`, as in the lease tool.
  The queue claims GPU 3 only when the card uses less than 2,000 MiB, so a vLLM server
  that outlived R2 stops it with exit 4 and the `nvidia-smi` table in its log.

Check on the host before the start, since each of these stops the queue at once:
`git -C /data/PROJECTS/bohdi status --porcelain` lists only files under `results/`, and
at most the host's copy of the tool in `scripts/` as changed; `git -C /data/PROJECTS/bohdi check-ignore -q
gpu_lease.sh` succeeds; and `/data/PROJECTS/bohdi/gpu_lease.sh status` shows the table.

To check on it:

```bash
tail -n 20 /data/PROJECTS/bohdi-queue/tier_a.log
cat /data/PROJECTS/bohdi-queue/tier_a.exit
/data/PROJECTS/bohdi/gpu_lease.sh status
```

## 5. Stop and resume

`tmux send-keys -t tier_a C-c` or `tmux kill-session -t tier_a` stops the queue as in
section 1: the running command gets SIGINT, the card is released and the marker says
why. To resume, start the same command again. The runner skips every stage whose
manifest is complete, so the queue goes on where it stopped. The checkout is already at
the commit, so the switch does nothing.

## 6. R7

R7's three lines in `configs/queues/tier_a.txt` start with `#` until the first author
answers on the 4,096-token pair ([BLOCKED.md](BLOCKED.md) item 1). After the answer:
change the R7 records if the answer adds the pair, remove the `#` from the three lines
here and in RUN_PLAN.md section 4.2, commit, make a new bundle, and start the queue
again at the new commit. If the first queue still runs, give the new one
`--after tier_a` and a new `--name`. The lines of R3 to R6 then find their stages
complete and skip them.

## 7. What it assumes about the host

- The lease tool takes `claim <gpu> <owner> <purpose>` and `release <gpu>`, records
  `CLAIM_PID`, and exits non-zero when it refuses. The last copy in this repo is
  `7e02348:scripts/gpu_lease.sh`; `tests/test_gpu_queue.py` runs that copy against the
  script.
- The lease file has one row per card. The row starts with the card's index and holds
  the owner and the PID as separate fields. The script reads nothing else from it.
- `tmux` is on PATH when `--after` is used, and `nvidia-smi` on the GPU host.
- One queue at a time switches a checkout. A queue switches it only when no job runs
  from it, which `--after` ensures for the job it waits for.

## 8. Tests

Since 2026-10-17 it also covers `--gpu any` (it waits while every row is busy, takes no
busy row, skips a free row on a card in use, and puts the card in `{gpu}`),
`--keep-lease` (the row stays busy under a live holder), a later queue that uses the
held card with no claim, a run without `--keep-lease` that gives a held card back, and a
failed claim that leaves no holder.

`tests/test_gpu_queue.py` runs the script with bash on the CPU: a fake lease tool, tmux
and `nvidia-smi` on PATH, and short shell commands as the queue. It covers the order and
the GPU pin, the stop at a failure, a refused claim, a card in use at the claim, the
wait with the marker check and with `--after-check`, a
line for another card, the env file, the three signals, a card that stays in use, the
lease tool of 7e02348, and the checkout switch: a dropped file, a changed file, a file kept out by a
sparse-checkout pattern, a changed file that is not kept, a folder
with files the commit no longer tracks, untracked files inside and outside
`--allow-untracked`, no `--keep-local`, and a commit the bundle lacks. `make test` runs
it.

On 2026-10-12 the script also ran by hand against a copy of the host's state: a
checkout at 854761f with a changed copy of the old lease tool marked skip-worktree and
an untracked R1 manifest, a bundle of c91d791 made as in section 4, the flags of section
4, a fake lease tool and a fake R2 session. It waited for the session, claimed the card,
switched the checkout clean (30 kept files copied back), and ran from the new commit,
with `python` from the env file, the dry runs of R2 at seed 42, of R4, and of R6 on the
2026-05 cell, whose stub judge graded the kept eval files. All three passed, and the
card was released.

## 9. Waiting for a card and keeping it (2026-10-17)

The first author's rule since 2026-10-17: take no card that someone else holds, wait in a
queue instead, and keep the card when our runs end. Two flags do it.

- `--gpu any` waits, a check every minute, until a card is free: its row says `free` and
  it uses less than 2,000 MiB. It claims that card through the lease tool, and the lines
  get its index through `{gpu}`. A card this owner already holds comes first.
- `--keep-lease` names a holder in the lease instead of the queue's shell: a tmux session
  `<owner>-hold-gpu<N>` that only sleeps. The lease tool frees a row only when its PID is
  gone and the card is empty, so the lease stays live after the queue ends, and the card
  stays ours, empty, until we give it back. The next queue with the same owner finds the
  card held by its holder and uses it with no claim.

This holds a card that no job uses between our runs, which the group's posted rules
("free GPUs the moment the job ends") do not do; the first author chose it. To give the
card back: `/data/PROJECTS/bohdi/gpu_lease.sh release <N>` and
`tmux kill-session -t PeterLi-hold-gpu<N>`. A run without `--keep-lease` on the held card
gives it back at its end.

The reruns after the EOS fix (RUN_PLAN.md section 4.4) start like this on the host, with
the bundle and the env file of section 4 in `/data/PROJECTS/bohdi/_queue`:

```bash
cd /data/PROJECTS/bohdi/_queue
tmux new-session -d -s eos_rerun "bash gpu_queue.sh --gpu any --keep-lease --owner PeterLi \
    --purpose 'BODHI EOS reruns R2-R5 (Peter Li)' \
    --queue configs/queues/eos_rerun.txt --log-dir /data/PROJECTS/bohdi/_queue --name eos_rerun \
    --repo /data/PROJECTS/bohdi --bundle eos.bundle --commit <commit> --allow-untracked results/ \
    --keep-local /data/PROJECTS/bohdi/scripts/gpu_lease.sh \
    --env /data/PROJECTS/bohdi/_queue/tier_a.env --lease-tool /data/PROJECTS/bohdi/gpu_lease.sh"
```

