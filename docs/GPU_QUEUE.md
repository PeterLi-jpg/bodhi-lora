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
   anything else, it stops and claims nothing.
2. **Claim.** It runs `<lease tool> claim <gpu> <owner> <purpose>` with `CLAIM_PID`
   set to its own PID. That shell lives as long as the queue, so the lease stays live
   between commands. The lease tool takes its own lock and refuses a card that another
   live job holds. If the tool refuses, or the lease file does not then show the card
   with this owner and PID, the queue stops and runs nothing.
3. **Switch the checkout** (with `--bundle` and `--commit`). It stops if `git status`
   shows a change to a tracked file, or an untracked file outside the `--allow-untracked`
   paths; ignored files are allowed. It makes this check at the start too, before the
   wait, so a checkout that cannot switch fails at once. It fetches the bundle's refs
   into `FETCH_HEAD` only, so no branch changes, and checks out the commit as a detached
   HEAD. After the switch the checkout must pass the same check, so every manifest
   records a clean commit. A `--keep-local` path (relative to the checkout, or absolute
   inside it) is a file or a folder, which means every file git tracks under it. Its
   files come through the switch by two rules:
   - The host's own copy (a file marked skip-worktree, or one git does not track) comes
     back whatever the commit holds. If the commit tracks the file, it gets
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
   `CUDA_VISIBLE_DEVICES` from `--gpus`. The queue stops at the first command that
   fails.
6. **Release and mark**, on every exit: success, a failed command, a refused setup step,
   or SIGINT, SIGTERM or SIGHUP. It releases the card only if the lease file still names
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
git bundle create ../bohdi-tier-a.bundle cleanup/resubmission ^854761f
git rev-parse cleanup/resubmission      # the commit for --commit
```

Copy the bundle and `scripts/gpu_queue.sh` to a folder outside the checkout on the host,
for example `/data/PROJECTS/bohdi-queue/`. The host's checkout at 854761f has no copy of
the script, and a copy inside it would show as an untracked file. Then start the queue
in its own tmux session on the host, while R2 still runs:

```bash
cd /data/PROJECTS/bohdi-queue
tmux new-session -d -s tier_a "bash gpu_queue.sh --gpu 3 --owner bohdi \
    --purpose 'BODHI R3-R7' --queue configs/queues/tier_a.txt --name tier_a \
    --log-dir /data/PROJECTS/bohdi-queue --repo /data/PROJECTS/bohdi \
    --after <R2 session> --after-marker <R2 marker> \
    --bundle bohdi-tier-a.bundle --commit <commit> --allow-untracked results/ \
    --keep-local /data/PROJECTS/bohdi/scripts/gpu_lease.sh \
    --keep-local /data/PROJECTS/bohdi/results/healthbench__medgemma-27b-text__bodhi__2026-05 \
    --env <run env file> --lease-tool /data/PROJECTS/bohdi/gpu_lease.sh"
```

- `<R2 session>` is the tmux session that runs R2. R2 was not started with this script,
  so `--after-marker` must name a file that R2's session writes when R2 ends, with R2's
  exit status on its first line. If R2's session writes no such file, the queue cannot
  tell how R2 ended, and it stops. Then start the queue without `--after` once R2's
  manifests show it complete.
- The env file must put the Python of `.venv` first on PATH (for example
  `. .venv/bin/activate`) and export `HF_TOKEN`. The stages take their interpreters from
  `configs/project.yaml`. Gemma 2's FlashAttention setting comes from its model record,
  not from the env file.
- `--allow-untracked results/` lets the switch pass the outputs of R1 and R2. Git does
  not ignore their manifests, `cell.yaml` files and tables, and no commit holds them yet.
  Any other untracked file stops the queue at the start: move it, or add it to
  `.git/info/exclude` on the host.
- The first `--keep-local` keeps the host's copy of the old lease tool in `scripts/`,
  which the host marks skip-worktree. The commit to run does not track the file, so the
  queue lists it in `.git/info/exclude`.
- The second keeps the 20 eval files and 5 grade files of the 2026-05 cell, which 854761f
  tracks and the new commits do not. The switch would delete them, and R6's last line
  reads them (RUN_PLAN.md section 4.3, item 4). It also keeps the four other files of
  that cell that the switch would delete and the host has now: three old analysis files
  and the answer key.
- RUN_PLAN.md section 4.3 lists the other checks before the queue starts: R2 complete,
  the eval files of the other recorded cells, the model downloads and the disk.
- `--lease-tool` is `/data/PROJECTS/bohdi/gpu_lease.sh` by default, and `--lease-file`
  is `GPU_LEASE_FILE` or `/home/nvidia/GPUS.txt`, as in the lease tool.

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

`tests/test_gpu_queue.py` runs the script with bash on the CPU: a fake lease tool, tmux
and `nvidia-smi` on PATH, and short shell commands as the queue. It covers the order and
the GPU pin, the stop at a failure, a refused claim, the wait and the marker check, a
line for another card, the env file, the three signals, a card that stays in use, the
lease tool of 7e02348, and the checkout switch: a dropped file, a changed file, a folder
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
