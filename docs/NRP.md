# Running on NRP

The reruns after the EOS fix (RUN_PLAN.md section 4.4, DECISIONS.md D29) run on the
National Research Platform's Kubernetes cluster, in the namespace `aiea-interns`, on at
most three A100-80GB cards at once. This page says how. The specs are in `deploy/nrp/`.

## 1. What runs where

- **The volume** (`deploy/nrp/pvc.yaml`): a 200 GiB CephFS claim, `bodhi-lora-rerun`,
  that every pod mounts at `/data`. It holds the checkout (`/data/bohdi`), the cell
  folders under `results/`, the task pool's state (`/data/pool/eos_rerun`), and both envs
  as one tar each (`/data/venvs`). The envs run from each pod's own disk: CephFS is slow
  with the tens of thousands of small files of a venv (a first install on the volume made
  no headway in 25 minutes), for the install and for every import. Each env is built at a
  fixed path (`/local/venv-train`, `/local/venv-infer`) and unpacked at the same path in
  every worker, and `.venv` and `.venv-infer` in the checkout are links to those paths, so
  the paths `configs/project.yaml` names work in every pod.
- **The setup job** (`deploy/nrp/setup-job.yaml`, CPU only): checks out the commit from a
  bundle, unpacks the inputs copied from the shared host, builds both envs from their
  lock files with `--no-deps` (so each package is the version that ran the cells) on its
  disk, runs `pip check` on both, saves them to the volume as tars, checks that the Hugging Face token reads the three models at their
  pinned revisions, and dry-runs every cell.
- **The workers** (`deploy/nrp/workers-job.yaml`): one Job of three pods, one A100-80GB
  each (`nvidia.com/a100` with the 80 GB products in the node affinity; the namespace
  quota is five). Each pod unpacks the two envs, then downloads the three models it loads (Mistral-Small-24B,
  Llama-3.1-8B, MedGemma-27B, at their records' revisions, without the duplicate
  full-model files) to its own disk, goes offline, and runs `scripts/task_pool.py` on
  `configs/queues/eos_rerun_nrp.yaml`.

Argo Workflows is installed on the cluster but not open to the namespace, and the
namespace cannot bind roles, so no controller pod can create Jobs. The task pool is the
scheduler instead: the pods share one state folder on the volume.

## 2. The task pool

`configs/queues/eos_rerun_nrp.yaml` lists 34 tasks, each one `python -m bodhi_lora run`
with `--gpus 0` (a pod sees its one card as 0), with the tasks it comes after and an
estimate of its A100-hours. A worker:

1. takes the ready task (all it comes after are done) with the most hours on its
   longest chain of later tasks, by an atomic `mkdir` of its claim folder;
2. runs it in the checkout, with its output in `logs/<task>.log`, touching a heartbeat
   in the claim every 30 s;
3. records the attempt in the state folder's `done` folder, or in `failed`, and then runs
   the task once more; after two failures every task after it is blocked;
4. checks, before it claims a task, that the card holds no memory (a vLLM server that
   outlived a killed task); if it does, the pod exits 75 and the Job replaces it;
5. exits when nothing is ready and the tasks left are no more than the other workers are
   running: those workers take them, and this card goes back to the cluster. Here a claim
   counts as running only with a heartbeat in the last 90 s, so the claim of a pod that
   died does not send the others away.

A claim with no heartbeat for 10 minutes (its pod was stopped or evicted) is moved to
`claims/.stale/`, and the task runs again on another worker. A worker touches or removes
only a claim whose `claim.json` names it; if its claim is taken over, it stops its own copy
of the task, so no task runs twice. A training task run again resumes from the newest
checkpoint the trainer finished writing (`bodhi_lora/pipeline/train.py`), and an input file
that is already right is never rewritten (`bodhi_lora/pipeline/runner.py`), since runners
in other pods may be reading it. The runner skips every stage
whose manifest is complete, and the eval stage keeps each reply as it comes, so a task run
again goes on where it stopped. Runner processes in different pods can write one cell
manifest at once; its temporary file carries the host and the PID
(`bodhi_lora/manifest.py`). The last task of each cell runs the cell whole once more: every
stage is complete, so it only writes the cell manifest with every stage in its index.

With the estimates, three workers finish in about 31 hours, with no card waiting idle
(a simulation of the rule above; about 92 A100-hours of work).

## 3. The steps

On this machine, with `kubectl` on the `nautilus` context:

```bash
kubectl apply -n aiea-interns -f deploy/nrp/pvc.yaml
git bundle create ../bodhi.bundle cleanup/resubmission
```

On the shared host, pack the inputs: each rerun cell's folder without its `train`, `eval`,
`grade` and `bench` folders, logs and cell manifest (so every condition, the base too,
runs again on NRP; D29) and without the H100 runs' records in `shared/eval`, and the
benchmark data the cells read. The payload of 2026-10-17 held no train, eval, grade, bench
or log file and kept every traces, filter and inputs manifest:

```bash
cd /data/PROJECTS/bohdi
CELLS="healthbench__mistral-small-24b__bodhi mediq-paired__mistral-small-24b__bodhi
       healthbench__mistral-small-24b__bodhi__labelled healthbench__mistral-small-24b__ask-first
       mediq-paired__mistral-small-24b__ask-first"
# --no-wildcards-match-slash: without it 'results/*/manifest.json' would also drop the
# traces, filter and inputs manifests, and the runner would make them again.
R=results
tar -czf /data/PROJECTS/bohdi/hf_cache/tmp/payload.tar.gz --no-wildcards-match-slash \
    --exclude="$R/*/seed_*/train" --exclude="$R/*/seed_*/eval" --exclude="$R/*/seed_*/grade" \
    --exclude="$R/*/seed_*/bench" --exclude="$R/*/shared/eval" --exclude="$R/*/logs" \
    --exclude="$R/*/manifest.json" \
    data/raw/healthbench data/raw/mediq-paired $(for c in $CELLS; do printf "%s/%s " "$R" "$c"; done)
```

Copy the bundle and the payload to the volume, then start the setup job with `COMMIT`
set to the bundle's commit, and the workers once it is done:

```bash
kubectl apply -n aiea-interns -f deploy/nrp/io-pod.yaml
kubectl exec -n aiea-interns bodhi-lora-io -- mkdir -p /data/incoming
kubectl cp ../bodhi.bundle aiea-interns/bodhi-lora-io:/data/incoming/bodhi.bundle
kubectl cp payload.tar.gz aiea-interns/bodhi-lora-io:/data/incoming/payload.tar.gz
kubectl delete pod -n aiea-interns bodhi-lora-io
kubectl apply -n aiea-interns -f deploy/nrp/setup-job.yaml
kubectl logs -n aiea-interns -f job/bodhi-lora-setup
kubectl apply -n aiea-interns -f deploy/nrp/workers-job.yaml
```

To check on it, from a copy pod or a worker:

```bash
kubectl get pods -n aiea-interns -l app=bodhi-lora
kubectl exec -n aiea-interns <worker pod> -- tail -n 20 /data/pool/eos_rerun/workers/<worker pod>.log
kubectl exec -n aiea-interns <worker pod> -- ls /data/pool/eos_rerun/done
```

Read a running task's logs only from the pod that runs it. When another node reads a
file that a pod is writing, CephFS takes back that pod's write caching, and the writer then
waits on the metadata server at each write until it gets the caching back. A vLLM server
writes a log line per request, so its throughput falls: on 2026-10-18, while one pod read
every pod's logs every 3 minutes, R4's grading on another node ran at about a quarter of
its speed, with its card idle. A plain `stat` of a log that another node wrote took about
0.9 s there, against under 0.1 ms for a pod's own logs. Files that no process writes any
more (the pool records, finished manifests and result files) are safe to read from any pod.

When every worker has exited, `/data/pool/eos_rerun/summary.json` lists the tasks done,
failed and blocked. Copy the cell folders back with the copy pod (`kubectl cp` of a tar of
`/data/bohdi/results/<cell>`), to the shared host and to this machine (RUN_PLAN.md section
7), then delete the Jobs. Keep the volume until the copies are checked.

## 4. Rules of the namespace

Names carry the project, never a person's name (NRP stamps the account on every object
anyway). Jobs, not Deployments, so a card goes back when its pod ends. At most three
A100s at once, of the namespace's five. The Hugging Face token is the namespace secret
`nfootprint-hf` (key `token`); it never goes into a file or a log.

## 5. R7 and R10 (2026-10-19)

R7 (the three MedGemma-27B cells and the 4,096-token pair) and R10 (the two NLM cells) run
from `configs/queues/r7_nrp.yaml` (36 tasks, about 110 A100-hours by the estimates) on the
same volume, envs and rules, with their own state folder `/data/pool/r7` (DECISIONS.md D30).
The volume's checkout holds the EOS reruns' outputs as untracked files, and the new commit
tracks some of them, so `deploy/nrp/setup-r7-job.yaml` checks that each one equals the
commit's copy (or differs only in the purged commit id) before it moves the checkout. Its
payload is only the NLM files:

```bash
git bundle create ../bodhi.bundle cleanup/resubmission
tar -czf ../payload_r7.tar.gz data/raw/nlm-clinical-questions
kubectl apply -n aiea-interns -f deploy/nrp/io-pod.yaml
kubectl cp ../bodhi.bundle aiea-interns/bodhi-lora-io:/data/incoming/bodhi.bundle
kubectl cp ../payload_r7.tar.gz aiea-interns/bodhi-lora-io:/data/incoming/payload_r7.tar.gz
kubectl delete pod -n aiea-interns bodhi-lora-io
kubectl apply -n aiea-interns -f deploy/nrp/setup-r7-job.yaml   # with COMMIT set
kubectl apply -n aiea-interns -f deploy/nrp/workers-r7-job.yaml
```

The workers download five models (MedGemma-27B, Qwen2.5-14B, Llama-3.1-8B, Phi-4 and
Mistral-Small-24B), about 180 GB, so each pod's local volume is 270 GiB. The namespace's
A100 quota is shared with other groups: while it is full, the Job's pods are not created,
and the Job controller tries again, so the run waits in line and takes no one's card.

**The D31 fix during the run (2026-10-21).** Seeds 7 and 13 ran out of card memory in the
validation loss at the end of their first epoch (DECISIONS.md D31). On the first author's yes,
the volume's checkout moved from d6ae732 to the commit with the fix (a bundle of the commits
between them, `git fetch` and `git checkout --detach`; a stale `.git/index.lock` from the
workers' start went to `/data/trash`). Seed 42, which had finished with batch 8, was set aside
as `seed_42/train.batch8-pre-d31` (its log as `logs/train_s42.batch8-pre-d31.log`, its done
record in `/data/trash`) and trains again; seeds 7 and 13 got their attempts back
(`attempts/<task>` set to 0), and seed 99 was stopped and queued again. A worker that finds
nothing ready while every other task is claimed exits and gives its card back: a hold made of
claim files cost one of the two cards on 2026-10-21, so never hold tasks that way.
On the same day, a one-pod copy of the workers Job (to queue for the lost card) was refused
by NRP's admission webhook: "Your pods resources utilization is too low" for the account
(stats at https://nrp.ai/userinfo/). Running Jobs keep running and still replace their pods;
new Jobs wait until the account's utilization recovers, so requests should stay close to use.
