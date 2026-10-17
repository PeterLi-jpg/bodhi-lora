# Running on NRP

The reruns after the EOS fix (RUN_PLAN.md section 4.4, DECISIONS.md D29) run on the
National Research Platform's Kubernetes cluster, in the namespace `aiea-interns`, on at
most three A100-80GB cards at once. This page says how. The specs are in `deploy/nrp/`.

## 1. What runs where

- **The volume** (`deploy/nrp/pvc.yaml`): a 200 GiB CephFS claim, `bodhi-lora-rerun`,
  that every pod mounts at `/data`. It holds the checkout (`/data/bohdi`), both envs
  inside it (`.venv` and `.venv-infer`, as `configs/project.yaml` names them), the cell
  folders under `results/`, and the task pool's state (`/data/pool/eos_rerun`).
- **The setup job** (`deploy/nrp/setup-job.yaml`, CPU only): checks out the commit from a
  bundle, unpacks the inputs copied from the shared host, builds both envs from their
  lock files with `--no-deps` (so each package is the version that ran the cells), runs
  `pip check` on both, checks that the Hugging Face token reads the three models at their
  pinned revisions, and dry-runs every cell.
- **The workers** (`deploy/nrp/workers-job.yaml`): one Job of three pods, one A100-80GB
  each (`nvidia.com/a100` with the 80 GB products in the node affinity; the namespace
  quota is five). Each pod downloads the three models it loads (Mistral-Small-24B,
  Llama-3.1-8B, MedGemma-27B, at their records' revisions, without the duplicate
  full-model files) to its own disk, goes offline, and runs `scripts/task_pool.py` on
  `configs/queues/eos_rerun_nrp.yaml`.

Argo Workflows is installed on the cluster but not open to the namespace, and the
namespace cannot bind roles, so no controller pod can create Jobs. The task pool is the
scheduler instead: the pods share one state folder on the volume.

## 2. The task pool

`configs/queues/eos_rerun_nrp.yaml` lists 35 tasks, each one `python -m bodhi_lora run`
with `--gpus 0` (a pod sees its one card as 0), with the tasks it comes after and an
estimate of its A100-hours. A worker:

1. takes the ready task (all it comes after are done) with the most hours on its
   longest chain of later tasks, by an atomic `mkdir` of its claim folder;
2. runs it in the checkout, with its output in `logs/<task>.log`, touching a heartbeat
   in the claim every 30 s;
3. records the attempt in the state folder's `done` folder, or in `failed`, and then runs
   the task once more; after two failures every task after it is blocked;
4. checks before each task that the card holds no memory (a vLLM server that outlived a
   killed task); if it does, the pod exits 75 and the Job replaces it;
5. exits when nothing is ready and the tasks left are no more than the other workers are
   running: those workers take them, and this card goes back to the cluster.

A claim with no heartbeat for 10 minutes (its pod was stopped or evicted) is moved to
`claims/.stale/`, and the task runs again on another worker. The runner skips every stage
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
runs again on NRP; D29), and the benchmark data the cells read:

```bash
cd /data/PROJECTS/bohdi
CELLS="healthbench__mistral-small-24b__bodhi mediq-paired__mistral-small-24b__bodhi
       healthbench__mistral-small-24b__bodhi__labelled healthbench__mistral-small-24b__ask-first
       mediq-paired__mistral-small-24b__ask-first"
tar -czf /data/PROJECTS/bohdi/hf_cache/tmp/payload.tar.gz \
    --exclude='seed_*/train' --exclude='seed_*/eval' --exclude='seed_*/grade' --exclude='seed_*/bench' \
    --exclude='*/logs' --exclude='results/*/manifest.json' \
    data/raw/healthbench data/raw/mediq-paired $(for c in $CELLS; do printf "%s/%s " results "$c"; done)
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

When every worker has exited, `/data/pool/eos_rerun/summary.json` lists the tasks done,
failed and blocked. Copy the cell folders back with the copy pod (`kubectl cp` of a tar of
`/data/bohdi/results/<cell>`), to the shared host and to this machine (RUN_PLAN.md section
7), then delete the Jobs. Keep the volume until the copies are checked.

## 4. Rules of the namespace

Names carry the project, never a person's name (NRP stamps the account on every object
anyway). Jobs, not Deployments, so a card goes back when its pod ends. At most three
A100s at once, of the namespace's five. The Hugging Face token is the namespace secret
`nfootprint-hf` (key `token`); it never goes into a file or a log.
