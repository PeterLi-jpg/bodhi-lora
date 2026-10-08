"""Modal entrypoint for BODHI Stage 4 (eval) + Stage 5 (epistemic) on A100 80GB.

Why this script exists
----------------------
TPU spot capacity in europe-west4-a got hammered the night of the NeurIPS
deadline; we lost 4 of 5 VMs to preemption / vLLM-TPU engine-init bugs and
couldn't get eval through. Stages 1-3 had landed successfully for all 5
seeds though (HF-format adapters at gs://.../seed_<N>/checkpoints/best/
for seeds 7, 13, 42, 99, 101), so this script just runs Stage 4 (eval
over 4 configs: base/lora x wrapper/no-wrapper) plus Stage 5 (epistemic)
for those 5 seeds on Modal A100 80GB. MedGemma-27B base + Llama-8B
grader fit comfortably on a single A100 80GB.

Cost budget: ~$13 of $30 free Modal credit (~$3.20/hr * 50min wall * 5
seeds-in-parallel = same as sequential since billing is per-GPU-second).

Setup (one-time)
----------------
1. ``modal token new`` (already done).
2. ``gcloud auth application-default login`` locally so ADC JSON exists.
3. Modal Secrets ``hf-token`` (HF_TOKEN + GH_TOKEN) and ``gcp-creds``
   (GOOGLE_APPLICATION_CREDENTIALS_JSON) are created from local files via
   ``modal secret create ...``. No compute consumed.

Run
---
``modal run modal_eval.py::run_all`` — fans out all 5 seeds in parallel
via .map() (max_containers=5). Each seed runs 4 Stage-4 configs of
base/lora x wrapper/no-wrapper with primary grader Llama-3.1-8B-Instruct
only (no Mistral cross-grader, dropped to shave ~30%% wall after deadline
crunch). Stage 5 epistemic runs at the end of each seed once all 4
Stage 4 JSONs are in place.

Single-grader caveat: cross-grader robustness analysis is not produced.
The paper can note this as a deadline-driven simplification with the
unlocked future-work claim that a cross-grader run reproduces the
direction (a follow-up sweep is cheap on the same Modal account).

Resumable: each eval JSON write is checked before invocation. If a seed
preempts mid-run (rare on Modal vs TPU spot, but possible), the next
``modal run ...`` skips already-completed configs.
"""

from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path

import modal

APP_NAME = "bohdi-eval"
GCS_BASE = "gs://bohdi-runs-tokyo-micron/prod-v39-tunix-27b-neurips"
# All 5 seeds successfully landed a HF-format adapter at
# gs://.../seed_<N>/checkpoints/best/ (verified each is 10 objects, 60.47 MiB,
# adapter_model.safetensors + adapter_config.json + the tokenizer bundle that
# PR #247 added). Eval-only run is just 4 configs x 5 seeds + Stage 5.
SEEDS = [7, 13, 42, 99, 101]

# vLLM + bodhi-llm version pins copied verbatim from scripts/launch_gpu.sh —
# these are the exact versions the GPU codepath was last tested against.
# launch_gpu.sh pins vllm explicitly because scripts/_vllm_engine.py spawns
# `vllm serve` as a subprocess inside the GPU pod (no host docker), so we
# need vllm pip-installed in the same env. pip auto-picks the CUDA wheel.
PYTHON_PINS = [
    "torch>=2.5,<2.9",
    "transformers==4.57.6",
    "peft==0.19.1",
    "trl==0.11.4",
    "accelerate==1.13.0",
    "bodhi-llm[all]==0.1.4",
    "vllm>=0.6.5,<0.10",
    "huggingface-hub",
    "google-cloud-storage",
    "datasets>=2.18.0,<4.0.0",
    "tqdm",
    "rich>=13.0,<15.0",
    "matplotlib>=3.7,<4.0",
    "pyyaml>=6.0,<7.0",
    "numpy>=1.24,<3.0",
    "ml_dtypes",
    "safetensors",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "git", "ca-certificates")
    .pip_install(*PYTHON_PINS)
    .run_commands(
        # Install gcloud SDK so we can `gsutil rsync` from inside the container.
        # CLOUDSDK_CORE_DISABLE_PROMPTS=1 ensures non-interactive install in Modal
        # build context (no TTY) — without it the script can deadlock on a y/n
        # prompt for "modify .bashrc?".
        "CLOUDSDK_CORE_DISABLE_PROMPTS=1 "
        "curl -sSL https://sdk.cloud.google.com 2>/dev/null | bash > /dev/null",
        "ln -sf /root/google-cloud-sdk/bin/gcloud /usr/local/bin/gcloud",
        "ln -sf /root/google-cloud-sdk/bin/gsutil /usr/local/bin/gsutil",
        # Patch vllm's ovis.py: it calls AutoConfig.register("aimv2", AIMv2Config)
        # without exist_ok=True. transformers 4.57+ ships its own native aimv2
        # config — the two collide and `vllm serve` crashes at import with
        # `ValueError: 'aimv2' is already used by a Transformers config`. Adding
        # exist_ok=True makes the registration tolerant of a pre-existing entry.
        # https://github.com/vllm-project/vllm/issues/<aimv2-conflict>
        # (vllm 0.10+ has this fix upstream; we're pinned <0.10 for repo
        # compatibility, so we patch the installed package directly.)
        'sed -i \'s/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/\' /usr/local/lib/python3.11/site-packages/vllm/transformers_utils/configs/ovis.py',
        # Verify the patch took (fail the build if not, vs failing silently at runtime).
        'python -c "import vllm" && echo "vllm imports cleanly"',
    )
)

app = modal.App(APP_NAME, image=image)

# Modal Volumes — persistent across runs. The HF cache catches MedGemma-27B
# (~54 GB) and Llama-8B (~16 GB) so we only download once. The data volume
# caches GCS-pulled adapters + HealthBench data so we don't pay GCS egress
# repeatedly.
hf_cache = modal.Volume.from_name("bohdi-hf-cache", create_if_missing=True)
data_cache = modal.Volume.from_name("bohdi-data-cache", create_if_missing=True)


def _gcs_client():
    """Build a google-cloud-storage Client from the Modal Secret.

    Modal Secret ``gcp-creds`` exposes ``GOOGLE_APPLICATION_CREDENTIALS_JSON``
    with the literal JSON text of either an ADC user-account credential or
    a service-account key. We write it to ``$GOOGLE_APPLICATION_CREDENTIALS``
    so google-auth can pick it up (the Python lib honors this env var for
    BOTH authorized_user and service_account types — unlike gsutil, which
    only honors it for service accounts; that's why we use the Python
    library instead of shelling out to gsutil).
    """
    creds_text = os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    creds_path = "/tmp/gcp-creds.json"
    with open(creds_path, "w") as f:
        f.write(creds_text)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

    parsed = json.loads(creds_text)
    qproj = parsed.get("quota_project_id") or "tokyo-micron-494016-s9"
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", qproj)

    from google.cloud import storage
    return storage.Client(project=qproj)


def _split_gs_uri(uri: str) -> tuple[str, str]:
    """``gs://bucket/path/to/thing`` -> ``("bucket", "path/to/thing")``."""
    assert uri.startswith("gs://"), uri
    rest = uri[len("gs://"):]
    bucket, _, blob = rest.partition("/")
    return bucket, blob


def _gcs_download_prefix(client, gs_uri: str, local_dir: str) -> int:
    """Download every blob under ``gs_uri`` into ``local_dir``. Returns count."""
    bucket_name, prefix = _split_gs_uri(gs_uri.rstrip("/") + "/")
    bucket = client.bucket(bucket_name)
    os.makedirs(local_dir, exist_ok=True)
    n = 0
    for blob in bucket.list_blobs(prefix=prefix):
        rel = blob.name[len(prefix):]
        if not rel:  # the prefix itself, if listed
            continue
        local_path = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(local_path) or local_dir, exist_ok=True)
        blob.download_to_filename(local_path)
        n += 1
    return n


def _gcs_download_blob(client, gs_uri: str, local_path: str) -> bool:
    """Download a single blob. Returns True on success, False if not found."""
    bucket_name, blob_name = _split_gs_uri(gs_uri)
    blob = client.bucket(bucket_name).blob(blob_name)
    if not blob.exists():
        return False
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    blob.download_to_filename(local_path)
    return True


def _gcs_upload_dir(client, local_dir: str, gs_uri_prefix: str) -> int:
    """Upload every file under ``local_dir`` to ``gs_uri_prefix/<rel>``."""
    bucket_name, prefix = _split_gs_uri(gs_uri_prefix.rstrip("/") + "/")
    bucket = client.bucket(bucket_name)
    n = 0
    for root, _, files in os.walk(local_dir):
        for f in files:
            local = os.path.join(root, f)
            rel = os.path.relpath(local, local_dir)
            blob = bucket.blob(prefix + rel.replace(os.sep, "/"))
            blob.upload_from_filename(local)
            n += 1
    return n


def _run(cmd, **kwargs):
    """Subprocess.run with check=True and stdout/stderr passthrough."""
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


@app.function(
    cpu=0.25,
    timeout=120,
    secrets=[modal.Secret.from_name("gcp-creds")],
)
def validate_auth() -> str:
    """CPU-only auth probe — runs in <2 sec on a fractional CPU.

    Validates that the gcp-creds Modal Secret actually authenticates against
    the target bucket BEFORE any A100 is allocated. Catches the class of
    bugs (wrong cred type, missing IAM role, expired token, wrong bucket
    name) that would otherwise burn GPU minutes during the rsync step of
    eval_seed. Failure cost: ~$0; success cost: ~$0.0001.
    """
    gcs = _gcs_client()
    bucket_name, _ = _split_gs_uri(GCS_BASE)
    bucket = gcs.bucket(bucket_name)
    # Listing one blob is cheaper than .exists() on the bucket and proves
    # storage.objects.list IAM is granted (the same op that gsutil rsync did).
    test_prefix = "prod-v39-tunix-27b-neurips/seed_7/checkpoints/best/"
    blobs = list(bucket.list_blobs(prefix=test_prefix, max_results=1))
    if not blobs:
        raise RuntimeError(
            f"GCS auth OK but no blobs at gs://{bucket_name}/{test_prefix} "
            "— check the path."
        )
    return f"GCS auth OK; sample blob: {blobs[0].name}"


# Models that need to land in the HF Volume before any A100 work.
HF_MODELS_TO_WARM = (
    "google/medgemma-27b-text-it",
    "meta-llama/Llama-3.1-8B-Instruct",
)


@app.function(
    cpu=2,
    memory=4096,
    timeout=7200,  # 2h cap; cold download of 70 GB is ~10-30 min in practice.
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("hf-token")],
)
def warm_hf_cache() -> str:
    """Pre-download model weights to the HF Volume on a CPU container.

    Why: The first ``vllm serve`` cold start on the GPU function spends ~10
    min downloading MedGemma-27B (54 GB) and ~3 min on Llama-8B (16 GB),
    all of which is GPU-billed at ~$3.20/h. Doing the download on a CPU
    container instead costs essentially nothing (CPU at $0.0036/min × 30
    min = $0.11) and committing to the shared Volume means every GPU
    container that follows reads from local disk in <30 sec.

    Idempotent: ``snapshot_download`` skips files already present.
    """
    import time
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set — needed for gated MedGemma access.")
    started = time.time()
    sizes = {}
    for repo in HF_MODELS_TO_WARM:
        print(f"snapshot_download: {repo}", flush=True)
        path = snapshot_download(
            repo_id=repo,
            token=token,
            # max_workers default 8 — lots of parallel downloads to saturate egress.
        )
        # Sum up file sizes for a quick sanity readout.
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
        sizes[repo] = total
        print(f"  -> {path} ({total / 1e9:.1f} GB)", flush=True)
    hf_cache.commit()
    elapsed = int(time.time() - started)
    summary = ", ".join(f"{r}={s/1e9:.1f}GB" for r, s in sizes.items())
    return f"HF cache warm in {elapsed}s: {summary}"


@app.function(
    gpu="A100-80GB",
    timeout=14400,  # 4h per seed cap; a clean run is ~50 min.
    # Run all 5 seeds in parallel. Per-second billing means concurrency is
    # free wall-clock-wise; cost is the same as sequential. Cap at 5 (matches
    # SEEDS) so .map() fans out fully.
    max_containers=5,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/data": data_cache,
    },
    secrets=[
        modal.Secret.from_name("hf-token"),     # exposes HF_TOKEN
        modal.Secret.from_name("gcp-creds"),    # exposes GOOGLE_APPLICATION_CREDENTIALS_JSON
    ],
)
def eval_seed(seed: int, max_examples: int | None = None) -> str:
    """Evaluate one seed: 4 Stage 4 configs + Stage 5 epistemic.

    Args:
      seed: which seed to eval (must be in SEEDS).
      max_examples: if not None, passed as --max-examples to both eval
        scripts. Use a small int (e.g. 3) for the smoke entrypoint so the
        whole pipeline runs end-to-end in minutes, not an hour.

    Skips configs whose output JSON already exists in the local volume
    (resume-friendly across Modal restarts).
    """
    gcs = _gcs_client()

    # Match the bumped vllm `--max-num-seqs=32` (see patches below). Without
    # this, eval_healthbench.py's ThreadPoolExecutor caps at 16 client threads
    # and we'd never exceed 16 in-flight requests no matter how high the
    # server cap is. Both ends need to lift together to actually use the
    # extra GPU concurrency.
    os.environ["EVAL_CONCURRENCY"] = "32"

    # 1. Repo: clone fresh into the container so we know the code revision.
    repo_dir = "/root/bohdi-lora"
    if not os.path.exists(f"{repo_dir}/.git"):
        # Public token-injection trick — keeps GH_TOKEN out of disk config.
        gh_token = os.environ.get("GH_TOKEN", "")
        if gh_token:
            url = (
                f"https://x-access-token:{gh_token}"
                "@github.com/PeterLi-jpg/bohdi-lora.git"
            )
        else:
            # Repo is private; fall back to read-only public mirror if GH_TOKEN
            # not set. In practice it IS set via the hf-token Modal Secret if
            # the user added it (we re-use the same Secret for both).
            url = "https://github.com/PeterLi-jpg/bohdi-lora.git"
        _run(["git", "clone", url, repo_dir])
    else:
        _run(["git", "-C", repo_dir, "fetch", "origin", "main"])
        _run(["git", "-C", repo_dir, "reset", "--hard", "origin/main"])
    os.chdir(repo_dir)

    # 1b. Patch repo files for the A100-80GB GPU runtime. These tweaks fit
    # the eval pipeline within Modal's GPU envelope without modifying the
    # repo on disk.
    #
    #   (a) max_model_len 8192 -> 4096: TPU v6e-8 had 256 GB HBM and could
    #       afford a long context. A100-80GB has ~26 GB free for KV cache
    #       after the 54 GB MedGemma weights. 4096 keeps us comfortably
    #       under the cap and roughly halves the per-token KV cost.
    #   (b) max_num_seqs 64 -> 8: with EVAL_CONCURRENCY=16 client threads,
    #       8 in-flight server slots is enough — vllm queues the rest.
    #   (c) Two TPU drain `time.sleep(30)` calls (eval_healthbench.py) only
    #       matter when the v6e-8 runtime hasn't released libtpu state. On
    #       Modal we're between subprocess vllms on a single A100 — no
    #       drain needed. Saves 30s × 4 configs = 2 min per seed.
    patches = [
        ("scripts/_vllm_engine.py",
         'max_model_len: int = 8192',
         'max_model_len: int = 4096'),
        # vllm reports "Maximum concurrency for 4,096 tokens per request: 11.72x"
        # on A100-80GB after MedGemma weights. Real avg req is ~2.5K tokens
        # (1.5K prompt + ≤1024 output, often shorter), so the practical cap
        # is more like 18-20× at typical utilization. Set to 32 to give vllm
        # plenty of in-flight slots for continuous batching. EVAL_CONCURRENCY
        # is bumped to match (see env var below) so the client side matches.
        # vllm gracefully preempts/swaps if a concurrent batch exceeds KV cache
        # at peak, which is rare given short typical responses.
        ("scripts/_vllm_engine.py",
         '"--max-num-seqs", "64"',
         '"--max-num-seqs", "32"'),
        ("scripts/eval_healthbench.py",
         '_t.sleep(30)',
         '_t.sleep(0)  # patched: no TPU drain on GPU'),
        # Bump vllm logging from WARNING -> INFO so the container log shows
        # download progress, engine init phases, and KV-cache allocation.
        # Without this, vllm runs silently for ~minutes during cold start
        # and we can't tell "downloading" from "hung".
        ("scripts/_vllm_engine.py",
         '"VLLM_LOGGING_LEVEL": "WARNING"',
         '"VLLM_LOGGING_LEVEL": "INFO"'),
        # Bump HTTP client timeout 180s -> 900s. The smoke caught 2/3
        # lora_bodhi inferences hitting `TimeoutError: timed out` — bodhi's
        # 2-pass (analysis pass + final pass) doubles generation work per
        # prompt, and LoRA adds slight overhead vs base. With 32-way client
        # concurrency competing for KV cache, an individual request can
        # easily exceed the default 180s. 900s gives generous headroom
        # without masking real hangs.
        ("scripts/_vllm_engine.py",
         "urlopen(req, timeout=180)",
         "urlopen(req, timeout=900)"),
    ]
    for path, old, new in patches:
        full = os.path.join(repo_dir, path)
        with open(full) as f:
            text = f.read()
        if old not in text:
            print(f"  patch SKIPPED (not found): {path} | {old[:40]}")
            continue
        with open(full, "w") as f:
            f.write(text.replace(old, new))
        print(f"  patched: {path} | {old[:40]} -> {new[:40]}")

    # Surface the vllm log file to the Modal container's stdout. _vllm_engine.py
    # redirects vllm subprocess output to ~/vllm_serve_{port}.log, which is
    # invisible to `modal app logs`. We spawn a `tail -F` (capital F handles
    # the file not yet existing) so each line vllm writes lands in our process
    # stdout and propagates to Modal Logs in real time. Started ONCE per
    # eval_seed call (the eval scripts reuse port 8000 across configs).
    log_paths = [
        os.path.expanduser("~/vllm_serve_8000.log"),  # both model and grader use 8000
    ]
    for lp in log_paths:
        # `touch` so tail -F has something to start polling immediately.
        open(lp, "a").close()
    subprocess.Popen(
        ["tail", "-F", "-n", "+1", *log_paths],
        stdout=None, stderr=subprocess.STDOUT,
    )

    # 2. Pull the seed's per-stage outputs from GCS into the persistent
    #    volume (cached after first run). GCS layout is:
    #      seed_<N>/
    #        checkpoints/best/{adapter_model.safetensors, ...}
    #        train.jsonl, val.jsonl, ...
    #    Only the trained adapter dir matters for eval; we skip the orbax/
    #    checkpoint dir (large, training-only) to save bandwidth.
    seed_local = f"/data/seed_{seed}"
    seed_gcs = f"{GCS_BASE}/seed_{seed}"
    os.makedirs(seed_local, exist_ok=True)
    # Only pull if the trained adapter isn't already on the volume.
    adapter_local = f"{seed_local}/checkpoints/best"
    adapter_marker = f"{adapter_local}/adapter_model.safetensors"
    if not os.path.exists(adapter_marker):
        print(f"[seed {seed}] pulling adapter {seed_gcs}/checkpoints/best/")
        n = _gcs_download_prefix(
            gcs, f"{seed_gcs}/checkpoints/best/", adapter_local,
        )
        print(f"[seed {seed}] pulled {n} adapter files")
        # Pull train/val jsonl too (small, useful for any sanity checks).
        for jl in ("train.jsonl", "val.jsonl"):
            dst = f"{seed_local}/{jl}"
            if not os.path.exists(dst):
                _gcs_download_blob(gcs, f"{seed_gcs}/{jl}", dst)
    else:
        print(f"[seed {seed}] cached on volume, skipping GCS pull")

    # 3. Layout for the repo: eval scripts expect a writeable eval/seed_<N>/
    #    directory. Adapter is referenced by absolute path via --lora-path.
    #    For smoke runs we use a separate output dir so the tiny smoke JSONs
    #    don't satisfy the real eval's idempotency guard and cause it to
    #    skip the real eval.
    eval_subdir = f"seed_{seed}_smoke" if max_examples is not None else f"seed_{seed}"
    repo_eval = f"{repo_dir}/eval/{eval_subdir}"
    os.makedirs(repo_eval, exist_ok=True)
    lora_dir = adapter_local  # absolute path inside the data Volume

    # data/sft per-seed, data/raw shared
    repo_data_sft = f"{repo_dir}/data/sft"
    os.makedirs(repo_data_sft, exist_ok=True)
    for jl in ("train.jsonl", "val.jsonl"):
        src = f"{seed_local}/{jl}"
        dst = f"{repo_data_sft}/{jl}"
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)

    # 4. Pull HealthBench data + per-seed eval IDs (the latter is shared
    #    deterministic-by-seed; produced by make_bootstrap_eval_ids.py).
    if not os.path.exists(f"{repo_dir}/data/raw/healthbench_hard.jsonl"):
        _run(["python", "scripts/download_data.py"])
    seed_ids = f"{repo_dir}/data/raw/hard_seed_{seed}.json"
    if not os.path.exists(seed_ids):
        _run([
            "python", "scripts/make_bootstrap_eval_ids.py",
            "--healthbench-jsonl", f"{repo_dir}/data/raw/healthbench_hard.jsonl",
            "--seed", str(seed),
            "--output", seed_ids,
        ])

    # 5. Run the 4 Stage-4 configs sequentially, skipping any whose JSON
    #    already exists in the volume. Single-grader (no Mistral) — see
    #    docstring rationale. lora_dir was set in step 3 above.
    configs = [
        # (tag, lora_args, use_bodhi)
        ("base_no_wrapper",  [],                          False),
        ("base_bodhi",       [],                          True),
        ("lora_no_wrapper",  ["--lora-path", lora_dir],   False),
        ("lora_bodhi",       ["--lora-path", lora_dir],   True),
    ]
    for tag, lora_args, use_bodhi in configs:
        out = f"{repo_eval}/{tag}.json"
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print(f"[seed {seed}] {tag}.json exists, skipping")
            continue
        cmd = [
            "python", "scripts/eval_healthbench.py",
            "--model", "google/medgemma-27b-text-it",
            "--sample-ids", seed_ids,
            "--grader-model", "meta-llama/Llama-3.1-8B-Instruct",
            "--output", out,
            "--seed", str(seed),
        ] + lora_args
        if use_bodhi:
            cmd.append("--use-bodhi")
        if max_examples is not None:
            cmd += ["--max-examples", str(max_examples)]
        print(f"\n[seed {seed}] === Stage 4 config {tag} ===\n")
        _run(cmd)

        # Mirror the config JSON back onto the volume so a mid-run
        # interruption doesn't lose progress.
        vol_dst = f"{seed_local}/eval"
        os.makedirs(vol_dst, exist_ok=True)
        _run(["cp", out, vol_dst])

    # 6. Stage 5 epistemic — only if all 4 configs exist.
    epistemic_out = f"{repo_eval}/epistemic_scores.json"
    inputs = [f"{repo_eval}/{name}.json" for name, *_ in configs]
    if os.path.exists(epistemic_out) and os.path.getsize(epistemic_out) > 0:
        print(f"[seed {seed}] epistemic_scores.json exists, skipping")
    elif all(os.path.exists(p) for p in inputs):
        print(f"\n[seed {seed}] === Stage 5 epistemic ===\n")
        epistemic_cmd = [
            "python", "scripts/eval_epistemic.py",
            "--response-files", *inputs,
            "--grader-model", "meta-llama/Llama-3.1-8B-Instruct",
            "--output", epistemic_out,
            "--seed", str(seed),
        ]
        if max_examples is not None:
            epistemic_cmd += ["--max-examples", str(max_examples)]
        _run(epistemic_cmd)
        _run(["cp", epistemic_out, f"{seed_local}/eval/"])
    else:
        missing = [p for p in inputs if not os.path.exists(p)]
        print(f"[seed {seed}] Stage 5 SKIPPED — missing {missing}")

    # 7. Push results back to GCS so the rest of the analysis pipeline
    #    can find them at the canonical seed_<N>/eval/ path. Smoke runs
    #    push to a smoke-only prefix so their tiny --max-examples results
    #    don't corrupt real eval. We DO exercise the upload write path on
    #    smoke so an auth/bucket-write bug surfaces here, not on the real
    #    run.
    push_uri = (
        f"{seed_gcs}/eval/" if max_examples is None
        else f"{seed_gcs}/eval_smoke/"
    )
    print(f"[seed {seed}] uploading {repo_eval}/ -> {push_uri}")
    n = _gcs_upload_dir(gcs, repo_eval, push_uri)
    print(f"[seed {seed}] uploaded {n} result files")

    # 8. Commit volume writes (Modal Volume.commit makes the changes
    #    visible to other functions / next runs).
    data_cache.commit()
    hf_cache.commit()

    return f"seed_{seed} eval complete; pushed to {seed_gcs}/eval/"


@app.local_entrypoint()
def check_auth():
    """Cheap CPU-only auth probe (~$0). Run this any time you change creds."""
    print(validate_auth.remote())


@app.local_entrypoint()
def warm_cache():
    """Pre-download MedGemma-27B + Llama-8B to the HF Volume on CPU.

    Run this ONCE before the first smoke / run_all to amortize the ~10 min
    HF download out of the GPU billing window. Idempotent — skips files
    already present, so re-running is cheap.
    """
    print(warm_hf_cache.remote())


@app.local_entrypoint()
def run_all():
    """Run all 5 seeds in parallel via .map().

    Modal bills per-GPU-second, so concurrency is free wall-clock-wise:
    fanning out to 5 finishes ~5x faster than sequential at the same cost.
    The function is capped at max_containers=5 to match len(SEEDS).

    Cold-cache caveat: each container reads the HF Volume at start and only
    commits at end. If all 5 start cold simultaneously they each download
    MedGemma-27B (~54 GB) + Llama-3.1-8B (~16 GB) independently. Modal
    doesn't charge for network egress, but the parallel downloads add ~5 min
    of GPU-billed wall time per container. To avoid that on the very first
    run, warm the cache once with ``modal run modal_eval.py::run_one
    --seed 7`` (writes to Volume on commit), then ``run_all`` reads from
    cache for all 5 seeds.
    """
    print("--- Auth probe (CPU, ~$0) ---")
    print(validate_auth.remote())
    print("--- Auth OK; fanning out 5 GPUs ---")
    for result in eval_seed.map(SEEDS):
        print(result)


@app.local_entrypoint()
def run_one(seed: int = 7):
    """Single-seed full-eval entrypoint (~75 min, ~$4)."""
    print("--- Auth probe (CPU, ~$0) ---")
    print(validate_auth.remote())
    print(eval_seed.remote(seed))


@app.local_entrypoint()
def smoke(seed: int = 7, max_examples: int = 3):
    """Cheap end-to-end smoke (~15 min, ~$0.80).

    Runs the FULL pipeline (image build, GCS auth, GCS pull, vLLM init,
    bodhi wrapper subprocess, PEFT adapter load, all 4 configs, Stage 5,
    GCS push-back) but with --max-examples=3 so each eval is ~30 sec
    instead of ~13 min. Validates every plumbing seam before we burn $20
    on the real run.

    Auth is probed on a CPU container first — if creds are wrong, no A100
    is ever allocated.

    Outputs go to a separate smoke prefix (eval/seed_<N>_smoke/ locally,
    eval_smoke/ in GCS) so real-run idempotency isn't broken. Note:
    HF Volume cache IS shared, so smoke warms MedGemma + Llama for the
    real run that follows.
    """
    print("--- Auth probe (CPU, ~$0) ---")
    print(validate_auth.remote())
    print("--- Auth OK; allocating A100 for smoke ---")
    print(eval_seed.remote(seed, max_examples=max_examples))
