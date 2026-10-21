"""Record what a stage run used, in a manifest next to its outputs.

Every stage writes <stage folder>/manifest.json when it finishes, and the runner writes
the cell's manifest.json. A manifest holds:
  - the resolved settings and their sha256,
  - the git commit, a dirty flag and the sha256 of the uncommitted diff,
  - the python, torch, transformers, vllm and peft versions of the env that did the work
    (the runner writes some manifests itself and names the stage's env, not its own),
  - each model by registry id, with its repo, family and the revision it loaded,
  - the adapter sha256 and the eval-id hashes,
  - input and output file hashes and stage counts (tau, kept traces, tokens, ...),
  - wall-clock time and the GPUs the stage saw,
  - the host name, as a sha256 only.
The cost analysis reads the time, GPU and token fields.
"""

import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
# 2: models by registry id, timing and GPU fields (schema 1 had repo ids only).
# 3: manifests the runner writes record the packages of the env that ran the stage
#    (schema 2 recorded the runner's own env there); the cell manifest records both envs.
MANIFEST_SCHEMA = 3

# Packages whose version changes what a run produces. vllm is read from the
# running interpreter: the engine starts the vllm binary that sits next to this
# interpreter, so the two versions agree.
TRACKED_PACKAGES = (
    "torch", "transformers", "vllm", "peft", "trl", "accelerate",
    "bitsandbytes", "datasets", "bodhi-llm", "huggingface-hub", "numpy",
)

# -- hashing -----------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj) -> str:
    """sha256 of a canonical JSON form, so key order and spacing do not matter."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(canonical)


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def file_digest(path) -> dict:
    """Path, size and sha256 of one file. Counts lines for JSONL files."""
    p = Path(path)
    if not p.is_file():
        return {"path": str(path), "exists": False}
    out = {"path": str(path), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
    if p.suffix == ".jsonl":
        with open(p, "rb") as f:
            out["n_lines"] = sum(1 for line in f if line.strip())
    return out


# -- environment ---------------------------------------------------------------

def _git(args, repo_root):
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True, timeout=60, check=True,
    ).stdout


def git_state(repo_root=REPO_ROOT) -> dict:
    """Commit, dirty flag and the sha256 of the diff against HEAD.

    dirty covers tracked files only. Untracked files are counted separately,
    because a run can import an untracked script.
    """
    try:
        sha = _git(["rev-parse", "HEAD"], repo_root).decode().strip()
    except (OSError, subprocess.SubprocessError) as e:
        return {"sha": None, "error": f"{type(e).__name__}: {e}"}
    try:
        status = _git(["status", "--porcelain"], repo_root).decode().splitlines()
        diff = _git(["diff", "HEAD", "--binary"], repo_root)
    except (OSError, subprocess.SubprocessError) as e:
        # The commit is known even when status times out, as it did on NRP's CephFS
        # volume (docs/NRP.md, 2026-10-21); only the dirty check is missing then.
        return {"sha": sha, "dirty": None, "error": f"{type(e).__name__}: {e}"}
    tracked = [line for line in status if not line.startswith("??")]
    untracked = [line for line in status if line.startswith("??")]
    return {
        "sha": sha,
        "dirty": bool(tracked),
        "diff_sha256": sha256_bytes(diff) if diff else None,
        "n_untracked": len(untracked),
    }


def package_versions(names=TRACKED_PACKAGES) -> dict:
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def interpreter_package_versions(python, names=TRACKED_PACKAGES) -> dict:
    """package_versions() as another interpreter sees it.

    The runner starts each stage in the env of configs/project.yaml (python.infer or
    python.train), which is not the env the runner itself runs in. A manifest the runner
    writes for a stage records that stage's env, so it asks that interpreter.
    """
    code = ("import json, platform, sys\n"
            "from importlib import metadata\n"
            "v = {'python': platform.python_version()}\n"
            "for n in sys.argv[1:]:\n"
            "    try:\n"
            "        v[n] = metadata.version(n)\n"
            "    except metadata.PackageNotFoundError:\n"
            "        v[n] = None\n"
            "print(json.dumps(v))\n")
    out = subprocess.run([str(python), "-c", code, *names], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def hostname_sha256() -> str:
    # The hostname identifies the machine, so only its hash is stored.
    return sha256_text(socket.gethostname())


# -- models and hardware ----------------------------------------------------------

def model_entry(model) -> dict:
    """One model record as the manifest keeps it: id, repo, family and revision."""
    return {"id": model.id, "repo_id": model.hf_id, "family": model.family,
            "revision": model.revision,
            "revision_source": "record" if model.revision else "unpinned",
            "unpinned_reason": model.unpinned_reason}


def gpu_info() -> dict:
    """The GPUs this process sees (CUDA_VISIBLE_DEVICES), by name, from nvidia-smi."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {"visible_devices": visible, "gpus": []}
    gpus = [dict(zip(("index", "name", "memory"), (x.strip() for x in line.split(","))))
            for line in out.splitlines() if line.strip()]
    if visible:
        keep = {v.strip() for v in visible.split(",")}
        gpus = [g for g in gpus if g["index"] in keep]
    return {"visible_devices": visible, "gpus": gpus}


class Timer:
    """Wall-clock start, end and seconds of a stage, for its manifest."""

    def __init__(self):
        self.start = datetime.now(timezone.utc)

    def record(self) -> dict:
        end = datetime.now(timezone.utc)
        return {"started_utc": self.start.isoformat(), "finished_utc": end.isoformat(),
                "seconds": round((end - self.start).total_seconds(), 3)}


# -- run artifacts ------------------------------------------------------------------

def load_id_list(path) -> list:
    """prompt_ids from a .json list, a {'prompt_ids': [...]} dict or a .jsonl file."""
    p = Path(path)
    if p.suffix == ".jsonl":
        with open(p) as f:
            return [json.loads(line)["prompt_id"] for line in f if line.strip()]
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data["prompt_ids"]
    return list(data)


def id_list_digest(path) -> dict:
    """Hash of the sorted, de-duplicated id set, plus the file hash.

    Eval code loads the ids as a set, so the set hash is the one to compare
    between runs. The file hash also catches a change in format.
    """
    p = Path(path)
    if not p.is_file():
        return {"path": str(path), "exists": False}
    ids = sorted(set(load_id_list(p)))
    return {"path": str(path), "n": len(ids), "ids_sha256": sha256_json(ids),
            "file_sha256": sha256_file(p)}


ADAPTER_FILES = ("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json")


def adapter_digest(adapter_dir) -> dict:
    """sha256 of the adapter weights and config, plus the LoRA shape."""
    d = Path(adapter_dir)
    if not d.is_dir():
        return {"path": str(adapter_dir), "exists": False}
    files = {name: sha256_file(d / name) for name in ADAPTER_FILES if (d / name).is_file()}
    combined = sha256_text("".join(f"{k}:{v}\n" for k, v in sorted(files.items())))
    out = {"path": str(adapter_dir), "sha256": combined, "files": files}
    cfg_path = d / "adapter_config.json"
    if cfg_path.is_file():
        with open(cfg_path) as f:
            cfg = json.load(f)
        for key in ("r", "lora_alpha", "lora_dropout", "target_modules",
                    "base_model_name_or_path", "revision", "use_dora", "use_rslora"):
            if key in cfg:
                out[key] = cfg[key]
    return out


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str))


def build_manifest(stage, *, cell: Optional[str] = None, seed: Optional[int] = None,
                   config=None, models: Optional[dict] = None, engines=None, adapter_dir=None,
                   eval_ids=None, inputs=None, outputs=None, counts=None, timer: Optional[Timer] = None,
                   extra=None, packages: Optional[dict] = None, repo_root=REPO_ROOT) -> dict:
    """Assemble a manifest dict. Every argument except stage is optional.

    models maps a role ("base", "grader", ...) to a model record (ModelConfig).
    packages defaults to this interpreter's versions; a caller that writes the manifest
    for work done in another env passes that env's versions.
    """
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "stage": stage,
        "cell": cell,
        "seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(repo_root),
        "packages": package_versions() if packages is None else packages,
        "host_sha256": hostname_sha256(),
        "hardware": gpu_info(),
    }
    if timer is not None:
        manifest["time"] = timer.record()
    if config is not None:
        manifest["config"] = _jsonable(config)
        manifest["config_sha256"] = sha256_json(manifest["config"])
    if models:
        manifest["models"] = {role: model_entry(m) for role, m in models.items()}
    if engines:
        manifest["engines"] = _jsonable(engines)
    if adapter_dir:
        manifest["adapter"] = adapter_digest(adapter_dir)
    if eval_ids:
        manifest["eval_ids"] = id_list_digest(eval_ids)
    if inputs:
        manifest["inputs"] = [file_digest(p) for p in inputs]
    if outputs:
        manifest["outputs"] = [file_digest(p) for p in outputs]
    if counts:
        manifest["counts"] = _jsonable(counts)
    if extra:
        manifest.update(_jsonable(extra))
    return manifest


def write_manifest(path, manifest) -> Path:
    """Write atomically, so a crash never leaves half a manifest.

    The temporary name carries the host and the PID: runner processes in different pods
    can write the same cell manifest at once (docs/NRP.md), and a shared name would let
    one process rename the other's half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
        f.write("\n")
    os.replace(tmp, path)
    return path
