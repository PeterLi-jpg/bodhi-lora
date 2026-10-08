"""Record what a pipeline run used, in a manifest next to its outputs.

Every stage that produces numbers (trace generation, filtering, training,
evaluation, re-grading) calls write_manifest() when it finishes. The cell
launcher rebuttal/launch/run_cell.sh uses the CLI below to write a cell
manifest and to run the grader-family guard.

A manifest holds:
  - the resolved config and its sha256,
  - the git commit, a dirty flag and the sha256 of the uncommitted diff,
  - the python, torch, transformers, vllm and peft versions,
  - each model id with its family and the HF revision it loaded,
  - the adapter sha256 and the eval-id and holdout-id hashes,
  - stage counts, for example tau and the kept and generated trace counts,
  - the hostname, as a sha256 only.

The model registry (family, pinned revision, system-role support) is
configs/families.yaml.

CLI:
    python scripts/run_manifest.py write --stage cell --out <dir>/manifest.json \\
        --config rebuttal/configs/x.yaml --model base=<repo>[@<rev>] \\
        --eval-ids ids.json --set BENCH=healthbench --include filter=<dir>/sft/manifest.json
    python scripts/run_manifest.py check-family --base <repo> --filter <repo> \\
        --evaluator <repo> [--allow-same-family] [--json-out f.json]
    python scripts/run_manifest.py revision <repo>       # prints the pinned revision
    python scripts/run_manifest.py system-role <repo>    # prints true or false
"""

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "configs" / "families.yaml"
MANIFEST_SCHEMA = 1

# Packages whose version changes what a run produces. vllm is read from the
# running interpreter: in subprocess mode _vllm_engine starts the vllm binary
# that sits next to this interpreter, so the two versions agree.
TRACKED_PACKAGES = (
    "torch", "transformers", "vllm", "peft", "trl", "accelerate",
    "bitsandbytes", "datasets", "bodhi-llm", "huggingface-hub", "numpy",
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


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
        status = _git(["status", "--porcelain"], repo_root).decode().splitlines()
        diff = _git(["diff", "HEAD", "--binary"], repo_root)
    except (OSError, subprocess.SubprocessError) as e:
        return {"sha": None, "error": f"{type(e).__name__}: {e}"}
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


def hostname_sha256() -> str:
    # The hostname identifies the machine, so only its hash is stored.
    return sha256_text(socket.gethostname())


# -- model registry --------------------------------------------------------------

_REGISTRY_CACHE = {}


def load_registry(path=REGISTRY_PATH) -> dict:
    path = Path(path)
    key = str(path.resolve())
    if key not in _REGISTRY_CACHE:
        with open(path) as f:
            _REGISTRY_CACHE[key] = yaml.safe_load(f) or {}
    return _REGISTRY_CACHE[key]


def _entry(repo_id, registry=None) -> dict:
    registry = registry if registry is not None else load_registry()
    return (registry.get("models") or {}).get(repo_id) or {}


def model_family(repo_id, registry=None):
    return _entry(repo_id, registry).get("family")


def pinned_revision(repo_id, registry=None):
    return _entry(repo_id, registry).get("revision")


def supports_system_role(repo_id, registry=None) -> bool:
    return bool(_entry(repo_id, registry).get("system_role", True))


def role_default(role, registry=None):
    registry = registry if registry is not None else load_registry()
    return (registry.get("roles") or {}).get(role)


def effective_revision(repo_id, requested=None, registry=None):
    """The revision a run should load: the flag value, else the registry pin.

    A local path has no hub revision, so it returns None for it.
    """
    if requested:
        return requested
    if repo_id and Path(repo_id).exists():
        return None
    return pinned_revision(repo_id, registry)


def resolve_model(repo_id, requested_revision=None, registry=None) -> dict:
    """Describe one model for the manifest: family and the revision used."""
    if repo_id and Path(repo_id).exists():
        return {"repo_id": repo_id, "family": None, "revision": None,
                "revision_source": "local_path"}
    revision = effective_revision(repo_id, requested_revision, registry)
    if requested_revision:
        source = "flag"
    elif revision:
        source = "registry"
    else:
        source = "unpinned"
    return {
        "repo_id": repo_id,
        "family": model_family(repo_id, registry),
        "revision": revision,
        "revision_is_commit": bool(revision and _SHA_RE.match(revision)),
        "revision_source": source,
    }


# -- grader-family guard --------------------------------------------------------

class FamilyError(Exception):
    """A run would grade or filter a model with a model of its own family."""


def family_check(base, filter_model, evaluator, allow_same_family=False,
                 registry=None) -> dict:
    """Check the asymmetric-grading rule for one cell.

    The base model must not share a family with the trace filter or with the
    evaluator, and the filter and the evaluator must differ. A model with no
    registry entry fails, so a new model must be classified before it runs.
    """
    roles = {"base": base, "filter": filter_model, "evaluator": evaluator}
    families = {role: model_family(m, registry) for role, m in roles.items()}
    unknown = [f"{role}={roles[role]}" for role, fam in families.items() if fam is None]
    if unknown:
        raise FamilyError(
            "no family in configs/families.yaml for " + ", ".join(unknown)
            + ". Add the model there before running.")
    clashes = []
    for role in ("filter", "evaluator"):
        if families["base"] == families[role]:
            clashes.append(f"base and {role} are both {families['base']}")
    if families["filter"] == families["evaluator"]:
        clashes.append(f"filter and evaluator are both {families['filter']}")
    result = {
        "models": {role: {"repo_id": m, "family": families[role]} for role, m in roles.items()},
        "clashes": clashes,
        "allow_same_family": bool(allow_same_family),
        "passed": not clashes,
        "waived": bool(clashes) and bool(allow_same_family),
    }
    if clashes and not allow_same_family:
        raise FamilyError(
            "grader-family guard: " + "; ".join(clashes)
            + ". Pass --allow-same-family to run anyway; the manifest records it.")
    return result


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


def manifest_path_for(output) -> Path:
    """Where a stage writes its manifest.

    A directory output gets <dir>/manifest.json. A file output gets
    <stem>.manifest.json beside it, so several outputs in one directory
    (for example the four eval configs) keep separate manifests.
    """
    p = Path(output)
    if p.is_dir() or not p.suffix:
        return p / "manifest.json"
    return p.with_name(p.stem + ".manifest.json")


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str))


def build_manifest(stage, *, config=None, config_path=None, models=None,
                   engines=None, adapter_dir=None, eval_ids=None,
                   holdout_ids=None, inputs=None, outputs=None, counts=None,
                   extra=None, repo_root=REPO_ROOT) -> dict:
    """Assemble a manifest dict. Every argument except stage is optional.

    models maps a role ('base', 'grader', ...) to (repo_id, requested_revision)
    or to a dict already built by resolve_model().
    """
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(repo_root),
        "packages": package_versions(),
        "host_sha256": hostname_sha256(),
    }
    if config_path:
        manifest["config_file"] = file_digest(config_path)
    if config is not None:
        manifest["config"] = _jsonable(config)
        manifest["config_sha256"] = sha256_json(manifest["config"])
    if models:
        resolved = {}
        for role, spec in models.items():
            if isinstance(spec, dict):
                resolved[role] = spec
            else:
                repo_id, revision = spec
                resolved[role] = resolve_model(repo_id, revision)
        manifest["models"] = resolved
    if engines:
        manifest["engines"] = _jsonable(engines)
    if adapter_dir:
        manifest["adapter"] = adapter_digest(adapter_dir)
    if eval_ids:
        manifest["eval_ids"] = id_list_digest(eval_ids)
    if holdout_ids:
        manifest["holdout_ids"] = id_list_digest(holdout_ids)
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
    """Write atomically, so a crash never leaves half a manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
        f.write("\n")
    os.replace(tmp, path)
    return path


# -- CLI ------------------------------------------------------------------------

def _parse_value(text):
    """'0.4' -> 0.4, 'true' -> True, anything else stays a string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _split_pair(item, sep="="):
    if sep not in item:
        raise SystemExit(f"expected KEY{sep}VALUE, got {item!r}")
    key, value = item.split(sep, 1)
    return key, value


def _cmd_write(args):
    settings = {}
    for item in args.set or []:
        key, value = _split_pair(item)
        settings[key] = _parse_value(value)
    config = None
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    models = {}
    for item in args.model or []:
        role, spec = _split_pair(item)
        repo_id, _, revision = spec.partition("@")
        models[role] = (repo_id, revision or None)
    includes = {}
    for item in args.include or []:
        key, path = _split_pair(item)
        p = Path(path)
        if p.is_file():
            with open(p) as f:
                includes[key] = json.load(f)
        else:
            includes[key] = {"path": path, "exists": False}
    extra = {}
    if settings:
        extra["settings"] = settings
        extra["settings_sha256"] = sha256_json(settings)
    if includes:
        extra["includes"] = includes
    manifest = build_manifest(
        args.stage, config=config, config_path=args.config, models=models or None,
        adapter_dir=args.adapter, eval_ids=args.eval_ids, holdout_ids=args.holdout_ids,
        inputs=args.input, extra=extra or None,
    )
    path = write_manifest(args.out, manifest)
    print(f"manifest -> {path}")


def _cmd_check_family(args):
    try:
        result = family_check(args.base, args.filter, args.evaluator,
                              allow_same_family=args.allow_same_family)
    except FamilyError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(3)
    if args.json_out:
        write_manifest(args.json_out, result)
    fams = ", ".join(f"{r}={v['family']}" for r, v in result["models"].items())
    if result["waived"]:
        print(f"family check WAIVED by --allow-same-family ({fams}): "
              + "; ".join(result["clashes"]))
    else:
        print(f"family check passed ({fams})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="write a manifest from launcher settings")
    w.add_argument("--stage", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--config", default=None, help="YAML config to embed and hash")
    w.add_argument("--model", action="append", help="ROLE=REPO[@REVISION]")
    w.add_argument("--adapter", default=None)
    w.add_argument("--eval-ids", default=None)
    w.add_argument("--holdout-ids", default=None)
    w.add_argument("--input", action="append", help="file to hash")
    w.add_argument("--set", action="append", help="KEY=VALUE launcher setting")
    w.add_argument("--include", action="append", help="KEY=PATH of a JSON file to embed")
    w.set_defaults(func=_cmd_write)

    c = sub.add_parser("check-family", help="grader-family guard for one cell")
    c.add_argument("--base", required=True)
    c.add_argument("--filter", required=True)
    c.add_argument("--evaluator", required=True)
    c.add_argument("--allow-same-family", action="store_true")
    c.add_argument("--json-out", default=None)
    c.set_defaults(func=_cmd_check_family)

    r = sub.add_parser("revision", help="print the pinned revision of a model")
    r.add_argument("repo_id")
    r.set_defaults(func=lambda a: print(pinned_revision(a.repo_id) or ""))

    s = sub.add_parser("system-role", help="print whether the chat template takes a system message")
    s.add_argument("repo_id")
    s.set_defaults(func=lambda a: print("true" if supports_system_role(a.repo_id) else "false"))

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
