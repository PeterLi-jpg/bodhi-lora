# Vendored MaxText

This directory contains a vendored copy of [AI-Hypercomputer/maxtext](https://github.com/AI-Hypercomputer/maxtext),
used as the JAX-native training baseline for Stage 3 (LoRA fine-tune of MedGemma-27B).

We vendor instead of submodule because the worktree-based `/batch` workflow
needs every checkout to be self-contained (a submodule pin would require an
extra `git submodule update` step in every worker container).

## Source

- Upstream: https://github.com/AI-Hypercomputer/maxtext
- Pinned commit: `ad2316c88ec67a04bcc28de8e4f9bff65cca341a`
- Pinned date: 2026-04-30 (UTC)
- License: Apache License 2.0 (see `LICENSE`)

## Layout

The upstream `pyproject.toml` declares the package at `src/maxtext`, so the
importable name is `maxtext` (lowercase). To use the vendored copy without
`pip install`:

```python
import sys
sys.path.insert(0, "third_party/maxtext/src")
import maxtext
```

(The original Stage-3 task description used `MaxText` as the import name; the
upstream package is actually `maxtext`. Both forms refer to the same library.)

## What was vendored

Included from upstream (runtime + light tooling):

- `src/` — Python package (`maxtext/` and supporting `dependencies/`).
- `tools/` — orchestration / data-generation helpers.
- `LICENSE`, `LICENSE_HEADER`, `AUTHORS`, `pyproject.toml`, `README.md`,
  `.gitignore` — preserved verbatim for attribution and packaging.

Excluded (dev-only, not needed at runtime):

- `tests/` — upstream's pytest suite (53 MB; we run our own tests).
- `docs/` — Sphinx docs (6.5 MB).
- `benchmarks/` — perf scripts.
- CI / lint configs: `.github/`, `.gemini/`, `.vscode/`, `.coveragerc`,
  `.dockerignore`, `.editorconfig`, `.pre-commit-config.yaml`,
  `.pre-commit-hooks.yaml`, `.readthedocs.yml`, `codecov.yml`, `pylintrc`,
  `pytest.ini`, `CONTRIBUTING.md`, `PREFLIGHT.md`.

## Refresh procedure

To bump to a newer pin:

```bash
git clone --depth 1 https://github.com/AI-Hypercomputer/maxtext /tmp/maxtext-vendor
cd /tmp/maxtext-vendor && git rev-parse HEAD   # note the new SHA
# overwrite vendored copy:
rm -rf third_party/maxtext/src third_party/maxtext/tools
cp -R /tmp/maxtext-vendor/src        third_party/maxtext/src
cp -R /tmp/maxtext-vendor/tools      third_party/maxtext/tools
cp     /tmp/maxtext-vendor/LICENSE   third_party/maxtext/LICENSE
cp     /tmp/maxtext-vendor/AUTHORS   third_party/maxtext/AUTHORS
cp     /tmp/maxtext-vendor/README.md third_party/maxtext/README.md
cp     /tmp/maxtext-vendor/pyproject.toml third_party/maxtext/pyproject.toml
find third_party/maxtext -name '__pycache__' -type d -prune -exec rm -rf {} +
# then update the `Pinned commit` and `Pinned date` lines above.
```
