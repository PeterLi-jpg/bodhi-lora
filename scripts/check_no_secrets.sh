#!/bin/bash
# Scan tracked files for common token patterns before opening a PR.
# Run: bash scripts/check_no_secrets.sh

set -euo pipefail

pattern='hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}'
tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT

git ls-files -z | xargs -0 grep -nE "$pattern" >"$tmp_file" || true

if [ -s "$tmp_file" ]; then
  echo "Potential secret detected in tracked files:"
  cat "$tmp_file"
  exit 1
fi

echo "No tracked secrets detected."
