#!/usr/bin/env bash
# Provisions the toolchain the quoridor work needs. Run by devcontainer.json's
# postCreateCommand, and safe to re-run by hand against a live container.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> apt packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12 python3.12-venv python3-pip

echo "==> wasm-pack"
command -v wasm-pack >/dev/null || \
  curl -sSf https://rustwasm.github.io/wasm-pack/installer/init.sh | sh

echo "==> python venv (3.12)"
# The pre-existing .venv was built on another machine without pip; rebuild it.
if [ ! -x "$REPO_ROOT/.venv/bin/pip" ]; then
  rm -rf "$REPO_ROOT/.venv"
  python3.12 -m venv "$REPO_ROOT/.venv"
fi
"$REPO_ROOT/.venv/bin/pip" install --upgrade pip

echo "==> python requirements"
# Full requirements (not ci_requirements) -- the latter pins torch==2.9.1+cpu
# and so cannot support GPU training. maturin is listed there, so it lands in
# the venv rather than needing a separate install.
"$REPO_ROOT/.venv/bin/pip" install -r "$REPO_ROOT/deep_quoridor/requirements.txt"

echo "==> done"
