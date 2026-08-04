#!/usr/bin/env bash
# Provisions the toolchain the ML projects in this repo need. Run by
# devcontainer.json's postCreateCommand, and safe to re-run by hand against a
# live container.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> apt packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12 python3.12-venv python3-pip

echo "==> python venv (3.12)"
# This workspace is bind-mounted from the host, so .venv is the host's own
# venv, not something built inside this container -- it is deliberately
# preserved and must never be deleted. Only create it if it's missing (e.g.
# a fresh clone on a machine with no venv yet).
if [ ! -d "$REPO_ROOT/.venv" ]; then
  python3.12 -m venv "$REPO_ROOT/.venv"
fi
# The venv's console scripts (pip, pytest, wandb, ...) carry host-absolute
# shebangs baked in on the host, so they can't execute inside the container
# under the bind mount. That's expected -- warn rather than "fixing" it by
# rewriting shebangs or recreating the venv. "python -m pip" sidesteps the
# problem entirely, since it runs via the venv's real interpreter.
if [ -x "$REPO_ROOT/.venv/bin/pip" ]; then
  pip_shebang="$(head -n1 "$REPO_ROOT/.venv/bin/pip")"
  pip_interpreter="${pip_shebang#"#!"}"
  if [ ! -x "$pip_interpreter" ]; then
    echo "note: $REPO_ROOT/.venv/bin/pip's shebang points at" >&2
    echo "  $pip_interpreter, which doesn't exist in this container --" >&2
    echo "  expected, since that shebang is host-absolute and this" >&2
    echo "  workspace is bind-mounted. Using '.venv/bin/python -m pip'" >&2
    echo "  instead; do the same by hand." >&2
  fi
fi
"$REPO_ROOT/.venv/bin/python" -m pip install --upgrade pip

# No single requirements.txt covers this repo -- each project under it
# (object_tracker_0/, annotation_utils/, deep_water_level/, camera_control/,
# pytorch_exercises/) has its own. Install per-project as needed rather than
# pinning this shared venv to one of them.

echo "==> done"
