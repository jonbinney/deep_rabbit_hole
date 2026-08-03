# Extract deep_quoridor into lll_alpha_quoridor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `deep_quoridor/` out of `adamantivm/deep_rabbit_hole` into a standalone public repo, `adamantivm/lll_alpha_quoridor`, verified to train, test, build and play exactly as it does today.

**Architecture:** Copy (not filter) the subtree into a staging directory, apply the six edits the move requires, land it as one initial commit, then run a verification gate that ends in a human playing a browser game against a freshly-trained model. Only after the gate is green do the improvement commits and the old-repo cleanup happen.

**Tech Stack:** Python 3.12 (pytest, PyTorch, FastAPI), Rust (cargo, wasm-pack, maturin), Svelte + Vite + onnxruntime-web, GitHub Actions, devcontainers.

**Spec:** `docs/superpowers/specs/2026-08-03-extract-quoridor-repo-design.md`

## Global Constraints

- **Nothing lands on `main` in `deep_rabbit_hole`.** All work there goes on branch `jac/extract-quoridor-repo` and reaches `main` via pull request. This applies to specs, plans, devcontainer configs and code alike.
- **Commit messages** start with `vibe: ` followed by an imperative subject, 50 chars max excluding the prefix. Non-trivial changes get a body separated by a blank line, wrapped at 72 chars, explaining **why**. Never enumerate modified files.
- **Separate functional commits from formatting/lint commits.** One commit per logical change.
- **Rust:** run `cargo fmt --all` before committing, then verify formatting, build and tests.
- **Python:** activate the virtualenv before running anything.
- **Python 3.12 exactly** — CI pins it and the torch versions in the requirements files target it. Julian's host runs Ubuntu 24.04, whose system Python is 3.12; the devcontainer's `devcontainers/base:ubuntu` tag was unpinned and drifted to 26.04, whose system Python is 3.14. Task 1 pins the image to 24.04 so the container matches the host.
- **New repo:** `adamantivm/lll_alpha_quoridor`, public, default branch `main`.
- **Nothing hardcodes the owner `adamantivm`** — it may move to an org later.
- **Per AGENTS.md**, finish by writing a results markdown file suitable for a PR description.

## Key Paths

| What | Where |
|---|---|
| Old repo (working dir) | `/workspaces/deep_rabbit_hole` |
| Old repo branch | `jac/extract-quoridor-repo` |
| New repo staging dir | `/workspaces/lll_alpha_quoridor` |
| Source subtree | `/workspaces/deep_rabbit_hole/deep_quoridor` |

**`/workspaces` is root-owned and container-local.** Creating the staging dir needs `sudo`, and it does **not** survive a container rebuild. This is why Task 1 (which forces a rebuild) comes first and why Task 6 pushes to GitHub immediately — after that, a wipe costs one `git clone`.

---

## File Structure

**Modified in `deep_rabbit_hole` (branch `jac/extract-quoridor-repo`):**

- `.devcontainer/devcontainer.json` — rebased onto a Python 3.12 image; adds Rust, GitHub CLI, CUDA, GPU passthrough, and a post-create hook.
- `.devcontainer/post-create.sh` — *(new)* installs `wasm-pack` and `maturin`, creates the venv, installs requirements. One script so a rebuild and a manual run do the same thing.
- `docs/superpowers/plans/2026-08-03-extract-quoridor-repo.md` — this file.
- `docs/superpowers/results/2026-08-03-extract-quoridor-repo-results.md` — *(new, Task 15)* PR body.

**Deleted from `deep_rabbit_hole` (Task 15):** `deep_quoridor/`, `docs/superpowers/specs|plans` (they travel with the work), `.github/workflows/{python-app,rust-ci}.yml`, `.github/prompts/`, `experiments/2026_05_23_jon_b9w10_performance/`.

**Created in `lll_alpha_quoridor`:** the target tree from the spec. Files needing edits beyond a straight copy:

| File | Why |
|---|---|
| `.github/workflows/python-app.yml` | strip `deep_quoridor/` from paths, PYTHONPATH, cache key |
| `.github/workflows/rust-ci.yml` | same, plus `working-directory` |
| `.gitignore` | derived subset; must keep the `lib/` carve-out |
| `CLAUDE.md` | points at a file that does not exist |
| `src/v2/ai_report.py` | **functional** — repo-root arithmetic + path prefixes |
| `scripts/bench_rust_selfplay.sh` | **functional** — binary path + `cd` target |
| `test/test_ai_report_paths.py` | *(new)* regression test for the above |
| `src/v2/config.py`, `rust/src/python_consistency.rs`, 6 docs | cosmetic stale prefixes |

---

## Task 1: Devcontainer with the full toolchain

Everything downstream needs Python 3.12, Rust, and a GPU. None are present. This task ends with a human rebuilding the container, so it must complete before any other work starts.

The root cause is a floating tag: `.devcontainer/devcontainer.json` asks for
`mcr.microsoft.com/devcontainers/base:ubuntu`, which now resolves to Ubuntu 26.04 and its
Python 3.14. Pinning to `ubuntu-24.04` restores Python 3.12, matches Julian's host, and
stops the same drift recurring.

**Files:**
- Modify: `/workspaces/deep_rabbit_hole/.devcontainer/devcontainer.json`
- Create: `/workspaces/deep_rabbit_hole/.devcontainer/post-create.sh`

**Interfaces:**
- Produces: a container with `python3.12` + venv at `/workspaces/deep_rabbit_hole/.venv`, `cargo`, `rustc`, `rustfmt`, `clippy`, `wasm-pack`, `gh`, `node`, `.venv/bin/maturin`, and CUDA visible to torch.

- [ ] **Step 1: Confirm the branch**

```bash
cd /workspaces/deep_rabbit_hole
git rev-parse --abbrev-ref HEAD   # must print: jac/extract-quoridor-repo
```

If it prints anything else, `git checkout jac/extract-quoridor-repo` before continuing. Do not proceed on `main`.

- [ ] **Step 2: Ask Julian to confirm the host prerequisite**

GPU passthrough needs NVIDIA Container Toolkit on the host, which cannot be checked from inside the container. Ask:

> "Is NVIDIA Container Toolkit installed on the host? (`nvidia-ctk --version` or `docker run --rm --gpus all ubuntu nvidia-smi` on the host will tell us.)"

If **no**, continue anyway — build the container without `runArgs`, and Task 10's training leg falls back to Julian running it on the host. Note the decision in the results doc.

**Answered 2026-08-03: yes.** `docker run --rm --gpus all ubuntu nvidia-smi` on the host
returned an NVIDIA GeForce RTX 3050 Laptop GPU, driver 560.35.03, CUDA 12.6. Include
`runArgs`. Note the card has **4 GB VRAM** with ~137 MB already used by the display — see
the note in Task 10 Step 2.

- [ ] **Step 3: Write the post-create script**

The existing `.venv` is a dead artifact from another machine (`pyvenv.cfg` points at `/home/julian/aaae/...`, built `--without-pip`). Delete and rebuild it.

```bash
cat > /workspaces/deep_rabbit_hole/.devcontainer/post-create.sh <<'EOF'
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
EOF
chmod +x /workspaces/deep_rabbit_hole/.devcontainer/post-create.sh
```

- [ ] **Step 4: Rewrite devcontainer.json**

Pin the base image to 24.04 — same as the host, and Python 3.12 natively. `runArgs` cannot be applied to a running container; this is what forces the rebuild.

```bash
cat > /workspaces/deep_rabbit_hole/.devcontainer/devcontainer.json <<'EOF'
{
  "image": "mcr.microsoft.com/devcontainers/base:ubuntu-24.04",
  "features": {
    "ghcr.io/devcontainers/features/node:1": {},
    "ghcr.io/devcontainers/features/rust:1": {
      "profile": "default"
    },
    "ghcr.io/devcontainers/features/github-cli:1": {},
    "ghcr.io/devcontainers/features/nvidia-cuda:1": {
      "installCudnn": true
    },
    "ghcr.io/anthropics/devcontainer-features/claude-code:1.0": {}
  },
  "runArgs": ["--gpus", "all"],
  "postCreateCommand": ".devcontainer/post-create.sh"
}
EOF
```

If Step 2 answered **no**, omit the `"runArgs"` line.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/deep_rabbit_hole
git add .devcontainer/devcontainer.json .devcontainer/post-create.sh
git commit -m "vibe: give the devcontainer a full build toolchain

The base image tag was unpinned and had drifted to Ubuntu 26.04, whose
Python 3.14 nothing here supports; the container also had no Rust,
wasm-pack or GPU access, so none of the extraction's verification steps
could run. Pins to 24.04 to match the host and provisions the rest from
one script that a rebuild and a manual run share."
```

- [ ] **Step 6: HUMAN GATE — rebuild the container**

Ask Julian to run **Dev Containers: Rebuild Container** in VS Code. The session restarts; this plan file on disk carries the context across.

> "Devcontainer is committed. Please run *Dev Containers: Rebuild Container*. It'll take a few minutes (CUDA + torch). When it's back, we pick up at Task 2."

---

## Task 2: Verify the toolchain

**Files:** none — verification only.

**Interfaces:**
- Consumes: the rebuilt container from Task 1.
- Produces: a recorded pass/fail for GPU availability, which Task 10 branches on.

- [ ] **Step 1: Check every tool resolves**

```bash
for t in cargo rustc rustfmt clippy-driver wasm-pack gh node npm; do
  printf '%-14s %s\n' "$t" "$(command -v "$t" || echo MISSING)"
done
python3.12 -V
ls /workspaces/deep_rabbit_hole/.venv/bin/maturin
```

Expected: no `MISSING`, `Python 3.12.x`, and maturin present. `maturin` is listed in
`requirements.txt`, so it lives in the venv rather than on the global PATH — every later
task invokes it as `.venv/bin/maturin` or from an activated venv.

- [ ] **Step 2: Check the venv is real**

```bash
cd /workspaces/deep_rabbit_hole
.venv/bin/python -V && .venv/bin/python -c "import torch, numpy, fastapi; print('torch', torch.__version__)"
```

Expected: `Python 3.12.x` and a torch version. If `.venv/bin/python` is missing, re-run `.devcontainer/post-create.sh` and read its output — do not proceed past a broken venv.

- [ ] **Step 3: Check the GPU**

```bash
nvidia-smi -L
.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected: at least one GPU listed and `cuda: True`.

If `cuda: False`, **do not sink time into debugging here.** Record it and carry on — Task 10 has a documented fallback where Julian runs the training leg on the host.

- [ ] **Step 4: Record the outcome**

Note in the working log whether GPU is available. Task 10 reads this. No commit — nothing changed on disk.

---

## Task 3: Stage the new repo contents

A plain copy, verified by file count. No edits yet — those are Task 4, so a reviewer can see exactly what the move changed versus what the copy brought.

**Files:**
- Create: `/workspaces/lll_alpha_quoridor/` (the full target tree)

**Interfaces:**
- Produces: `/workspaces/lll_alpha_quoridor` containing the target tree, not yet a git repo.

- [ ] **Step 1: Create the staging directory**

```bash
sudo mkdir -p /workspaces/lll_alpha_quoridor
sudo chown "$(id -u):$(id -g)" /workspaces/lll_alpha_quoridor
```

- [ ] **Step 2: Copy the tracked subtree**

Copy from **git**, not the filesystem — the working tree holds gitignored `runs/`, `wandb/`, and `__pycache__/` that must not travel.

```bash
cd /workspaces/deep_rabbit_hole
git archive --format=tar HEAD deep_quoridor \
  | tar -x -C /workspaces/lll_alpha_quoridor --strip-components=1
```

- [ ] **Step 3: Copy the root-level files**

```bash
cd /workspaces/deep_rabbit_hole
mkdir -p /workspaces/lll_alpha_quoridor/.github/workflows \
         /workspaces/lll_alpha_quoridor/.github/prompts \
         /workspaces/lll_alpha_quoridor/docs/superpowers \
         /workspaces/lll_alpha_quoridor/.devcontainer

cp AGENTS.md CLAUDE.md pytest.ini ruff.toml /workspaces/lll_alpha_quoridor/
cp .github/workflows/python-app.yml .github/workflows/rust-ci.yml \
   /workspaces/lll_alpha_quoridor/.github/workflows/
cp .github/prompts/selfplay_rust_python_debugging.md \
   /workspaces/lll_alpha_quoridor/.github/prompts/
cp -r docs/superpowers/specs docs/superpowers/plans \
   /workspaces/lll_alpha_quoridor/docs/superpowers/
cp .devcontainer/devcontainer.json .devcontainer/post-create.sh \
   /workspaces/lll_alpha_quoridor/.devcontainer/
cp -r experiments/2026_05_23_jon_b9w10_performance \
   /workspaces/lll_alpha_quoridor/experiments/
```

- [ ] **Step 4: Verify the copy**

```bash
cd /workspaces/lll_alpha_quoridor
echo "top-level:"; ls -A
echo "counts:"; for d in src rust frontend test experiments coding-agents scripts docs .github; do
  printf '  %-14s %s\n' "$d" "$(find "$d" -type f | wc -l)"
done
echo "must NOT exist:"; ls runs wandb .env 2>&1 | head -3
echo "fixtures:"; ls -la rust/fixtures/
```

Expected: `src` 130, `rust` 60 (plus `fixtures/`), `frontend` 24, `test` 12, `coding-agents` 13, `scripts` 1; `docs` 19; `.github` 3; `runs`/`wandb`/`.env` all report "No such file"; both `alphazero_B5W2_mv1.*` fixtures present at ~412 KB.

Counts drifting by a file or two is fine — a missing whole directory is not.

- [ ] **Step 5: No commit**

Not a git repo yet. Task 4 edits, Task 5 initialises.

---

## Task 4: Apply the six required changes

The only edits the move requires. Two are functional; the rest keep docs and messages honest.

**Files:**
- Modify: `.github/workflows/python-app.yml`, `.github/workflows/rust-ci.yml`, `.gitignore` *(created)*, `CLAUDE.md`, `src/v2/ai_report.py`, `scripts/bench_rust_selfplay.sh`, `src/v2/config.py`, `rust/src/python_consistency.rs`, `AGENTS.md`, `frontend/README.md`, `rust/README.md`, `rust/RUST_FOR_PYTHONISTAS.md`, `src/v2/play_server_web/README.md`, `src/ai_report_cli.py`, `src/run_benchmarks_v2.py`, `src/v2/wandb_metrics.py`
- Create: `test/test_ai_report_paths.py`
- All paths relative to `/workspaces/lll_alpha_quoridor`

**Interfaces:**
- Consumes: the staged tree from Task 3.
- Produces: a tree where `git grep -n 'deep_quoridor/'` returns only `src/osaz/logs/*.md`, `coding-agents/*`, and `docs/superpowers/*`.

### 4a. The functional fix in ai_report.py — test first

- [ ] **Step 1: Write the failing test**

```bash
cat > /workspaces/lll_alpha_quoridor/test/test_ai_report_paths.py <<'EOF'
"""The AI report feeds real file paths to an LLM backend. When the repo was
extracted from deep_rabbit_hole these paths lost a directory level, and
nothing caught it -- the failure is a bad prompt, not an exception."""

from pathlib import Path

from v2 import ai_report


def test_repo_root_is_the_repository_root():
    # ai_report.py lives at <repo>/src/v2/ai_report.py
    expected = Path(ai_report.__file__).resolve().parent.parent.parent
    assert (expected / "src" / "v2" / "config.py").is_file()
    assert (expected / "pytest.ini").is_file()


def test_prompt_source_paths_all_exist():
    repo_root = Path(ai_report.__file__).resolve().parent.parent.parent
    prompt = ai_report._build_on_demand_prompt(
        project="p", group="g",
        metrics_snapshot=Path("/tmp/metrics.json"),
        repo_root=repo_root,
    )
    for line in prompt.splitlines():
        if line.startswith("- /"):
            assert Path(line[2:]).is_file(), f"prompt cites a missing file: {line[2:]}"
EOF
```

- [ ] **Step 2: Run it and watch it fail**

```bash
cd /workspaces/lll_alpha_quoridor
PYTHONPATH=src /workspaces/deep_rabbit_hole/.venv/bin/pytest test/test_ai_report_paths.py -v
```

Expected: `test_prompt_source_paths_all_exist` FAILS with `prompt cites a missing file: .../deep_quoridor/src/v2/config.py`.

- [ ] **Step 3: Fix the path list**

In `src/v2/ai_report.py`, `_build_on_demand_prompt`, drop the prefix from all four entries:

```python
    sources = [
        "src/v2/config.py",
        "src/v2/trainer.py",
        "src/v2/benchmarks.py",
        "src/metrics.py",
    ]
```

- [ ] **Step 4: Fix the repo-root arithmetic**

Same file, in `generate_on_demand_report` (~line 449). Three parents, not four:

```python
    # ai_report.py lives at <repo>/src/v2/ai_report.py
    # -> parents: v2 -> src -> <repo root>
    repo_root = Path(__file__).resolve().parent.parent.parent
```

- [ ] **Step 5: Fix the inline reference**

Same file, ~line 413:

```python
    f"Read {repo_root / 'src/v2/config.py'} for the "
```

- [ ] **Step 6: Fix the stale comment on `_src_root`**

Same file, ~line 526. The logic is correct in both layouts; only the comment lies:

```python
def _src_root() -> Path:
    # This file lives at src/v2/ai_report.py
    return Path(__file__).resolve().parent.parent
```

- [ ] **Step 7: Run the test and watch it pass**

```bash
cd /workspaces/lll_alpha_quoridor
PYTHONPATH=src /workspaces/deep_rabbit_hole/.venv/bin/pytest test/test_ai_report_paths.py -v
```

Expected: 2 passed.

### 4b. The functional fix in the bench script

- [ ] **Step 8: Fix the paths**

In `scripts/bench_rust_selfplay.sh`, line 10 and line 13:

```bash
BIN="rust/target/release/selfplay"
```
```bash
    (cd rust && cargo build --release --features binary --bin selfplay)
```

Read the surrounding lines before editing — reproduce the existing flags exactly, changing only the path.

- [ ] **Step 9: Verify by inspection**

```bash
cd /workspaces/lll_alpha_quoridor
grep -n 'deep_quoridor' scripts/bench_rust_selfplay.sh || echo "CLEAN"
bash -n scripts/bench_rust_selfplay.sh && echo "SYNTAX OK"
```

Expected: `CLEAN` and `SYNTAX OK`. The script is exercised for real in Task 8.

### 4c. CI workflows

- [ ] **Step 10: Strip the prefix from both workflows**

In `.github/workflows/python-app.yml` and `.github/workflows/rust-ci.yml`, remove every `deep_quoridor/` prefix. Change **paths only** — not versions, flags, or step logic. The `rust-ci.yml` comment explaining why `gpu` is excluded stays verbatim.

```bash
cd /workspaces/lll_alpha_quoridor
sed -i 's|deep_quoridor/||g' .github/workflows/python-app.yml .github/workflows/rust-ci.yml
```

- [ ] **Step 11: Review the diff by eye**

```bash
grep -nE 'paths:|- .|PYTHONPATH|working-directory|cache-dependency-path|run:' \
  .github/workflows/python-app.yml .github/workflows/rust-ci.yml
```

Check specifically that `working-directory:` now reads `rust`, `PYTHONPATH` points at `$(pwd)/src`, and `cache-dependency-path` has no prefix. `sed` is blunt — confirm it did not touch anything inside a URL or a comment.

### 4d. .gitignore

- [ ] **Step 12: Write the derived .gitignore**

The root file covers seven projects. This is the quoridor-relevant subset. The `!lib/` carve-out is load-bearing: the Python template's `lib/` pattern would otherwise swallow `frontend/src/lib/` (commit `5b0cd1b` in the old repo).

```bash
cat > /workspaces/lll_alpha_quoridor/.gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
# The Python template's lib/ rule above would swallow the frontend's source
# directory, which is not a Python build artifact.
!frontend/src/lib/
.venv/
venv/
.pytest_cache/
.ruff_cache/
.coverage
.coverage.*
htmlcov/

# Rust
target/
rust/quoridor-wasm/pkg/

# Node / frontend
node_modules/
frontend/dist/

# Training output and experiment tracking
runs/
wandb/
wandbmodels/
mlruns/
models/

# Local environment
.env
.claude/settings.local.json
EOF
```

- [ ] **Step 13: Prove the carve-out works**

```bash
cd /workspaces/lll_alpha_quoridor
git init -q 2>/dev/null || true
git check-ignore -v frontend/src/lib/api.ts && echo "BAD: frontend lib is ignored" || echo "GOOD: frontend lib is tracked"
git check-ignore -v runs/x .venv/x target/x >/dev/null && echo "GOOD: build output ignored"
```

Expected: `GOOD: frontend lib is tracked` and `GOOD: build output ignored`.

### 4e. Cosmetic stale references

- [ ] **Step 14: Fix CLAUDE.md**

It currently points at `@deep_quoridor/agents.md`, which does not exist in either repo.

```bash
echo "@AGENTS.md" > /workspaces/lll_alpha_quoridor/CLAUDE.md
```

- [ ] **Step 15: Fix the remaining prose and messages**

Strip `deep_quoridor/` from these, reading each line in context rather than blind-replacing:

- `src/v2/config.py:412` — build hint, becomes `cd rust && cargo build --release --features binary --bin selfplay`
- `rust/src/python_consistency.rs` — six `.expect("rust crate should live under deep_quoridor/")` messages become `.expect("rust crate should live under the repo root")`; **leave the surrounding logic alone**, it is relative and already correct
- `AGENTS.md` lines 19–20 — `rust/` and `src/`
- `frontend/README.md` — nine occurrences across build, run and test commands
- `rust/README.md:9`, `rust/RUST_FOR_PYTHONISTAS.md:3`
- `src/v2/play_server_web/README.md:28,49`
- `src/ai_report_cli.py:11`, `src/run_benchmarks_v2.py:4` — usage docstrings
- `src/v2/wandb_metrics.py:10` — references `deep_quoridor/src/metrics_reader.py`, a file that does not exist; the real one is `src/metrics.py`

**Do not touch `src/osaz/logs/*.md`** — historical tracebacks with absolute paths from Julian's old machine. They record what happened.

- [ ] **Step 16: Verify only the intended references remain**

```bash
cd /workspaces/lll_alpha_quoridor
grep -rn 'deep_quoridor' --include='*' . 2>/dev/null \
  | grep -v '^./src/osaz/logs/' | grep -v '^./coding-agents/' \
  | grep -v '^./docs/superpowers/' | grep -v '^./.git/'
```

Expected: only `wandb_project`/`project=` **string values** (`"deep_quoridor"` with no trailing slash) in `src/agents/alphazero/alphazero.py`, `src/train_sb3.py`, `src/osaz/alphazero_quoridor.py`, `src/v2/common.py`. Those are wandb project names, not paths — **leave them**, renaming would orphan the existing dashboards.

- [ ] **Step 17: Format check**

```bash
cd /workspaces/lll_alpha_quoridor
/workspaces/deep_rabbit_hole/.venv/bin/python -m ruff check . && \
/workspaces/deep_rabbit_hole/.venv/bin/python -m ruff format --check .
cargo fmt --manifest-path rust/Cargo.toml --all -- --check
```

Expected: clean. If `ruff format` wants changes, apply them — but they belong in a **separate commit** per AGENTS.md, so note which files and hold them for Task 5 Step 3.

---

## Task 5: Initial commit

**Files:** all of `/workspaces/lll_alpha_quoridor`

**Interfaces:**
- Produces: a git repo with one (or two) commits on `main`, not yet pushed.

- [ ] **Step 1: Initialise and stage**

Task 4 Step 13 already ran `git init` to test the `.gitignore`, so the repo may exist with
whatever default branch git chose. `git branch -M main` is unconditional for that reason —
`git init -b main` does **not** rename the branch of an already-initialised repo.

```bash
cd /workspaces/lll_alpha_quoridor
git init -q 2>/dev/null || true
git branch -M main
git rev-parse --abbrev-ref HEAD   # must print: main
git add -A
git status --short | head -30
echo "staged files: $(git diff --cached --name-only | wc -l)"
```

Expected: ~275 staged files. Nothing under `runs/`, `wandb/`, `node_modules/`, `target/`, `__pycache__/`.

- [ ] **Step 2: Commit the move**

```bash
cd /workspaces/lll_alpha_quoridor
git -c user.name="Julian Cerruti" -c user.email="jcerruti@gmail.com" \
  commit -q -m "vibe: import deep_quoridor as a standalone repository

Extracted from adamantivm/deep_rabbit_hole, where the project shared a repo
with six unrelated ML projects and could not be published on its own. This
is that code unchanged, apart from the path fixes the move forces: CI
prefixes, a derived .gitignore, and the repo-root arithmetic in ai_report.py
and the rust bench script, both of which assumed a deep_quoridor/ parent.

The full history stays in deep_rabbit_hole."
git log --stat --oneline -1 | tail -3
```

- [ ] **Step 3: Commit any formatting separately**

Only if Task 4 Step 17 produced changes:

```bash
cd /workspaces/lll_alpha_quoridor
git add -A && git commit -q -m "vibe: apply ruff formatting"
```

---

## Task 6: Create and push the GitHub repo

Push early — the staging directory does not survive a container rebuild, and after this it costs one `git clone` to recover.

**Files:** none locally.

**Interfaces:**
- Produces: `https://github.com/adamantivm/lll_alpha_quoridor` with `main` pushed and Actions running.

- [ ] **Step 1: Check gh auth**

```bash
gh auth status
```

If not authenticated, ask Julian to run `gh auth login` — do not attempt to authenticate on their behalf.

- [ ] **Step 2: HUMAN GATE — confirm before creating**

Creating a public repo is outward-facing and hard to undo. Confirm first:

> "Ready to create **public** repo `adamantivm/lll_alpha_quoridor` and push the initial commit. Go ahead?"

- [ ] **Step 3: Create and push**

```bash
cd /workspaces/lll_alpha_quoridor
gh repo create adamantivm/lll_alpha_quoridor --public --source=. --remote=origin --push
gh repo view adamantivm/lll_alpha_quoridor --json url,visibility,defaultBranchRef
```

Expected: `"visibility": "PUBLIC"`, default branch `main`.

- [ ] **Step 4: Watch the first CI run**

```bash
sleep 20 && gh run list --repo adamantivm/lll_alpha_quoridor --limit 5
```

Both workflows should have triggered. They are checked properly in Task 9 — this only confirms the path filters match, i.e. that the workflows fire at all. **Zero runs means the `paths:` filters are still wrong.**

---

## Task 7: Gate — Python

**Files:** none — verification only.

**Interfaces:**
- Consumes: the pushed repo.
- Produces: a green Python stack.

- [ ] **Step 1: Point the venv at the new repo**

The venv from Task 1 lives in the old repo but is version-appropriate and dependency-complete. Reuse it; only `PYTHONPATH` changes.

```bash
cd /workspaces/lll_alpha_quoridor
export VENV=/workspaces/deep_rabbit_hole/.venv/bin
$VENV/python -V
```

- [ ] **Step 2: Run the test suite**

```bash
cd /workspaces/lll_alpha_quoridor
PYTHONPATH=src $VENV/pytest test -v
```

Expected: all pass, including the new `test_ai_report_paths.py`. Record any failure verbatim before touching it — per the spec there is no baseline, so a failure gets fixed here, but it must be reported honestly rather than quietly patched.

- [ ] **Step 3: Sanity games**

```bash
cd /workspaces/lll_alpha_quoridor
PYTHONPATH=src $VENV/python src/play.py -p greedy mcts -t 2
```

Expected: two games play to completion with a result table.

- [ ] **Step 4: CI-scale training run**

```bash
cd /workspaces/lll_alpha_quoridor
PYTHONPATH=src $VENV/python src/train_v2.py experiments/ci.yaml
```

Expected: runs ~2 minutes (`finish_after: 2 minutes`), writes `runs/ci-*/`, emits benchmark output. This is the same command CI runs.

- [ ] **Step 5: Record, no commit**

Nothing changed on disk except gitignored `runs/`. Note results for the results doc.

---

## Task 8: Gate — Rust, WASM, and the extension

**Files:** none — verification only.

**Interfaces:**
- Produces: `rust/quoridor-wasm/pkg/` for Task 9's frontend build.

- [ ] **Step 1: Format, lint, build**

```bash
cd /workspaces/lll_alpha_quoridor/rust
cargo fmt --all -- --check
cargo clippy --all-targets --all-features
cargo build --verbose
```

- [ ] **Step 2: Tests**

The cross-language tests shell out to `src/action_reference.py` and `src/step_trace_reference.py`, so they exercise the `CARGO_MANIFEST_DIR.parent()/src` discovery the move could have broken. This is the real check on that path, not the `.expect()` message.

```bash
cd /workspaces/lll_alpha_quoridor/rust
PATH="/workspaces/deep_rabbit_hole/.venv/bin:$PATH" RUST_BACKTRACE=1 \
  cargo test --features binary --verbose
```

Expected: all pass. A failure mentioning `action_reference.py` or `step_trace_reference.py` not found means path discovery **did** break — revisit Task 4 Step 15.

- [ ] **Step 3: Release build**

```bash
cd /workspaces/lll_alpha_quoridor/rust && cargo build --release --verbose
```

- [ ] **Step 4: Python extension**

```bash
cd /workspaces/lll_alpha_quoridor/rust
/workspaces/deep_rabbit_hole/.venv/bin/maturin build --release
/workspaces/deep_rabbit_hole/.venv/bin/pip install --force-reinstall target/wheels/*.whl
/workspaces/deep_rabbit_hole/.venv/bin/python -c "import quoridor_rs; print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: The bench script**

Task 4 only syntax-checked it. Run it for real — it is the one place the fixed paths execute.

```bash
cd /workspaces/lll_alpha_quoridor && bash scripts/bench_rust_selfplay.sh
```

Expected: finds or builds `rust/target/release/selfplay` and prints timings. No "No such file or directory".

- [ ] **Step 6: Build the wasm package**

```bash
cd /workspaces/lll_alpha_quoridor
wasm-pack build rust/quoridor-wasm --target web --release
ls -la rust/quoridor-wasm/pkg/
```

Expected: `pkg/` contains `quoridor_wasm_bg.wasm`, `quoridor_wasm.js`, `package.json`.

---

## Task 9: Gate — frontend, play server, and CI

**Files:** none — verification only.

**Interfaces:**
- Consumes: `rust/quoridor-wasm/pkg/` from Task 8.
- Produces: `frontend/dist/` and a running play server for Task 10.

- [ ] **Step 1: Install and test the frontend**

```bash
cd /workspaces/lll_alpha_quoridor
npm --prefix frontend install
npm --prefix frontend run test
```

Expected: vitest passes (board model, eval marshalling, api client). `npm install` rather than `ci` — the lockfile references `quoridor-wasm` by relative path and must resolve against the just-built `pkg/`.

- [ ] **Step 2: Build the SPA**

```bash
cd /workspaces/lll_alpha_quoridor
npm --prefix frontend run build
ls frontend/dist frontend/dist/ort | head -20
```

Expected: `dist/index.html`, hashed JS assets, and `dist/ort/*.wasm` copied by `vite-plugin-static-copy`.

- [ ] **Step 3: Build a play directory from the fixture**

```bash
cd /workspaces/lll_alpha_quoridor
export PLAY=/tmp/playdir && rm -rf $PLAY && mkdir -p $PLAY/models
printf 'run_id: play\nquoridor:\n  board_size: 5\n  max_walls: 2\n  max_steps: 50\nalphazero:\n  mcts_n: 200\n  mcts_c_puct: 1.4\nself_play:\n  num_processes: 1\n  games_per_process: 1\ntraining:\n  games_per_training_step: 1.0\n  learning_rate: 0.001\n  batch_size: 64\n  weight_decay: 0.0001\n  replay_buffer_size: 1000\n' > $PLAY/config.yaml
cp rust/fixtures/alphazero_B5W2_mv1.onnx $PLAY/models/model_1.onnx
```

- [ ] **Step 4: Start the server**

```bash
cd /workspaces/lll_alpha_quoridor
PYTHONPATH=src /workspaces/deep_rabbit_hole/.venv/bin/python src/run_play_server_web.py \
  $PLAY --static-dir frontend/dist --port 8080 &
sleep 5
```

- [ ] **Step 5: Check every route**

```bash
for p in / /api/config /api/models /models/model_1.onnx /ort/ort-wasm-simd-threaded.jsep.wasm; do
  printf '%-45s %s\n' "$p" "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:8080$p")"
done
echo "--- config ---"; curl -s http://localhost:8080/api/config
echo; echo "--- models ---"; curl -s http://localhost:8080/api/models
```

Expected: all `200`. `/api/config` returns `board_size: 5`, `max_walls: 2` and a `defaults` object; `/api/models` returns `{"models":["model_1.onnx"],"default":"model_1.onnx"}`. If the `/ort/` filename 404s, list `frontend/dist/ort/` and retry with an actual filename — the exact name tracks the onnxruntime-web version.

- [ ] **Step 6: Stop the server**

```bash
kill %1 2>/dev/null || pkill -f run_play_server_web.py
```

- [ ] **Step 7: Confirm CI is green**

```bash
gh run list --repo adamantivm/lll_alpha_quoridor --limit 6
gh run watch --repo adamantivm/lll_alpha_quoridor "$(gh run list --repo adamantivm/lll_alpha_quoridor --limit 1 --json databaseId -q '.[0].databaseId')" || true
```

Expected: both `Python application` and `Rust CI` conclude `success`. On failure, `gh run view --log-failed` and fix — a CI-only failure usually means a path the local runs did not exercise.

---

## Task 10: Gate — the end-to-end chain

Train a model, export it, serve it, and have Julian play it. One continuous chain, and the same one project 2 will publish.

**Files:**
- Create: `/tmp/gate-run.yaml` (throwaway, not committed)

**Interfaces:**
- Consumes: a green Task 7–9 and the GPU status from Task 2.
- Produces: a run directory with exported ONNX, and human sign-off.

- [ ] **Step 1: Write a time-boxed config**

Based on `experiments/B5W2/cucu-01.yaml` — the proven B5W2 recipe — with a short budget, ONNX export on, and wandb dropped so no credentials are needed.

```bash
cat > /tmp/gate-run.yaml <<'EOF'
run_id: gate-$DATETIME
quoridor:
  board_size: 5
  max_walls: 2
  max_steps: 50
alphazero:
  network:
    type: resnet
    num_blocks: 2
    num_channels: 32
  mcts_n: 400
  mcts_c_puct: 1.2
self_play:
  num_processes: 4
  games_per_process: 40
  alphazero:
    mcts_noise_epsilon: 0.25
training:
  games_per_training_step: 25.0
  learning_rate: 0.001
  batch_size: 2048
  weight_decay: 0.0001
  replay_buffer_size: 10000
  save_onnx: true
  finish_after: 10 minutes
benchmarks:
  - every: 2 models
    jobs:
      - type: tournament
        prefix: raw
        alphazero:
          mcts_n: 0
        times: 20
        opponents:
          - random
          - greedy
      - type: dumb_score
        prefix: raw
        alphazero:
          mcts_n: 0
EOF
```

- [ ] **Step 2: Run it**

If Task 2 reported `cuda: True`:

```bash
cd /workspaces/lll_alpha_quoridor
PYTHONPATH=src /workspaces/deep_rabbit_hole/.venv/bin/python src/train_v2.py /tmp/gate-run.yaml 2>&1 | tee /tmp/gate-run.log
```

**VRAM watch.** The card is a 4 GB RTX 3050 Laptop with the display already on it. The
network is tiny, so `batch_size: 2048` is not the concern — four self-play processes each
holding a CUDA context is. On `CUDA out of memory`, drop `self_play.num_processes` to 2,
then 1, and record the value that worked. A smaller process count still satisfies the gate:
the claim is that the machinery turns, not that it trains fast.

If Task 2 reported `cuda: False`, **stop and hand off**:

> "GPU isn't reachable from the container. Please clone `adamantivm/lll_alpha_quoridor` on the host, run `PYTHONPATH=src python src/train_v2.py /tmp/gate-run.yaml` (config above) for ~10 minutes, and copy the resulting `runs/gate-*/` back to `/workspaces/lll_alpha_quoridor/runs/`."

The host runs Ubuntu 24.04, so its system Python is already 3.12 — the same version the
container targets. A plain `python3 -m venv .venv && pip install -r requirements.txt` on
the host is sufficient; no version juggling needed.

- [ ] **Step 3: Check the success signals**

The claim is "the machinery turns", not "the model is good".

```bash
cd /workspaces/lll_alpha_quoridor
RUN=$(ls -dt runs/gate-* | head -1); echo "run: $RUN"
echo "--- checkpoints ---"; ls -la $RUN/models/checkpoints/ | head
echo "--- onnx ---"; ls $RUN/models/checkpoints/*.onnx | wc -l
echo "--- loss trend ---"; grep -iE 'loss' /tmp/gate-run.log | tail -10
echo "--- benchmarks ---"; grep -iE 'dumb_score|tournament|win' /tmp/gate-run.log | tail -10
echo "--- gpu was used ---"; grep -iE 'cuda|device' /tmp/gate-run.log | head -3
```

Expected: several checkpoints, at least one `.onnx`, loss values trending down across steps, benchmark rows for `dumb_score` and the random/greedy tournament, and evidence the device was CUDA.

- [ ] **Step 4: Serve the freshly-trained model**

```bash
cd /workspaces/lll_alpha_quoridor
RUN=$(ls -dt runs/gate-* | head -1)
cp $RUN/config.yaml /tmp/gate-play.yaml 2>/dev/null || cp /tmp/gate-run.yaml /tmp/gate-play.yaml
mkdir -p /tmp/gate-play/models && cp /tmp/gate-play.yaml /tmp/gate-play/config.yaml
cp $RUN/models/checkpoints/*.onnx /tmp/gate-play/models/
PYTHONPATH=src /workspaces/deep_rabbit_hole/.venv/bin/python src/run_play_server_web.py \
  /tmp/gate-play --static-dir frontend/dist --port 8080 &
sleep 5 && curl -s http://localhost:8080/api/models
```

Expected: the models list contains the freshly-trained checkpoints.

- [ ] **Step 5: HUMAN GATE — Julian plays a game**

Forward port 8080 and ask:

> "Server's up on http://localhost:8080 with the model we just trained. Please play a full game and confirm: the AI responds to your moves, wall placement and hover work, undo works, choosing P1 vs P2 works, the thinking progress bar streams, and the game ends with a winner. WebGPU browser preferred; it'll fall back to wasm-CPU (slower) otherwise."

**Do not proceed to Task 11 until Julian confirms.** If anything is broken, fix it and re-run from Step 4.

- [ ] **Step 6: Stop the server and record**

```bash
pkill -f run_play_server_web.py
```

The gate is now green. Everything after this is improvement.

---

## Task 11: PR #1 — branch and README

**Files:**
- Create: `/workspaces/lll_alpha_quoridor/README.md`

**Interfaces:**
- Produces: branch `improvements` in the new repo, with the first improvement commit.

- [ ] **Step 1: Branch**

```bash
cd /workspaces/lll_alpha_quoridor && git checkout -b improvements
```

- [ ] **Step 2: Write the README**

The repo's public front door. Cover, in this order: one-paragraph description (AlphaZero-style Quoridor with Python trainer, Rust engine, browser frontend); a quickstart that plays against the bundled `rust/fixtures/alphazero_B5W2_mv1.onnx` (the exact commands verified in Tasks 8–9); how to train (`src/train_v2.py experiments/<config>.yaml`, pointing at `experiments/B5W2/cucu-01.yaml` as the proven recipe); the repo layout table from this plan's File Structure section; and pointers into `docs/superpowers/specs/` and `docs/superpowers/plans/`.

Use commands verified in Tasks 7–9, not remembered ones. Every command in the README must have been run.

- [ ] **Step 3: Verify the quickstart from scratch**

Follow your own README top to bottom in a clean shell. Any step that does not work as written is a README bug.

- [ ] **Step 4: Commit**

```bash
cd /workspaces/lll_alpha_quoridor
git add README.md && git commit -q -m "vibe: add a README front door

The repo is public now and had no top-level README at all -- the only entry
points were scattered per-directory docs. Leads with playing a game, since
that is the fastest way to see whether any of this works."
```

---

## Task 12: PR #1 — Cargo workspace

`rust/Cargo.toml` is a package and `quoridor-wasm` is a path-dep subdirectory, so they build into separate target dirs with separate lockfiles. Project 2's Pages job builds both.

**Files:**
- Modify: `rust/Cargo.toml`, `rust/quoridor-wasm/Cargo.toml`

- [ ] **Step 1: Read both manifests first**

```bash
cd /workspaces/lll_alpha_quoridor
cat rust/Cargo.toml | head -40
cat rust/quoridor-wasm/Cargo.toml
```

`rust/Cargo.toml` opens with a `[profile.dev]` block and a `[package]` with several `[[bin]]` targets behind a `binary` feature. Preserve all of it.

- [ ] **Step 2: Add the workspace table**

At the top of `rust/Cargo.toml`, above `[profile.dev]`:

```toml
[workspace]
members = ["quoridor-wasm"]
```

A root package plus `members` makes this a root-package workspace: `quoridor-rs` stays buildable exactly as before, and `quoridor-wasm` joins it.

- [ ] **Step 3: Verify nothing regressed**

```bash
cd /workspaces/lll_alpha_quoridor/rust
cargo fmt --all -- --check
cargo clippy --all-targets --all-features
cargo build --release
RUST_BACKTRACE=1 PATH="/workspaces/deep_rabbit_hole/.venv/bin:$PATH" cargo test --features binary
ls target/ && test ! -d quoridor-wasm/target && echo "GOOD: single target dir"
```

Expected: everything passes and `quoridor-wasm/target/` no longer exists.

- [ ] **Step 4: Rebuild wasm and re-test the frontend**

The workspace changes where wasm-pack writes intermediates. Confirm the frontend still resolves the package.

```bash
cd /workspaces/lll_alpha_quoridor
wasm-pack build rust/quoridor-wasm --target web --release
npm --prefix frontend install && npm --prefix frontend run build && npm --prefix frontend run test
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/lll_alpha_quoridor
git add rust/Cargo.toml rust/quoridor-wasm/Cargo.toml rust/Cargo.lock
git commit -q -m "vibe: make rust a cargo workspace

quoridor-wasm was a path dependency outside the workspace, so the two
crates kept separate target dirs and lockfiles and shared no build cache.
Publishing to Pages will build both on every deploy."
```

---

## Task 13: PR #1 — prune historical material

Two separate commits so a load-bearing deletion reverts without taking the other with it.

**Files:**
- Delete: `coding-agents/` (13 files), and whichever of the reference/train scripts prove dead

- [ ] **Step 1: Delete coding-agents/**

Thirteen agent planning and results documents from past features. Their value is historical, and the history now lives in `deep_rabbit_hole`.

```bash
cd /workspaces/lll_alpha_quoridor
grep -rn "coding-agents" --include='*.py' --include='*.rs' --include='*.yml' \
  --include='*.yaml' --include='*.sh' --include='*.md' . | grep -v '^./coding-agents/' \
  | grep -v '^./docs/superpowers/' || echo "NO REFERENCES"
git rm -r -q coding-agents
git commit -q -m "vibe: drop the coding-agents planning archive

Thirteen point-in-time agent plans for features that shipped long ago. The
repo they document is deep_rabbit_hole, which still has them."
```

If the grep finds a live reference, stop and report it rather than deleting.

- [ ] **Step 2: Establish which scripts are actually dead**

Reference counts alone mislead here: these are CLI entry points, so zero imports does not mean unused. Check docs, configs, CI and READMEs too.

```bash
cd /workspaces/lll_alpha_quoridor
for f in mcts_game_reference step_trace_reference selfplay_real_model_reference \
         train_alphazero train_sb3 tune_selfplay upload_policy_db \
         train_policy_db_evaluator; do
  n=$(grep -rn "$f" src test rust experiments scripts .github docs README.md \
      2>/dev/null | grep -v "^src/$f.py:" | wc -l)
  printf '%-32s %s\n' "$f" "$n"
done
```

Counts measured during planning, against the pre-move tree:

| Script | Refs | Verdict |
|---|---|---|
| `action_reference.py` | 4 | **KEEP** — named in `rust-ci.yml`'s path filter, invoked by the Rust tests |
| `mcts_game_reference.py` | 6 | **KEEP** — Rust cross-language tests shell out to it |
| `step_trace_reference.py` | 6 | **KEEP** — same, and documented in `.github/prompts/` |
| `train_policy_db_evaluator.py` | 2 | **KEEP** — paired with `upload_policy_db.py`; investigate before touching |
| `selfplay_real_model_reference.py` | 1 | investigate — the one reference may be a docs mention |
| `train_alphazero.py` | 0 | candidate |
| `train_sb3.py` | 0 | candidate |
| `tune_selfplay.py` | 0 | candidate |
| `upload_policy_db.py` | 0 | candidate, but pairs with `train_policy_db_evaluator.py` |

- [ ] **Step 3: Delete only confirmed-dead candidates**

Zero references does **not** by itself mean dead — these are CLI entry points nobody imports. For each candidate, additionally confirm it is absent from `README.md`, `docs/superpowers/`, `experiments/*.yaml` and `.github/`, and that it is not the counterpart of a script being kept. `upload_policy_db.py` writes the artifact `train_policy_db_evaluator.py` reads, so either both go or neither does.

When in doubt, **keep it** and record the doubt in the results doc — a wrong deletion costs more than a stale file. Ask Julian if any candidate is a workflow they still use.

```bash
cd /workspaces/lll_alpha_quoridor
# One `git rm -q src/<name>.py` per script that survived the checks above.
# If none did, skip to Step 4 and record that nothing was pruned.
```

- [ ] **Step 4: Re-run the Python gate**

```bash
cd /workspaces/lll_alpha_quoridor
PYTHONPATH=src /workspaces/deep_rabbit_hole/.venv/bin/pytest test -v
PYTHONPATH=src /workspaces/deep_rabbit_hole/.venv/bin/python src/play.py -p greedy mcts -t 2
```

- [ ] **Step 5: Commit**

```bash
cd /workspaces/lll_alpha_quoridor
git commit -q -m "vibe: remove unreferenced entry-point scripts

Nothing in the tree, CI, configs or docs invokes these. The cross-language
reference scripts are kept -- the rust tests shell out to them."
```

If Step 3 found nothing safely deletable, skip this commit and say so.

---

## Task 14: PR #1 — fix the requirements files

Three errors, all pre-existing: two imports are missing and one heavyweight dependency is unused. CI passes today only because it never reaches the affected code.

**Files:**
- Modify: `requirements.txt`, `ci_requirements.txt`

- [ ] **Step 1: Confirm each finding**

```bash
cd /workspaces/lll_alpha_quoridor
echo "networkx used by:"; grep -rn "import networkx" src test --include='*.py'
echo "absl used by:";     grep -rn "from absl\|import absl" src test --include='*.py'
echo "tensorflow used by:"; grep -rn "import tensorflow\|from tensorflow" src test --include='*.py' || echo "  (nothing)"
echo "--- currently listed ---"
grep -nE 'networkx|absl|tensorflow' requirements.txt ci_requirements.txt || echo "  none of the three in either file"
```

Expected: `networkx` in `src/agents/alphazero/mcts_gexf.py`; `absl` in three `src/osaz/` modules; `tensorflow` nowhere; and of the three, only `tensorflow` currently listed.

- [ ] **Step 2: Prove the gap is real**

```bash
/workspaces/deep_rabbit_hole/.venv/bin/pip download --no-deps -d /tmp/pd networkx absl-py >/dev/null 2>&1
cd /workspaces/lll_alpha_quoridor
PYTHONPATH=src /workspaces/deep_rabbit_hole/.venv/bin/python -c "import agents.alphazero.mcts_gexf" \
  && echo "imports (networkx already present in this venv)" \
  || echo "CONFIRMED: import fails without networkx"
```

Either result confirms the listing gap — the venv installed `requirements.txt`, so if the import succeeds, `networkx` arrived as a transitive dependency and is therefore unpinned and unguaranteed.

- [ ] **Step 3: Edit both files**

In `requirements.txt`: add `networkx` and `absl-py` in alphabetical position; remove `tensorflow`.

In `ci_requirements.txt`: add `networkx` and `absl-py`; remove `tensorflow`. Leave the `--extra-index-url` CPU-wheel line and the `torch==2.9.1+cpu` pin untouched — CI depends on both, and the file's comment explains why (runner disk space).

- [ ] **Step 4: Verify a clean install still works**

Build a throwaway venv from the edited CI file — the fast one, and the one CI actually uses.

```bash
python3.12 -m venv /tmp/reqcheck && /tmp/reqcheck/bin/pip install -q --upgrade pip
/tmp/reqcheck/bin/pip install -q -r /workspaces/lll_alpha_quoridor/ci_requirements.txt
cd /workspaces/lll_alpha_quoridor
PYTHONPATH=src /tmp/reqcheck/bin/pytest test -q
PYTHONPATH=src /tmp/reqcheck/bin/python -c "import agents.alphazero.mcts_gexf; print('mcts_gexf ok')"
rm -rf /tmp/reqcheck
```

Expected: tests pass and the import succeeds. If dropping `tensorflow` breaks something, restore it and record why in the results doc.

- [ ] **Step 5: Commit**

```bash
cd /workspaces/lll_alpha_quoridor
git add requirements.txt ci_requirements.txt
git commit -q -m "vibe: fix three errors in the requirements files

networkx and absl-py are imported but were listed nowhere, so they only
ever arrived as unpinned transitive dependencies; tensorflow was listed in
both files and imported by nothing, at real cost to install time and CI
disk. CI never caught it because it does not reach that code."
```

- [ ] **Step 6: Open PR #1**

```bash
cd /workspaces/lll_alpha_quoridor
git push -u origin improvements
gh pr create --repo adamantivm/lll_alpha_quoridor --base main --head improvements \
  --title "Post-extraction improvements: README, cargo workspace, pruning, requirements" \
  --body "$(cat <<'BODY'
First round of improvements after the extraction landed and the verification
gate went green. One commit per change, reviewable independently.

- **README** — the repo had no front door; leads with playing a game.
- **Cargo workspace** — `quoridor-wasm` joins the workspace so the two crates
  share a target dir and lockfile. Project 2's Pages job builds both.
- **Pruning** — the `coding-agents/` planning archive and unreferenced entry
  points. The cross-language reference scripts are kept; the rust tests shell
  out to them.
- **Requirements** — `networkx` and `absl-py` were imported but unlisted;
  `tensorflow` was listed but unused.

Verification: the full gate from
`docs/superpowers/plans/2026-08-03-extract-quoridor-repo.md` was re-run after
the workspace and pruning commits.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```

- [ ] **Step 7: Confirm CI is green on the PR**

```bash
gh pr checks --repo adamantivm/lll_alpha_quoridor improvements --watch
```

---

## Task 15: Cleanup PR on deep_rabbit_hole

Only once the new repo is verified. Back in the **old** repo, on the existing branch.

**Files:**
- Delete: `deep_quoridor/`, `docs/superpowers/specs/`, `docs/superpowers/plans/`, `.github/workflows/{python-app,rust-ci}.yml`, `.github/prompts/`, `experiments/2026_05_23_jon_b9w10_performance/`
- Modify: `README.md`, `.devcontainer/devcontainer.json`, `.devcontainer/post-create.sh`
- Create: `docs/superpowers/results/2026-08-03-extract-quoridor-repo-results.md`

- [ ] **Step 1: HUMAN GATE — colleagues warned?**

Jon Binney, Alejandro Marcu, Diego Belfer and Nick Fragale authored 470 of the 937 commits. Ask:

> "Ready to open the PR removing `deep_quoridor/` from `deep_rabbit_hole`. Have you given Jon, Alejandro, Diego and Nick a heads-up?"

Wait for confirmation.

- [ ] **Step 2: Confirm the branch, again**

```bash
cd /workspaces/deep_rabbit_hole
git rev-parse --abbrev-ref HEAD   # must print: jac/extract-quoridor-repo
```

- [ ] **Step 3: Remove the moved content**

```bash
cd /workspaces/deep_rabbit_hole
git rm -r -q deep_quoridor docs/superpowers/specs docs/superpowers/plans \
  .github/workflows/python-app.yml .github/workflows/rust-ci.yml \
  .github/prompts experiments/2026_05_23_jon_b9w10_performance
```

- [ ] **Step 4: Point the README at the new home**

Add a short section to `README.md` recording that the Quoridor project moved to
`https://github.com/adamantivm/lll_alpha_quoridor`, that its history remains in this repo,
and that PRs should go to the new repo.

- [ ] **Step 5: Trim the devcontainer**

Per the spec, the container stays but loses the quoridor-only tooling. Remove the Rust
feature from `devcontainer.json` and the `wasm-pack`/`maturin` blocks from
`post-create.sh`. Keep the Python 3.12 base, node, GitHub CLI, CUDA and GPU passthrough —
the remaining ML projects benefit. Update the pip install to reference a requirements file
that still exists, or drop that block if none applies.

- [ ] **Step 6: Verify nothing dangling references the removed tree**

```bash
cd /workspaces/deep_rabbit_hole
grep -rn "deep_quoridor" --include='*.py' --include='*.yml' --include='*.yaml' \
  --include='*.sh' --include='*.json' --include='*.md' . 2>/dev/null \
  | grep -v '^./.git/' | grep -v '^./README.md' | grep -v '^./wandb/' \
  | grep -v '^./mlruns/' || echo "CLEAN"
```

Expected: `CLEAN`, or only the intentional README pointer. `.env`'s `PYTHONPATH` may mention it — that file is untracked-by-intent and out of scope.

- [ ] **Step 7: Commit**

```bash
cd /workspaces/deep_rabbit_hole
git add -A
git commit -q -m "vibe: move the quoridor project to its own repo

deep_quoridor now lives at adamantivm/lll_alpha_quoridor, where it can be
published as a static GitHub Pages site -- impossible while it shared a repo
with six unrelated projects. Full history stays here; the new repo starts
fresh with a pointer back."
```

- [ ] **Step 8: Write the results document**

Create `docs/superpowers/results/2026-08-03-extract-quoridor-repo-results.md` covering:
what moved and where; the six required changes with the two functional breaks called out
(`ai_report.py` repo-root arithmetic, the bench script paths); every gate check with its
actual result; what was pruned and what was deliberately kept; the three requirements
fixes; and — stated plainly — what remains unverified: the Rust `gpu` feature, wandb
logging, and full-scale training. If the GPU fallback was used, say so.

Per AGENTS.md this is the PR body source. Report failures and skipped steps honestly.

- [ ] **Step 9: Commit the results doc and open the PR**

```bash
cd /workspaces/deep_rabbit_hole
git add docs/superpowers/results
git commit -q -m "vibe: record the extraction results"
git push -u origin jac/extract-quoridor-repo
gh pr create --repo adamantivm/deep_rabbit_hole --base main --head jac/extract-quoridor-repo \
  --title "Move deep_quoridor to its own repository" \
  --body-file docs/superpowers/results/2026-08-03-extract-quoridor-repo-results.md
```

- [ ] **Step 10: Report back**

Give Julian the two PR URLs, the new repo URL, and the list of unverified items. Do **not**
merge either PR — colleagues review first.

---

## What this plan does not cover

Project 2, specified separately: static-ifying the frontend (build-time config and model
manifest, base-relative paths) and the GitHub Pages deploy. The spec's closing section
records the findings it will need — cross-origin isolation is not a blocker, the play
server is already static-shaped, and two absolute paths in `ai.worker.ts` need rebasing
onto `import.meta.env.BASE_URL`.
