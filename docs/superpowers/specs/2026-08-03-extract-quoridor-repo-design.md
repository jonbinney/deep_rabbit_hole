# Extract deep_quoridor into its own repository

**Date:** 2026-08-03
**Status:** Approved, ready for planning

## Goal

Move `deep_quoridor/` out of `adamantivm/deep_rabbit_hole` into a standalone repository,
`adamantivm/lll_alpha_quoridor`, without breaking anything. The new repo must train, test,
build, and play exactly as the current one does before any cleanup happens.

This is the first of two projects. The second — publishing the frontend as a static GitHub
Pages site with no server dependency — gets its own spec once this lands.

## Working conventions

Nothing lands directly on `main` in `deep_rabbit_hole`. All work — specs, plans,
devcontainer configs, code — goes onto a feature branch and reaches `main` through a pull
request, so Julian's colleagues can review and approve it. This spec and the devcontainer
are on `jac/extract-quoridor-repo`.

## Non-goals

- Static-ifying the frontend or deploying to Pages. That is project 2.
- Remote model hosting. The ONNX models are 400 KB – 1.3 MB and GitHub Pages allows 1 GB
  per site, so HuggingFace solves a problem this project does not have. Cut, not deferred.
- Preserving git history. A single initial commit is fine (decided explicitly; see
  "History" below).
- Rotating the leaked MLflow credential in `deep_rabbit_hole`'s `.env`. Flagged, out of scope.

## Findings that shaped the design

**`deep_quoridor/` is self-contained.** 251 tracked files, 3.0 MB, and zero source
references outside its own directory — verified by grepping `../..`, absolute paths, and
`deep_rabbit_hole` across `.py`, `.rs`, `.ts`, `.toml`, and `.sh`. The only outward-looking
value in the code is `wandb_dir: "wandbmodels"` (`src/agents/alphazero/alphazero.py:132`
and two peers), which is relative to the working directory. wandb project and entity are
config-driven.

**Both CI workflows are already scoped to `deep_quoridor/**`** via `paths:` filters, so
they move as-is with their path prefixes stripped.

**All 19 files under `docs/superpowers/` are quoridor work.** A naive `deep_quoridor/` path
filter would have left them behind. Copying rather than filtering history handles this for
free.

**Both requirements files are wrong in three ways.** `networkx` (used by
`src/agents/alphazero/mcts_gexf.py`) and `absl-py` (used by `src/osaz/*`) are imported but
listed in neither file, while `tensorflow` is listed in both and imported nowhere. CI passes
today because it never exercises those code paths, so these are pre-existing latent bugs
rather than move regressions — they belong in PR #1's requirements cleanup, not the initial
commit.

**The dev environment is currently broken.** `.venv/pyvenv.cfg` points at
`/home/julian/aaae/...` — a different machine — and was built `--without-pip`; its `python`
symlink now resolves to system 3.14.4 rather than the 3.12 it was created with, and
`import torch` fails. The devcontainer has node but no `cargo`, `rustc`, `wasm-pack`,
`maturin`, or `gh`, and no GPU passthrough. Nothing is verifiable today. Building a working
toolchain is therefore a prerequisite of this project, not a footnote.

## History

Not preserved. The new repo starts with a single initial commit.

`deep_rabbit_hole` retains the full 937-commit, 6-author history (Jon Binney 392,
Alejandro Marcu 245, Julian Cerruti 225, Diego Belfer 52, Jonathan Binney 20, Nick
Fragale 3), and the pointer README added by the cleanup PR preserves the trail back to it.

A side effect of copying rather than filtering: the root `.env`, which contains
`MLFLOW_TRACKING_PASSWORD`, cannot leak into the new repository.

## Deliverables

Four, in order. Each is a gate for the next.

1. **Initial commit** in `adamantivm/lll_alpha_quoridor` — today's code, working.
2. **Verification gate** — everything passes before any improvement is written.
3. **PR #1** on the new repo — README, Cargo workspace, pruning, requirements cleanup.
4. **Cleanup PR** on `deep_rabbit_hole` — remove the moved content, leave a pointer.

**Assumed unless corrected:** the repo is public from the first push. The source is already
public in `deep_rabbit_hole`, so nothing new is exposed, and free Pages requires it. The
owner may move later, so nothing hardcodes `adamantivm`.

## Deliverable 1: the initial commit

### Target tree

~275 files, ~3.1 MB.

```
.github/workflows/{python-app,rust-ci}.yml   from root, path prefixes stripped
.github/prompts/selfplay_rust_python_debugging.md   from root, quoridor-specific
.devcontainer/                                extended: Rust, wasm-pack, maturin, gh, GPU
.gitignore                                    derived subset
AGENTS.md  CLAUDE.md  pytest.ini  ruff.toml   from root
requirements.txt  ci_requirements.txt         from deep_quoridor/
MODEL_SAVE_OPTIONS.md
src/           130 files   training, agents, env, v2/, play server
rust/           60 files   quoridor-rs + quoridor-wasm/ + fixtures/*.onnx
frontend/       24 files   Svelte SPA
test/           12 files
experiments/     9 files   plus the B9W10 config from the root experiments/
scripts/         1 file    bench_rust_selfplay.sh
coding-agents/  13 files
docs/superpowers/{specs,plans}/   19 files
```

The tracked Rust fixtures (`rust/fixtures/alphazero_B5W2_mv1.{onnx,pt}`, 412 KB each) come
along. They are the only model in git and the only thing the new repo can demo itself with
before a training run produces one.

`.devcontainer/` is committed to `deep_rabbit_hole` on `jac/extract-quoridor-repo` so it is
reviewable there, and copied into the new repo. The cleanup PR keeps it in
`deep_rabbit_hole` with the quoridor-specific tooling (Rust, `wasm-pack`, `maturin`)
stripped out — GPU passthrough, Python 3.12, node and `gh` stay useful for the other ML
projects that remain.

### Excluded

`.env` (MLflow credentials), the root `README.md`, the root `scripts/` and the non-quoridor
entries under the root `experiments/`, everything under `external/ datasets/ models/
mlflow/ wandb*/`, and the six unrelated project directories.

### Required changes

"Verbatim" cannot be literal. These six are the complete set of edits in the initial commit:

1. **CI workflow paths.** Both workflows hardcode `deep_quoridor/` in `paths:` filters,
   `cache-dependency-path`, `PYTHONPATH`, and `working-directory`. Strip the prefix. No
   logic changes.
2. **`.gitignore`.** The root file is 3743 bytes covering seven projects. Derive a subset,
   keeping the fix from commit `5b0cd1b` that stops the Python template's `lib/` pattern
   from swallowing `frontend/src/lib/`.
3. **`CLAUDE.md`.** Its single line references `@deep_quoridor/agents.md`, which does not
   exist — already broken today. Point it at `AGENTS.md`.
4. **Devcontainer.** Rust toolchain, `wasm-pack`, `maturin`, `gh`, Python 3.12 with a real
   venv, and GPU passthrough (see below). Without this there is no way to run the gate.
5. **Stale `deep_quoridor/` prefixes** in `AGENTS.md` and `frontend/README.md`.
6. **Hardcoded `deep_quoridor/` paths in code.** Two of these are functional breaks, not
   prose:
   - `src/v2/ai_report.py` — line 450 computes `repo_root` by walking four parents
     (`v2 → src → deep_quoridor → repo root`), which lands one level too high in the new
     layout, and lines 355–358 and 413 then join a `deep_quoridor/` prefix that no longer
     exists. Fix: three parents, and drop the prefix from all five paths.
   - `scripts/bench_rust_selfplay.sh` — `BIN="deep_quoridor/rust/target/release/selfplay"`
     (line 10) and `cd deep_quoridor/rust` (line 13), both resolved from the repo root.

   Cosmetic but user-facing, fixed in the same commit: the build hint in
   `src/v2/config.py:412`, the `.expect("rust crate should live under deep_quoridor/")`
   messages in `rust/src/python_consistency.rs` (the surrounding logic is relative and
   survives the move), and docstrings in `src/ai_report_cli.py`,
   `src/run_benchmarks_v2.py`, `src/v2/wandb_metrics.py`, `src/v2/play_server_web/README.md`,
   `rust/README.md`, and `rust/RUST_FOR_PYTHONISTAS.md`.

   Left alone: `src/osaz/logs/*.md`, which are historical tracebacks containing absolute
   paths from Julian's old machine. They are a record of what happened, not instructions.

### Devcontainer

Extend `.devcontainer/devcontainer.json` (currently just Ubuntu base + node + claude-code):

- Rust toolchain with `rustfmt` and `clippy`, plus `wasm-pack`, `maturin`, and `gh`.
- Python 3.12 with a working venv, installing the **full** `requirements.txt` — not
  `ci_requirements.txt`, which pins `torch==2.9.1+cpu` and so cannot support GPU training.
- GPU passthrough: `"runArgs": ["--gpus", "all"]` plus the `nvidia-cuda` feature. Torch's
  CUDA wheels ship their own runtime libs, so only the host driver needs exposing.

**Host prerequisite, unverifiable from inside the container:** NVIDIA Container Toolkit must
be installed on the GPU host. Julian confirms.

**`onnxruntime-gpu` is the fiddly half** — it wants cuDNN 9 on `LD_LIBRARY_PATH`, usually
satisfiable by pointing at torch's bundled `nvidia-*` packages. Time-box this. If GPU
passthrough or `onnxruntime-gpu` resists, fall back to Julian running the training portion
of the gate on the host against a clone. The gate is satisfied either way.

## Deliverable 2: the verification gate

Nothing in PR #1 gets written until this is green. No baseline is established against the
old repo — green is green. If a pre-existing failure surfaces, it gets fixed here.

### Automated

| Stack | Checks |
|---|---|
| Python | `pytest test`; `play.py -p greedy mcts -t 2`; `train_v2.py experiments/ci.yaml` |
| Rust | `cargo fmt --check`; `clippy --all-targets --all-features`; `build`; `test --features binary`; `build --release` |
| Extension | `maturin build --release`, then `import quoridor_rs` |
| WASM | `wasm-pack build rust/quoridor-wasm --target web --release` |
| Frontend | `npm ci`; `npm run build`; `npm run test` (vitest) |
| Play server | launch against a fixture play dir; assert `/`, `/api/config`, `/api/models`, `/models/*.onnx`, `/ort/*.wasm` return 200 with the expected shape |
| CI | both workflows green on the new repo |

### The end-to-end chain

The training check and the play check are one continuous chain, which is also the chain
project 2 will publish:

**train on GPU → export ONNX → serve that run directory → play the freshly-trained model
in a browser.**

Run `experiments/B5W2/cucu-01.yaml` — the proven B5W2 recipe, whose own comment records it
beating the optimal-strategy policy_db as P2 — with a short `finish_after` and
`save_onnx: true`. It writes `runs/<id>/models/checkpoints/*.onnx`, which is exactly the
input `run_play_server_web.py` takes.

Success signals from the short run: self-play games produced, loss trending down across
several training steps, benchmark rows appearing (`dumb_score`, tournaments vs.
random/greedy), a checkpoint written, ONNX exported. The claim is "the machinery turns,"
not "the model is good."

### Human check

Julian plays a full game in a real browser against the model just trained (falling back to
the bundled fixture if the training leg is deferred to the host): AI responds to moves,
wall placement and hover work, undo works, the P1/P2 choice works, the progress bar
streams, and the game reaches a winner. No automated check covers the WebGPU inference path
or clicking.

### Explicitly not covered

Stated so it does not read as passing:

- **Rust `gpu` feature** (`Cargo.toml:65`, `ort/cuda + ort/load-dynamic`). rust-ci
  deliberately skips it. Out of the gate even with a GPU present.
- **wandb logging.** Needs credentials. Unchanged code, config-driven, unverified.
- **Full-scale training.** Only the short time-boxed run, not a multi-hour one.

## Deliverable 3: PR #1 on the new repo

One commit per change, per `AGENTS.md`. Splitting the deletions keeps each reviewable on
its own, so a load-bearing one reverts cleanly without taking the README or workspace with it.

1. **README** — a real front door: what the project is, quickstart to play against the
   bundled fixture, how to train, repo layout, pointers into `docs/superpowers/`.
2. **Cargo workspace** — `rust/` becomes a workspace with `quoridor-rs` and `quoridor-wasm`
   as members. Today `quoridor-wasm` is a path-dep subdirectory, so they build into
   separate target dirs with separate lockfiles; unifying them pays off directly when
   project 2's Pages job has to build both.
3. **Prune `coding-agents/`** — 13 historical agent planning documents.
4. **Prune dead code** — the `*_reference.py` scripts (`mcts_game_reference.py`,
   `step_trace_reference.py`, `selfplay_real_model_reference.py`) and unused `train_*.py`
   variants. Each checked for live references first.
5. **Fix the requirements files** — add the missing `networkx` and `absl-py`, drop the
   unused `tensorflow`, and review the remainder for anything else unused.

## Deliverable 4: cleanup PR on deep_rabbit_hole

Only after the new repo is verified green.

Remove `deep_quoridor/`, `docs/superpowers/`, `.github/workflows/{python-app,rust-ci}.yml`,
and `experiments/2026_05_23_jon_b9w10_performance/`. Rewrite the root `README.md` with a
pointer to `adamantivm/lll_alpha_quoridor`. Strip the quoridor-specific tooling from
`.devcontainer/` but leave the container itself in place.

Julian gives Jon, Alejandro, Diego, and Nick a heads-up before merging — between them they
authored 470 of the 937 commits.

## Risks

| Risk | Mitigation |
|---|---|
| GPU passthrough or `onnxruntime-gpu` resists setup | Time-box; fall back to Julian running the training leg on the host |
| A pre-existing failure surfaces and looks like a move regression | No baseline by decision; check the old repo only for that specific failure |
| A pruned file turns out to be load-bearing | Separate commits per deletion category; check references before deleting |
| Contributors open PRs against the old copy | Cleanup PR removes it and leaves a pointer; heads-up sent first |

## Notes for project 2

Recorded now so the findings are not re-derived:

- **Cross-origin isolation is not a blocker.** The server sets COOP/COEP for
  `SharedArrayBuffer`, and GitHub Pages cannot set response headers — but `quoridor-wasm`
  has no threading (no rayon, no atomics) and onnxruntime-web auto-degrades to
  `numThreads = 1` when `crossOriginIsolated` is false. Pages works with no service-worker
  shim; the only cost is a slower CPU fallback on browsers without WebGPU.
- **The play server is already static-shaped.** `/api/config` derives a JSON view from
  `config.yaml`, `/api/models` is a directory listing, and `/models/*.onnx` and
  `/ort/*.wasm` are plain static mounts. Both endpoints become build-time JSON.
- **Two absolute paths need rebasing** onto `import.meta.env.BASE_URL`:
  `ai.worker.ts:8` (`ort.env.wasm.wasmPaths = "/ort/"`) and `ai.worker.ts:33`
  (`/models/${model}`). A project Pages site serves at `/lll_alpha_quoridor/`.
- **The Pages base path lives in one place**, since the repo owner may change.
