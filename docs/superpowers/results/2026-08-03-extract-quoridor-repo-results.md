# Results: extracting deep_quoridor into its own repository

## What moved and where

`deep_quoridor/` has moved out of this monorepo into a new standalone public
repository: https://github.com/adamantivm/lll_alpha_quoridor. The motivation
was that the project needs to be publishable as a static GitHub Pages site,
which was impossible while it shared a repository with six unrelated
projects (annotation_utils, camera_control, deep_water_level, datasets,
object_tracker_0, pytorch_exercises, and others).

The new repo was built by taking a `git archive` snapshot of the tracked
`deep_quoridor/` subtree (not a filesystem copy, to avoid carrying over
untracked artifacts), applying only the edits the move itself required, and
committing that as the new repo's first commit. Full project history up to
the move remains here, in this repo; the new repo starts fresh with a
pointer back to this history. This repo keeps a pointer forward to the new
repo (see the new README section).

PR #1 on the new repo (branch `improvements` against `main`) carries
additional post-move improvements: a public-facing README, pruning of
dead/internal-only files, and a requirements fix. Both that PR and this one
are open for review; neither has been merged.

## The six required changes, with the two predicted functional breaks

Six changes were required to make the extracted tree work standalone:

1. `rust/src/*.rs` path-discovery logic (`CARGO_MANIFEST_DIR.parent()/src`)
   had to keep resolving correctly at the new repo root. It did, unchanged —
   confirmed by all five cross-language Rust tests passing against the
   Python reference scripts.
2. Six `.expect()` path strings and two `assert!` messages in
   `rust/src/python_consistency.rs` referencing `deep_quoridor/rust/fixtures/`
   were updated to the new, shallower paths.
3. **`src/v2/ai_report.py` (functional break, predicted).** Its repo-root
   arithmetic used one `.parent()` call too many, and four hardcoded source
   paths carried a stale `deep_quoridor/` prefix left over from the
   monorepo layout. Both were wrong in the new repo — a fresh clone would
   have produced a report citing files that don't exist. Fixed test-first:
   a regression test, `test/test_ai_report_paths.py`, was written RED first
   (failing against the broken code, citing a nonexistent
   `.../deep_quoridor/src/v2/config.py`), then made GREEN by fixing the
   arithmetic and stripping the stale prefix from all four source entries.
4. **`scripts/bench_rust_selfplay.sh` (functional break, predicted).** The
   binary path and the `cd` target both assumed the old monorepo-relative
   layout. Fixed to the new repo-relative paths. Note for anyone re-running
   this script: it has always required two positional arguments
   (`CONFIG`, `MODEL`) via a `${1:?usage}` guard — this is not new, and
   running it bare (as an earlier draft of the plan assumed) fails at the
   usage check before ever reaching the fixed paths. Run as e.g.
   `bash scripts/bench_rust_selfplay.sh experiments/B5W2/cucu-01.yaml rust/fixtures/alphazero_B5W2_mv1.onnx 15`.
5. `.devcontainer/post-create.sh`'s install path for `requirements.txt`
   (not in the original file list; found while auditing the move).
6. `.github/prompts/selfplay_rust_python_debugging.md`'s references to the
   old paths (also not in the original file list; found the same way).

## A third functional break the plan did not predict

`test/os_pz_conversion_test.py` imported
`from deep_quoridor.test.quoridor_test import parse_board`. In the monorepo
this resolved because `deep_quoridor` behaved as an implicit Python
namespace package rooted at the monorepo's own repo root; that root does not
exist in the extracted repo, so the import fails there.

This was initially misdiagnosed twice — first by whoever implemented the
move, then again on review — as a pre-existing, out-of-scope issue. It is
not: pytest's collection errors abort the entire test run, so this alone
would have failed CI on the new repo. It was proven move-induced by running
pytest collection the same way CI does it in both places:

- Monorepo, CI-style invocation (`cd deep_rabbit_hole`, with
  `deep_quoridor/src` on `PYTHONPATH`, `pytest deep_quoridor/test
  --collect-only -q`): 101 collected, no error.
- Extracted repo, same style: 100 collected, 1 error, "Interrupted: 1 error
  during collection."

Fixed by changing the import to `from quoridor_test import parse_board`
(`test/` has no `__init__.py`, so pytest's default "prepend" import mode
puts `test/` itself on `sys.path`). Verified green under both the plan's
local invocation and CI's.

With this fix, the full suite in the new repo passed at 103 tests (the
monorepo's 101, plus the two new `ai_report` tests).

## Three plan premises that proved false

The plan that guided this extraction made three factual claims about the
old repo that turned out to be wrong. All three were caught during
execution, not before:

1. **The devcontainer's `.venv` is not a dead artifact.** The plan assumed
   it was leftover from a different machine (its `pyvenv.cfg` names a
   `/home/julian/...` path) and could be freely rebuilt. It is not dead: the
   devcontainer workspace is bind-mounted from the host, so that path is
   *this* machine's host path, seen through the mount, and the venv is the
   user's live, dependency-complete, GPU-capable host virtualenv. The
   original provisioning script's `rm -rf` on a detected stale `.venv`
   would have destroyed it. This was caught and fixed before it could fire:
   the script no longer deletes `.venv` under any condition, only creates
   it if missing, and drives all installs through
   `.venv/bin/python -m pip`/`-m pytest` rather than the venv's console
   scripts (whose shebangs are host-absolute and unrunnable inside the
   container).
2. **The Rust Cargo workspace change was a no-op.** The plan called for
   adding a `[workspace]` table to `rust/Cargo.toml` to give the `wasm`
   crate its own target directory and lockfile. `rust/Cargo.toml` has
   carried a `[workspace]` table since monorepo commit `05d384c2`, and it
   already includes `default-members = ["."]` — a refinement the plan's
   prescribed version lacked, and a better one: it keeps a bare `cargo
   build`/`cargo test` scoped to the native crate, since the wasm member
   targets `wasm32` and cannot compile for a native host at all. Applying
   the plan's edit verbatim would have produced a duplicate `[workspace]`
   table (a hard Cargo error) or, if merged by hand, would have dropped
   `default-members` and broken plain `cargo build`. No change was made;
   the existing state was verified already correct.
3. **`tensorflow` is not an unused dependency.** The plan listed it as dead
   weight to prune from `requirements.txt`/`ci_requirements.txt`. Removing
   it broke test collection in a clean virtualenv:
   `open_spiel.python.algorithms.alpha_zero.model` does
   `import tensorflow.compat.v1 as tf` internally, and that path is reached
   from `src/agents/alphazero_os.py`, imported by
   `test/os_pz_conversion_test.py`. It is a real, if indirect, dependency
   and was kept in both requirements files.

## Gate checks and their actual results

- **pytest**: 103 passed (101 pre-existing plus the 2 new `ai_report` tests),
  in both the monorepo-style and the extracted-repo-style invocation.
- **Sanity games** (`play.py`, scripted): 2/2 completed with a result table.
- **CI-scale training** (`experiments/ci.yaml`): completed cleanly, ~2.5
  minutes, clean shutdown message and full process drain. 222 `.pt`
  checkpoints. Loss fell from ~5.1 to ~1.7-2.2. Tournament, AgentEvolution
  and dumb-score benchmarks all emitted metrics. wandb ran mocked
  (`MockWandb`) — no credentials were available or used. Device is not
  confirmed at this scale; `ci.yaml` is CPU-scale and GPU use was confirmed
  separately below.
- **Rust/WASM/extension gate**: `cargo fmt` clean; `cargo clippy` 0 errors
  (warnings only, not required to be zero); `cargo test --features binary`:
  182/182 passed, including all five cross-language tests
  (`test_action_encoding_matches_python`,
  `test_initial_action_mask_matches_python`,
  `test_mcts_game_trace_matches_python`,
  `test_real_model_selfplay_trace_and_npz_matches_python`,
  `test_step_trace_matches_python`) — proof the `CARGO_MANIFEST_DIR`-based
  path discovery survived the move. Release build succeeded; the maturin
  wheel built and `import quoridor_rs` succeeded; `wasm-pack build` produced
  a working `pkg/` (`.wasm`, `.js`, `package.json`, types).
  A real environment gap was found and fixed along the way: `pkg-config`
  and `libssl-dev` were missing from the devcontainer but required to build
  `openssl-sys`, pulled in transitively via `ort -> ureq -> native-tls` when
  building the Rust crate's `binary` feature. Added to the provisioning
  script.
- **Frontend/play-server gate**: `vitest` 4 files / 10 tests passed. SPA
  build produced `dist/` plus `dist/ort/` (26 files, including the ONNX
  Runtime WASM/threaded worker files, no filename substitution needed). All
  five routes served 200 with correct bodies: `/`, `/api/config`,
  `/api/models`, a model file under `/models/`, and the ORT wasm asset.
- **Full-scale training gate** (the real end-to-end chain): run
  `runs/gate-20260804-1733`. Clean shutdown, zero tracebacks, zero OOM. 130
  `.pt` and 130 `.onnx` checkpoints — this is the run that proves ONNX
  export works (the CI-scale run above does not export ONNX at all).
  Loss fell from 5.09 to 2.20-2.26, clearly downward and plateauing late.
  Benchmarks show the model actually learning: win rate against
  random/greedy opponents climbed from 0% at checkpoint 0 to 40-47.5% by
  checkpoints 115-128. GPU use was confirmed directly with `nvidia-smi`
  (5-6 resident `.venv/bin/python` processes reported as CUDA compute
  apps, 949-1537 MiB of 4096 MiB used on the RTX 3050 Ti) rather than
  inferred from `torch.cuda.is_available()`. **No fallback to a lower
  `self_play.num_processes` was needed** — training ran at
  `num_processes: 4` throughout, using 1537/4096 MiB at peak. The training
  log itself never prints a device line in any of these runs; every device
  claim above is externally verified via `nvidia-smi`, not read out of
  application logs.
- **Browser gate**: a full game was played in-browser against
  `model_129.onnx` from the full-scale run above, with AI responses, wall
  placement with hover preview, undo, player-1/player-2 selection, a
  streaming "thinking" progress indicator, and a clean game-over state with
  a winner. Confirmed working directly by playing it, not just by serving
  the routes.
- **CI on the new repo**: both GitHub Actions workflows (Python application,
  Rust CI) ran green via `workflow_dispatch` on `main` before any PR
  existed, and green again via the `improvements` branch's pull request —
  the latter being the first real exercise of the workflows' `paths:`
  filters, which only evaluate against an actual base commit to diff
  against.

## What was pruned, and what was deliberately kept

Pruned from the new repo (all on its `improvements` branch, not yet merged):
- `coding-agents/` (13 internal planning/results files) — only doc mentions
  referenced it, no live code path.
- `src/train_alphazero.py`, `src/train_sb3.py`, `src/tune_selfplay.py` —
  zero references anywhere in the tree, CI, docs, experiments, or configs.

Deliberately kept, with the reasoning recorded:
- `src/selfplay_real_model_reference.py` — the plan flagged it as possibly
  a docs-only mention, but it is a live dependency of the Rust test
  `test_real_model_selfplay_trace_and_npz_matches_python`, which shells out
  to it. Deleting it would have broken `cargo test`.
- `src/upload_policy_db.py` — pairs with
  `src/train_policy_db_evaluator.py` (which reads the wandb artifact this
  script writes). Per the "either both go or neither does" rule, it stays.
- `tensorflow` in both requirements files — see above; it is a real
  transitive dependency, not dead weight.
- The Cargo workspace configuration — already correct; see above.

## What remains unverified

- **The Rust `gpu` feature** was never exercised. All Rust testing above ran
  under the default/`binary` feature set on CPU-only Rust code paths; no
  gate in this plan built or ran the crate with CUDA-backed inference from
  Rust.
- **wandb logging with real credentials.** Every training run in this plan,
  at every scale, ran with `MockWandb` — no `wandb login` was performed and
  no run was ever pushed to a real wandb project. Whether metrics logging
  against the real wandb backend works end-to-end is unverified.
- **Full-scale training at production duration/scale beyond the single gate
  run described above.** The gate run (130 checkpoints, ~seen through
  checkpoint 129) demonstrated a real, GPU-backed, ONNX-exporting, clearly
  learning training run, but it was not carried to whatever scale a full
  production run would use.

The GPU fallback path (dropping `self_play.num_processes` to reduce VRAM
pressure) was never needed at any point: every GPU run in this plan
completed at `num_processes: 4` without CUDA OOM, using at most 1537 of
4096 MiB.

## Requirements changes

`networkx` and `absl-py` were added as explicit direct dependencies to both
`requirements.txt` and `ci_requirements.txt` (alphabetized) — both were
already in effect via transitive resolution but were undeclared. `tensorflow`
was reconsidered for removal and reinstated in both files, as described
above. `ci_requirements.txt`'s `--extra-index-url` and pinned
`torch==2.9.1+cpu` were left untouched. A clean virtualenv built from the
edited `ci_requirements.txt` was verified to pass the full test suite (103
passed) and produce a valid `mcts.gexf` output.

## Changes on this side (deep_rabbit_hole)

Beyond removing `deep_quoridor/`, this repo's `.github/workflows/{python-app,rust-ci}.yml`,
`.github/prompts/`, `docs/superpowers/specs/`, `docs/superpowers/plans/`, and
`experiments/2026_05_23_jon_b9w10_performance/` were removed as part of the
same move (all superseded by, or specific to, the extracted project).

The devcontainer was trimmed rather than removed, since it still serves this
repo's other ML projects: the Rust feature and its wasm-pack/pkg-config/
libssl-dev provisioning were dropped (nothing left in this repo needs them),
while the Python 3.12 base, Node, GitHub CLI, CUDA feature and GPU
passthrough (`--gpus all`) were kept. The provisioning script's earlier fix
— never deleting the live host `.venv`, only creating it if absent, and
driving pip through `.venv/bin/python -m pip` rather than the venv's console
scripts — was preserved as-is. The script's final `pip install -r
deep_quoridor/requirements.txt` step was dropped rather than repointed: no
single requirements file applies to this repo any more; each remaining
project (`object_tracker_0/`, `annotation_utils/`, `deep_water_level/`,
`camera_control/`, `pytorch_exercises/`) carries its own, and none is a
sensible default for a shared devcontainer venv.

A handful of now-dangling references to the removed tree were also found and
fixed, beyond the plan's own file list: this repo's root `CLAUDE.md`
consisted solely of an import of `deep_quoridor/agents.md`, which no longer
exists, and is now empty; `AGENTS.md` carried a Rust-specific rule scoped to
`deep_quoridor/rust`, now removed; and `.vscode/settings.json` pointed
`rust-analyzer.linkedProjects` at `deep_quoridor/rust/Cargo.toml`, now
removed. A final grep for `deep_quoridor` across tracked source, config and
doc files turned up only the intentional pointer in this repo's `README.md`.

Two nested `.gitignore` files inside `deep_quoridor/` (`frontend/.gitignore`,
covering `node_modules` and `dist`; `rust/quoridor-wasm/.gitignore`, covering
`/pkg` and `/target`) were removed along with the rest of the tracked tree.
Since those rules no longer existed, the untracked build artifacts they had
been excluding (installed `node_modules`, build output, Rust target
directories, old training run and wandb directories under `deep_quoridor/`)
became visible as ordinary untracked files. None of this was ever tracked by
git; it was deleted from the working tree directly, rather than staged, to
avoid it being swept into a commit by an incautious `git add -A`.

`.devcontainer/devcontainer-lock.json` carries an unrelated, pre-existing
uncommitted modification from before this work began. It was left alone and
is not part of this change.
