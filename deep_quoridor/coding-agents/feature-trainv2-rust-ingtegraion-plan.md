# Plan: Integrate Rust Self-Play with Python Training

**TL;DR:** Unify the Rust self-play binary with the Python training pipeline by (1) switching Python to use the same npz replay format as Rust, (2) adding a continuous mode with model hot-reload to the Rust binary, and (3) adding a `--rust-selfplay` flag to the Python training entry point that spawns the Rust binary instead of Python self-play workers. ONNX export is auto-enabled when Rust self-play is selected. All field names in npz files are aligned between Rust and Python.

## Steps

### Commit 1: Switch Python self-play and trainer to use npz instead of pickle

1. In `deep_quoridor/src/agents/alphazero/alphazero.py`, modify `end_game_batch_and_save_replay_buffers` (around L498–L530):
   - Instead of pickling a list of dicts, stack the arrays and write an `.npz` file using `numpy.savez_compressed` with the same keys Rust uses: `input_arrays`, `policies`, `action_masks`, `values`, `players`
   - Change file extension from `.pkl` to `.npz`
   - Keep the same atomic write strategy (write to tmp dir, rename to ready dir)
   - Keep writing the companion `.yaml` first (same `GameInfo` schema)

2. In `deep_quoridor/src/v2/trainer.py`, update the replay reading logic:
   - Change glob from `*.pkl` to `*.npz` (around L107)
   - Change sequential rename from `game_{N:07d}.pkl` to `game_{N:07d}.npz`
   - Update the sampling function to load npz files via `numpy.load()` and index into stacked arrays, producing dicts with the keys expected by `compute_losses`: `input_array`, `mcts_policy`, `action_mask`, `value`, `player`

3. Verify that `compute_losses` in `deep_quoridor/src/agents/alphazero/nn_evaluator.py` does not need changes (it expects per-sample dicts — the trainer adapts the format).

4. **Verification run**: Run a short training using the existing `deep_quoridor/experiments/ci.yaml` config (or a variant with `finish_after: 2 models` and `benchmarks: []`) with Python self-play to confirm the npz format works end-to-end. Verify training completes, model versions increment, and losses are reasonable.

### Commit 2: Add `model_version` CLI parameter to Rust self-play

1. In `deep_quoridor/rust/src/bin/selfplay.rs`, add a `--model-version` CLI argument (integer, default 0).
2. Pass it through to `write_game_yaml()` instead of the hardcoded `0` at L174.
3. Run `cargo fmt`, build, and test.

### Commit 3: Add continuous mode with model hot-reload to Rust self-play

1. In `deep_quoridor/rust/src/bin/selfplay.rs`, add new CLI arguments:
   - `--continuous`: flag to enable continuous mode (plays games indefinitely)
   - `--latest-model-yaml <PATH>`: path to the `latest.yaml` file for model polling
   - `--shutdown-file <PATH>`: path to `.shutdown` file for graceful exit
   - When `--continuous` is set, `--model-path` becomes optional (loaded from `latest.yaml` instead)

2. Add a `LatestModel` YAML reader in a new module or within `selfplay_config.rs`:
   - Parse `filename` and `version` fields (matching Python's `LatestModel` schema)
   - Poll the file between game batches

3. Implement continuous loop in `main()`:
   - After each game (or small batch of games), check `latest.yaml` for a new version
   - If version changed: load new ONNX model, update agents, update `model_version` for replay metadata
   - Check for `.shutdown` file existence → break and exit gracefully
   - Use atomic write pattern: write npz to a `tmp/` subdirectory within `output-dir`, then rename to `output-dir`

4. Run `cargo fmt`, build, and test.

### Commit 4: Add `--rust-selfplay` flag to Python training entry point

1. In `deep_quoridor/src/v2/config.py`:
   - Add `rust_selfplay_binary: Optional[str]` field to `SelfPlayConfig` (default: `None`, meaning use Python self-play)
   - When set, this is the path to the pre-built Rust binary

2. In `deep_quoridor/src/train_v2.py`:
   - Add `--rust-selfplay` CLI argument (optional path to Rust binary, default: `rust/target/release/selfplay` relative to `deep_quoridor/`)
   - When `--rust-selfplay` is used:
     - Auto-set `save_onnx = True` in the training config (force ONNX export)
     - Instead of spawning Python `self_play` workers, spawn `num_workers` Rust self-play processes in continuous mode
     - Pass the correct args: `--continuous`, `--config <config.yaml>`, `--latest-model-yaml <path>`, `--output-dir <replay_buffers_ready>`, `--shutdown-file <path>`
   - Wire up graceful shutdown: the existing `ShutdownSignal` writes a `.shutdown` file, which the Rust binary watches

3. In `deep_quoridor/src/v2/trainer.py`:
   - No additional changes needed (already updated in Commit 1 to read npz)
   - The initial model save already respects `save_onnx` — verify it exports `model_0.onnx` correctly

### Commit 5: End-to-end test runs (both Python and Rust self-play)

Use the existing `deep_quoridor/experiments/ci.yaml` config as a base, with overrides for a quick run (e.g. `finish_after: 2 models`, `benchmarks: []`, `save_onnx: true`).

1. **Build the Rust binary**: `cd deep_quoridor/rust && cargo build --release --features binary --bin selfplay`

2. **Test run with Python self-play**:
   - `python src/train_v2.py ci.yaml -o training.finish_after="2 models" -o training.save_onnx=true -o benchmarks=[]`
   - Verify: training completes, npz files generated, models increment, losses are reasonable

3. **Test run with Rust self-play**:
   - `python src/train_v2.py ci.yaml --rust-selfplay -o training.finish_after="2 models" -o benchmarks=[]`
   - Verify:
     - Rust binary spawns and produces `.npz` + `.yaml` files in `replay_buffers_ready/`
     - Trainer picks up the files, moves them, trains successfully
     - Model versions increment, ONNX files are exported
     - Rust binary detects new model versions and reloads
     - Shutdown signal works (Rust processes exit cleanly)
     - Losses are reasonable for the short run

4. **Compare outputs**: Spot-check that npz files from both Python and Rust self-play have the same array keys, shapes, and dtypes.

## Verification
- `cargo fmt --check && cargo build --release --features binary && cargo test` for Rust changes
- `python -m pytest deep_quoridor/test/` for Python changes (if applicable tests exist)
- Two manual end-to-end runs (Python self-play + Rust self-play) using `ci.yaml` with minimal overrides
- Inspect npz files with `python -c "import numpy as np; d = np.load('file.npz'); print({k: d[k].shape for k in d})"` to verify format consistency

## Decisions
- **npz field names**: Aligned on Rust conventions (`input_arrays`, `policies`, `action_masks`, `values`, `players`) since they are more natural for stacked numpy arrays. Python trainer adapts to per-sample dict keys (`input_array`, `mcts_policy`, etc.) at read time.
- **Continuous mode over repeated spawning**: Rust binary gets a continuous loop with model polling, avoiding process spawn overhead and matching Python self-play behavior. This is its own commit.
- **Pre-built binary**: User must build the Rust binary before training. Default path is `rust/target/release/selfplay` relative to `deep_quoridor/`. Configurable via `--rust-selfplay <path>`.
- **Atomic writes in Rust**: Rust continuous mode uses tmp→rename pattern matching Python convention for race-condition safety.
- **Test config**: Reuse existing `deep_quoridor/experiments/ci.yaml` with CLI overrides rather than creating a new config file.

## Notes
- Per `deep_quoridor/agents.md`, this plan should be saved to `deep_quoridor/coding-agents/` before implementation begins.
- Each commit message should start with `"vibe: "`.
- Rust changes must pass `cargo fmt`, build, and tests.
- Use Resnet network format for testing everything, defer using MLP format for later by writing it in backlog.md
