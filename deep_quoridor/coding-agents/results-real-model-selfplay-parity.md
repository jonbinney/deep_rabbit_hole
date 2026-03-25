# Real-Model Selfplay Parity: Implementation Results

I'm using AGENTS.md

## Summary
Implemented a cross-language real-model parity harness that:
- Runs Python selfplay trace generation using a real `.pt` fixture model.
- Runs Rust selfplay trace generation using a real `.onnx` fixture model.
- Writes one replay `.npz` on both sides.
- Compares trace snapshots and (when trace remains aligned) compares NPZ contents.

Cleanup pass completed:
- Removed temporary checkpoint metadata normalization from Python reference runner.
- Removed deterministic override of Python random tie sampling.
- Kept only passive instrumentation for trace capture/logging.

Parity difference found and fixed:
- Rust temperature=0 tie handling was first-max deterministic.
- Python temperature=0 tie handling samples among max-visit ties.
- Rust was updated to sample among max-visit ties to match Python behavior.

Deterministic parity mode added (non-default):
- Toggle via `DEEP_QUORIDOR_PARITY_DETERMINISTIC_TIES=1` in Rust parity test runs.
- Mode is threaded to both sides:
   - Rust uses deterministic tie-break (first max-visit child) in parity harness.
   - Python parity runner enables deterministic tie-break in AlphaZero action selection.
- Default behavior remains unchanged (stochastic tie sampling among max-visit ties).

## Implemented
1. Added Python runner:
   - File: `deep_quoridor/src/selfplay_real_model_reference.py`
   - Uses real `AlphaZeroAgent` + MCTS path and real `agent.get_action_batch([(0, observation)])`.
   - Captures root outputs by wrapping `agent.mcts.search_batch` for trace logging only.
   - Emits trace format: `CFG/G/P/W/C/M/T/RM/RT/V/Q/A`.
   - Writes one replay game to `ready/*.npz` + `ready/*.yaml`.
   - Does not patch model checkpoint metadata.
   - Does not patch random tie-break behavior.

2. Extended Rust parity harness:
   - File: `deep_quoridor/rust/src/python_consistency.rs`
   - Added fixture path resolution:
     - `DEEP_QUORIDOR_PT_MODEL` (default fixtures `.pt`)
     - `DEEP_QUORIDOR_ONNX_MODEL` (default fixtures `.onnx`)
   - Added Python invocation for real-model runner.
   - Added Rust real-model trace generation + replay writing.
   - Added tolerant float comparisons for real-model traces.
   - Added NPZ load and compare helpers (shape + contents with tolerances).
   - Added test: `test_real_model_selfplay_trace_and_npz_matches_python`.

3. Added plan artifact:
   - File: `deep_quoridor/coding-agents/real-model-selfplay-parity-plan.md`

## Latest Changes
Production-path rotation alignment was implemented in Rust:
- `rust/src/agents/alphazero/evaluator.rs`
   - On evaluator input, player 1 states are rotated to current-player-downward orientation.
   - The evaluated priors are remapped back to original action-index space before returning.
- `rust/src/game_runner.rs`
   - Action selection now runs in original orientation for both players.
   - Replay storage remains current-player-downward by rotating player 1 replay state/policy/mask.
- `rust/src/python_consistency.rs`
   - Real-model parity trace generation mirrors the production behavior above.
   - Deterministic tie logic in the mock-trace test path was preserved to match Python reference behavior.

NPZ compatibility fix:
- `action_masks.npy` loading now accepts `float32`, `bool`, `int8` (`|i1`), and `uint8`.

Reviewer follow-up fixes:
- Rotation remapping in the parity harness now reuses `create_rotation_mapping` + `remap_policy` from `rotation.rs` instead of duplicating local mapping code.
- Board-size-invariant rotation mappings are cached/reused in hot paths:
   - `OnnxEvaluator` caches rotated-to-original policy mappings by board size.
   - `game_runner` computes original-to-rotated mapping once per game.
   - `python_consistency` computes original-to-rotated mapping once per parity trace.
- `apply_temperature_and_sample_with_mode` now asserts non-empty, equal-length inputs with explicit messages.
- The real-model parity test is now deterministic by default in-code, so `cargo test --all-features` no longer depends on `DEEP_QUORIDOR_PARITY_DETERMINISTIC_TIES`.

Production-path observer refactor:
- `rust/src/game_runner.rs`
   - Added an optional observer hook so parity tests can capture per-step state snapshots and chosen actions from the production self-play loop.
- `rust/src/agents/mod.rs`
   - Added a small `ActionSelectionTrace` payload so agents can expose read-only metadata from the last production move selection.
- `rust/src/agents/alphazero/agent.rs`
   - Records the MCTS root value from the production action-selection path for parity tracing.
- `rust/src/rotation.rs`
   - Added a shared rotated-state builder used by evaluator, game runner, and parity code.
- `rust/src/python_consistency.rs`
   - The real-model parity path now runs `play_game_with_observer(...)` with real `AlphaZeroAgent`s instead of reimplementing the self-play loop locally.

## Current Test Outcome
Commands used:

```bash
cd deep_quoridor/rust
source /home/julian/aaae/deep-rabbit-hole/code/deep_rabbit_hole/.venv/bin/activate
cargo fmt && cargo fmt --check
cargo check --features "python binary"
export DEEP_QUORIDOR_PARITY_DETERMINISTIC_TIES=1
cargo test --features "python binary" test_mcts_game_trace_matches_python -- --nocapture
cargo test --features "python binary" test_real_model_selfplay_trace_and_npz_matches_python -- --nocapture
cargo test --features "python binary"
```

Result:
- `test_mcts_game_trace_matches_python`: PASS
- `test_real_model_selfplay_trace_and_npz_matches_python`: PASS
- Full feature-enabled Rust suite (`--features "python binary"`): PASS (`132 passed, 0 failed`)

After reviewer follow-up:
- Full Rust all-features suite: PASS (`cargo test --all-features`, `132 passed, 0 failed`)

After production-path observer refactor:
- `cargo fmt`: PASS
- Full Rust all-features suite: PASS (`cargo test --all-features --quiet`, `132 passed, 0 failed`)

## Interpretation
The parity harness now runs with production-faithful Rust rotation timing and action-index handling, while preserving deterministic behavior only where the Python mock reference is deterministic by construction. The Rust real-model trace no longer depends on a duplicated self-play loop, so future production changes in `game_runner` are much less likely to drift away from parity coverage. The previous NPZ mask dtype blocker is resolved.

## Next Recommended Debug Step
- If deeper parity confidence is needed, run multiple deterministic and non-deterministic seeds with the same fixture pair and capture the first divergence step plus root-policy deltas.
