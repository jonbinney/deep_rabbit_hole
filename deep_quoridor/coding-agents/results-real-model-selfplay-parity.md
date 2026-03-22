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

## Current Test Outcome
Command used:

```bash
cd deep_quoridor/rust
source /home/julian/aaae/deep-rabbit-hole/code/deep_rabbit_hole/.venv/bin/activate
cargo test --features "python binary" test_real_model_selfplay_trace_and_npz_matches_python -- --nocapture
```

Result:
- Test now fails during Python model load under the unmodified production path.
- Failure is a checkpoint inconsistency:
  - The fixture state_dict has ResNet keys.
  - Checkpoint params cause Python to construct `MLPNetwork`.
  - `load_state_dict` errors with missing/unexpected keys.

After fixture replacement and tie-handling alignment:
- Python model loading issue is resolved with refreshed PT fixture.
- Parity test still diverges at step 0 due random tie resolution using independent RNG streams.

## Interpretation
After cleanup, the runner is faithful to production behavior. The previous temporary metadata patch was masking a fixture-format inconsistency. To continue parity comparison without harness-side patching, the `.pt` fixture itself must be made internally consistent for the Python loader.

## Next Recommended Debug Step
- Fix or regenerate the `.pt` fixture so its saved params match its ResNet weights.
- Then rerun parity and evaluate first action-level divergence without any harness overrides.
