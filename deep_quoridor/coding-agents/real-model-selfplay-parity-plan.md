I'm using AGENTS.md

# Real-Model Selfplay Parity Plan

## Goal
Implement deterministic cross-language parity checks using real model inference:
- Python selfplay path uses paired `.pt` fixture.
- Rust selfplay path uses paired `.onnx` fixture.
- Compare both per-step traces and generated replay `.npz` content.

## Scope
- Keep production code paths as intact as possible.
- Add focused test hooks/runners with minimal refactoring.
- Deterministic config only (no Dirichlet noise, temperature 0, fixed MCTS settings).

## Phases
1. Add a dedicated Python parity runner script reusing `AlphaZeroAgent` + selfplay loop behavior.
2. Extend Rust `python_consistency.rs` to invoke Python runner and generate Rust trace with real ONNX evaluator.
3. Persist one-game NPZ outputs for both sides and compare fields/shape/dtypes/values.
4. Add fixture resolution with explicit CLI args and deterministic defaults in `rust/fixtures`.
5. Run targeted parity test and document results.

## Expected Artifacts
- New Python runner script under `deep_quoridor/src`.
- Extended Rust parity test(s) in `deep_quoridor/rust/src/python_consistency.rs`.
- Updated debugging prompt context to reflect expanded Idea D.
- Results markdown under `deep_quoridor/coding-agents`.

## Constraints
- Fail hard if required fixtures are missing.
- Maintain existing trace format compatibility (`CFG/G/P/W/C/M/T/RM/RT/V/Q/A`).
- Compare structural fields exactly; float fields with small epsilon.

## Cleanup Pass (Current)
1. Remove non-production behavior from Python reference runner:
	- Remove temporary checkpoint metadata normalization.
	- Remove deterministic override of numpy random selection.
2. Keep only instrumentation that does not alter decisions:
	- Keep trace emission and root-policy reconstruction for logging/parity assertions.
3. Re-run real-model parity test with unchanged production behavior and record new first divergence.
4. Document why any remaining helper exists and whether it is decision-affecting or logging-only.

## Next Alignment Step
1. Update Rust temperature=0 action selection to match Python semantics:
	- Sample uniformly among max-visit children instead of first-max deterministic pick.
2. Re-run `test_real_model_selfplay_trace_and_npz_matches_python` and stop at the first remaining mismatch for user review.
