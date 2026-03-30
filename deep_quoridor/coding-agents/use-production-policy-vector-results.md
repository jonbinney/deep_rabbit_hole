# Use Production Policy Vector Results

## What changed
- Added a production helper method on `AlphaZeroAgent` to construct a full action-space policy vector from MCTS root children visit counts.
- Updated self-play training data collection to use the new helper instead of duplicating policy-vector logic inline.
- Updated `selfplay_real_model_reference.py` to use the production helper, removing its local policy-vector implementation.

## Why
- The reference trace path was re-implementing policy-vector construction already present in production behavior.
- Reusing production code reduces drift risk and keeps parity checks aligned with real agent behavior.

## Validation
- Checked both modified files with workspace diagnostics; no errors reported.

## Follow-up cleanup
- Removed duplicated visit-count normalization logic from `AlphaZeroAgent.get_action_batch`.
- Added a single shared helper for root-child visit-probability normalization and reused it in both:
	- policy-vector construction
	- action-selection probability computation
- Re-ran diagnostics on `alphazero.py`; no errors reported.

## Notes
- Behavior remains unchanged: same normalization, same dtype (`float32`), and same error condition when no MCTS visits are available.
