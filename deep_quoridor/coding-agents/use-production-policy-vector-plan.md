I'm using AGENTS.md

# Use Production Policy Vector Plan

## Goal
Update the real-model selfplay reference script to use production AlphaZero policy-vector logic instead of re-implementing it.

## Steps
1. Add the smallest possible helper in `AlphaZeroAgent` that builds a full action-space policy vector from MCTS root children.
2. Replace the local policy builder in `selfplay_real_model_reference.py` with a call to that helper.
3. Keep behavior unchanged (same normalization and dtype) and avoid unrelated refactors.
4. Run targeted validation for changed Python files.

## Expected Outcome
- No duplicated policy-vector implementation in the reference script.
- Reference test path uses production code for policy reconstruction.
- Minimal, review-friendly diff.
