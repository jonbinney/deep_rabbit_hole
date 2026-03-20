# Step-Trace Consistency Test: Results

## What was done

Added a deterministic, step-by-step cross-language consistency test that compares the Python and Rust Quoridor game engines at every step of 4 scripted action sequences on a B5W2 board (board_size=5, max_walls=2).

## Files changed

### New: `deep_quoridor/src/step_trace_reference.py`
Python CLI script that plays a sequence of actions and emits, at each step:
- Grid (hex-encoded int8 bytes)
- Player positions
- Walls remaining
- Current player
- Action mask (bitmask)
- ResNet input tensor (hex-encoded float32)
- Rotated mask and tensor when current player is Player 1

### Modified: `deep_quoridor/rust/src/python_consistency.rs`
Extended with:
- `StepSnapshot` struct and parser for the Python output
- `grid_to_hex()` / `tensor_to_hex()` for comparable encoding
- `build_rotated_state()` mirroring game_runner.rs rotation logic
- `assert_snapshot_matches()` comparing all 7 fields
- `test_step_trace_matches_python` test that runs 4 sequences

## Test sequences

| Sequence | Steps | Description |
|----------|-------|-------------|
| A: Moves only | 6 | Both players advance and diverge |
| B: Walls + moves | 8 | Mix of vertical/horizontal walls and moves |
| C: Straight jump | 4 | Player 1 jumps over Player 0 |
| D: Diagonal jump | 6 | Wall forces diagonal jump |

## Results

All 93 tests pass (including the 3 cross-language consistency tests). The step-trace test confirms that **at every step of all 4 sequences**, Python and Rust produce identical:
- Grid state
- Player positions
- Walls remaining
- Current player
- Action masks
- ResNet input tensors
- Rotated masks and tensors (for Player 1 turns)

This means the basic game engine operations (moves, walls, jumps, rotation, tensor construction) are consistent between the two implementations for these deterministic sequences.

## Next steps

The test currently covers the "pure game engine" path without the neural network or MCTS. To find the training regression bug, next areas to investigate:
1. **Replay buffer format** — verify that the Rust replay writer produces data the Python trainer reads correctly
2. **MCTS tree search** — compare MCTS visit counts / policy outputs given identical NN evaluations
3. **Value assignment / game termination** — verify win/loss value backfill logic matches
4. **Edge cases** — add sequences that reach game-over (win condition) to test terminal state handling
