# Plan: Train Evaluator from Policy DB

## Context
The policy DB (`PolicyDb` in `rust/src/compact/policy_db.rs`) stores compact bit-packed game states and their minimax values (P0-absolute: 1=P0 wins, 0=draw, -1=P1 wins). We want to train a neural network that approximates this lookup, so it can evaluate unseen or larger-board positions quickly without a full DB.

Two metrics are required:
1. **Loss**: MSE between model predictions and DB values on a held-out test set.
2. **Accuracy**: For a random sample of test states, check whether the child state the model rates highest is the same child state the DB rates highest (i.e., does the model pick the best move?).

## New Rust Python Bindings (lib.rs)

### `compact_state_to_features(state, board_size, max_walls, max_steps) -> np.ndarray`
Decodes a compact state `bytes` object into a `float32` feature vector.

### `get_compact_child_states(state, board_size, max_walls, max_steps) -> list[tuple[int, int, int, bytes]]`
Returns `(row, col, action_type, child_state_bytes)` for every valid action.

## Python Training Script
**New file**: `src/train_policy_db_evaluator.py`

Trains an MLP evaluator on policy DB data, logging loss and accuracy.
