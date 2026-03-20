# Debugging: Rust Self-Play Produces Bad Training Data

## Problem Statement

Training with Rust self-play does not improve: **neither loss nor win-rate improves** when using the Rust self-play implementation. The same training pipeline works correctly with Python self-play.

The board configuration where this has been observed is **B5W2** (board_size=5, max_walls=2).

## Architecture Overview

The system has two self-play implementations that are supposed to produce equivalent training data:

### Python Path
- `deep_quoridor/src/v2/self_play.py` — orchestrates games
- `deep_quoridor/src/agents/alphazero/alphazero.py` — AlphaZero agent with `get_action_batch()`
- `deep_quoridor/src/agents/alphazero/nn_evaluator.py` — handles board rotation for P1 via `rotate_if_needed_to_point_downwards()`, feeds tensor to NN, rotates policy back
- `deep_quoridor/src/agents/alphazero/mcts.py` — Python MCTS tree search
- `deep_quoridor/src/agents/alphazero/resnet_network.py` — `game_to_input_array()` builds 5-channel tensor
- `deep_quoridor/src/quoridor.py` — game engine (`Quoridor`, `Board`, action encoding)

### Rust Path
- `deep_quoridor/rust/src/game_runner.rs` — `play_game()` loop, handles rotation explicitly for P1
- `deep_quoridor/rust/src/agents/alphazero/mcts.rs` — Rust MCTS implementation
- `deep_quoridor/rust/src/game_state.rs` — `GameState` struct, `step()`, `get_action_mask()`
- `deep_quoridor/rust/src/grid_helpers.rs` — `grid_game_state_to_resnet_input()` builds 5-channel tensor
- `deep_quoridor/rust/src/rotation.rs` — `rotate_grid_180()`, `rotate_player_positions()`, `rotate_goal_rows()`, `rotate_action_coords()`
- `deep_quoridor/rust/src/replay_writer.rs` — writes NPZ files read by Python trainer
- `deep_quoridor/rust/src/actions.rs` — action encoding/decoding

### Training Pipeline
- Rust self-play writes `.npz` files (arrays: `input_arrays`, `policies`, `action_masks`, `values`, `players`) plus `.yaml` metadata
- Python trainer (`deep_quoridor/src/v2/trainer.py`) loads these NPZ files via `Sampler` class
- The trainer doesn't know or care whether data came from Python or Rust self-play

## Key Architecture Details

### Board Rotation
The NN always sees "current player moving downward." When it's Player 1's turn:
- **Python**: `NNEvaluator.rotate_if_needed_to_point_downwards()` copies game, calls `game.rotate_board()` (which does `np.rot90(grid, k=2)`), then rotates the policy back after NN inference
- **Rust**: `game_runner.rs` explicitly builds a rotated `work_state` using `rotate_grid_180()`, `rotate_player_positions()`, `rotate_goal_rows()`, computes mask/tensor on that, then un-rotates the action coords back

### ResNet Input Tensor (5 channels, shape `(5, M, M)` where `M = board_size*2+3`)
1. Walls (1 where wall, 0 elsewhere) — from grid
2. Current player position — 1-hot at `(row*2+2, col*2+2)`
3. Opponent position — 1-hot
4. Current player walls remaining — scalar fill
5. Opponent walls remaining — scalar fill

### Action Encoding (flat index)
- Indices `[0, bs²)` — move actions: `idx = row*bs + col`
- Indices `[bs², bs² + (bs-1)²)` — vertical walls: `idx = bs² + row*(bs-1) + col`
- Indices `[bs² + (bs-1)², bs² + 2*(bs-1)²)` — horizontal walls

### Value Backfill (Rust)
- Winner's moves get `+1.0`, loser's get `-1.0`
- Truncated games (max_steps) get `0.0` for all moves

### Replay Buffer Storage
Both paths store the same data. Rust writes NPZ directly in `replay_writer.rs`. The replay items are stored **in the rotated frame** (the frame the NN saw), not the original board frame.

## Bug Candidate Surfaces

We identified 6 main areas where Python and Rust could diverge:

| # | Surface | Description |
|---|---------|-------------|
| 1 | **Game engine** | `step()`, `get_action_mask()`, jump logic, wall placement |
| 2 | **Board rotation** | 180° rotation of grid, positions, goals for P1 |
| 3 | **Tensor construction** | 5-channel ResNet input from game state |
| 4 | **MCTS tree search** | UCB selection, expansion, backprop, Dirichlet noise |
| 5 | **Replay format** | NPZ field ordering, shapes, dtypes, value backfill |
| 6 | **Action encoding** | Index ↔ (row, col, type) mapping, rotation remapping |

## Testing Approaches Discussed

We discussed 4 ideas and chose to start with Idea A:

### Idea A: Deterministic Scripted Game Trace (IMPLEMENTED ✅)
Replay scripted action sequences (no NN/MCTS) and compare every field at every step between Python and Rust. Pure game-engine + rotation + tensor comparison.

### Idea B: Single-Game Replay Comparison
Record a real Rust self-play game, replay it in Python, compare at each step.

### Idea C: Round-Trip Replay Buffer
Write game data from Rust, read it back in Python, verify fields are identical.

### Idea D: Identical-NN MCTS Comparison
Give both implementations the same NN weights with deterministic seeding, compare MCTS visit counts and policies.

## What Has Been Done

### 1. Step-Trace Consistency Test (Idea A) — PASSED ✅

**Commit**: Added `step_trace_reference.py` + extended `python_consistency.rs`

Created a cross-language test with 4 deterministic sequences on B5W2:

| Sequence | Steps | Description |
|----------|-------|-------------|
| A: Moves only | 6 | Both players advance and diverge |
| B: Walls + moves | 8 | Mix of vertical/horizontal walls and moves |
| C: Straight jump | 4 | P1 jumps over P0 |
| D: Diagonal jump | 6 | Wall forces diagonal jump |

At **every step** of all 4 sequences, Python and Rust produce **identical**:
- Grid state
- Player positions
- Walls remaining
- Current player
- Action masks
- ResNet input tensors
- Rotated masks and tensors (for Player 1 turns)

**Conclusion**: The basic game engine, rotation, and tensor construction are consistent. The bug is elsewhere.

### 2. CI Pipeline Fix — IN PROGRESS
The `rust-ci.yml` needed updating to use `deep_quoridor/ci_requirements.txt` instead of manual `pip install` since `step_trace_reference.py` imports `torch`.

## What To Try Next

Priority-ordered based on likelihood of being the root cause:

### Priority 1: Replay Buffer Round-Trip (Idea C)
Verify that the Rust NPZ writer produces data the Python trainer reads correctly. Check:
- Array shapes and dtypes match expectations
- Field names match (`input_arrays`, `policies`, `action_masks`, `values`, `players`)
- Values are in the correct range
- The rotated frame is stored correctly (Rust stores replay items in the rotated frame)

### Priority 2: Value Backfill / Game Termination
Compare win/loss/truncation value assignment between Python and Rust:
- Does Rust correctly detect wins?
- Is the value backfill (+1/-1/0) applied the same way?
- Are game lengths comparable?
- Add test sequences that reach game-over (win condition)

### Priority 3: MCTS Comparison (Idea D)
Given identical NN evaluations, do Python and Rust MCTS produce the same visit counts and policies?
- Dirichlet noise seeding
- UCB formula differences
- Tree expansion order
- Policy normalization

### Priority 4: Subtle Edge Cases
- Wall placement pathfinding validation (do both check path existence the same way?)
- Games that end in exactly `max_steps` 
- Action mask for states with 0 walls remaining

## Key Files Reference

| File | Purpose |
|------|---------|
| `deep_quoridor/rust/src/python_consistency.rs` | Cross-language consistency tests |
| `deep_quoridor/src/step_trace_reference.py` | Python reference script for step trace |
| `deep_quoridor/src/action_reference.py` | Python reference script for action encoding |
| `deep_quoridor/rust/src/game_runner.rs` | Rust self-play loop with rotation |
| `deep_quoridor/src/v2/self_play.py` | Python v2 self-play orchestrator |
| `deep_quoridor/rust/src/replay_writer.rs` | Rust NPZ replay writer |
| `deep_quoridor/src/v2/trainer.py` | Python trainer (reads NPZ) |
| `deep_quoridor/rust/src/agents/alphazero/mcts.rs` | Rust MCTS |
| `deep_quoridor/src/agents/alphazero/mcts.py` | Python MCTS |
| `deep_quoridor/src/agents/alphazero/nn_evaluator.py` | Python NN evaluator with rotation |
| `deep_quoridor/rust/src/grid_helpers.rs` | Rust ResNet tensor construction |
| `deep_quoridor/src/agents/alphazero/resnet_network.py` | Python ResNet tensor construction |
| `deep_quoridor/ci_requirements.txt` | CI Python deps (CPU-only torch) |
| `deep_quoridor/coding-agents/results-step-trace-consistency-test.md` | Results of Idea A |
