# PR #346 Review — Implementation Plan

## Reviewer Comments Being Addressed

1. `evaluator.rs:70` — reuse `masked_softmax` inside `OnnxEvaluator::evaluate`
2. `game_state.rs:93` — replace all multi-param calls with `&GameState`
3. `evaluator.rs:40` — use `&GameState` in `grid_game_state_to_resnet_input`
4. `mcts.rs:125` — (doc question, no code change needed)
5/6. `selfplay.rs:19` — replace `--agent-type` string with `--use-raw-onnx-agent` bool flag
7. `selfplay.rs:138` — print board config (size and walls) in stats line

---

## Commit 1: Use `&GameState` everywhere, reuse `masked_softmax`

### `grid_helpers.rs`
- Change `grid_game_state_to_resnet_input` signature from 4 raw params
  `(grid, player_positions, walls_remaining, current_player)` to `(state: &GameState)`.
- Bind local vars at the top of the function body; rest of body unchanged.
- Remove `use ndarray::{ArrayView1, ArrayView2}` top-level import (no longer in signature).
- Add `use crate::game_state::GameState;` import.
- Update all 11 call sites in tests to use `GameState::new(board_size, 3)`.
  Tests that mutate grid/walls mutate `state.grid` / `state.walls_remaining` directly (fields are `pub`).
  Perspective-swap test clones state and sets `state.current_player = 1`.
- Remove `create_test_board` helper; remove stale `Array1, Array2` test imports where no longer needed.

### `agents/onnx_agent.rs`
- Change `grid_game_state_to_resnet_input(...)` 4-arg call to `grid_game_state_to_resnet_input(state)`.

### `agents/alphazero/evaluator.rs`
- Change `grid_game_state_to_resnet_input(...)` 4-arg call to `grid_game_state_to_resnet_input(state)`.
- Replace inline mask-then-softmax block with `masked_softmax(policy_logits.1, action_mask)`.
- Remove `use crate::agents::onnx_agent::softmax` import (no longer used).

### `game_runner.rs`
- Replace `compute_full_action_mask(...)` 6-arg call + manual mask allocation
  with `let mask = work_state.get_action_mask();`.
- Change `grid_game_state_to_resnet_input(...)` 4-arg call to `grid_game_state_to_resnet_input(&work_state)`.
- Remove `compute_full_action_mask` from imports.

### `actions.rs` tests
- Replace `create_test_game()` tuple + `compute_full_action_mask(...)` calls
  with `GameState::new(...)` + `state.get_action_mask()`.
- Add `use crate::game_state::GameState;` in test module.

Note: `compute_full_action_mask` itself keeps its low-level raw-array signature —
it is an internal helper called only by `GameState::get_action_mask()`.

---

## Commit 2: `selfplay.rs` CLI simplification + board config print

### `selfplay.rs`
- Remove `agent_type: String` (default `"onnx"`).
- Add `use_raw_onnx_agent: bool` flag (default `false`).
  - `false` (default) → AlphaZero agent.
  - `true` → legacy raw ONNX greedy agent.
- Simplify `create_agent` accordingly.
- Update `p2` handling: `--p2 random` remains; otherwise P2 uses same mode as P1.
- Print board config (`board_size`, `max_walls`) in the opening stats block.
- Update doc-comment examples at top of file.
