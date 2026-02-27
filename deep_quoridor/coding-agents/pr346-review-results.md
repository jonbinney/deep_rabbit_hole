# PR #346 Review — Changes Made

This document summarises the changes made in response to the reviewer's comments on PR #346.

---

## Reviewer Comments Addressed

### Comment: `evaluator.rs` — `masked_softmax` not reused

`OnnxEvaluator::evaluate` was manually applying the mask and calling `softmax` inline, duplicating
the logic of the `masked_softmax` function defined just below it. Fixed by replacing the inline block
with a direct call to `masked_softmax(logits, action_mask)`.

### Comments: use `&GameState` instead of individual array parameters

`grid_game_state_to_resnet_input` and `compute_full_action_mask` were both called throughout the
codebase with 4–6 individual array parameters. All call sites have been updated to pass `&GameState`
instead:

- **`grid_helpers.rs`**: `grid_game_state_to_resnet_input` signature changed from
  `(grid, player_positions, walls_remaining, current_player)` to `(state: &GameState)`.
  All 11 test call sites updated to use `GameState::new(...)`.

- **`evaluator.rs`**: call site updated to `grid_game_state_to_resnet_input(state)`.

- **`onnx_agent.rs`**: call site updated to `grid_game_state_to_resnet_input(state)`.

- **`game_runner.rs`**: the 6-arg `compute_full_action_mask(...)` call replaced with
  `work_state.get_action_mask()`. The 4-arg resnet call replaced with
  `grid_game_state_to_resnet_input(&work_state)`. The `compute_full_action_mask` import removed.

- **`actions.rs` tests**: direct `compute_full_action_mask(...)` calls replaced with
  `GameState::new(...).get_action_mask()`.

Note: `compute_full_action_mask` itself keeps its low-level raw-array signature — it is an internal
helper called only by `GameState::get_action_mask()`, so no circular dependency is introduced.

### Comment: `mcts.rs` — clarify "Arena-based"

The doc comment on `NodeArena` already explains the rationale. No code change needed; the answer is:
an arena (`Vec<Node>` + integer indices) is the standard Rust pattern for graph structures with parent
pointers, avoiding self-referential borrows and per-node heap allocation overhead that
`Rc<RefCell<Node>>` would incur.

### Comments: `selfplay.rs` — simplify agent selection, print board config

- Removed `--agent-type: String`. Added `--use-raw-onnx-agent: bool` (default `false`).
  AlphaZero MCTS is now the unconditional default; the flag opts back into the legacy raw-ONNX
  greedy-argmax path.
- `--p2 random` continues to work as before.
- Board configuration (`board_size`, `max_walls`) is now printed in the startup summary block.
- Doc-comment examples at the top of the file updated to reflect the new CLI.

---

## Commits

| Hash | Summary |
|------|---------|
| `97c2db6` | use `&GameState` in all call sites, reuse `masked_softmax` in evaluator |
| `2f08ff0` | replace `--agent-type` with `--use-raw-onnx-agent` flag, alphazero is now default |
| `6bf0244` | fix build warnings — remove unused `rand::Rng` import, add explicit lifetimes to `GameState` accessors |

---

## Verification

- `cargo fmt` — no formatting changes
- `cargo build --features binary` — clean, zero warnings
- `cargo test` — 84 tests pass, 0 failures
