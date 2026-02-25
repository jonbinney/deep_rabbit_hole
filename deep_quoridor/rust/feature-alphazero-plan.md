# Rust AlphaZero MCTS Agent — Implementation Plan

Port the Python AlphaZero agent (MCTS + NN evaluation) to Rust, behind the `binary` feature flag. Single-game search for now; batch search deferred to a backlog. The work begins with foundational refactors — `GameState` wrapper in the core module and `ActionSelector` simplification — before building the AlphaZero-specific components. Each self-contained step is its own git commit.

---

## Commit 1: `GameState` struct + `ActionSelector` refactor

Introduce a `GameState` wrapper and modernize all existing code to use it.

### Steps

1. Add `rand_distr` to `Cargo.toml` as optional dependency behind `binary` feature (needed later for Dirichlet noise, but adding it now avoids touching Cargo.toml again).

2. Create `GameState` struct in `game_state.rs` — bundles `grid: Array2<i8>`, `player_positions: Array2<i32>`, `walls_remaining: Array1<i32>`, `goal_rows: Array1<i32>`, `current_player: i32`, `board_size: i32`, `completed_steps: usize`. Derives `Clone`. Methods:
   - `new(board_size, max_walls) -> Self` — delegates to existing `create_initial_state`
   - `step(&mut self, action: [i32; 3])` — calls `apply_action`, increments `completed_steps`, swaps `current_player`
   - `clone_and_step(&self, action) -> Self` — clone + step (for MCTS lazy expansion)
   - `is_game_over(&self) -> bool` — either player has won
   - `check_win(&self, player: i32) -> bool` — delegates to existing `check_win`
   - `get_action_mask(&self) -> Vec<bool>` — delegates to `compute_full_action_mask`
   - `get_fast_hash(&self) -> u64` — hash of grid + positions + walls + current_player (for visited-state sets)
   - `policy_size(&self) -> usize` — delegates to `actions::policy_size`
   - Accessor methods for grid, positions, walls, goals, current_player as views

3. Update `ActionSelector` trait in `agents/mod.rs` — change signature to:
   ```rust
   fn select_action(&mut self, state: &GameState, action_mask: &[bool]) -> Result<(usize, Vec<f32>)>
   ```
   The agent can access all game state fields via `GameState` accessors. `action_mask` stays a separate param because the game runner pre-computes it.

4. Update `RandomAgent` in `random_agent.rs` — adapt `select_action` to new signature. Extract needed fields from `&GameState`.

5. Update `OnnxAgent` in `onnx_agent.rs` — adapt to new signature. Calls `grid_game_state_to_resnet_input` using `state.grid()`, `state.player_positions()`, etc.

6. Update `play_game` in `game_runner.rs` — construct a `GameState` at the start, use its methods for stepping/win-checking. Build rotated `GameState` copies for Player 1 calls. Pass `&GameState` to `select_action`. Keep `ReplayBufferItem` construction using `grid_game_state_to_resnet_input` on the rotated state as before.

7. Update `selfplay.rs` — if affected by `play_game` signature changes.

8. **Tests**: Existing tests in `game_state.rs`, `random_agent.rs`, `onnx_agent.rs`, `game_runner.rs` must still pass (adapted to use `GameState`). Add new tests for `GameState`: construction, step, clone_and_step, check_win, is_game_over, action_mask validity, hash stability (same state → same hash, different state → different hash).

---

## Commit 2: `Evaluator` trait + MCTS tree search

Build the core MCTS engine as a reusable module, decoupled from ONNX via a trait.

### Steps

1. Create `agents/alphazero/mod.rs` — declares submodules `evaluator` and `mcts` (and `agent` later).

2. Create `Evaluator` trait in `agents/alphazero/evaluator.rs`:
   - `trait Evaluator { fn evaluate(&self, state: &GameState, action_mask: &[bool]) -> Result<(f32, Vec<f32>)>; }` — returns `(value_for_current_player, priors)` where priors are masked + softmaxed.
   - `OnnxEvaluator` struct: wraps `ort::Session`. `evaluate()` builds ResNet input via `grid_game_state_to_resnet_input`, runs inference, extracts `"value"` and `"policy_logits"`, applies mask (`logits[i] = -1e32` where invalid), softmax, returns `(value, priors)`.
   - Reuse `softmax()` from `onnx_agent.rs` — consider moving it to a shared utility or `pub use` it.

3. Create MCTS in `agents/alphazero/mcts.rs`, matching Python `alphazero/mcts.py`:
   - **Arena-based tree**: `NodeArena` wraps `Vec<Node>`, methods to allocate/access nodes by index. Avoids Rust lifetime issues with parent pointers.
   - `QuoridorKey` — `hash: u64` for visited-state set lookups.
   - `Node` — `game: Option<GameState>`, `parent: Option<usize>` (arena index), `action_taken: Option<[i32; 3]>`, `children: Vec<usize>`, `visit_count: u32`, `value_sum: f64`, `wins: u32`, `losses: u32`, `ucb_c: f32`, `prior: f32`. Lazy game: on first access, clone parent's game + step.
   - `should_expand()` → `children.is_empty()`
   - `expand(priors, arena)` — for each nonzero prior, create a child node with `game: None`, push to arena, add index to `self.children`.
   - `select(visited_states, arena) -> usize` — PUCT: `q_value + prior * c * sqrt(parent_visits) / (child_visits + 1) - penalty`. Penalty = 1.0 if child's game hash is in `visited_states`.
   - `backpropagate(value, arena)` — update `value_sum`, `visit_count`, recurse to parent with `-value`.
   - `backpropagate_result(value, arena)` — same + tracks `wins`/`losses`.
   - `MCTSConfig` — `n`, `k`, `ucb_c`, `noise_epsilon`, `noise_alpha`, `max_steps`.
   - `MCTS::search(config, game, evaluator, visited_states) -> (Vec<ChildInfo>, f32)`:
     - Create root node, iterate `n` (or `k * num_valid_actions`) times.
     - Each iteration: select → terminal check (game_over → `backpropagate_result(1)`, max_steps → `backpropagate_result(0)`) → evaluate with `evaluator` → expand → `backpropagate(-value)`.
     - Dirichlet noise at root on first expansion (if `noise_epsilon > 0`).
     - `n=0` special case: evaluate root, expand, set `visit_count = (prior * 1000) as u32`.
     - Return `ChildInfo { action: [i32; 3], action_index: usize, visit_count: u32 }` for each root child + root value.

4. **Tests**:
   - `OnnxEvaluator`: masked softmax correctness (invalid actions get ~0 prob, valid sum to ~1). (Full ONNX inference test can be integration-level, skip if no model file.)
   - `Node` expansion: expand with known priors `[0.0, 0.5, 0.0, 0.3, 0.2]`, verify 3 children created with correct priors.
   - `Node` UCB/select: create children with controlled visit_counts/priors, verify highest-UCB child is selected.
   - `Node` backpropagation: 3-node chain, backpropagate `+1` from leaf, verify value alternation and visit counts.
   - `Node` backpropagate_result: verify wins/losses tracked correctly with alternating signs.
   - Visited-state penalty: two children, one in visited set, verify the other is selected.
   - MCTS search with mock evaluator: `MockEvaluator` returning fixed `(value=0.0, uniform_priors)`. Run short search (`n=10`), verify root has children, visit counts > 0, root value is finite.
   - MCTS `n=0`: verify visit counts are proportional to priors × 1000.
   - Dirichlet noise: apply to priors, verify priors changed from original, still sum to ~1.0, only nonzero-prior entries affected.

---

## Commit 3: AlphaZero agent + config + selfplay integration

Wire everything together as a usable agent.

### Steps

1. Create `AlphaZeroAgent` in `agents/alphazero/agent.rs`:
   - Struct: `evaluator: OnnxEvaluator`, `config: AlphaZeroConfig`, `visited_states: HashSet<u64>`.
   - Constructor: `new(model_path, config) -> Result<Self>`.
   - Implements `ActionSelector`:
     1. Construct action mask from `state.get_action_mask()` (or use passed-in mask).
     2. Create `MCTS`, call `search(state, &self.evaluator, &self.visited_states)`.
     3. Compute visit-count probabilities: `visit_count[i] / total_visits`.
     4. Apply temperature: if `T == 0` → argmax; else `probs[i]^(1/T)`, renormalize, sample from distribution.
     5. Check `drop_t_on_step`: if `state.completed_steps >= threshold`, force `T = 0`.
     6. Optionally add chosen state's hash to `visited_states` (if `penalized_visited_states` is true).
     7. Return `(action_index, full_policy_vector)`.
   - `reset_game()` method to clear `visited_states` between games.

2. Update `agents/alphazero/mod.rs` — add `pub mod agent;`, re-export `AlphaZeroAgent`.

3. Update `agents/mod.rs` — add `#[cfg(feature = "binary")] pub mod alphazero;`.

4. Extend `selfplay_config.rs`:
   - Add `AlphaZeroConfig` struct (serde): `mcts_n: Option<u32>` (default 100), `mcts_k: Option<u32>`, `mcts_c_puct: f32` (default 1.4, maps to `ucb_c`), `mcts_noise_epsilon: f32` (default 0.25), `mcts_noise_alpha: Option<f32>` (auto-computed if None), `temperature: Option<f32>`, `drop_t_on_step: Option<usize>`, `penalized_visited_states: bool` (default false), `max_steps: Option<i32>`.
   - YAML keys match the Python config exactly so configs are reusable.
   - Add `alphazero: Option<AlphaZeroConfig>` to `PipelineConfig`.
   - Also handle `self_play.alphazero` nesting for training-specific overrides (e.g., noise epsilon during self-play).

5. Update `selfplay.rs`:
   - Add `--agent-type` CLI arg: `"onnx"` (default, current behavior) or `"alphazero"`.
   - When `alphazero`: load `AlphaZeroConfig` from YAML's `alphazero` section (merged with `self_play.alphazero` overrides), create `AlphaZeroAgent`, call `reset_game()` between games.

6. **Tests**:
   - Temperature: greedy (T=0) always picks highest visit count; proportional (T=1) returns normalized visit counts; high T flattens distribution.
   - Config parsing: deserialize a sample YAML string, verify all AlphaZero fields populated correctly with defaults.
   - Integration-level: with a `MockEvaluator`, run `AlphaZeroAgent.select_action` on a fresh `GameState`, verify valid action returned.

---

## Commit 4: Backlog

1. Create `backlog.md` documenting:
   - Batch search (multiple games per MCTS iteration, batched NN inference)
   - NN evaluation caching (LRU cache keyed by game hash, matching Python `NNEvaluator`)
   - Potential `undo_action`-based tree traversal instead of cloning

---

## Verification

- `cargo test` — all new + existing tests pass
- `cargo build --features binary` — compiles with AlphaZero + ONNX
- `cargo build` (default `python` feature) — compiles without AlphaZero (behind feature flag)
- Manual: `selfplay --agent-type alphazero --config experiments/ci.yaml --model-path <model.onnx> --num-games 1 --trace`

## Key Decisions

- `GameState` lives in core `game_state.rs`, not inside alphazero module
- `ActionSelector` updated to take `&GameState`, all agents + game_runner refactored
- `Evaluator` is a separate MCTS-internal trait (not a replacement for `ActionSelector`)
- Arena-based tree (`Vec<Node>` + indices) for MCTS nodes
- Each major step is its own git commit
- `rand_distr` added for Dirichlet noise
- Config YAML keys match Python format for reusability
- Single-game search only; batch search deferred to backlog
- All behind `#[cfg(feature = "binary")]` feature flag (depends on `ort`)
