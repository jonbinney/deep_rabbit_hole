# Plan: Multi-threaded Parallel Games & NN Cache for Rust Self-Play

**TL;DR:** Add two features to the Rust selfplay binary: (1) an NN evaluation
cache backed by an LRU map keyed on `GameState::get_fast_hash()`, and (2)
multi-threaded parallel game execution where N games run on separate threads
sharing one `CachingEvaluator`. The work is split into refactoring commits
(with partial verification at each step) followed by feature commits. Both
batch and continuous modes gain `--parallel-games` and `--max-cache-size` CLI
flags, wired to the already-parsed YAML `parallel_games` config.

**Note:** This plan implements thread-level parallelism (independent games on
separate threads) and per-evaluation caching. It does NOT implement batched NN
inference (collecting leaves across multiple MCTS trees into one forward pass),
which remains in the backlog as a complementary future optimization.

## Steps

### Refactoring

1. **Change `Evaluator` trait from `&mut self` to `&self`**
   - In `evaluator.rs`, change `fn evaluate(&mut self, ...)` to
     `fn evaluate(&self, ...)`. `OnnxEvaluator` compiles trivially since
     `Session::run()` already takes `&self`.
   - Update `mcts.rs search()` from `evaluator: &mut E` to `evaluator: &E`.
   - Update `agent.rs select_action()` call from `&mut self.evaluator` to
     `&self.evaluator`.
   - Update `MockEvaluator` in mcts.rs tests — use `Cell`/`RefCell` if it
     tracks internal call counts.
   - **Pre-commit:** `cargo fmt && cargo build && cargo test`
   - **Commit:** `vibe: change Evaluator trait to &self`

2. **Refactor `AlphaZeroAgent` to borrow evaluator instead of owning it**
   - Change `AlphaZeroAgent` to `AlphaZeroAgent<'a>` holding
     `evaluator: &'a (dyn Evaluator + Sync)` instead of owning
     `OnnxEvaluator`.
   - Update `new()` to accept an evaluator reference. Callers create the
     evaluator externally and pass a reference.
   - Update `selfplay.rs` `create_agent` and `BoxedAgent` to own the
     evaluator separately and pass a reference into the agent.
   - **Pre-commit:** `cargo fmt && cargo build && cargo test`. Run a quick
     batch selfplay with a CI ONNX model (`--num-games 5`).
   - **Commit:** `vibe: refactor AlphaZeroAgent to borrow evaluator`

### Features

3. **Add `CachingEvaluator` with LRU cache**
   - Add `lru` crate to `Cargo.toml`.
   - Create `CachingEvaluator` in `evaluator.rs`:
     - `session: RwLock<Session>` — evaluate takes read lock (no contention
       between threads), model reload takes write lock.
     - `cache: Mutex<LruCache<u64, (f32, Vec<f32>)>>` — keyed by
       `state.get_fast_hash()`, default max size 200,000.
   - Implement `Evaluator for CachingEvaluator`: hash state, check cache
     under mutex, on miss acquire session read lock, run inference, insert
     result into cache.
   - Add `clear_cache(&self)` and
     `reload_model(&self, model_path) -> Result<()>` (acquires session write
     lock, replaces session, clears cache).
   - Add unit tests.
   - **Pre-commit:** `cargo fmt && cargo build && cargo test`.
   - **Commit:** `vibe: add CachingEvaluator with LRU cache`

4. **Add `play_games_parallel` with multi-threaded game execution**
   - In `game_runner.rs`, add `play_games_parallel()`.
   - Uses `std::thread::scope` to spawn `parallel_games` threads. Each
     thread creates its own `AlphaZeroAgent` borrowing the shared evaluator,
     and plays `ceil(num_games / parallel_games)` games sequentially.
   - Keep old `play_game(agent_p1, agent_p2, ...)` for backward compat.
   - Add tests with `MockEvaluator`.
   - **Pre-commit:** `cargo fmt && cargo build && cargo test`.
   - **Commit:** `vibe: add play_games_parallel with threading`

5. **Update `selfplay.rs` to use parallel games and caching**
   - Add CLI flags: `--parallel-games` (default from YAML), `--max-cache-size`
     (default 200,000).
   - Batch mode: `CachingEvaluator` + `play_games_parallel` (fallback to
     sequential for `--p2`).
   - Continuous mode: `CachingEvaluator` + `parallel_games` threads via
     `std::thread::scope`, with `AtomicI64` version tracking + model reload.
   - **Pre-commit:** `cargo fmt && cargo build && cargo test`.
   - **Commit:** `vibe: use parallel games and caching in selfplay binary`

6. **Update backlog** — remove NN Evaluation Caching, update Batch Search
   to "Batched NN Inference".
   - **Commit:** `vibe: update backlog for completed work`

7. **Formatting** — `cargo fmt` if any remaining changes.

8. **Write results** — `coding-agents/parallel-games-and-cache-results.md`.
   - **Commit:** `vibe: add results for parallel games and cache`

## Verification

1. `cargo fmt && cargo build && cargo test` passes at every commit.
2. Batch selfplay: `--num-games 20 --parallel-games 4 --max-cache-size 50000`
3. Regression: `--parallel-games 1` vs current sequential.
4. Full training: `train_v2.py` with `ci.yaml` + `self_play.program: rust`.

## Decisions

- **Threading:** `std::thread::scope` — clean borrowing without `Arc`.
- **Session sharing:** `RwLock<Session>` — evaluations take read lock (zero
  contention), model reload takes write lock (briefly blocks).
- **Cache:** `Mutex<lru::LruCache>` — O(1) LRU eviction, negligible
  contention since NN inference dominates.
- **Backward compat:** Old `play_game(agent_p1, agent_p2, ...)` stays for
  `--p2 random` and existing tests.
- **Batched inference deferred:** Tracked in backlog as separate optimization.
