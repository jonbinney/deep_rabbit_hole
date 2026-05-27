# Self-play MCTS Metrics → W&B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface per-model-version MCTS diagnostics from Rust self-play (terminal/truncation sim fraction, tree depth, root-visit spread, nodes/branching, unique games) to a W&B run in the training group, reset on each model update.

**Architecture:** Rust `search()` returns lightweight `SearchStats`; `play_game_async` folds them per game and hashes the move sequence; a per-process `SelfPlayAccumulator` aggregates by model version and flushes a JSON record to a metrics dir on each model-version change and on shutdown. A new Python process spawned by `train_v2.py` polls the dir, aggregates per-version records across processes, and logs to W&B (`group=run_id`, x-axis `Model version`).

**Tech Stack:** Rust (edition 2024, `serde_json`, `clap`, `tokio`), Python (`wandb`, `pydantic` config), pytest, cargo test.

**Spec:** `docs/superpowers/specs/2026-05-26-selfplay-mcts-metrics-design.md`

---

## File structure

- `deep_quoridor/rust/src/agents/alphazero/selfplay_mcts.rs` — add `SearchStats`, return it from `search()`. (modify)
- `deep_quoridor/rust/src/agents/alphazero/selfplay_game.rs` — add `GameMetrics` + move-sequence hashing; `play_game_async` returns `(GameResult, GameMetrics)`. (modify)
- `deep_quoridor/rust/src/agents/alphazero/selfplay_metrics.rs` — `SelfPlayAccumulator` (fold + flush JSON). (create)
- `deep_quoridor/rust/src/agents/alphazero/mod.rs` — register new module. (modify)
- `deep_quoridor/rust/Cargo.toml` — add `serde_json` to the `binary` feature. (modify)
- `deep_quoridor/rust/src/bin/selfplay.rs` — `--metrics-dir` arg; create accumulator; fold per game; flush on version change + shutdown. (modify)
- `deep_quoridor/src/v2/selfplay_metrics.py` — `aggregate_records()` + `run_selfplay_metrics()`. (create)
- `deep_quoridor/src/v2/__init__.py` — export `run_selfplay_metrics`. (modify)
- `deep_quoridor/src/train_v2.py` — pass `--metrics-dir`, spawn the metrics process. (modify)
- `deep_quoridor/test/test_selfplay_metrics.py` — Python aggregator unit tests. (create)

Build/test env note: Rust builds/tests that touch the eval pipeline need `--features binary,gpu` and, at runtime, the GPU env vars. For the tests in this plan that use the **stub coordinator** (no real ONNX), `--features binary` is enough and **no** env vars are needed. Run Rust commands with the sandbox disabled and long timeouts. Commit style (AGENTS.md): `vibe: ` imperative subject ≤50 chars; separate functional vs formatting commits; run `cargo fmt` before committing Rust.

---

## Task 1: `SearchStats` from `search()`

**Files:**
- Modify: `deep_quoridor/rust/src/agents/alphazero/selfplay_mcts.rs`

- [ ] **Step 1: Add the `SearchStats` struct**

At the top of `selfplay_mcts.rs`, after the `LeafParallelConfig` struct (around line 30), add:

```rust
/// Lightweight per-search diagnostics, accumulated cheaply during one `search()`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchStats {
    /// Number of MCTS simulations (selected leaves) this search.
    pub sims: u32,
    /// Simulations whose selected leaf was a win-terminal game state.
    pub terminal_wins: u32,
    /// Simulations whose selected leaf hit the max_steps cap.
    pub truncations: u32,
    /// Deepest selection path length (nodes from root to leaf, inclusive).
    pub max_depth: u32,
    /// Sum of selection-path lengths (divide by `sims` for mean depth).
    pub sum_depth: u64,
    /// Total nodes in the arena at search end.
    pub nodes: u32,
    /// Arena nodes that have at least one child (internal/expanded nodes).
    pub internal_nodes: u32,
    /// Entropy (nats) of the root child visit distribution.
    pub root_visit_entropy: f64,
    /// Fraction of root visits on the single most-visited child.
    pub top_move_visit_frac: f64,
}
```

- [ ] **Step 2: Change `search()` signature and instrument it**

In `selfplay_mcts.rs`, change the `search` return type (around line 102-107) from:

```rust
    pub async fn search(
        &mut self,
        root_data: CompactState,
        mechanics: &QGameMechanics,
        visited_states: &HashSet<CompactState>,
    ) -> Result<(Vec<ChildInfo>, f32)> {
```

to:

```rust
    pub async fn search(
        &mut self,
        root_data: CompactState,
        mechanics: &QGameMechanics,
        visited_states: &HashSet<CompactState>,
    ) -> Result<(Vec<ChildInfo>, f32, SearchStats)> {
```

Immediately before the `let mut iters_done: u32 = 0;` line (around line 147), add:

```rust
        let mut stats = SearchStats::default();
```

Inside the selection loop `for _ in 0..outer { ... }`, right after `let leaf_data = arena.get(leaf_idx).data;` (around line 177), add depth/sim accounting:

```rust
                let depth = path.len() as u32;
                stats.sims += 1;
                stats.sum_depth += depth as u64;
                if depth > stats.max_depth {
                    stats.max_depth = depth;
                }
```

In the same loop, in the win-terminal branch (where `Item::Terminal { path, value: v }` with `v = 1.0` is pushed for `mechanics.winner(...).is_some()`), increment after pushing:

```rust
                    items.push(Item::Terminal { path, value: v });
                    if v > 0.0 {
                        stats.terminal_wins += 1;
                    } else {
                        stats.truncations += 1;
                    }
                    continue;
```

And in the explicit max_steps branch (`items.push(Item::Terminal { path, value: 0.0 });` under `if let Some(max) = self.cfg.max_steps`), add `stats.truncations += 1;` before its `continue;`.

(Note: the win branch already computes `v` as `1.0` for a winner else `0.0`; the `v > 0.0` test above classifies win vs draw-terminal. The separate max_steps branch is always a truncation.)

- [ ] **Step 3: Compute spread + node stats and return them**

Replace the children-extraction tail of `search()` (around line 275-300) — the block that builds `children` and ends with `Ok((children, computed_root_value))` — so it computes the spread/node stats and returns the 3-tuple:

```rust
        // Extract children info.
        let bs = mechanics.repr().board_size() as i32;
        let root = arena.get(0);
        let computed_root_value = if root.visit_count > 0 {
            -(root.value_sum / root.visit_count as f64) as f32
        } else {
            root_value
        };
        let children: Vec<ChildInfo> = root
            .children
            .iter()
            .map(|&ci| {
                let c = arena.get(ci);
                let ai = c.action_index.expect("child node must have action_index");
                ChildInfo {
                    action: crate::actions::action_index_to_action(bs, ai),
                    action_index: ai,
                    visit_count: c.visit_count,
                }
            })
            .collect();

        // Root visit spread (entropy in nats + top-move fraction).
        let total_visits: u64 = children.iter().map(|c| c.visit_count as u64).sum();
        if total_visits > 0 {
            let mut entropy = 0.0f64;
            let mut max_v = 0u32;
            for c in &children {
                if c.visit_count > 0 {
                    let p = c.visit_count as f64 / total_visits as f64;
                    entropy -= p * p.ln();
                    if c.visit_count > max_v {
                        max_v = c.visit_count;
                    }
                }
            }
            stats.root_visit_entropy = entropy;
            stats.top_move_visit_frac = max_v as f64 / total_visits as f64;
        }
        stats.nodes = arena.len() as u32;
        stats.internal_nodes = (0..arena.len())
            .filter(|&i| !arena.get(i).children.is_empty())
            .count() as u32;

        // Stash the arena for tree reuse on the next call.
        self.arena = Some(arena);

        Ok((children, computed_root_value, stats))
```

- [ ] **Step 4: Update the 4 in-module tests to destructure the 3-tuple**

In the `#[cfg(test)] mod tests` of `selfplay_mcts.rs`, the four `mcts.search(...).await.unwrap()` calls currently bind `(children, _root_value)` / `(children, _)`. Change each to add a third binding `_stats`:
- `let (children, _root_value, _stats) = mcts.search(data, &mech, &visited).await.unwrap();` (test_leaf_parallel_k1...)
- `let (children, _, _stats) = mcts.search(data, &mech, &visited).await.unwrap();` (test_leaf_parallel_k8...)
- `let (children_1, _, _stats) = mcts.search(data, &mech, &visited).await.unwrap();` and `let (children_2, _, _stats) = mcts.search(next_state, &mech, &visited).await.unwrap();` (test_tree_reuse...)
- `let _ = mcts.search(data, &mech, &visited).await.unwrap();` (test_note_model_version... — unchanged, already discards)

- [ ] **Step 5: Add a `SearchStats` sanity test**

Append this test inside the `mod tests` block of `selfplay_mcts.rs` (reuses the existing `spawn_stub_coordinator`):

```rust
    #[test]
    fn test_search_stats_are_sane() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let mech = QGameMechanics::new(5, 0, 200);
            let data = mech.create_initial_state();
            let cache = Arc::new(EvalCache::new());
            let (tx, rx) = tokio_mpsc::channel::<FrontMsg>(128);
            let stub = spawn_stub_coordinator(rx, Arc::clone(&cache));

            let mcts_cfg = MCTSConfig {
                n: Some(40),
                ucb_c: 1.4,
                noise_epsilon: 0.0,
                ..Default::default()
            };
            let lp_cfg = LeafParallelConfig {
                leaf_parallelism: 4,
                virtual_loss: 1,
                enable_tree_reuse: false,
            };
            let mut mcts = LeafParallelMCTS::new(mcts_cfg, lp_cfg, tx.clone(), Arc::clone(&cache));
            let visited = std::collections::HashSet::new();
            let (_children, _v, stats) = mcts.search(data, &mech, &visited).await.unwrap();

            assert_eq!(stats.sims, 40, "sims should equal mcts_n");
            assert!(stats.max_depth >= 1, "max_depth must be >= 1");
            assert!(stats.sum_depth >= stats.sims as u64, "each sim has depth >= 1");
            assert!(stats.nodes >= 1);
            assert!(stats.internal_nodes >= 1);
            assert!(stats.root_visit_entropy >= 0.0);
            assert!(
                stats.top_move_visit_frac > 0.0 && stats.top_move_visit_frac <= 1.0,
                "top_move_visit_frac in (0,1], got {}",
                stats.top_move_visit_frac
            );

            drop(mcts);
            drop(tx);
            let _ = stub.await;
        });
    }
```

- [ ] **Step 6: Build + run the tests**

Run:
```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo test --no-default-features --features binary selfplay_mcts -- --nocapture
```
Expected: all `selfplay_mcts::tests::*` pass, including `test_search_stats_are_sane`, printing no failures.

- [ ] **Step 7: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/rust/src/agents/alphazero/selfplay_mcts.rs
git commit -m "vibe: return per-search MCTS SearchStats"
```

---

## Task 2: `GameMetrics` + move hashing in `play_game_async`

**Files:**
- Modify: `deep_quoridor/rust/src/agents/alphazero/selfplay_game.rs`
- Modify: `deep_quoridor/rust/src/bin/selfplay.rs` (two `play_game_async` call sites)

- [ ] **Step 1: Add `GameMetrics`, the opening constant, and the hash helper**

In `selfplay_game.rs`, after the `GameSettings` struct (around line 74), add:

```rust
/// Number of leading plies that define a game's "opening" for uniqueness.
pub const OPENING_PLIES: usize = 8;

/// Per-game MCTS diagnostics, summed over the game's searches, plus move-sequence hashes.
#[derive(Debug, Clone, Default)]
pub struct GameMetrics {
    pub sims: u64,
    pub terminal_wins: u64,
    pub truncations: u64,
    pub max_depth: u32,
    pub sum_depth: u64,
    pub moves: u64,
    pub sum_root_entropy: f64,
    pub sum_top_move_frac: f64,
    pub sum_nodes: u64,
    pub sum_internal_nodes: u64,
    pub full_hash: u64,
    pub opening_hash: u64,
}

/// Deterministic (within-process) hash of a move-index sequence.
fn hash_actions(actions: &[usize]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    actions.hash(&mut h);
    h.finish()
}
```

Add the `SearchStats` import to the existing `use` of `selfplay_mcts` at the top:

```rust
use crate::agents::alphazero::selfplay_mcts::{LeafParallelMCTS, SearchStats};
```

- [ ] **Step 2: Make `run_az_select` return its `SearchStats`**

In `selfplay_game.rs`, change `run_az_select` to thread the stats out. Its first line becomes:

```rust
    let (children, _root_value, stats) = mcts.search(data, mechanics, visited).await?;
```

and its return type and final expression:

```rust
) -> Result<(usize, Vec<f32>, SearchStats)> {
```
...
```rust
    Ok((action_idx, policy, stats))
```

- [ ] **Step 3: Fold stats + record moves in `play_game_async`, return `(GameResult, GameMetrics)`**

Change the `play_game_async` return type:

```rust
) -> Result<(GameResult, GameMetrics)> {
```

After `let mut winner: Option<i32> = None;` (around line 90), add:

```rust
    let mut gm = GameMetrics::default();
    let mut actions: Vec<usize> = Vec::new();
```

Replace the action-selection block (the `let (action_idx, policy) = if current_player == 0 { ... } else { ... };`, around line 101-110) with a version that captures optional stats:

```rust
        let (action_idx, policy, stats) = if current_player == 0 {
            let (a, p, s) =
                run_az_select(p1, data, &mechanics, &visited, settings, step as usize).await?;
            (a, p, Some(s))
        } else {
            match p2 {
                P2::AlphaZero(m) => {
                    let (a, p, s) =
                        run_az_select(m, data, &mechanics, &visited, settings, step as usize)
                            .await?;
                    (a, p, Some(s))
                }
                P2::Random => {
                    let (a, p) = random_select(&mask);
                    (a, p, None)
                }
            }
        };
        if let Some(s) = stats {
            gm.moves += 1;
            gm.sims += s.sims as u64;
            gm.terminal_wins += s.terminal_wins as u64;
            gm.truncations += s.truncations as u64;
            gm.max_depth = gm.max_depth.max(s.max_depth);
            gm.sum_depth += s.sum_depth;
            gm.sum_root_entropy += s.root_visit_entropy;
            gm.sum_top_move_frac += s.top_move_visit_frac;
            gm.sum_nodes += s.nodes as u64;
            gm.sum_internal_nodes += s.internal_nodes as u64;
        }
        actions.push(action_idx);
```

- [ ] **Step 4: Single return point with hashes**

Replace the win-branch early return (around line 143-157) so it sets the winner and `break`s instead of returning:

```rust
        if mechanics.check_win(data, current_player as usize) {
            winner = Some(current_player);
            break;
        }
```

Then replace the function's tail (the final `Ok(GameResult { winner, num_turns: max_steps, replay_items })`, around line 159-163) with value assignment + hashing + a single tuple return:

```rust
    if let Some(w) = winner {
        for item in replay_items.iter_mut() {
            item.value = if item.player == w { 1.0 } else { -1.0 };
        }
    }
    gm.full_hash = hash_actions(&actions);
    gm.opening_hash = hash_actions(&actions[..actions.len().min(OPENING_PLIES)]);
    let num_turns = if winner.is_some() {
        actions.len() as i32
    } else {
        max_steps
    };
    Ok((
        GameResult {
            winner,
            num_turns,
            replay_items,
        },
        gm,
    ))
```

(This preserves the prior behavior: on a win, `num_turns` is the number of moves played; on truncation, `max_steps`. Value targets are ±1 from the winner's perspective, identical to before.)

- [ ] **Step 5: Update the two call sites in `selfplay.rs`**

In `run_continuous_batched` (around line 659): change
```rust
                    let result = play_game_async(&mut p1, &mut p2, settings, board_size, max_walls, max_steps).await?;
                    write_replay(&output_dir, Some(&tmp_dir), &result, mv, idx, pid)?;
```
to
```rust
                    let (result, _game_metrics) = play_game_async(&mut p1, &mut p2, settings, board_size, max_walls, max_steps).await?;
                    write_replay(&output_dir, Some(&tmp_dir), &result, mv, idx, pid)?;
```
(The `_game_metrics` binding is replaced with real folding in Task 4; keep the underscore for now so this task builds independently.)

In the batch-mode game loop inline in `main` (the non-`--continuous` path, around line 451): change
```rust
                    let result = play_game_async(&mut p1, &mut p2, settings, board_size, max_walls, max_steps).await?;
```
to
```rust
                    let (result, _game_metrics) = play_game_async(&mut p1, &mut p2, settings, board_size, max_walls, max_steps).await?;
```

- [ ] **Step 6: Add a hashing unit test**

Append to the bottom of `selfplay_game.rs` (create a `#[cfg(test)] mod tests` block if none exists):

```rust
#[cfg(test)]
mod tests {
    use super::{hash_actions, OPENING_PLIES};

    #[test]
    fn identical_sequences_hash_equal_different_differ() {
        let a = vec![3usize, 7, 1, 9, 2, 4, 8, 0, 5, 6];
        let b = a.clone();
        let mut c = a.clone();
        c[9] = 99; // differs only after the opening

        assert_eq!(hash_actions(&a), hash_actions(&b), "identical games hash equal");
        assert_ne!(hash_actions(&a), hash_actions(&c), "different full games differ");

        // Same opening (first OPENING_PLIES), different tail -> opening hashes equal.
        let open_a = hash_actions(&a[..a.len().min(OPENING_PLIES)]);
        let open_c = hash_actions(&c[..c.len().min(OPENING_PLIES)]);
        assert_eq!(open_a, open_c, "same opening hashes equal");
    }
}
```

- [ ] **Step 7: Build + test**

Run:
```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo test --no-default-features --features binary selfplay_game -- --nocapture && cargo build --no-default-features --features binary --bin selfplay
```
Expected: the hashing test passes and the `selfplay` binary builds (confirms both call sites updated).

- [ ] **Step 8: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/rust/src/agents/alphazero/selfplay_game.rs deep_quoridor/rust/src/bin/selfplay.rs
git commit -m "vibe: collect per-game MCTS metrics and move hashes"
```

---

## Task 3: `SelfPlayAccumulator` + JSON flush

**Files:**
- Create: `deep_quoridor/rust/src/agents/alphazero/selfplay_metrics.rs`
- Modify: `deep_quoridor/rust/src/agents/alphazero/mod.rs`
- Modify: `deep_quoridor/rust/Cargo.toml`

- [ ] **Step 1: Add `serde_json` to the `binary` feature**

In `deep_quoridor/rust/Cargo.toml`, under `[dependencies]` add (near the other optional deps):

```toml
serde_json = { version = "1", optional = true }
```

and add `"serde_json"` to the `binary` feature list, so it reads:

```toml
binary = ["clap", "ort", "serde_yaml", "serde_json", "ndarray-npy", "zip", "rand_distr", "tokio", "futures"]
```

- [ ] **Step 2: Create the accumulator module**

Create `deep_quoridor/rust/src/agents/alphazero/selfplay_metrics.rs`:

```rust
//! Per-process self-play metric accumulator.
//!
//! Folds `GameMetrics` for the current model version, then flushes a raw-aggregate
//! JSON record per `(version, pid)` so the Python side can combine processes and
//! compute final metrics. Reset happens on each model-version change and on shutdown.

use std::collections::HashSet;

use anyhow::{Context, Result};

use crate::agents::alphazero::selfplay_game::GameMetrics;

/// Running raw aggregates for one model version within one process.
pub struct SelfPlayAccumulator {
    version: i64,
    sims: u64,
    terminal_wins: u64,
    truncations: u64,
    max_depth: u32,
    sum_depth: u64,
    moves: u64,
    sum_root_entropy: f64,
    sum_top_move_frac: f64,
    sum_nodes: u64,
    sum_internal_nodes: u64,
    games_generated: u64,
    full_hashes: HashSet<u64>,
    opening_hashes: HashSet<u64>,
}

impl SelfPlayAccumulator {
    pub fn new(version: i64) -> Self {
        Self {
            version,
            sims: 0,
            terminal_wins: 0,
            truncations: 0,
            max_depth: 0,
            sum_depth: 0,
            moves: 0,
            sum_root_entropy: 0.0,
            sum_top_move_frac: 0.0,
            sum_nodes: 0,
            sum_internal_nodes: 0,
            games_generated: 0,
            full_hashes: HashSet::new(),
            opening_hashes: HashSet::new(),
        }
    }

    pub fn set_version(&mut self, v: i64) {
        self.version = v;
    }

    pub fn fold_game(&mut self, gm: &GameMetrics) {
        self.sims += gm.sims;
        self.terminal_wins += gm.terminal_wins;
        self.truncations += gm.truncations;
        self.max_depth = self.max_depth.max(gm.max_depth);
        self.sum_depth += gm.sum_depth;
        self.moves += gm.moves;
        self.sum_root_entropy += gm.sum_root_entropy;
        self.sum_top_move_frac += gm.sum_top_move_frac;
        self.sum_nodes += gm.sum_nodes;
        self.sum_internal_nodes += gm.sum_internal_nodes;
        self.games_generated += 1;
        self.full_hashes.insert(gm.full_hash);
        self.opening_hashes.insert(gm.opening_hash);
    }

    fn clear_counts(&mut self) {
        self.sims = 0;
        self.terminal_wins = 0;
        self.truncations = 0;
        self.max_depth = 0;
        self.sum_depth = 0;
        self.moves = 0;
        self.sum_root_entropy = 0.0;
        self.sum_top_move_frac = 0.0;
        self.sum_nodes = 0;
        self.sum_internal_nodes = 0;
        self.games_generated = 0;
        self.full_hashes.clear();
        self.opening_hashes.clear();
    }

    fn to_json(&self, pid: u32) -> serde_json::Value {
        serde_json::json!({
            "model_version": self.version,
            "pid": pid,
            "sims": self.sims,
            "terminal_wins": self.terminal_wins,
            "truncations": self.truncations,
            "max_depth": self.max_depth,
            "sum_depth": self.sum_depth,
            "moves": self.moves,
            "sum_root_entropy": self.sum_root_entropy,
            "sum_top_move_frac": self.sum_top_move_frac,
            "sum_nodes": self.sum_nodes,
            "sum_internal_nodes": self.sum_internal_nodes,
            "games_generated": self.games_generated,
            "unique_full": self.full_hashes.len(),
            "unique_opening": self.opening_hashes.len(),
        })
    }

    /// Write the current version's record to `<dir>/v{version}_pid{pid}.json` (atomic
    /// tmp+rename), then clear counts. No-op (just clears) when nothing was accumulated.
    pub fn flush_and_reset(&mut self, dir: &str, pid: u32) -> Result<()> {
        if self.moves == 0 && self.games_generated == 0 {
            self.clear_counts();
            return Ok(());
        }
        let path = format!("{}/v{}_pid{}.json", dir, self.version, pid);
        let tmp = format!("{}.tmp", path);
        let bytes = serde_json::to_vec(&self.to_json(pid)).context("serialize metrics record")?;
        std::fs::write(&tmp, &bytes).with_context(|| format!("write {}", tmp))?;
        std::fs::rename(&tmp, &path).with_context(|| format!("rename to {}", path))?;
        self.clear_counts();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::alphazero::selfplay_game::GameMetrics;

    fn sample_game(full: u64, opening: u64) -> GameMetrics {
        GameMetrics {
            sims: 100,
            terminal_wins: 5,
            truncations: 2,
            max_depth: 10,
            sum_depth: 300,
            moves: 20,
            sum_root_entropy: 12.0,
            sum_top_move_frac: 8.0,
            sum_nodes: 500,
            sum_internal_nodes: 250,
            full_hash: full,
            opening_hash: opening,
        }
    }

    #[test]
    fn flush_writes_expected_aggregates() {
        let dir = std::env::temp_dir().join(format!("spm_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dir_s = dir.to_str().unwrap().to_string();

        let mut acc = SelfPlayAccumulator::new(7);
        acc.fold_game(&sample_game(1, 100)); // unique full=1, opening=100
        acc.fold_game(&sample_game(2, 100)); // distinct full, same opening
        acc.fold_game(&sample_game(2, 100)); // duplicate of the previous
        acc.flush_and_reset(&dir_s, 4242).unwrap();

        let path = format!("{}/v7_pid4242.json", dir_s);
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(v["model_version"], 7);
        assert_eq!(v["games_generated"], 3);
        assert_eq!(v["sims"], 300);
        assert_eq!(v["unique_full"], 2); // hashes {1,2}
        assert_eq!(v["unique_opening"], 1); // hashes {100}
        assert_eq!(v["max_depth"], 10);

        // After flush the accumulator is empty: a second flush writes nothing new.
        std::fs::remove_file(&path).unwrap();
        acc.flush_and_reset(&dir_s, 4242).unwrap();
        assert!(!std::path::Path::new(&path).exists(), "empty flush writes nothing");

        std::fs::remove_dir_all(&dir).ok();
    }
}
```

- [ ] **Step 3: Register the module**

In `deep_quoridor/rust/src/agents/alphazero/mod.rs`, add alongside the other `pub mod` lines:

```rust
pub mod selfplay_metrics;
```

- [ ] **Step 4: Build + test**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo test --no-default-features --features binary selfplay_metrics -- --nocapture
```
Expected: `flush_writes_expected_aggregates` passes (downloads `serde_json` on first build).

- [ ] **Step 5: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/rust/Cargo.toml deep_quoridor/rust/src/agents/alphazero/selfplay_metrics.rs deep_quoridor/rust/src/agents/alphazero/mod.rs
git commit -m "vibe: add SelfPlayAccumulator with JSON metric flush"
```

---

## Task 4: Wire metrics into the continuous self-play loop

**Files:**
- Modify: `deep_quoridor/rust/src/bin/selfplay.rs`

- [ ] **Step 1: Add the `--metrics-dir` CLI arg**

In the `Cli` struct (`selfplay.rs`, after the `profile_counters` field around line 125), add:

```rust
    /// Directory to write per-model-version MCTS metric JSON records. When omitted,
    /// metric collection is disabled.
    #[arg(long)]
    metrics_dir: Option<String>,
```

- [ ] **Step 2: Create the accumulator before the game tasks spawn**

In `run_continuous_batched`, inside the `rt.block_on(async move { ... })`, after `let pid = std::process::id();` (around line 595), add:

```rust
        use quoridor_rs::agents::alphazero::selfplay_metrics::SelfPlayAccumulator;
        let metrics_dir = cli.metrics_dir.clone();
        if let Some(ref d) = metrics_dir {
            std::fs::create_dir_all(d)?;
        }
        let metrics = std::sync::Arc::new(std::sync::Mutex::new(SelfPlayAccumulator::new(
            initial_version,
        )));
```

- [ ] **Step 3: Clone the handle into the game-task closure and fold each game**

In the `for _tid in 0..rust_cfg.games_per_process { ... }` loop, alongside the other `let ... = std::sync::Arc::clone(&...)` clones (around line 632-643), add:

```rust
            let metrics = std::sync::Arc::clone(&metrics);
            let metrics_enabled = metrics_dir.is_some();
```

Then change the game body (the `_game_metrics` binding added in Task 2, around line 659) to fold when enabled:

```rust
                    let (result, game_metrics) = play_game_async(&mut p1, &mut p2, settings, board_size, max_walls, max_steps).await?;
                    write_replay(&output_dir, Some(&tmp_dir), &result, mv, idx, pid)?;
                    if metrics_enabled {
                        metrics.lock().unwrap().fold_game(&game_metrics);
                    }
```

- [ ] **Step 4: Flush on version change and on shutdown in the poll task**

Clone the handle for the poll task. Alongside the poll task's existing clones (around line 668-670), add:

```rust
            let metrics = std::sync::Arc::clone(&metrics);
            let metrics_dir = metrics_dir.clone();
            let pid_for_metrics = pid;
```

In the poll task body, in the new-model branch (right after `model_version.store(latest.version, ...)`, around line 685), flush the just-finished version then point the accumulator at the new one:

```rust
                                if let Some(ref d) = metrics_dir {
                                    let mut m = metrics.lock().unwrap();
                                    if let Err(e) = m.flush_and_reset(d, pid_for_metrics) {
                                        eprintln!("selfplay-metrics: flush failed: {:#}", e);
                                    }
                                    m.set_version(latest.version);
                                }
```

And in the shutdown branch (right after `shutdown.store(true, ...)`, around line 676), flush the final partial version:

```rust
                        if let Some(ref d) = metrics_dir {
                            let mut m = metrics.lock().unwrap();
                            if let Err(e) = m.flush_and_reset(d, pid_for_metrics) {
                                eprintln!("selfplay-metrics: final flush failed: {:#}", e);
                            }
                        }
```

- [ ] **Step 5: Build**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo build --no-default-features --features binary --bin selfplay
```
Expected: builds with no errors or warnings.

- [ ] **Step 6: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/rust/src/bin/selfplay.rs
git commit -m "vibe: flush self-play MCTS metrics per model version"
```

---

## Task 5: Python aggregator + logger process

**Files:**
- Create: `deep_quoridor/src/v2/selfplay_metrics.py`
- Create: `deep_quoridor/test/test_selfplay_metrics.py`

- [ ] **Step 1: Write the failing aggregator test**

Create `deep_quoridor/test/test_selfplay_metrics.py`:

```python
import math

from v2.selfplay_metrics import aggregate_records


def _record(**kw):
    base = dict(
        model_version=3, pid=1, sims=0, terminal_wins=0, truncations=0,
        max_depth=0, sum_depth=0, moves=0, sum_root_entropy=0.0,
        sum_top_move_frac=0.0, sum_nodes=0, sum_internal_nodes=0,
        games_generated=0, unique_full=0, unique_opening=0,
    )
    base.update(kw)
    return base


def test_aggregate_combines_two_processes():
    r1 = _record(
        sims=100, terminal_wins=10, truncations=5, max_depth=12, sum_depth=400,
        moves=20, sum_root_entropy=20.0, sum_top_move_frac=10.0, sum_nodes=600,
        sum_internal_nodes=300, games_generated=2, unique_full=2, unique_opening=1,
    )
    r2 = _record(
        sims=300, terminal_wins=30, truncations=15, max_depth=18, sum_depth=1200,
        moves=60, sum_root_entropy=66.0, sum_top_move_frac=36.0, sum_nodes=1800,
        sum_internal_nodes=900, games_generated=6, unique_full=5, unique_opening=2,
    )
    agg = aggregate_records([r1, r2])

    sims, moves = 400, 80
    assert agg["selfplay/terminal_sim_frac"] == 40 / sims
    assert agg["selfplay/truncation_sim_frac"] == 20 / sims
    assert agg["selfplay/max_tree_depth"] == 18
    assert agg["selfplay/mean_tree_depth"] == 1600 / sims
    assert agg["selfplay/root_visit_entropy"] == 86.0 / moves
    assert agg["selfplay/root_visit_perplexity"] == math.exp(86.0 / moves)
    assert agg["selfplay/top_move_visit_frac"] == 46.0 / moves
    assert agg["selfplay/mean_nodes_per_search"] == 2400 / moves
    assert agg["selfplay/mean_branching"] == (2400 - moves) / 1200
    assert agg["selfplay/games_generated"] == 8
    assert agg["selfplay/unique_games_full"] == 7
    assert agg["selfplay/unique_games_opening"] == 3
    assert agg["selfplay/unique_frac_full"] == 7 / 8
    assert agg["selfplay/unique_frac_opening"] == 3 / 8


def test_aggregate_skips_empty():
    assert aggregate_records([_record(moves=0, sims=0)]) is None
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=$PYTHONPATH:$(pwd)/deep_quoridor/src .venv/bin/python -m pytest deep_quoridor/test/test_selfplay_metrics.py -q
```
Expected: FAIL with `ModuleNotFoundError: No module named 'v2.selfplay_metrics'`.

- [ ] **Step 3: Implement the module**

Create `deep_quoridor/src/v2/selfplay_metrics.py`:

```python
"""Self-play MCTS metrics: aggregate per-process Rust JSON records and log to W&B.

The Rust self-play binary writes one raw-aggregate record per (model_version, pid)
to a metrics directory. This module combines a version's records across processes,
computes the final metrics, and logs them to a W&B run in the training group.
"""
import glob
import json
import math
import os
import re
import time
from typing import Optional

import wandb

from v2.common import MockWandb, ShutdownSignal
from v2.config import Config

_FILE_RE = re.compile(r"v(\d+)_pid\d+\.json$")


def metrics_dir_for(config: Config) -> str:
    """Directory shared by the Rust writer and this reader."""
    return str(config.paths.run_dir / "selfplay_metrics")


def aggregate_records(records: list[dict]) -> Optional[dict]:
    """Combine per-process raw records for one model version into final metrics.

    Returns None if no searches happened (nothing to log).
    """
    sims = sum(r["sims"] for r in records)
    moves = sum(r["moves"] for r in records)
    if moves == 0 or sims == 0:
        return None

    sum_entropy = sum(r["sum_root_entropy"] for r in records)
    sum_nodes = sum(r["sum_nodes"] for r in records)
    sum_internal = sum(r["sum_internal_nodes"] for r in records)
    games = sum(r["games_generated"] for r in records)
    unique_full = sum(r["unique_full"] for r in records)
    unique_opening = sum(r["unique_opening"] for r in records)
    mean_entropy = sum_entropy / moves

    out = {
        "selfplay/terminal_sim_frac": sum(r["terminal_wins"] for r in records) / sims,
        "selfplay/truncation_sim_frac": sum(r["truncations"] for r in records) / sims,
        "selfplay/max_tree_depth": max(r["max_depth"] for r in records),
        "selfplay/mean_tree_depth": sum(r["sum_depth"] for r in records) / sims,
        "selfplay/root_visit_entropy": mean_entropy,
        "selfplay/root_visit_perplexity": math.exp(mean_entropy),
        "selfplay/top_move_visit_frac": sum(r["sum_top_move_frac"] for r in records) / moves,
        "selfplay/mean_nodes_per_search": sum_nodes / moves,
        "selfplay/mean_branching": (sum_nodes - moves) / sum_internal if sum_internal else 0.0,
        "selfplay/games_generated": games,
        "selfplay/unique_games_full": unique_full,
        "selfplay/unique_games_opening": unique_opening,
        "selfplay/unique_frac_full": unique_full / games if games else 0.0,
        "selfplay/unique_frac_opening": unique_opening / games if games else 0.0,
    }
    return out


def _scan(metrics_dir: str) -> dict[int, list[str]]:
    """Map model_version -> list of record file paths present on disk."""
    by_version: dict[int, list[str]] = {}
    for path in glob.glob(os.path.join(metrics_dir, "v*_pid*.json")):
        m = _FILE_RE.search(os.path.basename(path))
        if m:
            by_version.setdefault(int(m.group(1)), []).append(path)
    return by_version


def _load(paths: list[str]) -> list[dict]:
    records = []
    for p in paths:
        try:
            with open(p) as f:
                records.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue  # mid-write or transient; picked up on a later poll
    return records


def run_selfplay_metrics(config: Config, poll_seconds: float = 5.0):
    """Poll the metrics dir and log each completed model version to W&B once."""
    metrics_dir = metrics_dir_for(config)
    os.makedirs(metrics_dir, exist_ok=True)

    if config.wandb:
        run_id = f"{config.run_id}-selfplay"
        wandb_run = wandb.init(
            project=config.wandb.project,
            job_type="selfplay",
            group=config.run_id,
            name=run_id,
            id=run_id,
            resume="allow",
        )
        wandb.define_metric("Model version", hidden=True)
        wandb.define_metric("*", "Model version")
    else:
        wandb_run = MockWandb()

    logged: set[int] = set()

    def flush(finalize_all: bool):
        by_version = _scan(metrics_dir)
        if not by_version:
            return
        max_version = max(by_version)
        for version in sorted(by_version):
            if version in logged:
                continue
            # A version is complete once a newer version exists (the writer moved on)
            # or we're finalizing on shutdown.
            if not finalize_all and version >= max_version:
                continue
            agg = aggregate_records(_load(by_version[version]))
            if agg is not None:
                agg["Model version"] = version
                wandb_run.log(agg)
            logged.add(version)

    while not ShutdownSignal.is_set(config):
        flush(finalize_all=False)
        time.sleep(poll_seconds)

    flush(finalize_all=True)  # log the last in-progress version on shutdown
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=$PYTHONPATH:$(pwd)/deep_quoridor/src .venv/bin/python -m pytest deep_quoridor/test/test_selfplay_metrics.py -q
```
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/v2/selfplay_metrics.py deep_quoridor/test/test_selfplay_metrics.py
git commit -m "vibe: add self-play metrics aggregator and W&B logger"
```

---

## Task 6: Spawn the logger and pass `--metrics-dir` from `train_v2.py`

**Files:**
- Modify: `deep_quoridor/src/v2/__init__.py`
- Modify: `deep_quoridor/src/train_v2.py`

- [ ] **Step 1: Export `run_selfplay_metrics` from the package**

In `deep_quoridor/src/v2/__init__.py`, add to `__all__` (the list that includes `"create_benchmark_processes"`):

```python
    "run_selfplay_metrics",
```

and add the import (next to `from v2.benchmarks import create_benchmark_processes`):

```python
from v2.selfplay_metrics import metrics_dir_for, run_selfplay_metrics
```

Also add `"metrics_dir_for"` to `__all__`.

- [ ] **Step 2: Import in `train_v2.py` and pass `--metrics-dir` to the Rust subprocess**

In `deep_quoridor/src/train_v2.py`, update the `from v2 import (...)` line (line 8) to also import the two new names:

```python
from v2 import (
    benchmarks,
    check_ai_available,
    load_config_and_setup_run,
    metrics_dir_for,
    run_ai_reporter,
    run_selfplay_metrics,
    self_play,
    train,
)
```

In the rust-spawn block (`if config.self_play.program == "rust":`, around line 93), before the `for i in range(...)` loop, compute the metrics dir:

```python
        selfplay_env = _selfplay_subprocess_env()
        if selfplay_env is not None:
            print(f"Self-play GPU env: ORT_DYLIB_PATH={selfplay_env['ORT_DYLIB_PATH']}")
        metrics_dir = metrics_dir_for(config)
        os.makedirs(metrics_dir, exist_ok=True)
        config_file_path = str(config.paths.config_file)
```

and add the two args to `cmd` (inside the list, after the `--shutdown-file` pair):

```python
                "--shutdown-file",
                str(ShutdownSignal.file_path(config)),
                "--metrics-dir",
                metrics_dir,
```

- [ ] **Step 3: Spawn the metrics logger process (rust path only)**

Still inside the `if config.self_play.program == "rust":` block, after the `for` loop that starts the rust subprocesses (after the `print(f"Started Rust self-play process {proc.pid}")` line), add:

```python
        selfplay_metrics_process = mp.Process(target=run_selfplay_metrics, args=[config])
        selfplay_metrics_process.start()
        self_play_processes.append(selfplay_metrics_process)
```

(Appending to `self_play_processes` means the existing shutdown/`is_alive()` accounting at lines 128-129 already tracks and waits on it.)

- [ ] **Step 4: Syntax check**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor && PYTHONPATH=$(pwd)/src ../.venv/bin/python -m py_compile src/train_v2.py src/v2/__init__.py src/v2/selfplay_metrics.py && echo PY_OK
```
Expected: `PY_OK`.

- [ ] **Step 5: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/v2/__init__.py deep_quoridor/src/train_v2.py
git commit -m "vibe: spawn self-play metrics logger from train_v2"
```

---

## Task 7: Formatting commit (per AGENTS.md)

- [ ] **Step 1: Format Rust and Python**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo fmt && cargo fmt --check && echo FMT_OK
cd /home/jbinney/ws/deep_rabbit_hole && .venv/bin/ruff format deep_quoridor/src/v2/selfplay_metrics.py deep_quoridor/test/test_selfplay_metrics.py deep_quoridor/src/train_v2.py deep_quoridor/src/v2/__init__.py 2>/dev/null || true
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo build --no-default-features --features binary --bin selfplay
```
Expected: `FMT_OK`, build succeeds.

- [ ] **Step 2: Commit only if formatting changed files**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git status --short deep_quoridor/rust deep_quoridor/src
# If any files changed:
git add -u deep_quoridor/rust deep_quoridor/src
git commit -m "vibe: cargo fmt + ruff format"
```
If nothing changed, skip the commit.

---

## Manual end-to-end verification (after all tasks)

Not a unit test — a smoke check the implementer should run once with a real model and the GPU env exported (`ORT_DYLIB_PATH` + `LD_LIBRARY_PATH` as in prior work), building with `--features binary,gpu`:

1. Start `train_v2.py` on the b9w10 config with a fresh `run_id` for a few minutes (enough for ≥2 model updates).
2. Confirm JSON records appear under `runs/<run_id>/selfplay_metrics/` named `v{N}_pid{PID}.json`.
3. Confirm a W&B run `${run_id}-selfplay` exists in group `${run_id}` with `selfplay/*` metrics plotted against `Model version`.
4. Sanity-check values: `terminal_sim_frac > 0`, `max_tree_depth ≥ mean_tree_depth ≥ 1`, `top_move_visit_frac ∈ (0,1]`, `unique_frac_full` near 1.0 early on.

## Self-review (completed during authoring)

- **Spec coverage:** terminal/truncation frac, max+mean depth, root entropy/perplexity, top-move frac, nodes/branching (Task 1 + Task 5 aggregation); unique full+opening with K=8 (Task 2 `OPENING_PLIES`, per-process dedup in Task 3, summed in Task 5); reset-on-model-update + shutdown flush (Task 4); file-based bridge + Python logger in the run group, x-axis Model version (Tasks 3/5/6); disabled unless `--metrics-dir` (Task 4); testing (Tasks 1/2/3/5) — all mapped.
- **Placeholder scan:** none — every step has concrete code/commands.
- **Type consistency:** `SearchStats` (Task 1) fields are consumed exactly by `GameMetrics` folding (Task 2) and `SelfPlayAccumulator.fold_game` (Task 3); JSON keys written by `to_json` (Task 3) match those read by `aggregate_records` (Task 5) and the test record (Task 5); `metrics_dir_for` defined in Task 5 is used in Task 6; `play_game_async`'s `(GameResult, GameMetrics)` return is consistently destructured at both call sites (Task 2) and folded (Task 4).
