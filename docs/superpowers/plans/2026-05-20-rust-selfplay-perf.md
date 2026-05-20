# Rust Self-Play Performance Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Saturate CPU and GPU on the Rust self-play binary by switching to a tokio task-per-game model with leaf-parallel MCTS (virtual loss), a pipelined eval coordinator, and tree reuse across moves.

**Architecture:** One tokio task per concurrent game; each game's MCTS issues K in-flight evals per outer iteration via virtual loss. Eval coordinator becomes a three-stage pipeline (Batcher → Inference → Post-process) so GPU work overlaps with batch assembly and parallel softmax/un-rotation. Trees are reused across consecutive moves within a single game.

**Tech Stack:** Rust 2021, tokio (rt-multi-thread + sync), rayon, smallvec, ort 2.0, dashmap; Python 3 with pydantic for config schema.

**Spec:** `docs/superpowers/specs/2026-05-20-rust-selfplay-perf-design.md`

**File overview:**

- **New (Rust):**
  - `deep_quoridor/rust/src/agents/alphazero/eval_pipeline.rs` — pipelined async coordinator (Batcher / Inference / Post-process stages)
  - `deep_quoridor/rust/src/agents/alphazero/selfplay_mcts.rs` — `LeafParallelMCTS` (async leaf-parallel search with virtual loss, tree reuse, model-version tracking)
  - `deep_quoridor/rust/src/agents/alphazero/selfplay_game.rs` — `play_game_async` (mirrors `play_game` semantics for replay capture)
- **Modified (Rust):**
  - `deep_quoridor/rust/Cargo.toml` (add tokio, smallvec, futures)
  - `deep_quoridor/rust/src/agents/alphazero/mod.rs` (exports)
  - `deep_quoridor/rust/src/agents/alphazero/mcts.rs` (virtual-loss helpers, refactor of Dirichlet noise to in-arena, `promote_subtree`)
  - `deep_quoridor/rust/src/agents/alphazero/eval_coordinator.rs` (deleted in cleanup task)
  - `deep_quoridor/rust/src/agents/alphazero/evaluator.rs` (`BatchingEvaluator` deleted in cleanup task)
  - `deep_quoridor/rust/src/selfplay_config.rs` (schema)
  - `deep_quoridor/rust/src/bin/selfplay.rs` (new CLI + tokio runtime + game tasks)
- **Modified (Python):**
  - `deep_quoridor/src/v2/config.py` (`SelfPlayConfig`)
  - `deep_quoridor/test/config_test.py`
  - `deep_quoridor/experiments/*.yaml`

The legacy `--use-raw-onnx-agent` path (which uses `OnnxAgent` directly without MCTS) is left untouched.

---

## Task 1: Add Rust dependencies

**Files:**
- Modify: `deep_quoridor/rust/Cargo.toml`

- [ ] **Step 1: Add tokio, smallvec, and futures to `[dependencies]`**

In `deep_quoridor/rust/Cargo.toml`, locate the `[dependencies]` section and add:

```toml
tokio = { version = "1", features = ["rt-multi-thread", "sync", "macros", "time"], optional = true }
futures = { version = "0.3", optional = true }
smallvec = "1.13"
```

Then update the `[features]` section so `binary` pulls in tokio and futures:

```toml
binary = ["clap", "ort", "serde_yaml", "ndarray-npy", "zip", "rand_distr", "tokio", "futures"]
```

`smallvec` stays out of features (used unconditionally in mcts.rs).

- [ ] **Step 2: Verify cargo resolves the new deps**

Run: `cd deep_quoridor/rust && cargo check --features binary --bin selfplay`
Expected: build succeeds; no warnings about unused new deps yet (they will be used in later tasks).

- [ ] **Step 3: Commit**

```bash
git add deep_quoridor/rust/Cargo.toml deep_quoridor/rust/Cargo.lock
git commit -m "Add tokio, futures, smallvec deps for selfplay refactor"
```

---

## Task 2: Update Python `SelfPlayConfig` schema

**Files:**
- Modify: `deep_quoridor/src/v2/config.py`
- Test: `deep_quoridor/test/config_test.py`

- [ ] **Step 1: Update the `SelfPlayConfig` model**

In `deep_quoridor/src/v2/config.py`, replace the existing `SelfPlayConfig` class with:

```python
class SelfPlayConfig(StrictBaseModel):
    num_processes: int
    games_per_process: int
    # Leaf-parallel MCTS knobs (Rust self-play only).
    leaf_parallelism: int = 16
    virtual_loss: int = 3
    enable_tree_reuse: bool = True
    mcts_worker_threads: Optional[int] = None
    eval_batch_size: int = 2048
    eval_max_wait_ms: int = 0
    eval_cache_max_size: int = 100000
    alphazero: Optional[AlphaZeroSelfPlayConfig] = None
    program: Literal["python", "rust"] = "python"
    rust_selfplay_binary: Optional[str] = None
```

Note: `threads_per_process` and `games_per_thread` are deleted.

- [ ] **Step 2: Update the existing `test/config_test.py` fixture**

In `deep_quoridor/test/config_test.py`, change:

```python
    "self_play": {"num_processes": 2, "games_per_thread": 8, "alphazero": {"mcts_noise_epsilon": 0.25}},
```

to:

```python
    "self_play": {"num_processes": 2, "games_per_process": 16, "alphazero": {"mcts_noise_epsilon": 0.25}},
```

- [ ] **Step 3: Run the Python config tests**

Run: `cd deep_quoridor && PYTHONPATH=src python -m pytest test/config_test.py -q`
Expected: 13 passed.

- [ ] **Step 4: Commit**

```bash
git add deep_quoridor/src/v2/config.py deep_quoridor/test/config_test.py
git commit -m "Rework SelfPlayConfig for leaf-parallel MCTS

Drops threads_per_process/games_per_thread; adds games_per_process,
leaf_parallelism, virtual_loss, enable_tree_reuse, mcts_worker_threads.
Raises eval_cache_max_size default from 0 to 100000."
```

---

## Task 3: Update Rust `SelfPlayWorkerConfig`

**Files:**
- Modify: `deep_quoridor/rust/src/selfplay_config.rs`

- [ ] **Step 1: Replace `SelfPlayWorkerConfig` fields**

In `deep_quoridor/rust/src/selfplay_config.rs`, replace the existing `SelfPlayWorkerConfig` struct (around line 77) with:

```rust
/// Self-play worker parameters from the YAML (subset of Python's `SelfPlayConfig`).
///
/// `num_processes` is the number of self-play subprocesses (Python-side concern).
/// Inside one Rust process, `games_per_process` async game tasks run concurrently,
/// each MCTS issuing `leaf_parallelism` in-flight evals per outer iteration with
/// `virtual_loss` applied during descent. The eval coordinator batches all
/// in-flight evals together (up to `eval_batch_size`, deadline-bounded by
/// `eval_max_wait_ms` from first request) and runs one ONNX inference per batch.
#[derive(Debug, Deserialize)]
pub struct SelfPlayWorkerConfig {
    #[serde(default)]
    pub num_processes: Option<usize>,
    #[serde(default)]
    pub games_per_process: Option<usize>,
    #[serde(default)]
    pub leaf_parallelism: Option<usize>,
    #[serde(default)]
    pub virtual_loss: Option<u32>,
    #[serde(default)]
    pub enable_tree_reuse: Option<bool>,
    #[serde(default)]
    pub mcts_worker_threads: Option<usize>,
    #[serde(default)]
    pub eval_batch_size: Option<usize>,
    #[serde(default)]
    pub eval_max_wait_ms: Option<u64>,
    #[serde(default)]
    pub eval_cache_max_size: Option<usize>,
    #[serde(default)]
    pub alphazero: Option<AlphaZeroSelfPlayConfig>,
}
```

- [ ] **Step 2: Update the in-file YAML test strings**

In the `tests` module of the same file, replace every occurrence of:

```yaml
  games_per_thread: 2
```

with:

```yaml
  games_per_process: 16
```

…and replace any other now-removed fields (`threads_per_process`) similarly. Verify test assertions still reference fields that exist on the new struct.

Also update the `test_load_python_compatible_config` assertion `assert_eq!(config.self_play.unwrap().games_per_thread, Some(2));` → `assert_eq!(config.self_play.unwrap().games_per_process, Some(16));`.

- [ ] **Step 3: Run the rust unit tests**

Run: `cd deep_quoridor/rust && cargo test --features binary --lib selfplay_config`
Expected: all `selfplay_config` tests pass.

- [ ] **Step 4: Commit**

```bash
git add deep_quoridor/rust/src/selfplay_config.rs
git commit -m "Rework SelfPlayWorkerConfig for leaf-parallel MCTS

Mirror Python schema changes: drops threads_per_process / games_per_thread,
adds games_per_process, leaf_parallelism, virtual_loss, enable_tree_reuse,
mcts_worker_threads."
```

---

## Task 4: Update experiment YAML configs

**Files:**
- Modify: `deep_quoridor/experiments/ci.yaml`
- Modify: `deep_quoridor/experiments/ci-resnet.yaml`
- Modify: `deep_quoridor/experiments/example.yaml`
- Modify: `deep_quoridor/experiments/B5W2/cucu-01.yaml`
- Modify: `deep_quoridor/experiments/B5W3/base.yaml`
- Modify: `deep_quoridor/experiments/B5W3/cucu-01.yaml`
- Modify: `deep_quoridor/experiments/B5W3/test_onnx_export.yaml`

- [ ] **Step 1: In each file, rename `games_per_thread` → `games_per_process`**

For every experiment YAML listed above, locate the `self_play:` block and:

- Rename `games_per_thread: N` → `games_per_process: M` where `M` is chosen as follows: pick a reasonable concurrent-games count (e.g., 16 for small/CI configs; 64 for production training configs). When the file previously had both `threads_per_process: T` and `games_per_thread: G`, set `games_per_process` to `T * G` to preserve the old aggregate concurrent-games count. Remove the `threads_per_process` line.

Example: `ci.yaml` `self_play:` block becomes:

```yaml
self_play:
  num_processes: 2
  games_per_process: 16
  leaf_parallelism: 16
  virtual_loss: 3
  alphazero:
    mcts_noise_epsilon: 0.25
```

(No need to set every new field explicitly — defaults are fine. Include `leaf_parallelism` and `virtual_loss` in CI to exercise non-default values.)

- [ ] **Step 2: Verify the example file loads cleanly**

Run: `cd deep_quoridor && PYTHONPATH=src python -c "from v2.config import load_user_config; print(load_user_config('experiments/ci.yaml').self_play)"`
Expected: prints a `SelfPlayConfig` instance with `games_per_process=16`, `leaf_parallelism=16`, `virtual_loss=3`.

- [ ] **Step 3: Commit**

```bash
git add deep_quoridor/experiments/
git commit -m "Update experiment YAMLs for new self_play schema"
```

---

## Task 5: Add `prior_clean` to `Node` (preserves un-noised network output)

**Files:**
- Modify: `deep_quoridor/rust/src/agents/alphazero/mcts.rs`

We store a separate `prior_clean` on each node so that, after tree reuse, we can re-noise the new root's children from a clean baseline. Today, root's children get noise baked into `prior` at expansion; deeper nodes are clean. With the invariant that noise is *only ever* applied to direct children of the current root, deeper-node `prior_clean` simply mirrors `prior`. The field is set at construction and never modified afterwards.

- [ ] **Step 1: Add the field**

In `mcts.rs`, modify `struct Node`:

```rust
#[derive(Debug)]
pub struct Node {
    pub data: CompactState,
    pub parent: Option<usize>,
    pub action_index: Option<usize>,
    pub children: Vec<usize>,
    pub visit_count: u32,
    pub value_sum: f64,
    pub wins: u32,
    pub losses: u32,
    pub prior: f32,
    /// Un-noised network prior; never modified after construction. Used to
    /// re-apply Dirichlet noise to new-root children after tree reuse.
    pub prior_clean: f32,
}
```

Update `Node::new_root` and `Node::new_child`:

```rust
impl Node {
    pub fn new_root(data: CompactState) -> Self {
        Self {
            data, parent: None, action_index: None, children: Vec::new(),
            visit_count: 0, value_sum: 0.0, wins: 0, losses: 0,
            prior: 1.0, prior_clean: 1.0,
        }
    }

    pub fn new_child(parent: usize, action_index: usize, data: CompactState, prior: f32) -> Self {
        Self {
            data, parent: Some(parent), action_index: Some(action_index),
            children: Vec::new(),
            visit_count: 0, value_sum: 0.0, wins: 0, losses: 0,
            prior, prior_clean: prior,
        }
    }
    // ...
}
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `cd deep_quoridor/rust && cargo test --features binary --lib agents::alphazero::mcts`
Expected: existing 12 mcts tests pass.

- [ ] **Step 3: Commit**

```bash
git add deep_quoridor/rust/src/agents/alphazero/mcts.rs
git commit -m "Store prior_clean on each MCTS node for tree-reuse re-noising"
```

---

## Task 6: Refactor `apply_dirichlet_noise` to operate on the arena's root children

**Files:**
- Modify: `deep_quoridor/rust/src/agents/alphazero/mcts.rs`

Today `apply_dirichlet_noise` mutates a priors `Vec<f32>` *before* `expand_node` is called, so the noise gets baked into children at expansion. We change this so the root is always expanded with clean priors, and noise is mixed into the root children's `prior` field after expansion. This preserves `prior_clean` correctly and makes promotion-noise reuse trivial.

- [ ] **Step 1: Write a failing test for the new arena-based function**

Append to the `tests` module in `mcts.rs`:

```rust
#[test]
fn test_apply_dirichlet_noise_arena_modifies_only_prior() {
    let mech = QGameMechanics::new(5, 0, 200);
    let data = mech.create_initial_state();
    let mut arena = NodeArena::new(data);
    let total = crate::actions::policy_size(5);
    let mut priors = vec![0.0f32; total];
    let mask = mech.get_action_mask_immut(data);
    for (i, &v) in mask.iter().enumerate() {
        if v {
            priors[i] = 1.0 / mask.iter().filter(|&&m| m).count() as f32;
        }
    }
    expand_node(&mut arena, 0, &priors, &mech);

    // Snapshot prior_clean before noise.
    let clean_before: Vec<f32> = arena.get(0).children.iter()
        .map(|&i| arena.get(i).prior_clean).collect();

    apply_dirichlet_noise_to_root_children(&mut arena, 0, 0.25, 0.5);

    // prior_clean unchanged, prior changed.
    let mut any_changed = false;
    for (offset, &child_idx) in arena.get(0).children.iter().enumerate() {
        let c = arena.get(child_idx);
        assert!((c.prior_clean - clean_before[offset]).abs() < 1e-6,
            "prior_clean must not be modified");
        if (c.prior - c.prior_clean).abs() > 1e-6 {
            any_changed = true;
        }
    }
    assert!(any_changed, "noise should change at least one child's prior");
}
```

- [ ] **Step 2: Run to confirm it fails**

Run: `cd deep_quoridor/rust && cargo test --features binary --lib agents::alphazero::mcts::tests::test_apply_dirichlet_noise_arena_modifies_only_prior`
Expected: FAIL — `apply_dirichlet_noise_to_root_children` does not exist.

- [ ] **Step 3: Implement `apply_dirichlet_noise_to_root_children`**

In `mcts.rs`, add (next to the existing `apply_dirichlet_noise`):

```rust
/// Mix Dirichlet noise into the `prior` field of all children of `root_idx`,
/// leaving `prior_clean` untouched. Iterates over children, samples a Dirichlet
/// over them, and replaces `prior[i] ← (1-ε) * prior_clean[i] + ε * noise[i]`.
pub fn apply_dirichlet_noise_to_root_children(
    arena: &mut NodeArena,
    root_idx: usize,
    epsilon: f32,
    alpha: f32,
) {
    let child_ids: Vec<usize> = arena.get(root_idx).children.clone();
    if child_ids.is_empty() {
        return;
    }
    let dirichlet = match Dirichlet::new_with_size(alpha, child_ids.len()) {
        Ok(d) => d,
        Err(_) => return,
    };
    let mut rng = rand::thread_rng();
    let noise: Vec<f32> = dirichlet.sample(&mut rng);
    for (i, &child_idx) in child_ids.iter().enumerate() {
        let c = arena.get_mut(child_idx);
        c.prior = (1.0 - epsilon) * c.prior_clean + epsilon * noise[i];
    }
}
```

- [ ] **Step 4: Update `search` to use the arena-based version**

Find this block in `search`:

```rust
    // Apply Dirichlet noise at root if configured
    if config.noise_epsilon > 0.0 {
        let alpha = config.noise_alpha.unwrap_or_else(|| {
            let num_valid = action_mask.iter().filter(|&&m| m).count();
            10.0 / num_valid.max(1) as f32
        });
        apply_dirichlet_noise(&mut root_priors, config.noise_epsilon, alpha);
    }
```

Move the noise application to AFTER root expansion (which happens inside the iteration loop today). Specifically: after `expand_node(&mut arena, 0, &leaf_priors, mechanics);` in the first-iteration branch (where `current_idx == 0`), call:

```rust
            if config.noise_epsilon > 0.0 && current_idx == 0 {
                let alpha = config.noise_alpha.unwrap_or_else(|| {
                    let num_valid = action_mask.iter().filter(|&&m| m).count();
                    10.0 / num_valid.max(1) as f32
                });
                apply_dirichlet_noise_to_root_children(&mut arena, 0, config.noise_epsilon, alpha);
            }
```

And delete the pre-expansion `apply_dirichlet_noise(&mut root_priors, …)` call. Also update the special `n=0` branch (`expand_node(&mut arena, 0, &root_priors, mechanics);`) to apply noise after expansion in the same way (if epsilon > 0).

- [ ] **Step 5: Run all mcts tests**

Run: `cd deep_quoridor/rust && cargo test --features binary --lib agents::alphazero::mcts`
Expected: all tests pass, including the new one.

- [ ] **Step 6: Commit**

```bash
git add deep_quoridor/rust/src/agents/alphazero/mcts.rs
git commit -m "Apply Dirichlet noise after root expansion using arena-based fn

Lays groundwork for tree reuse: prior_clean stays untouched, and noise
can be re-mixed into the new root's children after subtree promotion."
```

---

## Task 7: Add virtual-loss apply/undo helpers

**Files:**
- Modify: `deep_quoridor/rust/src/agents/alphazero/mcts.rs`

- [ ] **Step 1: Write the failing test**

Append to the `tests` module:

```rust
#[test]
fn test_virtual_loss_apply_undo_round_trip() {
    let (_, data) = make_mech_state();
    let mut arena = NodeArena::new(data);
    let c1 = arena.alloc_child(0, 0, data, 0.5);
    let c2 = arena.alloc_child(c1, 0, data, 0.5);
    arena.get_mut(0).children = vec![c1];
    arena.get_mut(c1).children = vec![c2];

    // Snapshot.
    let (root_v, root_s) = (arena.get(0).visit_count, arena.get(0).value_sum);
    let (c1_v, c1_s)     = (arena.get(c1).visit_count, arena.get(c1).value_sum);
    let (c2_v, c2_s)     = (arena.get(c2).visit_count, arena.get(c2).value_sum);

    let path = vec![0usize, c1, c2];
    apply_virtual_loss(&mut arena, &path, 3);

    assert_eq!(arena.get(0).visit_count, root_v + 3);
    assert!((arena.get(0).value_sum - (root_s - 3.0)).abs() < 1e-9);
    assert_eq!(arena.get(c1).visit_count, c1_v + 3);
    assert_eq!(arena.get(c2).visit_count, c2_v + 3);

    undo_virtual_loss(&mut arena, &path, 3);

    assert_eq!(arena.get(0).visit_count, root_v);
    assert!((arena.get(0).value_sum - root_s).abs() < 1e-9);
    assert_eq!(arena.get(c1).visit_count, c1_v);
    assert!((arena.get(c1).value_sum - c1_s).abs() < 1e-9);
    assert_eq!(arena.get(c2).visit_count, c2_v);
    assert!((arena.get(c2).value_sum - c2_s).abs() < 1e-9);
}
```

- [ ] **Step 2: Run to confirm it fails**

Run: `cd deep_quoridor/rust && cargo test --features binary --lib agents::alphazero::mcts::tests::test_virtual_loss_apply_undo_round_trip`
Expected: FAIL — `apply_virtual_loss` undefined.

- [ ] **Step 3: Implement the helpers**

In `mcts.rs`, add:

```rust
/// Apply a virtual loss along `path` (root → leaf inclusive). For each node,
/// `visit_count += vl` and `value_sum -= vl`. Call `undo_virtual_loss` before
/// real backprop to restore the baseline.
pub fn apply_virtual_loss(arena: &mut NodeArena, path: &[usize], vl: u32) {
    let vl_f = vl as f64;
    for &idx in path {
        let n = arena.get_mut(idx);
        n.visit_count += vl;
        n.value_sum -= vl_f;
    }
}

/// Reverse `apply_virtual_loss`.
pub fn undo_virtual_loss(arena: &mut NodeArena, path: &[usize], vl: u32) {
    let vl_f = vl as f64;
    for &idx in path {
        let n = arena.get_mut(idx);
        n.visit_count -= vl;
        n.value_sum += vl_f;
    }
}
```

- [ ] **Step 4: Run the test**

Run: `cd deep_quoridor/rust && cargo test --features binary --lib agents::alphazero::mcts::tests::test_virtual_loss_apply_undo_round_trip`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add deep_quoridor/rust/src/agents/alphazero/mcts.rs
git commit -m "Add virtual-loss apply/undo helpers for leaf-parallel MCTS"
```

---

## Task 8: Add `select_leaf_with_vl` (leaf-parallel selection primitive)

**Files:**
- Modify: `deep_quoridor/rust/src/agents/alphazero/mcts.rs`

This function descends from `root_idx` using UCB, applying virtual loss at each node it touches. It returns the path (root → leaf) and the leaf index. The caller is responsible for either expanding/evaluating the leaf and then `undo_virtual_loss` + `backpropagate`, or treating the leaf as terminal.

- [ ] **Step 1: Write the failing diversification test**

```rust
#[test]
fn test_select_leaf_with_vl_diversifies_concurrent_selections() {
    use smallvec::SmallVec;
    let mech = QGameMechanics::new(5, 0, 200);
    let data = mech.create_initial_state();
    let mut arena = NodeArena::new(data);

    let mask = mech.get_action_mask_immut(data);
    let total = crate::actions::policy_size(5);
    let mut priors = vec![0.0f32; total];
    // Three valid actions with similar priors so vl can spread them out.
    let valid: Vec<usize> = mask.iter().enumerate()
        .filter_map(|(i, &v)| if v { Some(i) } else { None }).collect();
    assert!(valid.len() >= 3);
    for &i in &valid[..3] { priors[i] = 1.0 / 3.0; }
    expand_node(&mut arena, 0, &priors, &mech);
    arena.get_mut(0).visit_count = 0;

    let visited = HashSet::new();
    let mut first_level_choices = std::collections::HashSet::new();
    for _ in 0..3 {
        let path: SmallVec<[usize; 32]> = select_leaf_with_vl(&mut arena, 0, 1.4, 3, &visited);
        // First-level child is path[1] (path[0] is the root).
        first_level_choices.insert(path[1]);
    }
    assert!(first_level_choices.len() >= 2,
        "vl should drive at least two distinct first-level child selections");
}
```

- [ ] **Step 2: Run to confirm it fails**

Run: `cd deep_quoridor/rust && cargo test --features binary --lib agents::alphazero::mcts::tests::test_select_leaf_with_vl_diversifies_concurrent_selections`
Expected: FAIL — `select_leaf_with_vl` undefined; `smallvec` import missing.

- [ ] **Step 3: Implement the function**

Add `use smallvec::SmallVec;` near the top of `mcts.rs`. Then:

```rust
/// Descend from `root_idx` to a leaf using PUCT, applying a virtual loss of
/// magnitude `vl` to every node touched (including the leaf). Returns the path
/// root→leaf (inclusive). The caller must eventually call `undo_virtual_loss`
/// on this path before doing real backprop.
pub fn select_leaf_with_vl(
    arena: &mut NodeArena,
    root_idx: usize,
    ucb_c: f32,
    vl: u32,
    visited_states: &HashSet<CompactState>,
) -> SmallVec<[usize; 32]> {
    let mut path: SmallVec<[usize; 32]> = SmallVec::new();
    let mut current = root_idx;
    loop {
        path.push(current);
        if arena.get(current).should_expand() {
            break;
        }
        current = select_child(arena, current, ucb_c, visited_states);
    }
    apply_virtual_loss(arena, &path, vl);
    path
}
```

- [ ] **Step 4: Run the test**

Run: `cd deep_quoridor/rust && cargo test --features binary --lib agents::alphazero::mcts::tests::test_select_leaf_with_vl_diversifies_concurrent_selections`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add deep_quoridor/rust/src/agents/alphazero/mcts.rs
git commit -m "Add select_leaf_with_vl: PUCT descent that applies virtual loss"
```

---

## Task 9: Add `promote_subtree` for tree reuse

**Files:**
- Modify: `deep_quoridor/rust/src/agents/alphazero/mcts.rs`

`promote_subtree` builds a new arena rooted at the chosen child, copying that subtree and dropping siblings.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn test_promote_subtree_keeps_only_chosen_branch() {
    let mech = QGameMechanics::new(5, 0, 200);
    let data = mech.create_initial_state();
    let mut arena = NodeArena::new(data);

    let mask = mech.get_action_mask_immut(data);
    let valid: Vec<usize> = mask.iter().enumerate()
        .filter_map(|(i, &v)| if v { Some(i) } else { None }).collect();
    assert!(valid.len() >= 2);

    let mut d1 = data;
    mech.apply_action_index(&mut d1, valid[0]);
    let mut d2 = data;
    mech.apply_action_index(&mut d2, valid[1]);
    let c1 = arena.alloc_child(0, valid[0], d1, 0.5);
    let c2 = arena.alloc_child(0, valid[1], d2, 0.5);
    arena.get_mut(0).children = vec![c1, c2];
    arena.get_mut(c1).visit_count = 7;
    arena.get_mut(c1).value_sum = 3.5;
    // Give c1 a grandchild so we can confirm it survives the promotion.
    let g = arena.alloc_child(c1, valid[0], d1, 0.5);
    arena.get_mut(c1).children = vec![g];
    arena.get_mut(g).visit_count = 4;

    let new_arena = promote_subtree(&arena, c1);

    // New arena has root + g (2 nodes).
    assert_eq!(new_arena.len(), 2);
    assert_eq!(new_arena.get(0).visit_count, 7);
    assert!((new_arena.get(0).value_sum - 3.5).abs() < 1e-9);
    assert!(new_arena.get(0).parent.is_none());
    assert_eq!(new_arena.get(0).children.len(), 1);
    let new_g = new_arena.get(0).children[0];
    assert_eq!(new_arena.get(new_g).visit_count, 4);
    assert_eq!(new_arena.get(new_g).parent, Some(0));
}
```

- [ ] **Step 2: Run to confirm it fails**

Run: `cd deep_quoridor/rust && cargo test --features binary --lib agents::alphazero::mcts::tests::test_promote_subtree_keeps_only_chosen_branch`
Expected: FAIL — `promote_subtree` undefined.

- [ ] **Step 3: Implement**

In `mcts.rs`, add:

```rust
/// Copy the subtree rooted at `new_root_idx` of `old_arena` into a fresh
/// `NodeArena`. The new arena's root is the copy of `new_root_idx`, with its
/// `parent` and `action_index` cleared. All descendants are copied; siblings
/// of `new_root_idx` (and their descendants) are dropped.
///
/// Statistics (`visit_count`, `value_sum`, `wins`, `losses`, `prior`,
/// `prior_clean`) carry over unchanged.
pub fn promote_subtree(old_arena: &NodeArena, new_root_idx: usize) -> NodeArena {
    let mut new_nodes: Vec<Node> = Vec::new();
    let mut idx_map: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();

    // BFS so parents always get assigned a new index before children.
    let mut queue: std::collections::VecDeque<usize> =
        std::collections::VecDeque::new();
    queue.push_back(new_root_idx);
    idx_map.insert(new_root_idx, 0);
    let old_root = old_arena.get(new_root_idx);
    new_nodes.push(Node {
        data: old_root.data,
        parent: None,
        action_index: None,
        children: Vec::new(),
        visit_count: old_root.visit_count,
        value_sum: old_root.value_sum,
        wins: old_root.wins,
        losses: old_root.losses,
        prior: old_root.prior,
        prior_clean: old_root.prior_clean,
    });

    while let Some(old_idx) = queue.pop_front() {
        let new_idx = idx_map[&old_idx];
        let old_node = old_arena.get(old_idx);
        let mut new_children: Vec<usize> = Vec::with_capacity(old_node.children.len());
        for &old_child in &old_node.children {
            let new_child_idx = new_nodes.len();
            idx_map.insert(old_child, new_child_idx);
            let c = old_arena.get(old_child);
            new_nodes.push(Node {
                data: c.data,
                parent: Some(new_idx),
                action_index: c.action_index,
                children: Vec::new(),
                visit_count: c.visit_count,
                value_sum: c.value_sum,
                wins: c.wins,
                losses: c.losses,
                prior: c.prior,
                prior_clean: c.prior_clean,
            });
            new_children.push(new_child_idx);
            queue.push_back(old_child);
        }
        new_nodes[new_idx].children = new_children;
    }

    NodeArena::from_nodes(new_nodes)
}
```

And add a `from_nodes` constructor to `NodeArena`:

```rust
impl NodeArena {
    /// Construct from a pre-filled `Vec<Node>` (used by `promote_subtree`).
    pub fn from_nodes(nodes: Vec<Node>) -> Self {
        Self { nodes }
    }
}
```

- [ ] **Step 4: Run the test**

Run: `cd deep_quoridor/rust && cargo test --features binary --lib agents::alphazero::mcts::tests::test_promote_subtree_keeps_only_chosen_branch`
Expected: PASS.

- [ ] **Step 5: Run the full mcts test module**

Run: `cd deep_quoridor/rust && cargo test --features binary --lib agents::alphazero::mcts`
Expected: all tests pass (previous tests untouched).

- [ ] **Step 6: Commit**

```bash
git add deep_quoridor/rust/src/agents/alphazero/mcts.rs
git commit -m "Add promote_subtree for tree reuse across moves"
```

---

## Task 10: Create new pipelined eval coordinator (`eval_pipeline.rs`)

**Files:**
- Create: `deep_quoridor/rust/src/agents/alphazero/eval_pipeline.rs`
- Modify: `deep_quoridor/rust/src/agents/alphazero/mod.rs`

The new pipeline lives alongside the existing `eval_coordinator.rs` for now; the old one will be deleted in Task 19.

- [ ] **Step 1: Create `eval_pipeline.rs`**

Create `deep_quoridor/rust/src/agents/alphazero/eval_pipeline.rs` with:

```rust
//! Pipelined eval coordinator: Batcher → Inference → Post-process.
//!
//! Game tasks send `EvalRequest`s over a tokio `mpsc`. A Batcher OS thread
//! pulls them (via `blocking_recv`), assembles batches up to
//! `eval_batch_size` (deadline-bounded by `eval_max_wait_ms` from the first
//! request), stacks features, and hands `BatchPayload` to an Inference OS
//! thread that owns the ORT session. Inference forwards `BatchOutputs` to a
//! Post-process OS thread that runs `finalize_policy` in parallel via rayon,
//! inserts to the cache, and fires each request's tokio oneshot.
//!
//! Control messages (`Reload(path)`, `Shutdown`) ride the front mpsc as
//! enum variants so ordering with batches is preserved.

use std::sync::mpsc::{sync_channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use dashmap::DashMap;
use ndarray::{Array4, Axis};
use ort::session::Session;
use rayon::prelude::*;
use tokio::sync::mpsc as tokio_mpsc;
use tokio::sync::oneshot;

use crate::agents::alphazero::evaluator::finalize_policy;
use crate::compact::q_bit_repr::CompactState;

/// Shared eval cache: maps a (compact) state to its (value, masked priors).
pub type EvalCache = DashMap<CompactState, EvalResult>;

#[derive(Clone, Debug)]
pub struct EvalResult {
    pub value: f32,
    pub priors: Vec<f32>,
}

/// A single eval request from a game task. The request owns its rotated
/// features + work_action_mask + un-rotation map; the coordinator just stacks
/// and runs them.
pub struct EvalRequest {
    pub state: CompactState,
    pub features: Array4<f32>,
    pub work_action_mask: Vec<bool>,
    pub rot_to_orig: Option<Vec<usize>>,
    pub responder: oneshot::Sender<Result<EvalResult>>,
}

/// Front-channel message: a regular request or a control directive.
pub enum FrontMsg {
    Req(EvalRequest),
    Reload(String),
    Shutdown,
}

#[derive(Debug, Clone, Copy)]
pub struct CoordinatorConfig {
    pub eval_batch_size: usize,
    pub eval_max_wait_ms: u64,
    /// 0 disables caching (lookups still cheap, inserts skipped).
    pub eval_cache_max_size: usize,
}

/// Internal pipeline messages between stages.
struct BatchPayload {
    stacked: Array4<f32>,
    reqs: Vec<EvalRequest>,
}

enum InferenceIn {
    Batch(BatchPayload),
    Reload(String),
    Shutdown,
}

struct BatchOutputs {
    values: Vec<f32>,
    policy: Vec<f32>,
    policy_size: usize,
    reqs: Vec<EvalRequest>,
}

enum PostIn {
    Outputs(BatchOutputs),
    Reload, // forwarded marker, post-process doesn't need the path
    Shutdown,
}

/// Open an ONNX `Session` from a file path.
pub fn load_session(model_path: &str) -> Result<Session> {
    Session::builder()
        .context("Failed to create ONNX session builder")?
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to load ONNX model from {}", model_path))
}

/// Spawn the three-stage pipeline. Returns join handles for graceful shutdown.
pub struct CoordinatorHandles {
    pub batcher: thread::JoinHandle<()>,
    pub inference: thread::JoinHandle<()>,
    pub post: thread::JoinHandle<()>,
}

pub fn spawn_coordinator(
    initial_session: Session,
    cache: Arc<EvalCache>,
    config: CoordinatorConfig,
    front_rx: tokio_mpsc::Receiver<FrontMsg>,
) -> CoordinatorHandles {
    let (inf_tx, inf_rx) = sync_channel::<InferenceIn>(1);
    let (post_tx, post_rx) = sync_channel::<PostIn>(1);

    let batcher = thread::Builder::new()
        .name("eval-batcher".to_string())
        .spawn(move || run_batcher(front_rx, inf_tx, config))
        .expect("spawn batcher");
    let inference = thread::Builder::new()
        .name("eval-inference".to_string())
        .spawn({
            let cache = Arc::clone(&cache);
            move || run_inference(initial_session, cache, inf_rx, post_tx)
        })
        .expect("spawn inference");
    let post = thread::Builder::new()
        .name("eval-post".to_string())
        .spawn(move || run_postprocess(cache, config.eval_cache_max_size, post_rx))
        .expect("spawn post");

    CoordinatorHandles { batcher, inference, post }
}

fn run_batcher(
    mut front_rx: tokio_mpsc::Receiver<FrontMsg>,
    inf_tx: std::sync::mpsc::SyncSender<InferenceIn>,
    config: CoordinatorConfig,
) {
    let batch_size = config.eval_batch_size.max(1);
    let max_wait = Duration::from_millis(config.eval_max_wait_ms);

    loop {
        // Block on first message.
        let first = match front_rx.blocking_recv() {
            Some(m) => m,
            None => break, // channel closed; drain done
        };
        let first_req = match first {
            FrontMsg::Req(r) => r,
            FrontMsg::Reload(p) => {
                let _ = inf_tx.send(InferenceIn::Reload(p));
                continue;
            }
            FrontMsg::Shutdown => {
                let _ = inf_tx.send(InferenceIn::Shutdown);
                break;
            }
        };

        let mut reqs: Vec<EvalRequest> = Vec::with_capacity(batch_size);
        reqs.push(first_req);

        // Fill batch up to `batch_size`, deadline = first arrival + max_wait.
        let deadline = Instant::now() + max_wait;
        while reqs.len() < batch_size {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            // Use try_recv-with-yield pattern: non-blocking try, sleep briefly,
            // re-check. tokio mpsc's blocking_recv has no deadline variant we can use here.
            match front_rx.try_recv() {
                Ok(FrontMsg::Req(r)) => reqs.push(r),
                Ok(FrontMsg::Reload(p)) => {
                    // flush current batch first, then forward reload
                    flush_batch(&inf_tx, std::mem::take(&mut reqs));
                    let _ = inf_tx.send(InferenceIn::Reload(p));
                    break;
                }
                Ok(FrontMsg::Shutdown) => {
                    flush_batch(&inf_tx, std::mem::take(&mut reqs));
                    let _ = inf_tx.send(InferenceIn::Shutdown);
                    return;
                }
                Err(tokio_mpsc::error::TryRecvError::Empty) => {
                    thread::sleep(Duration::from_micros(50));
                }
                Err(tokio_mpsc::error::TryRecvError::Disconnected) => break,
            }
        }

        if !reqs.is_empty() {
            flush_batch(&inf_tx, reqs);
        }
    }
}

fn flush_batch(
    inf_tx: &std::sync::mpsc::SyncSender<InferenceIn>,
    reqs: Vec<EvalRequest>,
) {
    if reqs.is_empty() {
        return;
    }
    let views: Vec<_> = reqs.iter().map(|r| r.features.view()).collect();
    let stacked = match ndarray::concatenate(Axis(0), &views) {
        Ok(a) => a,
        Err(e) => {
            let msg = format!("Failed to concatenate features: {}", e);
            for r in reqs {
                let _ = r.responder.send(Err(anyhow::anyhow!(msg.clone())));
            }
            return;
        }
    };
    let payload = BatchPayload { stacked, reqs };
    let _ = inf_tx.send(InferenceIn::Batch(payload));
}

fn run_inference(
    mut session: Session,
    cache: Arc<EvalCache>,
    inf_rx: Receiver<InferenceIn>,
    post_tx: std::sync::mpsc::SyncSender<PostIn>,
) {
    while let Ok(msg) = inf_rx.recv() {
        match msg {
            InferenceIn::Batch(BatchPayload { stacked, reqs }) => {
                let shape = stacked.shape().to_vec();
                let input_data: Vec<f32> = stacked.iter().copied().collect();
                let batch_len = reqs.len();
                let input_value = match ort::value::Value::from_array(
                    (shape.as_slice(), input_data),
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        let msg = format!("Failed to build ONNX input: {}", e);
                        for r in reqs {
                            let _ = r.responder.send(Err(anyhow::anyhow!(msg.clone())));
                        }
                        continue;
                    }
                };
                let outputs = match session.run(ort::inputs!["input" => input_value]) {
                    Ok(o) => o,
                    Err(e) => {
                        let msg = format!("Failed to run ONNX inference: {}", e);
                        for r in reqs {
                            let _ = r.responder.send(Err(anyhow::anyhow!(msg.clone())));
                        }
                        continue;
                    }
                };
                let value_tensor = match outputs["value"].try_extract_tensor::<f32>() {
                    Ok(t) => t,
                    Err(e) => {
                        let msg = format!("Failed to extract value tensor: {}", e);
                        for r in reqs {
                            let _ = r.responder.send(Err(anyhow::anyhow!(msg.clone())));
                        }
                        continue;
                    }
                };
                let policy_tensor = match outputs["policy_logits"].try_extract_tensor::<f32>() {
                    Ok(t) => t,
                    Err(e) => {
                        let msg = format!("Failed to extract policy logits: {}", e);
                        for r in reqs {
                            let _ = r.responder.send(Err(anyhow::anyhow!(msg.clone())));
                        }
                        continue;
                    }
                };
                let values: Vec<f32> = value_tensor.1.to_vec();
                let policy: Vec<f32> = policy_tensor.1.to_vec();
                let policy_size = policy.len() / batch_len;
                let outputs = BatchOutputs { values, policy, policy_size, reqs };
                let _ = post_tx.send(PostIn::Outputs(outputs));
            }
            InferenceIn::Reload(path) => match load_session(&path) {
                Ok(s) => {
                    session = s;
                    cache.clear();
                    eprintln!("eval-pipeline: loaded model {} (cache cleared)", path);
                    let _ = post_tx.send(PostIn::Reload);
                }
                Err(e) => eprintln!("eval-pipeline: failed to load model {}: {:#}", path, e),
            },
            InferenceIn::Shutdown => {
                let _ = post_tx.send(PostIn::Shutdown);
                return;
            }
        }
    }
}

fn run_postprocess(
    cache: Arc<EvalCache>,
    cache_max: usize,
    post_rx: Receiver<PostIn>,
) {
    while let Ok(msg) = post_rx.recv() {
        match msg {
            PostIn::Outputs(out) => {
                let BatchOutputs { values, policy, policy_size, reqs } = out;
                // Parallel finalize over the request batch.
                let finalized: Vec<EvalResult> = reqs
                    .par_iter()
                    .enumerate()
                    .map(|(i, req)| {
                        let logits = &policy[i * policy_size..(i + 1) * policy_size];
                        let priors = finalize_policy(
                            logits,
                            &req.work_action_mask,
                            req.rot_to_orig.as_deref(),
                        );
                        EvalResult { value: values[i], priors }
                    })
                    .collect();
                // Insert into cache (serial — DashMap is sharded internally, parallel
                // inserts have contention; serial is fine here).
                for (req, res) in reqs.iter().zip(finalized.iter()) {
                    if cache_max > 0 && cache.len() < cache_max {
                        cache.insert(req.state, res.clone());
                    }
                }
                // Fire oneshots.
                for (req, res) in reqs.into_iter().zip(finalized.into_iter()) {
                    let _ = req.responder.send(Ok(res));
                }
            }
            PostIn::Reload => continue,
            PostIn::Shutdown => return,
        }
    }
}
```

- [ ] **Step 2: Add the module export**

In `deep_quoridor/rust/src/agents/alphazero/mod.rs`, add:

```rust
pub mod eval_pipeline;
```

- [ ] **Step 3: Verify it compiles**

Run: `cd deep_quoridor/rust && cargo check --features binary --bin selfplay`
Expected: compiles with no errors (warnings about unused code in `eval_pipeline.rs` are OK and will be resolved when it's wired up).

- [ ] **Step 4: Commit**

```bash
git add deep_quoridor/rust/src/agents/alphazero/eval_pipeline.rs deep_quoridor/rust/src/agents/alphazero/mod.rs
git commit -m "Add pipelined eval coordinator: Batcher -> Inference -> Post-process

Three dedicated OS threads connected by capacity-1 sync_channels, fronted
by a tokio mpsc. finalize_policy is parallelized via rayon in the
post-process stage. Old eval_coordinator.rs remains until cleanup task."
```

---

## Task 11: Unit-test the post-process stage (finalize parity)

**Files:**
- Modify: `deep_quoridor/rust/src/agents/alphazero/eval_pipeline.rs`

Verify the parallel finalize produces the same outputs as a serial finalize over the same inputs.

- [ ] **Step 1: Add a tests module to `eval_pipeline.rs`**

At the bottom of `eval_pipeline.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;
    use tokio::sync::oneshot;

    fn make_request(policy_size: usize, board_size: i32) -> (EvalRequest, oneshot::Receiver<Result<EvalResult>>) {
        let m = (board_size * 2 + 3) as usize;
        let features = Array4::<f32>::zeros((1, 5, m, m));
        let mask = vec![true; policy_size];
        let (tx, rx) = oneshot::channel();
        let req = EvalRequest {
            state: CompactState::default(),
            features,
            work_action_mask: mask,
            rot_to_orig: None,
            responder: tx,
        };
        (req, rx)
    }

    #[test]
    fn test_postprocess_parallel_matches_serial_finalize() {
        // Synthesise a batch and feed it directly through run_postprocess via a
        // hand-driven channel pair; then compare against a serial reference.
        let cache = Arc::new(EvalCache::new());
        let (post_tx, post_rx) = sync_channel::<PostIn>(1);

        let batch_len = 8usize;
        let policy_size = 10usize;
        let mut reqs = Vec::with_capacity(batch_len);
        let mut rxs = Vec::with_capacity(batch_len);
        for _ in 0..batch_len {
            let (r, rx) = make_request(policy_size, 5);
            reqs.push(r);
            rxs.push(rx);
        }
        let values: Vec<f32> = (0..batch_len).map(|i| i as f32 * 0.1 - 0.4).collect();
        let policy: Vec<f32> = (0..batch_len * policy_size).map(|i| (i as f32).sin()).collect();

        // Expected from serial loop using finalize_policy directly.
        let mut expected: Vec<Vec<f32>> = Vec::with_capacity(batch_len);
        for i in 0..batch_len {
            let logits = &policy[i * policy_size..(i + 1) * policy_size];
            expected.push(finalize_policy(logits, &vec![true; policy_size], None));
        }

        post_tx.send(PostIn::Outputs(BatchOutputs {
            values: values.clone(),
            policy: policy.clone(),
            policy_size,
            reqs,
        })).unwrap();
        post_tx.send(PostIn::Shutdown).unwrap();

        // Drive the post stage on this thread.
        run_postprocess(Arc::clone(&cache), 1024, post_rx);

        for (i, rx) in rxs.into_iter().enumerate() {
            let res = rx.blocking_recv().unwrap().unwrap();
            assert!((res.value - values[i]).abs() < 1e-6);
            for (p, e) in res.priors.iter().zip(expected[i].iter()) {
                assert!((p - e).abs() < 1e-6);
            }
        }
    }
}
```

- [ ] **Step 2: Run the test**

Run: `cd deep_quoridor/rust && cargo test --features binary --lib agents::alphazero::eval_pipeline`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add deep_quoridor/rust/src/agents/alphazero/eval_pipeline.rs
git commit -m "Test post-process stage: rayon finalize matches serial reference"
```

---

## Task 12: Create `selfplay_mcts.rs` with `LeafParallelMCTS::search`

**Files:**
- Create: `deep_quoridor/rust/src/agents/alphazero/selfplay_mcts.rs`
- Modify: `deep_quoridor/rust/src/agents/alphazero/mod.rs`

This is the async leaf-parallel MCTS. It owns an arena and a per-board-size rotation-mapping cache. Tree reuse and model-version tracking come in Task 14.

- [ ] **Step 1: Create the module skeleton**

Create `deep_quoridor/rust/src/agents/alphazero/selfplay_mcts.rs`:

```rust
//! Async leaf-parallel MCTS for self-play.
//!
//! Each MCTS instance owns its arena and dispatches K eval requests per outer
//! iteration through the shared eval pipeline. Virtual loss is applied during
//! descent and undone before real backprop.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use futures::future::try_join_all;
use smallvec::SmallVec;
use tokio::sync::mpsc as tokio_mpsc;
use tokio::sync::oneshot;

use crate::agents::alphazero::eval_pipeline::{EvalCache, EvalRequest, EvalResult, FrontMsg};
use crate::agents::alphazero::evaluator::prepare_eval_input;
use crate::agents::alphazero::mcts::{
    apply_dirichlet_noise_to_root_children, backpropagate, backpropagate_result, expand_node,
    promote_subtree, select_leaf_with_vl, undo_virtual_loss, ChildInfo, MCTSConfig, Node,
    NodeArena,
};
use crate::compact::q_bit_repr::CompactState;
use crate::compact::q_game_mechanics::QGameMechanics;

/// Tunables specific to leaf-parallel batched MCTS.
#[derive(Debug, Clone, Copy)]
pub struct LeafParallelConfig {
    pub leaf_parallelism: u32,
    pub virtual_loss: u32,
    pub enable_tree_reuse: bool,
}

/// One LeafParallelMCTS per game agent. Lives across moves so tree reuse can
/// preserve the subtree of the chosen child.
pub struct LeafParallelMCTS {
    pub cfg: MCTSConfig,
    pub lp: LeafParallelConfig,
    sender: tokio_mpsc::Sender<FrontMsg>,
    cache: Arc<EvalCache>,
    rotation_mappings: std::collections::HashMap<i32, (Vec<usize>, Vec<usize>)>,
    /// Persistent arena across moves when tree reuse is enabled. None means
    /// the next search should start from a fresh arena.
    arena: Option<NodeArena>,
}

impl LeafParallelMCTS {
    pub fn new(
        cfg: MCTSConfig,
        lp: LeafParallelConfig,
        sender: tokio_mpsc::Sender<FrontMsg>,
        cache: Arc<EvalCache>,
    ) -> Self {
        Self {
            cfg,
            lp,
            sender,
            cache,
            rotation_mappings: std::collections::HashMap::new(),
            arena: None,
        }
    }

    /// Discard any retained tree. Call between games or on model reload.
    pub fn reset_tree(&mut self) {
        self.arena = None;
    }

    /// After the caller picks `action_idx` at the root, promote that child's
    /// subtree to the new root for the next search.
    pub fn advance_root(&mut self, action_idx: usize) {
        if !self.lp.enable_tree_reuse {
            self.arena = None;
            return;
        }
        let Some(old) = self.arena.take() else { return; };
        let root = old.get(0);
        let chosen = root.children.iter().copied().find(|&c| {
            old.get(c).action_index == Some(action_idx)
        });
        self.arena = chosen.map(|c| promote_subtree(&old, c));
    }

    /// Run one search starting from `root_data`. Returns `(children, root_value)`.
    pub async fn search(
        &mut self,
        root_data: CompactState,
        mechanics: &QGameMechanics,
        visited_states: &HashSet<CompactState>,
    ) -> Result<(Vec<ChildInfo>, f32)> {
        // Fresh arena unless we have a reusable one matching the root state.
        let mut arena = match self.arena.take() {
            Some(a) if a.get(0).data == root_data => a,
            _ => NodeArena::new(root_data),
        };

        // If root has no children, expand it via one eval first.
        let action_mask = mechanics.get_action_mask_immut(root_data);
        let root_value;
        if arena.get(0).children.is_empty() {
            let (v, priors) = self.evaluate_once(root_data, mechanics, &action_mask).await?;
            expand_node(&mut arena, 0, &priors, mechanics);
            root_value = v;
        } else {
            // Tree-reuse path: synthesise root_value from existing stats.
            let r = arena.get(0);
            root_value = if r.visit_count > 0 {
                -(r.value_sum / r.visit_count as f64) as f32
            } else {
                0.0
            };
        }

        // (Re-)apply Dirichlet noise to the new root's children.
        if self.cfg.noise_epsilon > 0.0 {
            let num_valid = action_mask.iter().filter(|&&m| m).count();
            let alpha = self.cfg.noise_alpha.unwrap_or_else(|| 10.0 / num_valid.max(1) as f32);
            apply_dirichlet_noise_to_root_children(&mut arena, 0, self.cfg.noise_epsilon, alpha);
        }

        let total = self
            .cfg
            .n
            .unwrap_or_else(|| self.cfg.k.unwrap_or(10) * action_mask.iter().filter(|&&m| m).count() as u32);

        let mut iters_done: u32 = 0;
        let k = self.lp.leaf_parallelism.max(1);
        let vl = self.lp.virtual_loss;

        while iters_done < total {
            let outer = ((total - iters_done) as u32).min(k);

            // Selection phase: pick `outer` leaves; classify each as Terminal, Hit, or Miss.
            enum Item {
                Terminal { path: SmallVec<[usize; 32]>, value: f64 },
                Hit { path: SmallVec<[usize; 32]>, leaf_idx: usize, result: EvalResult },
                Miss {
                    path: SmallVec<[usize; 32]>,
                    leaf_idx: usize,
                    rx: oneshot::Receiver<Result<EvalResult>>,
                },
            }
            let mut items: Vec<Item> = Vec::with_capacity(outer as usize);
            let mut to_send: Vec<(usize, EvalRequest)> = Vec::new();

            for _ in 0..outer {
                let path = select_leaf_with_vl(&mut arena, 0, self.cfg.ucb_c, vl, visited_states);
                let leaf_idx = *path.last().unwrap();
                let leaf_data = arena.get(leaf_idx).data;

                // Terminal?
                if mechanics.is_game_over(leaf_data) {
                    let v = if mechanics.winner(leaf_data).is_some() { 1.0 } else { 0.0 };
                    items.push(Item::Terminal { path, value: v });
                    continue;
                }
                if let Some(max) = self.cfg.max_steps {
                    if mechanics.repr().get_completed_steps(leaf_data) >= max as usize {
                        items.push(Item::Terminal { path, value: 0.0 });
                        continue;
                    }
                }

                // Cache hit?
                if let Some(entry) = self.cache.get(&leaf_data) {
                    items.push(Item::Hit {
                        path,
                        leaf_idx,
                        result: EvalResult {
                            value: entry.value,
                            priors: entry.priors.clone(),
                        },
                    });
                    continue;
                }

                // Miss: build features in this task and queue a send.
                let leaf_mask = mechanics.get_action_mask_immut(leaf_data);
                let prep = prepare_eval_input(mechanics, leaf_data, &leaf_mask, &mut self.rotation_mappings);
                let (tx, rx) = oneshot::channel();
                let req = EvalRequest {
                    state: leaf_data,
                    features: prep.features,
                    work_action_mask: prep.work_action_mask,
                    rot_to_orig: prep.rot_to_orig,
                    responder: tx,
                };
                to_send.push((items.len(), req));
                // Placeholder; rx assigned after this loop so we can borrow mutably below.
                items.push(Item::Miss { path, leaf_idx, rx });
            }

            // Send all miss requests.
            for (_, req) in to_send {
                // Sender is async; await is OK here.
                self.sender.send(FrontMsg::Req(req)).await.map_err(|_| {
                    anyhow::anyhow!("eval pipeline front channel closed")
                })?;
            }

            // Process items in selection order. For Misses, await the oneshot.
            for item in items {
                match item {
                    Item::Terminal { path, value } => {
                        undo_virtual_loss(&mut arena, &path, vl);
                        let leaf = *path.last().unwrap();
                        backpropagate_result(&mut arena, leaf, value);
                        iters_done += 1;
                    }
                    Item::Hit { path, leaf_idx, result } => {
                        undo_virtual_loss(&mut arena, &path, vl);
                        expand_node(&mut arena, leaf_idx, &result.priors, mechanics);
                        backpropagate(&mut arena, leaf_idx, -result.value as f64);
                        iters_done += 1;
                    }
                    Item::Miss { path, leaf_idx, rx } => {
                        let result = rx.await
                            .map_err(|_| anyhow::anyhow!("eval responder dropped"))??;
                        undo_virtual_loss(&mut arena, &path, vl);
                        expand_node(&mut arena, leaf_idx, &result.priors, mechanics);
                        backpropagate(&mut arena, leaf_idx, -result.value as f64);
                        iters_done += 1;
                    }
                }
            }
        }

        // Extract children info.
        let bs = mechanics.repr().board_size() as i32;
        let root = arena.get(0);
        let computed_root_value = if root.visit_count > 0 {
            -(root.value_sum / root.visit_count as f64) as f32
        } else {
            root_value
        };
        let children: Vec<ChildInfo> = root.children.iter().map(|&ci| {
            let c = arena.get(ci);
            let ai = c.action_index.expect("child node must have action_index");
            ChildInfo {
                action: crate::actions::action_index_to_action(bs, ai),
                action_index: ai,
                visit_count: c.visit_count,
            }
        }).collect();

        // Stash the arena for tree reuse on the next call.
        self.arena = Some(arena);

        Ok((children, computed_root_value))
    }

    /// Run a single eval through the pipeline (used to seed root expansion).
    async fn evaluate_once(
        &mut self,
        data: CompactState,
        mechanics: &QGameMechanics,
        action_mask: &[bool],
    ) -> Result<(f32, Vec<f32>)> {
        if let Some(entry) = self.cache.get(&data) {
            return Ok((entry.value, entry.priors.clone()));
        }
        let prep = prepare_eval_input(mechanics, data, action_mask, &mut self.rotation_mappings);
        let (tx, rx) = oneshot::channel();
        let req = EvalRequest {
            state: data,
            features: prep.features,
            work_action_mask: prep.work_action_mask,
            rot_to_orig: prep.rot_to_orig,
            responder: tx,
        };
        self.sender.send(FrontMsg::Req(req)).await.map_err(|_| {
            anyhow::anyhow!("eval pipeline front channel closed")
        })?;
        let res = rx.await.map_err(|_| anyhow::anyhow!("responder dropped"))??;
        Ok((res.value, res.priors))
    }
}

// Silence unused-warning during incremental build.
#[allow(dead_code)]
fn _force_use(_: Node, _: ChildInfo) {}
```

- [ ] **Step 2: Export the new module**

In `deep_quoridor/rust/src/agents/alphazero/mod.rs`, add:

```rust
pub mod selfplay_mcts;
```

- [ ] **Step 3: Verify it compiles**

Run: `cd deep_quoridor/rust && cargo check --features binary --bin selfplay`
Expected: compiles. There will be warnings about unused items in selfplay_mcts.rs until it is wired in; those are acceptable for now.

- [ ] **Step 4: Commit**

```bash
git add deep_quoridor/rust/src/agents/alphazero/selfplay_mcts.rs deep_quoridor/rust/src/agents/alphazero/mod.rs
git commit -m "Add LeafParallelMCTS: async leaf-parallel search with virtual loss

Owns its arena across calls so subsequent task can enable tree reuse.
Each outer iteration selects up to K leaves with virtual loss, classifies
them as Terminal/Hit/Miss, sends miss requests to the eval pipeline, then
awaits in selection order while undoing VL + backpropping real results."
```

---

## Task 13: Integration test — `LeafParallelMCTS` with K=1, vl=0 ≈ sequential MCTS

**Files:**
- Modify: `deep_quoridor/rust/src/agents/alphazero/selfplay_mcts.rs`

We exercise the full async pipeline end-to-end with a stub "inference" thread that does deterministic uniform priors, no GPU required. With K=1, vl=0, and `enable_tree_reuse=false`, the leaf-parallel path should produce the same total visit count and the same set of children as the existing sync `search` would on the same state and config.

- [ ] **Step 1: Add the integration test**

Append to the `#[cfg(test)] mod tests` of `selfplay_mcts.rs` (create the module if absent):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::alphazero::eval_pipeline::EvalCache;
    use crate::compact::q_bit_repr::CompactState;
    use crate::compact::q_game_mechanics::QGameMechanics;
    use std::sync::Arc;
    use tokio::sync::mpsc as tokio_mpsc;

    /// Stub coordinator: replies to every request with uniform priors over the
    /// valid actions in the request's mask, value=0.
    fn spawn_stub_coordinator(
        mut rx: tokio_mpsc::Receiver<FrontMsg>,
        cache: Arc<EvalCache>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                match msg {
                    FrontMsg::Req(req) => {
                        let n_valid = req.work_action_mask.iter().filter(|&&v| v).count();
                        let p = if n_valid > 0 { 1.0 / n_valid as f32 } else { 0.0 };
                        let mut priors = vec![0.0f32; req.work_action_mask.len()];
                        for (i, &v) in req.work_action_mask.iter().enumerate() {
                            if v { priors[i] = p; }
                        }
                        // If rot_to_orig is provided, undo rotation.
                        let priors = match req.rot_to_orig.as_ref() {
                            Some(map) => crate::rotation::remap_policy(&priors, map),
                            None => priors,
                        };
                        let res = EvalResult { value: 0.0, priors };
                        let _ = cache.insert(req.state, res.clone());
                        let _ = req.responder.send(Ok(res));
                    }
                    FrontMsg::Reload(_) | FrontMsg::Shutdown => break,
                }
            }
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_leaf_parallel_k1_matches_sequential_visit_total() {
        let mech = QGameMechanics::new(5, 0, 200);
        let data = mech.create_initial_state();
        let cache = Arc::new(EvalCache::new());
        let (tx, rx) = tokio_mpsc::channel::<FrontMsg>(64);
        let stub = spawn_stub_coordinator(rx, Arc::clone(&cache));

        let mcts_cfg = MCTSConfig { n: Some(20), ucb_c: 1.4, noise_epsilon: 0.0, ..Default::default() };
        let lp_cfg = LeafParallelConfig { leaf_parallelism: 1, virtual_loss: 0, enable_tree_reuse: false };
        let mut mcts = LeafParallelMCTS::new(mcts_cfg.clone(), lp_cfg, tx.clone(), Arc::clone(&cache));

        let visited = std::collections::HashSet::new();
        let (children, _root_value) = mcts.search(data, &mech, &visited).await.unwrap();
        let total: u32 = children.iter().map(|c| c.visit_count).sum();
        // Should be exactly n iterations of MCTS expansion under the root.
        assert!(total >= 20, "expected ≥20 child visits, got {}", total);

        drop(tx);
        let _ = stub.await;
    }
}
```

Also add `tokio = { version = "1", features = ["rt-multi-thread", "sync", "macros", "test-util"] }` as a `dev-dependency` (or include `test-util` in the existing tokio feature list under `binary`). For dev-only test infrastructure, add to `[dev-dependencies]` in `Cargo.toml`:

```toml
tokio = { version = "1", features = ["rt-multi-thread", "sync", "macros", "time"] }
```

(This duplicates the optional-binary tokio dep so tests can use `#[tokio::test]` without the binary feature.)

- [ ] **Step 2: Run the test**

Run: `cd deep_quoridor/rust && cargo test --features binary --lib agents::alphazero::selfplay_mcts`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add deep_quoridor/rust/src/agents/alphazero/selfplay_mcts.rs deep_quoridor/rust/Cargo.toml
git commit -m "Test LeafParallelMCTS with stub coordinator: K=1 visit total"
```

---

## Task 14: Add tree reuse + diversification integration tests

**Files:**
- Modify: `deep_quoridor/rust/src/agents/alphazero/selfplay_mcts.rs`

- [ ] **Step 1: Add a diversification test (K=8, vl=3)**

In the `tests` module of `selfplay_mcts.rs`:

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_leaf_parallel_k8_diversifies_children() {
    let mech = QGameMechanics::new(5, 0, 200);
    let data = mech.create_initial_state();
    let cache = Arc::new(EvalCache::new());
    let (tx, rx) = tokio_mpsc::channel::<FrontMsg>(128);
    let stub = spawn_stub_coordinator(rx, Arc::clone(&cache));

    let mcts_cfg = MCTSConfig { n: Some(64), ucb_c: 1.4, noise_epsilon: 0.0, ..Default::default() };
    let lp_cfg = LeafParallelConfig { leaf_parallelism: 8, virtual_loss: 3, enable_tree_reuse: false };
    let mut mcts = LeafParallelMCTS::new(mcts_cfg, lp_cfg, tx.clone(), Arc::clone(&cache));

    let visited = std::collections::HashSet::new();
    let (children, _) = mcts.search(data, &mech, &visited).await.unwrap();
    let visited_top: u32 = children.iter().filter(|c| c.visit_count > 0).count() as u32;
    assert!(visited_top >= 2, "K=8 vl=3 should spread visits across ≥2 root children");

    drop(tx);
    let _ = stub.await;
}
```

- [ ] **Step 2: Add a tree-reuse test**

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_tree_reuse_promotes_subtree_between_moves() {
    let mech = QGameMechanics::new(5, 0, 200);
    let data = mech.create_initial_state();
    let cache = Arc::new(EvalCache::new());
    let (tx, rx) = tokio_mpsc::channel::<FrontMsg>(128);
    let stub = spawn_stub_coordinator(rx, Arc::clone(&cache));

    let mcts_cfg = MCTSConfig { n: Some(32), ucb_c: 1.4, noise_epsilon: 0.0, ..Default::default() };
    let lp_cfg = LeafParallelConfig { leaf_parallelism: 4, virtual_loss: 1, enable_tree_reuse: true };
    let mut mcts = LeafParallelMCTS::new(mcts_cfg, lp_cfg, tx.clone(), Arc::clone(&cache));

    let visited = std::collections::HashSet::new();
    // First search at the initial state.
    let (children_1, _) = mcts.search(data, &mech, &visited).await.unwrap();
    let chosen = children_1.iter().max_by_key(|c| c.visit_count).unwrap();

    // Advance root and search again from the resulting state.
    let mut next_state = data;
    mech.apply_action_index(&mut next_state, chosen.action_index);
    mcts.advance_root(chosen.action_index);
    let (children_2, _) = mcts.search(next_state, &mech, &visited).await.unwrap();
    assert!(!children_2.is_empty());

    drop(tx);
    let _ = stub.await;
}
```

- [ ] **Step 3: Run all selfplay_mcts tests**

Run: `cd deep_quoridor/rust && cargo test --features binary --lib agents::alphazero::selfplay_mcts`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add deep_quoridor/rust/src/agents/alphazero/selfplay_mcts.rs
git commit -m "Test leaf-parallel diversification (K=8 vl=3) and tree reuse"
```

---

## Task 15: Add model-version tracking to `LeafParallelMCTS`

**Files:**
- Modify: `deep_quoridor/rust/src/agents/alphazero/selfplay_mcts.rs`

When the active model version changes mid-game, the retained tree's value estimates were produced by the old network and should be discarded. We add a `current_model_version: i64` field and a helper `note_model_version(v)` that resets the tree when `v` differs from the last seen version.

- [ ] **Step 1: Extend `LeafParallelMCTS`**

In `selfplay_mcts.rs`, add a field and method:

```rust
pub struct LeafParallelMCTS {
    // ...existing fields...
    last_model_version: Option<i64>,
}
```

Update `new` to initialize `last_model_version: None`, and add:

```rust
impl LeafParallelMCTS {
    /// Inform the MCTS of the current model version. If it has changed since
    /// the last call, the retained tree is discarded (its values were from
    /// the previous network).
    pub fn note_model_version(&mut self, v: i64) {
        match self.last_model_version {
            Some(prev) if prev == v => {}
            _ => {
                self.arena = None;
                self.last_model_version = Some(v);
            }
        }
    }
}
```

- [ ] **Step 2: Add a unit test**

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_note_model_version_clears_tree_on_change() {
    let mech = QGameMechanics::new(5, 0, 200);
    let data = mech.create_initial_state();
    let cache = Arc::new(EvalCache::new());
    let (tx, rx) = tokio_mpsc::channel::<FrontMsg>(64);
    let stub = spawn_stub_coordinator(rx, Arc::clone(&cache));

    let mcts_cfg = MCTSConfig { n: Some(8), ucb_c: 1.4, noise_epsilon: 0.0, ..Default::default() };
    let lp_cfg = LeafParallelConfig { leaf_parallelism: 2, virtual_loss: 1, enable_tree_reuse: true };
    let mut mcts = LeafParallelMCTS::new(mcts_cfg, lp_cfg, tx.clone(), Arc::clone(&cache));

    mcts.note_model_version(1);
    let visited = std::collections::HashSet::new();
    let _ = mcts.search(data, &mech, &visited).await.unwrap();
    assert!(mcts.arena.is_some());

    mcts.note_model_version(2);
    assert!(mcts.arena.is_none(), "tree should be cleared on version change");

    drop(tx);
    let _ = stub.await;
}
```

- [ ] **Step 3: Run the test**

Run: `cd deep_quoridor/rust && cargo test --features binary --lib agents::alphazero::selfplay_mcts::tests::test_note_model_version_clears_tree_on_change`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add deep_quoridor/rust/src/agents/alphazero/selfplay_mcts.rs
git commit -m "LeafParallelMCTS: clear tree on model version change"
```

---

## Task 16: Create async game runner `selfplay_game.rs`

**Files:**
- Create: `deep_quoridor/rust/src/agents/alphazero/selfplay_game.rs`
- Modify: `deep_quoridor/rust/src/agents/alphazero/mod.rs`

Mirrors the existing `play_game` but works with two `LeafParallelMCTS` instances and is async. Produces the same `GameResult` so the existing replay writer is unchanged.

- [ ] **Step 1: Create the module**

Create `deep_quoridor/rust/src/agents/alphazero/selfplay_game.rs`:

```rust
//! Async self-play game runner.
//!
//! Plays a complete game between two `LeafParallelMCTS` agents (or one MCTS +
//! one "random" baseline). The replay buffer mirrors `game_runner::play_game`'s
//! current-player-downward storage so downstream training is unchanged.

use std::collections::HashSet;

use anyhow::Result;
use ndarray::Axis;

use crate::actions::action_index_to_action;
use crate::agents::alphazero::selfplay_mcts::LeafParallelMCTS;
use crate::compact::q_bit_repr::CompactState;
use crate::compact::q_game_mechanics::QGameMechanics;
use crate::game_runner::{GameResult, ReplayBufferItem};
use crate::grid_helpers::compact_state_to_resnet_input;
use crate::rotation::{create_rotation_mapping, remap_mask, remap_policy, rotate_compact_state};

/// P2 agent variants. Today only AlphaZero (same as P1) or `Random` are wired up.
pub enum P2 {
    AlphaZero(LeafParallelMCTS),
    Random,
}

/// Apply temperature-and-sample to visit counts. Same semantics as
/// `agent::apply_temperature_and_sample`; duplicated here so we don't depend on
/// the sync `AlphaZeroAgent` plumbing.
fn sample_action(
    visit_counts: &[u32],
    action_indices: &[usize],
    temperature: f32,
    deterministic: bool,
) -> usize {
    use rand::Rng;
    assert!(!visit_counts.is_empty() && !action_indices.is_empty());
    if temperature == 0.0 {
        let max_v = visit_counts.iter().max().copied().unwrap_or(0);
        let tied: Vec<usize> = visit_counts.iter().enumerate()
            .filter_map(|(i, &v)| if v == max_v { Some(i) } else { None })
            .collect();
        if deterministic {
            return action_indices[tied[0]];
        }
        let mut rng = rand::thread_rng();
        return action_indices[tied[rng.gen_range(0..tied.len())]];
    }
    let total: f64 = visit_counts.iter().map(|&v| (v as f64).powf(1.0 / temperature as f64)).sum();
    if total <= 0.0 {
        return action_indices[0];
    }
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen();
    let mut cum = 0.0;
    for (i, &v) in visit_counts.iter().enumerate() {
        cum += (v as f64).powf(1.0 / temperature as f64) / total;
        if r < cum {
            return action_indices[i];
        }
    }
    action_indices[action_indices.len() - 1]
}

/// Per-game settings for action selection.
#[derive(Debug, Clone, Copy)]
pub struct GameSettings {
    pub temperature: f32,
    pub drop_t_on_step: Option<usize>,
    pub deterministic_tie_break: bool,
}

pub async fn play_game_async(
    p1: &mut LeafParallelMCTS,
    p2: &mut P2,
    settings: GameSettings,
    board_size: i32,
    max_walls: i32,
    max_steps: i32,
) -> Result<GameResult> {
    let mechanics = QGameMechanics::new(board_size as usize, max_walls as usize, max_steps as usize);
    let mut data = mechanics.create_initial_state();
    let (orig_to_rot, _) = create_rotation_mapping(board_size);
    let mut replay_items: Vec<ReplayBufferItem> = Vec::new();
    let visited = HashSet::new();
    let mut winner: Option<i32> = None;

    for step in 0..max_steps {
        let current_player = mechanics.repr().get_current_player(data) as i32;
        let mask = mechanics.get_action_mask_immut(data);
        if !mask.iter().any(|&m| m) { break; }

        let resnet_input = compact_state_to_resnet_input(&mechanics, data);

        let (action_idx, policy) = if current_player == 0 {
            run_az_select(p1, data, &mechanics, &visited, settings, step as usize).await?
        } else {
            match p2 {
                P2::AlphaZero(m) => {
                    run_az_select(m, data, &mechanics, &visited, settings, step as usize).await?
                }
                P2::Random => random_select(&mask),
            }
        };

        // Replay capture (current-player-downward frame).
        let (stored_input_3d, stored_policy, stored_mask) = if current_player == 1 {
            let rotated_data = rotate_compact_state(&mechanics, data);
            let rotated_input = compact_state_to_resnet_input(&mechanics, rotated_data)
                .index_axis(Axis(0), 0).to_owned();
            let rotated_policy = remap_policy(&policy, &orig_to_rot);
            let rotated_mask = remap_mask(&mask, &orig_to_rot);
            (rotated_input, rotated_policy, rotated_mask)
        } else {
            (resnet_input.index_axis(Axis(0), 0).to_owned(), policy.clone(), mask.clone())
        };
        replay_items.push(ReplayBufferItem {
            input_array: stored_input_3d,
            policy: stored_policy,
            action_mask: stored_mask,
            value: 0.0,
            player: current_player,
        });

        // Apply action, then advance roots for tree reuse on the next move.
        mechanics.apply_action_index(&mut data, action_idx);
        p1.advance_root(action_idx);
        if let P2::AlphaZero(m) = p2 {
            m.advance_root(action_idx);
        }

        if mechanics.check_win(data, current_player as usize) {
            winner = Some(current_player);
            for item in replay_items.iter_mut() {
                item.value = if item.player == current_player { 1.0 } else { -1.0 };
            }
            return Ok(GameResult {
                winner,
                num_turns: step + 1,
                replay_items,
            });
        }
    }
    Ok(GameResult { winner, num_turns: max_steps, replay_items })
}

async fn run_az_select(
    mcts: &mut LeafParallelMCTS,
    data: CompactState,
    mechanics: &QGameMechanics,
    visited: &HashSet<CompactState>,
    settings: GameSettings,
    step: usize,
) -> Result<(usize, Vec<f32>)> {
    let (children, _root_value) = mcts.search(data, mechanics, visited).await?;
    let visit_counts: Vec<u32> = children.iter().map(|c| c.visit_count).collect();
    let action_indices: Vec<usize> = children.iter().map(|c| c.action_index).collect();
    let temperature = match settings.drop_t_on_step {
        Some(t) if step >= t => 0.0,
        _ => settings.temperature,
    };
    let action_idx = sample_action(&visit_counts, &action_indices, temperature, settings.deterministic_tie_break);

    // Build full policy vector from visit counts.
    let total_visits: u32 = visit_counts.iter().sum();
    let mask = mechanics.get_action_mask_immut(data);
    let mut policy = vec![0.0f32; mask.len()];
    if total_visits > 0 {
        for c in &children {
            policy[c.action_index] = c.visit_count as f32 / total_visits as f32;
        }
    }
    Ok((action_idx, policy))
}

fn random_select(mask: &[bool]) -> (usize, Vec<f32>) {
    use rand::Rng;
    let valid: Vec<usize> = mask.iter().enumerate()
        .filter_map(|(i, &v)| if v { Some(i) } else { None }).collect();
    let mut rng = rand::thread_rng();
    let idx = valid[rng.gen_range(0..valid.len())];
    let mut p = vec![0.0f32; mask.len()];
    p[idx] = 1.0;
    (idx, p)
}

#[allow(dead_code)]
fn _action_used(_a: [i32; 3]) {}
```

(The `action_index_to_action` import is required by some downstream consumers; if the compiler reports it unused you can drop the import line — leave it in if compile succeeds with `cargo check`.)

- [ ] **Step 2: Export the new module**

In `deep_quoridor/rust/src/agents/alphazero/mod.rs`:

```rust
pub mod selfplay_game;
```

- [ ] **Step 3: Verify it compiles**

Run: `cd deep_quoridor/rust && cargo check --features binary --bin selfplay`
Expected: compiles.

- [ ] **Step 4: Commit**

```bash
git add deep_quoridor/rust/src/agents/alphazero/selfplay_game.rs deep_quoridor/rust/src/agents/alphazero/mod.rs
git commit -m "Add play_game_async: tokio-based self-play game runner"
```

---

## Task 17: Update `bin/selfplay.rs` CLI

**Files:**
- Modify: `deep_quoridor/rust/src/bin/selfplay.rs`

Replace the existing `Cli` struct fields that no longer correspond to the schema, and add the new ones.

- [ ] **Step 1: Update CLI struct**

In `selfplay.rs`, locate the `Cli` struct (~line 51) and modify the relevant fields:

```rust
    /// Number of concurrent game tasks per process (default: 1, or YAML self_play.games_per_process).
    #[arg(long)]
    games_per_process: Option<usize>,

    /// Leaf-parallel batch size: number of in-flight evals per game per outer iteration.
    #[arg(long)]
    leaf_parallelism: Option<usize>,

    /// Virtual loss magnitude applied during descent.
    #[arg(long)]
    virtual_loss: Option<u32>,

    /// Disable tree reuse across moves (default: enabled).
    #[arg(long, default_value = "false")]
    no_tree_reuse: bool,

    /// Tokio worker threads (default: hardware threads, or YAML self_play.mcts_worker_threads).
    #[arg(long)]
    mcts_worker_threads: Option<usize>,

    /// Max eval batch size at the coordinator (default: 1).
    #[arg(long)]
    eval_batch_size: Option<usize>,

    /// Max wait (ms) for batch to fill after first request (default: 0).
    #[arg(long)]
    eval_max_wait_ms: Option<u64>,

    /// Max entries in the shared eval cache; 0 disables caching (default: 0).
    #[arg(long)]
    eval_cache_max_size: Option<usize>,
```

Remove the old `threads_per_process` and `games_per_thread` fields entirely.

- [ ] **Step 2: Update `ResolvedRustConfig`**

```rust
#[derive(Debug, Clone, Copy)]
struct ResolvedRustConfig {
    games_per_process: usize,
    leaf_parallelism: usize,
    virtual_loss: u32,
    enable_tree_reuse: bool,
    mcts_worker_threads: usize,
    eval_batch_size: usize,
    eval_max_wait_ms: u64,
    eval_cache_max_size: usize,
}

impl ResolvedRustConfig {
    fn resolve(cli: &Cli, yaml: Option<&SelfPlayWorkerConfig>) -> Self {
        let pick_usize = |c: Option<usize>, y: Option<usize>, d: usize| c.or(y).unwrap_or(d).max(1);
        let pick_usize_zero_ok = |c: Option<usize>, y: Option<usize>, d: usize| c.or(y).unwrap_or(d);
        let pick_u32 = |c: Option<u32>, y: Option<u32>, d: u32| c.or(y).unwrap_or(d);
        let pick_u64 = |c: Option<u64>, y: Option<u64>, d: u64| c.or(y).unwrap_or(d);
        let pick_bool = |yaml_v: Option<bool>, d: bool| yaml_v.unwrap_or(d);
        let default_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self {
            games_per_process: pick_usize(cli.games_per_process, yaml.and_then(|c| c.games_per_process), 1),
            leaf_parallelism: pick_usize(cli.leaf_parallelism, yaml.and_then(|c| c.leaf_parallelism), 1),
            virtual_loss: pick_u32(cli.virtual_loss, yaml.and_then(|c| c.virtual_loss), 3),
            enable_tree_reuse: if cli.no_tree_reuse {
                false
            } else {
                pick_bool(yaml.and_then(|c| c.enable_tree_reuse), true)
            },
            mcts_worker_threads: pick_usize(cli.mcts_worker_threads, yaml.and_then(|c| c.mcts_worker_threads), default_workers),
            eval_batch_size: pick_usize(cli.eval_batch_size, yaml.and_then(|c| c.eval_batch_size), 1),
            eval_max_wait_ms: pick_u64(cli.eval_max_wait_ms, yaml.and_then(|c| c.eval_max_wait_ms), 0),
            eval_cache_max_size: pick_usize_zero_ok(cli.eval_cache_max_size, yaml.and_then(|c| c.eval_cache_max_size), 100000),
        }
    }

    fn total_in_flight(&self) -> usize {
        self.games_per_process * self.leaf_parallelism
    }
}
```

(Remove the old `total_workers` method; in-flight is `games_per_process * leaf_parallelism` now.)

- [ ] **Step 3: Verify compilation**

Run: `cd deep_quoridor/rust && cargo check --features binary --bin selfplay`
Expected: errors at the call sites of the old fields (`rust_cfg.num_threads`, `rust_cfg.games_per_thread`, `rust_cfg.total_workers()`). That's expected — Task 18 fixes those.

- [ ] **Step 4: Do NOT commit yet** — continue with Task 18 to land a buildable binary.

---

## Task 18: Replace `run_batch_batched` and `run_continuous_batched` with tokio implementations

**Files:**
- Modify: `deep_quoridor/rust/src/bin/selfplay.rs`

This is the big switch-over. The new versions create a tokio runtime sized to `mcts_worker_threads`, spawn the eval pipeline, then spawn `games_per_process` async game tasks that each loop playing games until shutdown.

- [ ] **Step 1: Replace the body of `run_batch_batched`**

Delete the old function body and replace with:

```rust
fn run_batch_batched(
    cli: &Cli,
    q: &QuoridorConfig,
    az_config: &AlphaZeroConfig,
    rust_cfg: ResolvedRustConfig,
) -> Result<()> {
    use quoridor_rs::agents::alphazero::eval_pipeline::{self, EvalCache, FrontMsg};
    use quoridor_rs::agents::alphazero::selfplay_game::{play_game_async, GameSettings, P2};
    use quoridor_rs::agents::alphazero::selfplay_mcts::{LeafParallelConfig, LeafParallelMCTS};
    use tokio::sync::mpsc as tokio_mpsc;

    let model_path = cli
        .model_path
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--model-path is required in batch mode"))?
        .to_string();
    let num_games = cli.num_games;
    let output_dir = cli.output_dir.clone();
    let p2_kind = cli.p2.clone();

    println!(
        "Self-play (leaf-parallel): board_size={}, max_walls={}, max_steps={}, num_games={}",
        q.board_size, q.max_walls, q.max_steps, num_games,
    );
    println!(
        "games_per_process={}, leaf_parallelism={}, virtual_loss={}, tree_reuse={}, eval_batch_size={}, eval_max_wait_ms={}, eval_cache_max_size={}, mcts_worker_threads={}",
        rust_cfg.games_per_process, rust_cfg.leaf_parallelism, rust_cfg.virtual_loss,
        rust_cfg.enable_tree_reuse, rust_cfg.eval_batch_size, rust_cfg.eval_max_wait_ms,
        rust_cfg.eval_cache_max_size, rust_cfg.mcts_worker_threads,
    );

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(rust_cfg.mcts_worker_threads)
        .enable_time()
        .build()?;

    rt.block_on(async move {
        let cache = std::sync::Arc::new(EvalCache::new());
        let (front_tx, front_rx) = tokio_mpsc::channel::<FrontMsg>(1024);
        let session = eval_pipeline::load_session(&model_path)?;
        let coord = eval_pipeline::spawn_coordinator(session, std::sync::Arc::clone(&cache),
            eval_pipeline::CoordinatorConfig {
                eval_batch_size: rust_cfg.eval_batch_size,
                eval_max_wait_ms: rust_cfg.eval_max_wait_ms,
                eval_cache_max_size: rust_cfg.eval_cache_max_size,
            },
            front_rx,
        );

        let mcts_cfg = az_config.to_agent_config(q.board_size, q.max_walls).mcts;
        let lp_cfg = LeafParallelConfig {
            leaf_parallelism: rust_cfg.leaf_parallelism as u32,
            virtual_loss: rust_cfg.virtual_loss,
            enable_tree_reuse: rust_cfg.enable_tree_reuse,
        };
        let settings = GameSettings {
            temperature: az_config.temperature.unwrap_or(1.0),
            drop_t_on_step: az_config.drop_t_on_step,
            deterministic_tie_break: false,
        };
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pid = std::process::id();
        let start = std::time::Instant::now();
        let stats = std::sync::Arc::new(std::sync::Mutex::new(Stats::default()));

        let mut handles = Vec::with_capacity(rust_cfg.games_per_process);
        for _ in 0..rust_cfg.games_per_process {
            let front_tx = front_tx.clone();
            let cache = std::sync::Arc::clone(&cache);
            let counter = std::sync::Arc::clone(&counter);
            let stats = std::sync::Arc::clone(&stats);
            let output_dir = output_dir.clone();
            let p2_kind = p2_kind.clone();
            let mcts_cfg = mcts_cfg.clone();
            let board_size = q.board_size;
            let max_walls = q.max_walls;
            let max_steps = q.max_steps as i32;
            let model_version = cli.model_version;
            handles.push(tokio::spawn(async move {
                let mut p1 = LeafParallelMCTS::new(mcts_cfg.clone(), lp_cfg, front_tx.clone(), std::sync::Arc::clone(&cache));
                let mut p2: P2 = match p2_kind.as_deref() {
                    Some("random") => P2::Random,
                    Some(other) => return Err(anyhow::anyhow!("Unknown --p2 agent: '{}'", other)),
                    None => P2::AlphaZero(LeafParallelMCTS::new(mcts_cfg, lp_cfg, front_tx.clone(), std::sync::Arc::clone(&cache))),
                };
                loop {
                    let idx = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if idx >= num_games { break; }
                    p1.reset_tree();
                    if let P2::AlphaZero(m) = &mut p2 { m.reset_tree(); }
                    let result = play_game_async(&mut p1, &mut p2, settings, board_size, max_walls, max_steps).await?;
                    write_replay(&output_dir, None, &result, model_version, idx, pid)?;
                    let mut s = stats.lock().unwrap();
                    match result.winner {
                        Some(0) => s.wins[0] += 1,
                        Some(1) => s.wins[1] += 1,
                        _ => s.draws += 1,
                    }
                    s.total_turns += result.num_turns as u64;
                    s.completed += 1;
                    let done = s.completed;
                    let p1w = s.wins[0];
                    let p2w = s.wins[1];
                    let draws = s.draws;
                    let avg_turns = s.total_turns as f64 / done.max(1) as f64;
                    drop(s);
                    if done % 10 == 0 || done == num_games {
                        let elapsed = start.elapsed().as_secs_f64();
                        let gps = done as f64 / elapsed;
                        println!("[{}/{}] P1 wins: {}, P2 wins: {}, draws: {}, avg turns: {:.1}, {:.1} games/s",
                            done, num_games, p1w, p2w, draws, avg_turns, gps);
                    }
                }
                Ok::<(), anyhow::Error>(())
            }));
        }
        drop(front_tx);
        for h in handles {
            h.await??;
        }
        // Send shutdown to the pipeline (sender is dropped above so blocking_recv will return None).
        let _ = coord.batcher.join();
        let _ = coord.inference.join();
        let _ = coord.post.join();
        Ok::<(), anyhow::Error>(())
    })?;

    println!("Done. {} games written to {}", num_games, cli.output_dir);
    Ok(())
}
```

- [ ] **Step 2: Replace the body of `run_continuous_batched`**

Same overall shape, but the game tasks loop forever (until shutdown), checking the shutdown file from the main task. Replace with:

```rust
fn run_continuous_batched(
    cli: &Cli,
    q: &QuoridorConfig,
    az_config: &AlphaZeroConfig,
    rust_cfg: ResolvedRustConfig,
) -> Result<()> {
    use quoridor_rs::agents::alphazero::eval_pipeline::{self, EvalCache, FrontMsg};
    use quoridor_rs::agents::alphazero::selfplay_game::{play_game_async, GameSettings, P2};
    use quoridor_rs::agents::alphazero::selfplay_mcts::{LeafParallelConfig, LeafParallelMCTS};
    use quoridor_rs::selfplay_config::load_latest_model;
    use tokio::sync::mpsc as tokio_mpsc;

    let latest_yaml_path = cli.latest_model_yaml.as_deref()
        .ok_or_else(|| anyhow::anyhow!("--latest-model-yaml is required with --continuous"))?
        .to_string();
    let shutdown_path = cli.shutdown_file.as_deref()
        .ok_or_else(|| anyhow::anyhow!("--shutdown-file is required with --continuous"))?
        .to_string();
    let tmp_dir = format!("{}/tmp", cli.output_dir);
    std::fs::create_dir_all(&tmp_dir)?;

    println!(
        "Continuous self-play (leaf-parallel): board_size={}, max_walls={}, max_steps={}",
        q.board_size, q.max_walls, q.max_steps,
    );
    println!(
        "games_per_process={}, leaf_parallelism={}, virtual_loss={}, tree_reuse={}, eval_batch_size={}, eval_max_wait_ms={}, eval_cache_max_size={}, mcts_worker_threads={}",
        rust_cfg.games_per_process, rust_cfg.leaf_parallelism, rust_cfg.virtual_loss,
        rust_cfg.enable_tree_reuse, rust_cfg.eval_batch_size, rust_cfg.eval_max_wait_ms,
        rust_cfg.eval_cache_max_size, rust_cfg.mcts_worker_threads,
    );
    println!("Polling: {}\nShutdown: {}\nOutput: {}", latest_yaml_path, shutdown_path, cli.output_dir);

    println!("Waiting for initial model...");
    loop {
        if std::path::Path::new(&shutdown_path).exists() {
            println!("Shutdown signal detected before model was available. Exiting.");
            return Ok(());
        }
        if std::path::Path::new(&latest_yaml_path).exists() {
            let onnx_path = pt_to_onnx_path(
                &load_latest_model(&latest_yaml_path).map(|m| m.filename).unwrap_or_default(),
            );
            if std::path::Path::new(&onnx_path).exists() { break; }
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    let latest = load_latest_model(&latest_yaml_path)?;
    let initial_version = latest.version;
    let initial_path = pt_to_onnx_path(&latest.filename);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(rust_cfg.mcts_worker_threads)
        .enable_time()
        .build()?;

    rt.block_on(async move {
        let cache = std::sync::Arc::new(EvalCache::new());
        let (front_tx, front_rx) = tokio_mpsc::channel::<FrontMsg>(1024);
        let session = eval_pipeline::load_session(&initial_path)?;
        let coord = eval_pipeline::spawn_coordinator(session, std::sync::Arc::clone(&cache),
            eval_pipeline::CoordinatorConfig {
                eval_batch_size: rust_cfg.eval_batch_size,
                eval_max_wait_ms: rust_cfg.eval_max_wait_ms,
                eval_cache_max_size: rust_cfg.eval_cache_max_size,
            },
            front_rx,
        );

        let mcts_cfg = az_config.to_agent_config(q.board_size, q.max_walls).mcts;
        let lp_cfg = LeafParallelConfig {
            leaf_parallelism: rust_cfg.leaf_parallelism as u32,
            virtual_loss: rust_cfg.virtual_loss,
            enable_tree_reuse: rust_cfg.enable_tree_reuse,
        };
        let settings = GameSettings {
            temperature: az_config.temperature.unwrap_or(1.0),
            drop_t_on_step: az_config.drop_t_on_step,
            deterministic_tie_break: false,
        };
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let model_version = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(initial_version));
        let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pid = std::process::id();

        let mut handles = Vec::with_capacity(rust_cfg.games_per_process);
        for _tid in 0..rust_cfg.games_per_process {
            let front_tx = front_tx.clone();
            let cache = std::sync::Arc::clone(&cache);
            let counter = std::sync::Arc::clone(&counter);
            let model_version = std::sync::Arc::clone(&model_version);
            let shutdown = std::sync::Arc::clone(&shutdown);
            let output_dir = cli.output_dir.clone();
            let tmp_dir = tmp_dir.clone();
            let p2_kind = cli.p2.clone();
            let mcts_cfg = mcts_cfg.clone();
            let board_size = q.board_size;
            let max_walls = q.max_walls;
            let max_steps = q.max_steps as i32;
            handles.push(tokio::spawn(async move {
                let mut p1 = LeafParallelMCTS::new(mcts_cfg.clone(), lp_cfg, front_tx.clone(), std::sync::Arc::clone(&cache));
                let mut p2: P2 = match p2_kind.as_deref() {
                    Some("random") => P2::Random,
                    Some(other) => return Err(anyhow::anyhow!("Unknown --p2 agent: '{}'", other)),
                    None => P2::AlphaZero(LeafParallelMCTS::new(mcts_cfg, lp_cfg, front_tx.clone(), std::sync::Arc::clone(&cache))),
                };
                loop {
                    if shutdown.load(std::sync::atomic::Ordering::Relaxed) { break; }
                    let mv = model_version.load(std::sync::atomic::Ordering::Relaxed);
                    p1.note_model_version(mv);
                    if let P2::AlphaZero(m) = &mut p2 { m.note_model_version(mv); }
                    p1.reset_tree();
                    if let P2::AlphaZero(m) = &mut p2 { m.reset_tree(); }
                    let idx = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let game_start = std::time::Instant::now();
                    let result = play_game_async(&mut p1, &mut p2, settings, board_size, max_walls, max_steps).await?;
                    let elapsed = game_start.elapsed().as_secs_f64();
                    println!("{}-{} - selfplay finished in {:.4}", pid, idx, elapsed);
                    write_replay(&output_dir, Some(&tmp_dir), &result, mv, idx, pid)?;
                }
                Ok::<(), anyhow::Error>(())
            }));
        }

        // Main coordinator-poll loop: watches latest.yaml + shutdown sentinel.
        let main_handle = {
            let front_tx = front_tx.clone();
            let shutdown = std::sync::Arc::clone(&shutdown);
            let model_version = std::sync::Arc::clone(&model_version);
            tokio::spawn(async move {
                let mut current = initial_version;
                loop {
                    if std::path::Path::new(&shutdown_path).exists() {
                        println!("Shutdown signal detected. Stopping workers...");
                        shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
                        break;
                    }
                    if let Ok(latest) = load_latest_model(&latest_yaml_path) {
                        if latest.version != current {
                            let new_path = pt_to_onnx_path(&latest.filename);
                            if std::path::Path::new(&new_path).exists() {
                                println!("New model detected: version {} -> {} ({})", current, latest.version, new_path);
                                current = latest.version;
                                model_version.store(latest.version, std::sync::atomic::Ordering::Relaxed);
                                let _ = front_tx.send(FrontMsg::Reload(new_path)).await;
                            }
                        }
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
                Ok::<(), anyhow::Error>(())
            })
        };

        drop(front_tx);
        let _ = main_handle.await?;
        for h in handles { h.await??; }
        let _ = coord.batcher.join();
        let _ = coord.inference.join();
        let _ = coord.post.join();
        Ok::<(), anyhow::Error>(())
    })?;

    Ok(())
}
```

- [ ] **Step 3: Remove now-unused helpers**

In `selfplay.rs`, delete:
- The old `BoxedAgent` enum and its impls
- `build_p1_agent_batched` and `build_p2_agent_batched`
- The old `spawn_coordinator` helper (in selfplay.rs — keep only the new one in eval_pipeline.rs)
- `create_agent_legacy` is KEPT — used by `--use-raw-onnx-agent` paths.

Also delete the `use quoridor_rs::agents::alphazero::eval_coordinator::{...}` imports that are no longer used. Keep only the imports `run_continuous_legacy` and `run_batch_legacy` need.

- [ ] **Step 4: Update the `main` dispatcher**

The existing `main()` already dispatches `cli.continuous` / `cli.use_raw_onnx_agent`. No change needed except verifying the imports compile.

- [ ] **Step 5: Verify compilation**

Run: `cd deep_quoridor/rust && cargo check --features binary --bin selfplay`
Expected: compiles cleanly (warnings about unused old code paths are OK).

- [ ] **Step 6: Run the full test suite**

Run: `cd deep_quoridor/rust && cargo test --features binary`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add deep_quoridor/rust/src/bin/selfplay.rs
git commit -m "Switch selfplay binary to tokio + leaf-parallel MCTS

Both batch and continuous modes now use the new pipelined eval coordinator,
spawn one async game task per concurrent game, and rely on LeafParallelMCTS
for per-game search. Old BatchingEvaluator/coordinator scaffolding is no
longer referenced; the legacy --use-raw-onnx-agent path is unchanged."
```

---

## Task 19: Delete legacy `eval_coordinator.rs` and `BatchingEvaluator`

**Files:**
- Delete: `deep_quoridor/rust/src/agents/alphazero/eval_coordinator.rs`
- Modify: `deep_quoridor/rust/src/agents/alphazero/evaluator.rs`
- Modify: `deep_quoridor/rust/src/agents/alphazero/mod.rs`

Now that nothing references the old coordinator or `BatchingEvaluator`, drop them.

- [ ] **Step 1: Remove `BatchingEvaluator`**

In `deep_quoridor/rust/src/agents/alphazero/evaluator.rs`, delete the entire `BatchingEvaluator` struct, its `impl` block, and the `impl Evaluator for BatchingEvaluator`. Also drop any unused imports (`sync_channel`, `EvalRequest`, `EvalCache`) at the top of the file.

- [ ] **Step 2: Delete `eval_coordinator.rs` and remove its export**

```bash
rm deep_quoridor/rust/src/agents/alphazero/eval_coordinator.rs
```

In `deep_quoridor/rust/src/agents/alphazero/mod.rs`, remove the line `pub mod eval_coordinator;`.

- [ ] **Step 3: Verify everything still compiles and tests pass**

Run: `cd deep_quoridor/rust && cargo build --features binary --bin selfplay && cargo test --features binary`
Expected: build + all tests pass.

- [ ] **Step 4: Commit**

```bash
git add -A deep_quoridor/rust/src/agents/alphazero
git commit -m "Remove legacy eval_coordinator.rs and BatchingEvaluator

The new pipelined coordinator in eval_pipeline.rs replaces it entirely."
```

---

## Task 20: Update Python `agent_evolution_tournament.py` / etc references if any depend on dropped fields

**Files:**
- Search: `deep_quoridor/`

The Python `train_v2.py` already uses `config.self_play.num_processes` (updated in the previous brainstorm session's rename) and does not depend on `threads_per_process` or `games_per_thread`. Confirm there are no remaining references.

- [ ] **Step 1: Search for any remaining references**

Run: `grep -rn "threads_per_process\|games_per_thread" deep_quoridor/ --include='*.py' --include='*.rs' --include='*.yaml' --include='*.yml'`
Expected: no matches (other than possibly `runs/*` historical snapshots, which are intentionally left alone).

If anything turns up, update it to use `games_per_process` or the appropriate new field.

- [ ] **Step 2: Run the Python test suite to confirm nothing else broke**

Run: `cd deep_quoridor && PYTHONPATH=src python -m pytest test/ -q`
Expected: all tests pass.

- [ ] **Step 3: If changes were made, commit**

```bash
git add deep_quoridor/
git commit -m "Drop stale references to threads_per_process / games_per_thread"
```

(If nothing changed, skip this commit.)

---

## Task 21: Add `--profile-counters` flag to the selfplay binary

**Files:**
- Modify: `deep_quoridor/rust/src/agents/alphazero/eval_pipeline.rs`
- Modify: `deep_quoridor/rust/src/bin/selfplay.rs`

Adds counters in the pipeline (atomic u64s for accumulated GPU time, batcher wait time, post-process parallel time, and batches processed). A new CLI flag prints them periodically.

- [ ] **Step 1: Add a `PipelineCounters` struct**

In `eval_pipeline.rs`, near the top, add:

```rust
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub struct PipelineCounters {
    pub gpu_ns: AtomicU64,
    pub batcher_wait_ns: AtomicU64,
    pub postprocess_ns: AtomicU64,
    pub batches: AtomicU64,
    pub items: AtomicU64,
}
```

Extend `spawn_coordinator` to accept `Arc<PipelineCounters>` and pass it to each stage. In `run_batcher`, time the wait from "ready to recv" until receipt of the first request and accumulate. In `run_inference`, time the `session.run` call. In `run_postprocess`, time the `par_iter` block.

Specifically: add this parameter and thread it through:

```rust
pub fn spawn_coordinator(
    initial_session: Session,
    cache: Arc<EvalCache>,
    config: CoordinatorConfig,
    front_rx: tokio_mpsc::Receiver<FrontMsg>,
    counters: Arc<PipelineCounters>,
) -> CoordinatorHandles {
    // ...
}
```

In each stage, after each block of work, do:

```rust
let t0 = std::time::Instant::now();
// ... work ...
counters.gpu_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
counters.batches.fetch_add(1, Ordering::Relaxed);
counters.items.fetch_add(batch_len as u64, Ordering::Relaxed);
```

(adapt per-stage). Update the two call sites in `bin/selfplay.rs` to construct an `Arc<PipelineCounters>` and pass it in.

- [ ] **Step 2: Add `--profile-counters` to the CLI**

In `bin/selfplay.rs`:

```rust
    /// Periodically print pipeline counters (GPU time, batcher wait, postprocess time).
    #[arg(long, default_value = "false")]
    profile_counters: bool,
```

When `cli.profile_counters` is true, spawn a tokio task that wakes every 5 seconds, snapshots the counters, computes deltas, and prints something like:

```
[pipe] batches=312 items=159744 avg_batch=512.0 gpu_busy=84.1% batch_wait_ms=12.4 post_ms=4.6
```

- [ ] **Step 3: Verify it compiles**

Run: `cd deep_quoridor/rust && cargo build --features binary --bin selfplay`
Expected: builds cleanly.

- [ ] **Step 4: Commit**

```bash
git add deep_quoridor/rust/src/agents/alphazero/eval_pipeline.rs deep_quoridor/rust/src/bin/selfplay.rs
git commit -m "Add --profile-counters: periodic GPU/batcher/postprocess timings"
```

---

## Task 22: Add a performance benchmark script

**Files:**
- Create: `deep_quoridor/scripts/bench_rust_selfplay.sh`

Wraps the Rust binary in a fixed-duration run that prints games/sec for before/after comparison.

- [ ] **Step 1: Create the script**

```bash
mkdir -p deep_quoridor/scripts
```

Then create `deep_quoridor/scripts/bench_rust_selfplay.sh`:

```bash
#!/usr/bin/env bash
# Run the Rust self-play binary in batch mode for a fixed wall-clock window
# and report games/sec. Pass a YAML config and an ONNX model path.
set -euo pipefail

CONFIG="${1:?usage: bench_rust_selfplay.sh CONFIG_YAML MODEL_ONNX [DURATION_SECONDS]}"
MODEL="${2:?usage: bench_rust_selfplay.sh CONFIG_YAML MODEL_ONNX [DURATION_SECONDS]}"
DURATION="${3:-60}"

BIN="deep_quoridor/rust/target/release/selfplay"
if [[ ! -x "$BIN" ]]; then
    echo "Building selfplay release binary..."
    (cd deep_quoridor/rust && cargo build --release --features binary --bin selfplay)
fi

OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

echo "Benchmarking for ${DURATION}s..."
timeout "$DURATION" "$BIN" \
    --config "$CONFIG" \
    --model-path "$MODEL" \
    --output-dir "$OUT" \
    --num-games 100000 \
    --profile-counters \
    || true

COUNT=$(find "$OUT" -maxdepth 1 -name '*.npz' | wc -l)
echo "Games completed in ${DURATION}s: $COUNT"
echo "Rate: $(python3 -c "print(${COUNT} / ${DURATION})") games/sec"
```

Then make it executable:

```bash
chmod +x deep_quoridor/scripts/bench_rust_selfplay.sh
```

- [ ] **Step 2: Commit**

```bash
git add deep_quoridor/scripts/bench_rust_selfplay.sh
git commit -m "Add bench_rust_selfplay.sh: fixed-duration games/sec measurement"
```

---

## Task 23: Final cleanup — `cargo fmt`, `cargo clippy --features binary`

**Files:** all touched

- [ ] **Step 1: Run `cargo fmt`**

```bash
cd deep_quoridor/rust && cargo fmt
```

Stage any formatting changes.

- [ ] **Step 2: Run `cargo clippy --features binary`**

```bash
cd deep_quoridor/rust && cargo clippy --features binary --bin selfplay
```
Expected: any new warnings introduced by the refactor are addressed inline (typically removing unused imports or adding `#[allow(dead_code)]` for intentional stubs). Pre-existing baseline warnings are out of scope for this work — flag them in the PR description if any are particularly noisy.

- [ ] **Step 3: Run the full Rust test suite**

```bash
cd deep_quoridor/rust && cargo test --features binary
```
Expected: all tests pass.

- [ ] **Step 4: Run the Python test suite**

```bash
cd deep_quoridor && PYTHONPATH=src python -m pytest test/ -q
```
Expected: all tests pass.

- [ ] **Step 5: Commit the cleanup**

```bash
git add -A
git commit -m "cargo fmt + clippy cleanup after selfplay perf refactor"
```

(If nothing changed, skip this commit.)

---

## Task 24: Performance verification (manual)

This is a **manual** step done by the engineer on the reference hardware (RTX 5080, 12c/24t CPU).

- [ ] **Step 1: Capture baseline games/sec on `main`**

```bash
git switch main
cd deep_quoridor/rust && cargo build --release --features binary --bin selfplay
./deep_quoridor/scripts/bench_rust_selfplay.sh experiments/B5W3/base.yaml <ONNX_MODEL_PATH> 60
```
Record the games/sec figure.

- [ ] **Step 2: Capture games/sec on the new branch**

```bash
git switch <perf-branch>
cd deep_quoridor/rust && cargo build --release --features binary --bin selfplay
./deep_quoridor/scripts/bench_rust_selfplay.sh experiments/B5W3/base.yaml <ONNX_MODEL_PATH> 60
```
Record the games/sec figure.

- [ ] **Step 3: Check counters**

While the new-branch run is going, observe the periodic `[pipe]` lines from `--profile-counters`. Expected:
- `gpu_busy` ≥ 70% (vs. roughly 15% on the baseline).
- `avg_batch` close to `eval_batch_size` (the GPU is fed full batches).

- [ ] **Step 4: Confirm ≥4× games/sec improvement**

If the multiplier is below 4×, capture profile counters and revisit one of: increase `games_per_process` (more in-flight evals); increase `leaf_parallelism`; enable tree reuse if disabled; raise `eval_cache_max_size`.

No commit at this step — performance data goes into the PR description.

---

## Self-Review Summary

After writing all tasks, the spec sections map to tasks as follows:

- **Spec § Goal / Bottleneck**: Tasks 24 (perf check verifies)
- **Spec § High-level architecture**: Tasks 10, 12, 16, 17, 18 (pipeline + tokio + game tasks)
- **Spec § Per-game MCTS detail**: Tasks 5–9, 12–15
- **Spec § Coordinator detail**: Tasks 10, 11, 21
- **Spec § Config schema**: Tasks 2, 3, 4, 17
- **Spec § Testing strategy**: Tasks 6, 7, 8, 9, 11, 13, 14, 15, 22, 23
- **Spec § Out of scope**: Confirmed by absence (no LRU, no dedup, no virtual mean tasks).
