# quoridor-wasm crate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile the project's Rust game+MCTS core to WebAssembly, exposing game session bindings and an async leaf-parallel MCTS search that runs neural-net evaluation through a caller-supplied (eventually JavaScript/onnxruntime-web) batched callback.

**Architecture:** Two-part. (1) Refactor `quoridor-rs` so its portable core (game mechanics, feature encoding, MCTS tree primitives, `prepare_eval_input`/`finalize_policy`) compiles with `--no-default-features`, by making the heavy deps (`arrow`, `parquet`, `rayon`, `dashmap`, `ort`) optional and feature-gating the modules that use them; and add a new **portable** async batched-search driver module to `quoridor-rs` that reuses the existing MCTS primitives and takes the NN forward pass as an injected async callback. (2) Add a new `quoridor-wasm` cdylib crate (workspace member) with `wasm-bindgen` bindings for game session + `run_search`, marshalling the eval batch to/from JS.

**Tech Stack:** Rust 2024, `wasm-bindgen`, `wasm-bindgen-futures`, `js-sys`, `serde-wasm-bindgen`, `getrandom` (js), `wasm-pack`, `wasm-bindgen-test`. No `tokio`/`rayon`/`ort` in the wasm build.

**Spec:** `docs/superpowers/specs/2026-07-05-browser-wasm-play-server-design.md` (this is Plan 1 of 3; Plans 2 and 3 are the Python server and the Svelte/worker frontend).

---

## File structure

**Modified in `quoridor-rs`:**
- `rust/Cargo.toml` — make `arrow`/`parquet`/`rayon`/`dashmap` optional; move `rand_distr` to always-on; add `policy_db` + `parallel` features; add `[workspace]`.
- `rust/src/lib.rs` — gate `mod minimax;` behind `parallel`.
- `rust/src/compact/mod.rs` — gate `policy_db` (feature `policy_db`) and `q_minimax` (feature `parallel`).
- `rust/src/agents/mod.rs` — make `alphazero` always compiled (inner submodules gated).
- `rust/src/agents/alphazero/mod.rs` — gate `agent`, `eval_pipeline`, `selfplay_*` behind `binary`; keep `mcts`, `evaluator`, new `batched_search` always compiled.
- `rust/src/agents/alphazero/evaluator.rs` — feature-gate the `ort`-dependent `OnnxEvaluator`; move `softmax` here (out of `onnx_agent`).
- `rust/src/agents/onnx_agent.rs` — re-export `softmax` from `evaluator` for existing callers.

**Created in `quoridor-rs`:**
- `rust/src/agents/alphazero/batched_search.rs` — portable async leaf-parallel batched MCTS driver (the one genuinely new algorithm), with native unit tests.

**Created — new crate `quoridor-wasm`:**
- `rust/quoridor-wasm/Cargo.toml`
- `rust/quoridor-wasm/src/lib.rs` — `#[wasm_bindgen]` module root.
- `rust/quoridor-wasm/src/view.rs` — `StateView` + `EnrichedAction` serde types (ported from `play_server/state.rs`).
- `rust/quoridor-wasm/src/game.rs` — `WasmGame` session (new/apply/undo/legal/state_view).
- `rust/quoridor-wasm/src/search.rs` — `run_search` binding wiring the JS eval + progress callbacks to `run_batched_search`.
- `rust/quoridor-wasm/tests/web.rs` — `wasm-bindgen-test` browser/node tests.

---

## Prerequisites (one-time, do before Task 2)

- [ ] **Step 1: Install the wasm target and wasm-pack**

Run:
```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```
Expected: both succeed; `wasm-pack --version` prints a version. (If `wasm-pack` is already installed, the install is a no-op.)

---

## Task 1: Make `quoridor-rs` core compile with `--no-default-features`

**Files:**
- Modify: `rust/Cargo.toml`
- Modify: `rust/src/lib.rs`
- Modify: `rust/src/compact/mod.rs`
- Modify: `rust/src/agents/mod.rs`
- Modify: `rust/src/agents/alphazero/mod.rs`
- Modify: `rust/src/agents/alphazero/evaluator.rs`
- Modify: `rust/src/agents/onnx_agent.rs`

This task is pure refactor: existing `python`/`binary` builds must remain byte-for-byte behaviorally identical, while a new bare core build becomes possible. The "tests" are compile checks in all three configurations plus the existing test suite.

- [ ] **Step 1: Baseline — confirm the existing build and tests are green before touching anything**

Run:
```bash
cd rust && cargo test --features binary 2>&1 | tail -20
```
Expected: builds and the existing tests pass. (Record this as the baseline; Task 1 must not change it.)

- [ ] **Step 2: Edit `rust/Cargo.toml` — make heavy deps optional, move `rand_distr` to always-on, add features**

In `[dependencies]`, change these four lines so the deps are optional:
```toml
dashmap = { version = "6", optional = true }
rayon = { version = "1.10", optional = true }
arrow = { version = "53", optional = true }
parquet = { version = "53", default-features = false, features = ["arrow", "zstd"], optional = true }
```
Remove `rand_distr` from the `binary` feature list and add it as an always-on dependency (the MCTS core uses it). In `[dependencies]` add:
```toml
rand_distr = "0.4"
```
and delete the old `rand_distr = { version = "0.4", optional = true }` line.

Replace the `[features]` block with:
```toml
[features]
default = ["python"]
# Portable core only (game mechanics, feature encoding, MCTS primitives,
# prepare_eval_input/finalize_policy, batched_search). No arrow/parquet/rayon/
# dashmap/ort/tokio. This is what the wasm crate builds against.
python = ["pyo3", "numpy", "policy_db", "parallel"]
binary = ["clap", "ort", "serde_yaml", "serde_json", "ndarray-npy", "zip", "tokio", "futures", "tiny_http", "policy_db", "parallel"]
gpu = ["binary", "ort/cuda", "ort/load-dynamic"]
# Optional heavy subsystems, split out so the core can drop them:
policy_db = ["dep:arrow", "dep:parquet", "dep:dashmap"]      # compact::policy_db
parallel = ["dep:rayon", "dep:dashmap"]                       # minimax, compact::q_minimax, eval_pipeline
```
Note: `binary` now lists `policy_db` + `parallel` explicitly so it keeps arrow/parquet/rayon/dashmap exactly as before. `rand_distr` was in the old `binary` list; it is now always-on, so its removal from the list is intentional.

- [ ] **Step 3: Gate `mod minimax;` in `rust/src/lib.rs`**

Find `mod minimax;` and change it to:
```rust
#[cfg(feature = "parallel")]
mod minimax;
```

- [ ] **Step 4: Gate the heavy `compact` submodules in `rust/src/compact/mod.rs`**

Find the `pub mod policy_db;` and `pub mod q_minimax;` declarations and gate them:
```rust
#[cfg(feature = "policy_db")]
pub mod policy_db;
#[cfg(feature = "parallel")]
pub mod q_minimax;
```
Leave the other `compact` submodules (`q_bit_repr`, `q_bit_repr_conversions`, `q_game_mechanics`, `mod`) unconditional.

- [ ] **Step 5: Move `softmax` into `evaluator.rs` and re-export it from `onnx_agent.rs`**

In `rust/src/agents/onnx_agent.rs`, locate the `pub fn softmax(...) -> Vec<f32>` definition, cut it out, and replace it with a re-export:
```rust
pub use crate::agents::alphazero::evaluator::softmax;
```
In `rust/src/agents/alphazero/evaluator.rs`, remove the line `use crate::agents::onnx_agent::softmax;` and paste the `softmax` definition near `masked_softmax`, made public:
```rust
/// Numerically-stable softmax over a slice.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    // (paste the exact body moved from onnx_agent.rs)
}
```

- [ ] **Step 6: Feature-gate the `ort` pieces of `evaluator.rs`**

At the top of `rust/src/agents/alphazero/evaluator.rs`, gate the ORT import:
```rust
#[cfg(feature = "binary")]
use ort::session::Session;
```
Gate the `OnnxEvaluator` struct, its `impl OnnxEvaluator`, and `impl Evaluator for OnnxEvaluator` blocks by prefixing each with:
```rust
#[cfg(feature = "binary")]
```
Leave `Evaluator` (trait), `UniformMockEvaluator`, `PreparedEvalInput`, `prepare_eval_input`, `finalize_policy`, `masked_softmax`, and `softmax` ungated (they are ort-free).

- [ ] **Step 7: Restructure `rust/src/agents/mod.rs` so `alphazero` always compiles**

Change the `alphazero` module declaration from binary-gated to unconditional (keep `onnx_agent` binary-gated as-is):
```rust
pub mod alphazero;
#[cfg(feature = "binary")]
pub mod onnx_agent;
```
(Leave `random_agent` and the `ActionSelector` trait unchanged.)

- [ ] **Step 8: Gate the heavy submodules in `rust/src/agents/alphazero/mod.rs`**

Rewrite the module declarations so the portable ones are always compiled and the tokio/ort ones are binary-gated:
```rust
pub mod evaluator;
pub mod mcts;
pub mod batched_search;                 // created in Task 3

#[cfg(feature = "binary")]
pub mod agent;
#[cfg(feature = "binary")]
pub mod eval_pipeline;
#[cfg(feature = "binary")]
pub mod selfplay_mcts;
#[cfg(feature = "binary")]
pub mod selfplay_game;
#[cfg(feature = "binary")]
pub mod selfplay_metrics;
```
Match the exact set of existing submodules (add/remove lines to mirror what the file currently declares; only the `#[cfg]` placement changes). If `batched_search` does not exist yet, temporarily omit its line and add it in Task 3.

- [ ] **Step 9: Verify the default (python) build is unchanged**

Run:
```bash
cd rust && cargo build 2>&1 | tail -20
```
Expected: builds successfully (python feature pulls `policy_db` + `parallel`, so arrow/parquet/rayon/dashmap are present exactly as before).

- [ ] **Step 10: Verify the binary build and the full test suite are unchanged**

Run:
```bash
cd rust && cargo test --features binary 2>&1 | tail -20
```
Expected: identical result to the Step 1 baseline (all tests pass).

- [ ] **Step 11: Verify the new bare core build compiles (native target)**

Run:
```bash
cd rust && cargo build --no-default-features 2>&1 | tail -30
```
Expected: builds successfully. If the compiler flags a module still referencing a gated dep, gate that module or its `use` line the same way (do not add the dep back to the core). Common culprits: a stray `use crate::compact::policy_db` / `q_minimax` / `minimax` in a non-gated file — gate the offending `use`/call site behind the matching feature.

- [ ] **Step 12: Commit**

```bash
cd rust && git add Cargo.toml src/lib.rs src/compact/mod.rs src/agents/mod.rs src/agents/alphazero/mod.rs src/agents/alphazero/evaluator.rs src/agents/onnx_agent.rs
git commit -m "refactor(rust): feature-gate heavy deps so core builds with --no-default-features"
```

---

## Task 2: Add the `quoridor-wasm` crate skeleton and prove the wasm toolchain builds

**Files:**
- Modify: `rust/Cargo.toml` (add `[workspace]`)
- Create: `rust/quoridor-wasm/Cargo.toml`
- Create: `rust/quoridor-wasm/src/lib.rs`

- [ ] **Step 1: Make `rust/` a workspace that excludes the wasm crate from default builds**

Append to `rust/Cargo.toml` (after the existing tables):
```toml
[workspace]
members = ["quoridor-wasm"]
# `cargo build`/`cargo test` at the root operate on quoridor-rs only; the wasm
# crate is built explicitly via wasm-pack (it targets wasm32).
default-members = ["."]
```

- [ ] **Step 2: Create `rust/quoridor-wasm/Cargo.toml`**

```toml
[package]
name = "quoridor-wasm"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
quoridor-rs = { path = "..", default-features = false }
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
js-sys = "0.3"
serde = { version = "1", features = ["derive"] }
serde-wasm-bindgen = "0.6"
anyhow = "1"
ndarray = "0.16"
# wasm needs the js backend for getrandom (pulled transitively by rand).
getrandom = { version = "0.2", features = ["js"] }
console_error_panic_hook = "0.1"

[dev-dependencies]
wasm-bindgen-test = "0.3"
```

- [ ] **Step 3: Create a minimal `rust/quoridor-wasm/src/lib.rs`**

```rust
use wasm_bindgen::prelude::*;

/// Call once from JS on startup to route Rust panics to `console.error`.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Smoke binding to prove the crate links against quoridor-rs core.
#[wasm_bindgen]
pub fn core_board_size(board_size: usize, max_walls: usize, max_steps: usize) -> usize {
    let mechanics =
        quoridor_rs::compact::q_game_mechanics::QGameMechanics::new(board_size, max_walls, max_steps);
    mechanics.repr().board_size()
}
```

- [ ] **Step 4: Build the wasm package**

Run:
```bash
cd rust/quoridor-wasm && wasm-pack build --target web --dev 2>&1 | tail -30
```
Expected: `wasm-pack` compiles `quoridor-wasm` (and `quoridor-rs` with no default features) for `wasm32-unknown-unknown` and emits a `pkg/` directory containing `quoridor_wasm.js` + `quoridor_wasm_bg.wasm`. If it fails on `getrandom`, confirm Step 2's `getrandom` line with `features = ["js"]` is present. If it fails compiling a `quoridor-rs` module, that module needs gating from Task 1 — fix there.

- [ ] **Step 5: Confirm the native workspace build still works**

Run:
```bash
cd rust && cargo build 2>&1 | tail -5
```
Expected: builds quoridor-rs only (the wasm crate is excluded via `default-members`), no attempt to compile `quoridor-wasm` for native.

- [ ] **Step 6: Commit**

```bash
cd rust && git add Cargo.toml quoridor-wasm/Cargo.toml quoridor-wasm/src/lib.rs
git commit -m "feat(wasm): add quoridor-wasm crate skeleton building against the core"
```

---

## Task 3: Portable async batched-search driver in `quoridor-rs`

**Files:**
- Create: `rust/src/agents/alphazero/batched_search.rs`
- Modify: `rust/src/agents/alphazero/mod.rs` (add the `batched_search` line from Task 1 Step 8 if omitted)

This is the one new algorithm. It reuses the existing `mcts.rs` primitives and takes the NN forward pass as an injected async callback, so it is runtime-agnostic (native tests drive it with a synchronous mock; wasm drives it with a JS promise).

- [ ] **Step 1: Write the module with the driver and a failing unit test**

Create `rust/src/agents/alphazero/batched_search.rs`:
```rust
//! Portable leaf-parallel batched MCTS driver.
//!
//! Reuses the `mcts` tree primitives but takes the neural-net forward pass as an
//! injected async callback (`eval_batch`), so the same driver runs natively (with
//! a synchronous mock) and in wasm (awaiting an onnxruntime-web call). It owns
//! `prepare_eval_input` (features + rotation) and `finalize_policy` (masked
//! softmax + un-rotation); the callback is a pure `features -> (value, logits)`.

use std::collections::HashMap;
use std::future::Future;

use anyhow::Result;
use ndarray::Array4;

use crate::actions::action_index_to_action;
use crate::compact::q_bit_repr::CompactState;
use crate::compact::q_game_mechanics::QGameMechanics;

use super::evaluator::{finalize_policy, prepare_eval_input};
use super::mcts::{
    backpropagate, backpropagate_result, expand_node, select_leaf_with_vl, undo_virtual_loss,
    ChildInfo, MCTSConfig, NodeArena,
};

/// Leaf-parallel batching knobs (play-mode subset of `LeafParallelConfig`).
#[derive(Debug, Clone, Copy)]
pub struct BatchedSearchConfig {
    pub leaf_parallelism: u32,
    pub virtual_loss: u32,
}

impl Default for BatchedSearchConfig {
    fn default() -> Self {
        Self { leaf_parallelism: 8, virtual_loss: 1 }
    }
}

/// Raw network outputs for one leaf. `policy_logits` is in the (possibly rotated)
/// work action space — the driver runs `finalize_policy` to get real priors.
#[derive(Clone, Debug)]
pub struct EvalOutput {
    pub value: f32,
    pub policy_logits: Vec<f32>,
}

/// Argmax visit count, first max wins on ties (temperature-0 / deterministic).
pub fn best_action(children: &[ChildInfo]) -> usize {
    children
        .iter()
        .max_by_key(|c| c.visit_count)
        .map(|c| c.action_index)
        .expect("best_action requires at least one child")
}

/// Run `cfg.n` simulations of leaf-parallel batched MCTS from `root_data`.
///
/// `eval_batch(features)` returns one `EvalOutput` per input, in order.
/// `progress(done, total)` is called after each backprop round.
/// Returns `(children, root_value)` where `children` carries per-move visit
/// counts (the play policy) and `root_value` is the negated mean root value.
///
/// Note: this driver always searches from a clean (empty) visited set — it does
/// NOT honor `cfg.penalize_visited_states` (that is a self-play concern). M1
/// play-mode callers leave it `false`.
pub async fn run_batched_search<E, EFut, P>(
    cfg: &MCTSConfig,
    bs_cfg: &BatchedSearchConfig,
    root_data: CompactState,
    mechanics: &QGameMechanics,
    mut eval_batch: E,
    mut progress: P,
) -> Result<(Vec<ChildInfo>, f32)>
where
    E: FnMut(Vec<Array4<f32>>) -> EFut,
    EFut: Future<Output = Result<Vec<EvalOutput>>>,
    P: FnMut(u32, u32),
{
    let mut arena = NodeArena::new(root_data);
    let mut rotation_mappings: HashMap<i32, (Vec<usize>, Vec<usize>)> = HashMap::new();
    let visited = std::collections::HashSet::new();

    let root_mask = mechanics.get_action_mask_immut(root_data);
    let num_valid = root_mask.iter().filter(|&&b| b).count() as u32;
    let total = cfg.n.unwrap_or_else(|| cfg.k.unwrap_or(1) * num_valid).max(1);
    let k = bs_cfg.leaf_parallelism.max(1);
    let vl = bs_cfg.virtual_loss;

    // Pre-expand the root with a single eval so the first parallel round descends
    // into real children instead of all `k` leaves colliding on the unexpanded
    // root (which would call expand_node k times and duplicate every child).
    // Mirrors the root handling in `selfplay_mcts`.
    if !mechanics.is_game_over(root_data) && arena.get(0).should_expand() {
        let prep = prepare_eval_input(mechanics, root_data, &root_mask, &mut rotation_mappings);
        let out = eval_batch(vec![prep.features])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("eval_batch returned no output for root"))?;
        let priors =
            finalize_policy(&out.policy_logits, &prep.work_action_mask, prep.rot_to_orig.as_deref());
        expand_node(&mut arena, 0, &priors, mechanics);
    }

    let mut done: u32 = 0;
    while done < total {
        let outer = (total - done).min(k);

        // Per-leaf bookkeeping for this round.
        struct Pending {
            path: smallvec::SmallVec<[usize; 32]>,
            leaf_idx: usize,
            work_action_mask: Vec<bool>,
            rot_to_orig: Option<Vec<usize>>,
        }
        enum Item {
            Terminal { path: smallvec::SmallVec<[usize; 32]>, value: f64 },
            Eval(Pending),
        }

        let mut items: Vec<Item> = Vec::with_capacity(outer as usize);
        let mut features_batch: Vec<Array4<f32>> = Vec::new();

        for _ in 0..outer {
            let path = select_leaf_with_vl(&mut arena, 0, cfg.ucb_c, vl, &visited);
            let leaf_idx = *path.last().unwrap();
            let leaf_data = arena.get(leaf_idx).data;

            if mechanics.is_game_over(leaf_data) {
                let v = if mechanics.winner(leaf_data).is_some() { 1.0 } else { 0.0 };
                items.push(Item::Terminal { path, value: v });
                continue;
            }
            if let Some(max) = cfg.max_steps {
                if mechanics.repr().get_completed_steps(leaf_data) >= max as usize {
                    items.push(Item::Terminal { path, value: 0.0 });
                    continue;
                }
            }

            let leaf_mask = mechanics.get_action_mask_immut(leaf_data);
            let prep = prepare_eval_input(mechanics, leaf_data, &leaf_mask, &mut rotation_mappings);
            features_batch.push(prep.features);
            items.push(Item::Eval(Pending {
                path,
                leaf_idx,
                work_action_mask: prep.work_action_mask,
                rot_to_orig: prep.rot_to_orig,
            }));
        }

        let outputs = if features_batch.is_empty() {
            Vec::new()
        } else {
            eval_batch(features_batch).await?
        };

        let mut out_iter = outputs.into_iter();
        for item in items {
            match item {
                Item::Terminal { path, value } => {
                    undo_virtual_loss(&mut arena, &path, vl);
                    let leaf = *path.last().unwrap();
                    backpropagate_result(&mut arena, leaf, value);
                }
                Item::Eval(p) => {
                    let out = out_iter
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("eval_batch returned too few outputs"))?;
                    let priors =
                        finalize_policy(&out.policy_logits, &p.work_action_mask, p.rot_to_orig.as_deref());
                    undo_virtual_loss(&mut arena, &p.path, vl);
                    // Guard against intra-round leaf collisions: two leaves selected
                    // in the same round can be the same not-yet-expanded node. Expand
                    // it only once; every colliding leaf still backpropagates its value.
                    if arena.get(p.leaf_idx).should_expand() {
                        expand_node(&mut arena, p.leaf_idx, &priors, mechanics);
                    }
                    backpropagate(&mut arena, p.leaf_idx, -out.value as f64);
                }
            }
            done += 1;
            if done >= total {
                break;
            }
        }
        progress(done, total);
    }

    // Build the child (visit-count) policy and root value.
    let bs = mechanics.repr().board_size() as i32;
    let root = arena.get(0);
    let mut children = Vec::with_capacity(root.children.len());
    for &ci in &root.children {
        let c = arena.get(ci);
        let ai = c.action_index.expect("child has an action index");
        children.push(ChildInfo {
            action: action_index_to_action(bs, ai),
            action_index: ai,
            visit_count: c.visit_count,
        });
    }
    let root_value = if root.visit_count > 0 {
        -(root.value_sum / root.visit_count as f64) as f32
    } else {
        0.0
    };
    Ok((children, root_value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compact::q_game_mechanics::QGameMechanics;
    use std::cell::Cell;
    use std::collections::HashSet;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    /// Minimal dependency-free executor for driving an immediately-ready future.
    fn block_on<F: Future>(mut fut: F) -> F::Output {
        fn noop(_: *const ()) {}
        fn clone(_: *const ()) -> RawWaker {
            RawWaker::new(std::ptr::null(), &VTABLE)
        }
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
        let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
        let mut cx = Context::from_waker(&waker);
        // Safety: we never move `fut` after pinning it here.
        let mut fut = unsafe { Pin::new_unchecked(&mut fut) };
        loop {
            if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
                return v;
            }
        }
    }

    /// Uniform mock: value 0, all-zero logits (→ uniform priors over legal moves).
    fn mock_eval(batch: Vec<Array4<f32>>) -> impl Future<Output = Result<Vec<EvalOutput>>> {
        // policy width = channels are irrelevant; we return zero logits sized to
        // the policy space. The driver's finalize_policy masks illegal actions,
        // and zero logits over the legal set give a uniform prior regardless of
        // vector length as long as it covers the action space. We size it to a
        // safe upper bound from the feature grid.
        let out: Vec<EvalOutput> = batch
            .iter()
            .map(|_| EvalOutput { value: 0.0, policy_logits: vec![0.0f32; 512] })
            .collect();
        std::future::ready(Ok(out))
    }

    #[test]
    fn search_returns_a_legal_move_and_counts_sum_to_n() {
        let mechanics = QGameMechanics::new(5, 2, 50);
        let root = mechanics.create_initial_state();
        let cfg = MCTSConfig { n: Some(64), noise_epsilon: 0.0, ..MCTSConfig::default() };
        let bs = BatchedSearchConfig { leaf_parallelism: 8, virtual_loss: 1 };

        let progress_calls = Cell::new(0u32);
        let last = Cell::new((0u32, 0u32));
        let (children, _root_value) = block_on(run_batched_search(
            &cfg,
            &bs,
            root,
            &mechanics,
            mock_eval,
            |d, t| {
                progress_calls.set(progress_calls.get() + 1);
                last.set((d, t));
            },
        ))
        .unwrap();

        assert!(!children.is_empty(), "root must have expanded children");
        let mask = mechanics.get_action_mask_immut(root);
        let chosen = best_action(&children);
        assert!(mask[chosen], "chosen action must be legal");

        // Each legal action appears exactly once — duplicate action_index values
        // are the signature of the leaf-collision / double-expand bug.
        let unique: HashSet<usize> = children.iter().map(|c| c.action_index).collect();
        assert_eq!(unique.len(), children.len(), "children must have distinct actions");

        // With the root pre-expanded, essentially every simulation lands a visit
        // on a child; allow one batch of slack for in-flight rounds.
        let total_visits: u32 = children.iter().map(|c| c.visit_count).sum();
        assert!(total_visits >= 64 - 8, "child visit sum ({total_visits}) should be ~n=64");

        assert!(progress_calls.get() >= 1, "progress fired at least once");
        assert_eq!(last.get(), (64, 64), "progress ends at (n, n)");
    }
}
```
Note: the mock's `policy_logits` length (512) must be ≥ the action-space size for a 5×5 board. Confirm `crate::actions::policy_size(5)` ≤ 512 (it is; 5×5 policy is well under 512). If `policy_size` is importable, prefer `vec![0.0f32; crate::actions::policy_size(5)]` in the mock.

- [ ] **Step 2: Ensure `batched_search` is declared in `mod.rs`**

Confirm `rust/src/agents/alphazero/mod.rs` contains `pub mod batched_search;` (add it if Task 1 Step 8 omitted it).

- [ ] **Step 3: Run the test — expect it to compile and pass**

Run:
```bash
cd rust && cargo test --no-default-features batched_search 2>&1 | tail -30
```
Expected: `search_returns_a_legal_move_and_counts_sum_to_n` passes. If `finalize_policy`/`prepare_eval_input`/primitive signatures differ from those referenced, adjust the call sites to the exact signatures in `mcts.rs`/`evaluator.rs` (all are `pub`).

- [ ] **Step 4: Add a determinism test (same inputs → same visit distribution)**

Append to the `tests` module in `batched_search.rs`:
```rust
#[test]
fn search_is_deterministic_without_noise() {
    let mechanics = QGameMechanics::new(5, 2, 50);
    let root = mechanics.create_initial_state();
    let cfg = MCTSConfig { n: Some(48), noise_epsilon: 0.0, ..MCTSConfig::default() };
    let bs = BatchedSearchConfig { leaf_parallelism: 4, virtual_loss: 1 };

    let run = || {
        block_on(run_batched_search(&cfg, &bs, root, &mechanics, mock_eval, |_, _| {}))
            .unwrap()
            .0
            .iter()
            .map(|c| (c.action_index, c.visit_count))
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run(), "no-noise search must be reproducible");
}
```

- [ ] **Step 5: Run both tests**

Run:
```bash
cd rust && cargo test --no-default-features batched_search 2>&1 | tail -20
```
Expected: both tests pass. Also run `cargo test --features binary 2>&1 | tail -5` to confirm the binary build still compiles with the new module.

- [ ] **Step 6: Commit**

```bash
cd rust && git add src/agents/alphazero/batched_search.rs src/agents/alphazero/mod.rs
git commit -m "feat(mcts): portable async leaf-parallel batched search driver with injected eval"
```

---

## Task 4: `WasmGame` session bindings

**Files:**
- Create: `rust/quoridor-wasm/src/view.rs`
- Create: `rust/quoridor-wasm/src/game.rs`
- Modify: `rust/quoridor-wasm/src/lib.rs`

- [ ] **Step 1: Create `rust/quoridor-wasm/src/view.rs` (ported serde view types)**

```rust
//! JSON-facing snapshot the JS client renders from. Ported from
//! quoridor-rs `play_server::state` (which is binary-gated and not on wasm).

use serde::Serialize;

use quoridor_rs::actions::{
    action_index_to_action, ACTION_MOVE, ACTION_WALL_HORIZONTAL, ACTION_WALL_VERTICAL,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WallOrientation {
    H,
    V,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum EnrichedAction {
    Move { index: u32, to: [i32; 2] },
    Wall { index: u32, row: i32, col: i32, orientation: WallOrientation },
}

#[derive(Debug, Clone, Serialize)]
pub struct WallEntry {
    pub row: i32,
    pub col: i32,
    pub orientation: WallOrientation,
}

#[derive(Debug, Clone, Serialize)]
pub struct StateView {
    pub board_size: i32,
    pub max_walls: i32,
    pub max_steps: i32,
    pub current_player: i32,
    pub p1_pos: [i32; 2],
    pub p2_pos: [i32; 2],
    pub p1_walls: i32,
    pub p2_walls: i32,
    pub walls: Vec<WallEntry>,
    pub legal_actions: Vec<EnrichedAction>,
    pub completed_steps: i32,
    pub winner: Option<i32>,
    pub human_player: i32,
    pub last_action: Option<EnrichedAction>,
    pub move_history: Vec<u32>,
}

pub fn enrich_action(board_size: i32, index: usize) -> EnrichedAction {
    let [row, col, action_type] = action_index_to_action(board_size, index);
    match action_type {
        ACTION_WALL_VERTICAL => EnrichedAction::Wall {
            index: index as u32, row, col, orientation: WallOrientation::V,
        },
        ACTION_WALL_HORIZONTAL => EnrichedAction::Wall {
            index: index as u32, row, col, orientation: WallOrientation::H,
        },
        ACTION_MOVE => EnrichedAction::Move { index: index as u32, to: [row, col] },
        other => panic!("unexpected action type {other} for index {index}"),
    }
}

pub fn enrich_legal_actions(board_size: i32, mask: &[bool]) -> Vec<EnrichedAction> {
    mask.iter()
        .enumerate()
        .filter(|&(_, legal)| *legal)
        .map(|(i, _)| enrich_action(board_size, i))
        .collect()
}
```
Confirm `ACTION_MOVE`, `ACTION_WALL_VERTICAL`, `ACTION_WALL_HORIZONTAL`, and `action_index_to_action` are `pub` in `quoridor_rs::actions` (they are used the same way in `play_server::state`). If `actions` is not a public module path, add `pub mod actions;` is already present per lib.rs.

- [ ] **Step 2: Create `rust/quoridor-wasm/src/game.rs` with `WasmGame` and native unit tests**

```rust
//! Per-game session exposed to JS. Owns mechanics + state + move history and
//! produces `StateView` snapshots.

use quoridor_rs::compact::q_bit_repr::{CompactState, WALL_HORIZONTAL, WALL_VERTICAL};
use quoridor_rs::compact::q_game_mechanics::QGameMechanics;

use crate::view::{
    enrich_action, enrich_legal_actions, EnrichedAction, StateView, WallEntry, WallOrientation,
};

pub struct WasmGame {
    mechanics: QGameMechanics,
    state: CompactState,
    board_size: i32,
    max_walls: i32,
    max_steps: i32,
    human_player: i32,
    last_action: Option<EnrichedAction>,
    move_history: Vec<u32>,
}

impl WasmGame {
    pub fn new(board_size: i32, max_walls: i32, max_steps: i32, human_player: i32) -> Self {
        let mechanics =
            QGameMechanics::new(board_size as usize, max_walls as usize, max_steps as usize);
        let state = mechanics.create_initial_state();
        Self {
            mechanics, state, board_size, max_walls, max_steps, human_player,
            last_action: None, move_history: Vec::new(),
        }
    }

    fn current_player(&self) -> i32 {
        self.mechanics.repr().get_current_player(self.state) as i32
    }

    fn is_game_over(&self) -> bool {
        self.mechanics.is_game_over(self.state)
            || self.mechanics.repr().get_completed_steps(self.state) >= self.max_steps as usize
    }

    pub fn legal_mask(&self) -> Vec<bool> {
        self.mechanics.get_action_mask_immut(self.state)
    }

    /// Apply any legal action (human or AI). Returns Err on illegal/over.
    pub fn apply_action(&mut self, action_index: u32) -> Result<(), String> {
        if self.is_game_over() {
            return Err("game is already over".into());
        }
        let mask = self.legal_mask();
        let idx = action_index as usize;
        if idx >= mask.len() || !mask[idx] {
            return Err(format!("action {action_index} is not legal"));
        }
        self.last_action = Some(enrich_action(self.board_size, idx));
        self.move_history.push(action_index);
        self.mechanics.apply_action_index(&mut self.state, idx);
        Ok(())
    }

    /// Undo the last `count` plies by replaying history from the initial state.
    pub fn undo(&mut self, count: usize) {
        let keep = self.move_history.len().saturating_sub(count);
        let replay: Vec<u32> = self.move_history[..keep].to_vec();
        self.state = self.mechanics.create_initial_state();
        self.move_history.clear();
        self.last_action = None;
        for a in replay {
            let idx = a as usize;
            self.last_action = Some(enrich_action(self.board_size, idx));
            self.move_history.push(a);
            self.mechanics.apply_action_index(&mut self.state, idx);
        }
    }

    pub fn state(&self) -> CompactState {
        self.state
    }
    pub fn mechanics(&self) -> &QGameMechanics {
        &self.mechanics
    }

    pub fn view(&self) -> StateView {
        let mask = self.legal_mask();
        let repr = self.mechanics.repr();
        let (p1r, p1c) = repr.get_player_position(self.state, 0);
        let (p2r, p2c) = repr.get_player_position(self.state, 1);
        let winner = if self.mechanics.check_win(self.state, 0) {
            Some(0)
        } else if self.mechanics.check_win(self.state, 1) {
            Some(1)
        } else {
            None
        };
        StateView {
            board_size: self.board_size,
            max_walls: self.max_walls,
            max_steps: self.max_steps,
            current_player: self.current_player(),
            p1_pos: [p1r as i32, p1c as i32],
            p2_pos: [p2r as i32, p2c as i32],
            p1_walls: repr.get_walls_remaining(self.state, 0) as i32,
            p2_walls: repr.get_walls_remaining(self.state, 1) as i32,
            walls: list_walls(&self.mechanics, self.state, self.board_size),
            legal_actions: enrich_legal_actions(self.board_size, &mask),
            completed_steps: repr.get_completed_steps(self.state) as i32,
            winner,
            human_player: self.human_player,
            last_action: self.last_action.clone(),
            move_history: self.move_history.clone(),
        }
    }
}

fn list_walls(mechanics: &QGameMechanics, state: CompactState, board_size: i32) -> Vec<WallEntry> {
    let mut out = Vec::new();
    let wall_size = (board_size - 1) as usize;
    for (orientation_const, orientation) in
        [(WALL_VERTICAL, WallOrientation::V), (WALL_HORIZONTAL, WallOrientation::H)]
    {
        for row in 0..wall_size {
            for col in 0..wall_size {
                if mechanics.repr().get_wall(state, row, col, orientation_const) {
                    out.push(WallEntry { row: row as i32, col: col as i32, orientation });
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_then_undo_restores_state() {
        let mut g = WasmGame::new(5, 2, 50, 0);
        let mask = g.legal_mask();
        let first = mask.iter().position(|&b| b).unwrap() as u32;
        g.apply_action(first).unwrap();
        assert_eq!(g.view().move_history, vec![first]);
        g.undo(1);
        let v = g.view();
        assert!(v.move_history.is_empty());
        assert!(v.last_action.is_none());
        assert_eq!(v.current_player, 0);
    }

    #[test]
    fn apply_rejects_illegal_action() {
        let mut g = WasmGame::new(5, 2, 50, 0);
        let err = g.apply_action(u32::MAX).unwrap_err();
        assert!(err.contains("not legal"));
    }
}
```

- [ ] **Step 3: Wire the modules and a `#[wasm_bindgen]` facade in `rust/quoridor-wasm/src/lib.rs`**

Replace `lib.rs` with:
```rust
use wasm_bindgen::prelude::*;

mod game;
mod view;

use game::WasmGame;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// JS-facing handle around a game session.
#[wasm_bindgen]
pub struct Game {
    inner: WasmGame,
}

#[wasm_bindgen]
impl Game {
    #[wasm_bindgen(constructor)]
    pub fn new(board_size: i32, max_walls: i32, max_steps: i32, human_player: i32) -> Game {
        Game { inner: WasmGame::new(board_size, max_walls, max_steps, human_player) }
    }

    /// Returns the `StateView` as a JS object.
    #[wasm_bindgen(js_name = stateView)]
    pub fn state_view(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.inner.view()).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen(js_name = applyAction)]
    pub fn apply_action(&mut self, action_index: u32) -> Result<JsValue, JsValue> {
        self.inner.apply_action(action_index).map_err(|e| JsValue::from_str(&e))?;
        self.state_view()
    }

    pub fn undo(&mut self, count: usize) -> Result<JsValue, JsValue> {
        self.inner.undo(count);
        self.state_view()
    }
}
```

- [ ] **Step 4: Run the native unit tests for the wasm crate's inner logic**

Run:
```bash
cd rust && cargo test -p quoridor-wasm 2>&1 | tail -20
```
Expected: `apply_then_undo_restores_state` and `apply_rejects_illegal_action` pass. (These test `WasmGame` directly on the native target — no browser needed. `cargo test -p` compiles the crate for native, where `wasm-bindgen` types still compile.)

- [ ] **Step 5: Rebuild the wasm package to confirm the bindings compile to wasm**

Run:
```bash
cd rust/quoridor-wasm && wasm-pack build --target web --dev 2>&1 | tail -15
```
Expected: `pkg/` regenerates with a `Game` class in `quoridor_wasm.js`.

- [ ] **Step 6: Commit**

```bash
cd rust && git add quoridor-wasm/src/view.rs quoridor-wasm/src/game.rs quoridor-wasm/src/lib.rs
git commit -m "feat(wasm): WasmGame session bindings (new/apply/undo/stateView) with native tests"
```

---

## Task 5: `run_search` binding — wire JS eval + progress to the driver

**Files:**
- Create: `rust/quoridor-wasm/src/search.rs`
- Modify: `rust/quoridor-wasm/src/lib.rs`
- Create: `rust/quoridor-wasm/tests/web.rs`

The JS side supplies an async batched-eval function `evalBatch(featuresFlat: Float32Array, batch: number, channels: number, size: number) -> Promise<{values: Float32Array, logits: Float32Array}>` and a `progress(done, total)` callback. This task marshals the Rust feature batch into a flat `Float32Array`, awaits the JS promise, and splits the result back into `EvalOutput`s.

- [ ] **Step 1: Create `rust/quoridor-wasm/src/search.rs`**

```rust
//! Wires the portable `run_batched_search` driver to JS callbacks: an async
//! batched NN eval (onnxruntime-web) and a progress reporter.

use js_sys::{Array, Float32Array, Object, Reflect};
use ndarray::Array4;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use quoridor_rs::agents::alphazero::batched_search::{
    best_action, run_batched_search, BatchedSearchConfig, EvalOutput,
};
use quoridor_rs::agents::alphazero::mcts::MCTSConfig;

use crate::game::WasmGame;

/// Flatten a batch of [1, C, H, W] feature tensors into one contiguous
/// Float32Array of shape [N, C, H, W] plus its dims.
fn flatten_batch(batch: &[Array4<f32>]) -> (Float32Array, usize, usize, usize) {
    let n = batch.len();
    let (c, h, w) = if n == 0 {
        (0, 0, 0)
    } else {
        let s = batch[0].shape();
        (s[1], s[2], s[3])
    };
    let per = c * h * w;
    let flat = Float32Array::new_with_length((n * per) as u32);
    for (i, arr) in batch.iter().enumerate() {
        let contiguous: Vec<f32> = arr.iter().copied().collect();
        flat.subarray((i * per) as u32, ((i + 1) * per) as u32)
            .copy_from(&contiguous);
    }
    (flat, c, h, w)
}

/// Split the JS result `{ values: Float32Array[N], logits: Float32Array[N*P] }`
/// into per-leaf `EvalOutput`s.
fn split_outputs(result: &JsValue, n: usize) -> Result<Vec<EvalOutput>, JsValue> {
    let values: Float32Array = Reflect::get(result, &JsValue::from_str("values"))?.dyn_into()?;
    let logits: Float32Array = Reflect::get(result, &JsValue::from_str("logits"))?.dyn_into()?;
    let values_v = values.to_vec();
    let logits_v = logits.to_vec();
    if values_v.len() != n {
        return Err(JsValue::from_str("eval result 'values' length != batch size"));
    }
    let p = if n == 0 { 0 } else { logits_v.len() / n };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(EvalOutput {
            value: values_v[i],
            policy_logits: logits_v[i * p..(i + 1) * p].to_vec(),
        });
    }
    Ok(out)
}

/// Run MCTS for the current position, calling `eval_batch` (async JS) for NN
/// forward passes and `progress` after each round. Resolves to
/// `{ action, rootValue, children: [{actionIndex, visitCount}] }`.
pub async fn run_search_js(
    game: &WasmGame,
    mcts_n: u32,
    c_puct: f32,
    leaf_parallelism: u32,
    virtual_loss: u32,
    eval_batch: js_sys::Function,
    progress: js_sys::Function,
) -> Result<JsValue, JsValue> {
    // Guard the boundary: searching a finished game leaves the root unexpanded,
    // so `run_batched_search` returns no children and `best_action` would panic
    // (a hard wasm abort under `panic = "abort"`). Surface a catchable JS error.
    if game.mechanics().is_game_over(game.state()) {
        return Err(JsValue::from_str("cannot run search: game is already over"));
    }

    let cfg = MCTSConfig {
        n: Some(mcts_n),
        k: None,
        ucb_c: c_puct,
        noise_epsilon: 0.0,
        noise_alpha: None,
        // Play mode terminates on a winner and the JS layer stops at game end,
        // so a hard step cap isn't needed here for M1.
        max_steps: None,
        penalize_visited_states: false,
    };
    let bs = BatchedSearchConfig { leaf_parallelism, virtual_loss };

    let eval = |batch: Vec<Array4<f32>>| {
        let eval_batch = eval_batch.clone();
        async move {
            let n = batch.len();
            let (flat, c, h, w) = flatten_batch(&batch);
            let args = Array::new();
            args.push(&flat);
            args.push(&JsValue::from_f64(n as f64));
            args.push(&JsValue::from_f64(c as f64));
            args.push(&JsValue::from_f64(h as f64));
            args.push(&JsValue::from_f64(w as f64));
            let promise = eval_batch
                .apply(&JsValue::NULL, &args)
                .map_err(|e| anyhow::anyhow!("eval_batch threw: {e:?}"))?;
            let promise: js_sys::Promise = promise
                .dyn_into()
                .map_err(|_| anyhow::anyhow!("eval_batch must return a Promise"))?;
            let result = JsFuture::from(promise)
                .await
                .map_err(|e| anyhow::anyhow!("eval_batch rejected: {e:?}"))?;
            split_outputs(&result, n).map_err(|e| anyhow::anyhow!("split_outputs: {e:?}"))
        }
    };

    let report = |done: u32, total: u32| {
        let _ = progress.call2(&JsValue::NULL, &JsValue::from_f64(done as f64), &JsValue::from_f64(total as f64));
    };

    let (children, root_value) =
        run_batched_search(&cfg, &bs, game.state(), game.mechanics(), eval, report)
            .await
            .map_err(|e| JsValue::from_str(&format!("{e:#}")))?;

    let action = best_action(&children);
    let out = Object::new();
    Reflect::set(&out, &JsValue::from_str("action"), &JsValue::from_f64(action as f64))?;
    Reflect::set(&out, &JsValue::from_str("rootValue"), &JsValue::from_f64(root_value as f64))?;
    let arr = Array::new();
    for c in &children {
        let o = Object::new();
        Reflect::set(&o, &JsValue::from_str("actionIndex"), &JsValue::from_f64(c.action_index as f64))?;
        Reflect::set(&o, &JsValue::from_str("visitCount"), &JsValue::from_f64(c.visit_count as f64))?;
        arr.push(&o);
    }
    Reflect::set(&out, &JsValue::from_str("children"), &arr)?;
    Ok(out.into())
}
```

- [ ] **Step 2: Expose `runSearch` on the `Game` class in `lib.rs`**

Add `mod search;` near the other `mod` lines, and add this method inside `#[wasm_bindgen] impl Game`:
```rust
    #[wasm_bindgen(js_name = runSearch)]
    pub async fn run_search(
        &self,
        mcts_n: u32,
        c_puct: f32,
        leaf_parallelism: u32,
        virtual_loss: u32,
        eval_batch: js_sys::Function,
        progress: js_sys::Function,
    ) -> Result<JsValue, JsValue> {
        crate::search::run_search_js(
            &self.inner, mcts_n, c_puct, leaf_parallelism, virtual_loss, eval_batch, progress,
        )
        .await
    }
```
Make `WasmGame::state` and `WasmGame::mechanics` accessible (they were made `pub` in Task 4 Step 2). The `Game` holds `inner: WasmGame`, so `&self.inner` is passed.

- [ ] **Step 3: Write a `wasm-bindgen-test` that drives `runSearch` with a JS mock eval**

Create `rust/quoridor-wasm/tests/web.rs`:
```rust
use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

// A JS mock eval that resolves to zero values + zero logits of width 512.
// NOTE on the runner: with `run_in_browser` above, run `wasm-pack test
// --headless --chrome` and keep the ESM `export function` form below. To run
// under Node instead (`wasm-pack test --node`), REMOVE the `run_in_browser`
// line and change the two `export function X` to `module.exports.X = function`
// (the Node harness is CommonJS). Pick one; they don't mix.
#[wasm_bindgen(inline_js = "
export function makeMockEval() {
  return function(flat, n, c, h, w) {
    const values = new Float32Array(n);
    const logits = new Float32Array(n * 512);
    return Promise.resolve({ values, logits });
  };
}
export function makeProgress() { return function(_d, _t) {}; }
")]
extern "C" {
    fn makeMockEval() -> js_sys::Function;
    fn makeProgress() -> js_sys::Function;
}

#[wasm_bindgen_test]
async fn run_search_returns_a_legal_action() {
    quoridor_wasm::init();
    let game = quoridor_wasm::Game::new(5, 2, 50, 0);
    let result = game
        .run_search(32, 1.4, 8, 1, makeMockEval(), makeProgress())
        .await
        .unwrap();

    let action = js_sys::Reflect::get(&result, &JsValue::from_str("action")).unwrap();
    let action = action.as_f64().unwrap() as u32;

    // Cross-check against the game's own legal mask via stateView.legal_actions.
    let view = game.state_view().unwrap();
    let legal = js_sys::Reflect::get(&view, &JsValue::from_str("legal_actions")).unwrap();
    let legal: js_sys::Array = legal.dyn_into().unwrap();
    let mut found = false;
    for i in 0..legal.length() {
        let a = legal.get(i);
        let idx = js_sys::Reflect::get(&a, &JsValue::from_str("index")).unwrap().as_f64().unwrap() as u32;
        if idx == action { found = true; break; }
    }
    assert!(found, "runSearch action must be one of the legal actions");
}
```
For this to compile, mark the crate's public items reachable from an integration test: `Game`, `Game::new`, `Game::run_search`, `Game::state_view`, and `init` must be `pub` (they are, via `#[wasm_bindgen]`).

- [ ] **Step 4: Run the browser test**

Run (Chrome/Chromium must be installed; wasm-pack drives a headless browser):
```bash
cd rust/quoridor-wasm && wasm-pack test --headless --chrome 2>&1 | tail -30
```
Expected: `run_search_returns_a_legal_action` passes. If no headless Chrome is available, run `wasm-pack test --node` instead (the mock uses only `Promise`/`Float32Array`, which work under Node). Document whichever runner is used.

- [ ] **Step 5: Confirm native crate tests still pass and the package still builds**

Run:
```bash
cd rust && cargo test -p quoridor-wasm 2>&1 | tail -10
cd quoridor-wasm && wasm-pack build --target web --dev 2>&1 | tail -8
```
Expected: native `WasmGame` tests pass; `pkg/` builds with `Game.runSearch` present.

- [ ] **Step 6: Commit**

```bash
cd rust && git add quoridor-wasm/src/search.rs quoridor-wasm/src/lib.rs quoridor-wasm/tests/web.rs
git commit -m "feat(wasm): runSearch binding wiring async JS eval + progress to batched MCTS"
```

---

## Task 6: Documentation and wrap-up

**Files:**
- Create: `rust/quoridor-wasm/README.md`

- [ ] **Step 1: Write `rust/quoridor-wasm/README.md`**

```markdown
# quoridor-wasm

WebAssembly bindings for the Quoridor game core + AlphaZero MCTS, for running the
AI in the browser. See `docs/superpowers/specs/2026-07-05-browser-wasm-play-server-design.md`.

## Build
```
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
cd rust/quoridor-wasm && wasm-pack build --target web --release
```
Output is in `pkg/` (import `quoridor_wasm.js`, which loads `quoridor_wasm_bg.wasm`).

## JS API
- `new Game(board_size, max_walls, max_steps, human_player)`
- `game.stateView()` → StateView object
- `game.applyAction(actionIndex)` → StateView
- `game.undo(count)` → StateView
- `game.runSearch(mctsN, cPuct, leafParallelism, virtualLoss, evalBatch, progress)` → `{ action, rootValue, children }`

`evalBatch(flat: Float32Array, n, c, h, w) => Promise<{values: Float32Array, logits: Float32Array}>`
runs the NN forward pass (onnxruntime-web / WebGPU) — supplied by the frontend in Plan 3.
`progress(done, total)` is called after each search round.

## Tests
- `cargo test -p quoridor-wasm` — native unit tests (game/undo logic).
- `wasm-pack test --headless --chrome` (or `--node`) — browser bindings test.
```

- [ ] **Step 2: Full verification sweep**

Run:
```bash
cd rust && cargo test --features binary 2>&1 | tail -5      # existing native suite unchanged
cargo test --no-default-features 2>&1 | tail -5             # core + batched_search
cargo test -p quoridor-wasm 2>&1 | tail -5                  # wasm crate native tests
cd quoridor-wasm && wasm-pack build --target web --release 2>&1 | tail -5
```
Expected: all green; release `pkg/` builds.

- [ ] **Step 3: Commit**

```bash
cd rust && git add quoridor-wasm/README.md
git commit -m "docs(wasm): quoridor-wasm README with build + JS API"
```

---

## Notes for the frontend (Plan 3) — not built here

- The model input tensor is named `"input"`, outputs are `"value"` and `"policy_logits"`; the resnet input is `[N, 5, M, M]` with `M = board_size*2+3`. `evalBatch` receives the flattened `[N,5,M,M]` features and must return `values[N]` and `logits[N, policy_size]` (raw logits — the Rust side applies masked softmax + un-rotation).
- `runSearch`'s result `children` (visit counts) is the play policy and the hook for the M2 policy heatmap.

## Follow-ups deferred (out of scope for Plan 1)

- **Real-model parity test** vs native `selfplay_mcts` (needs a real `.onnx` + the ort pipeline). The determinism test here guards the driver logic; full parity belongs with the integration spike.
- **Eval caching** (a per-search `HashMap<CompactState, EvalOutput>`) — self-play uses one; skipped for M1 simplicity, easy to add to `run_batched_search`.
- **Dirichlet noise** — play mode uses `noise_epsilon = 0`, so no root noise is applied. (Root pre-expansion itself *is* implemented — Task 3 pre-expands the root before the parallel loop to avoid leaf-collision duplicate children.)
