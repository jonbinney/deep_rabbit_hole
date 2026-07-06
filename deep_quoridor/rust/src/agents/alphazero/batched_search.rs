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
    ChildInfo, MCTSConfig, NodeArena, backpropagate, backpropagate_result, expand_node,
    select_leaf_with_vl, undo_virtual_loss,
};

/// Leaf-parallel batching knobs (play-mode subset of `LeafParallelConfig`).
#[derive(Debug, Clone, Copy)]
pub struct BatchedSearchConfig {
    pub leaf_parallelism: u32,
    pub virtual_loss: u32,
}

impl Default for BatchedSearchConfig {
    fn default() -> Self {
        Self {
            leaf_parallelism: 8,
            virtual_loss: 1,
        }
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
/// Note: this driver ignores `cfg.penalize_visited_states`; it always searches
/// with a clean (empty) visited set. Penalizing visited states is a self-play
/// concern; M1 play-mode leaves it false.
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
    let total = cfg
        .n
        .unwrap_or_else(|| cfg.k.unwrap_or(1) * num_valid)
        .max(1);
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
        let priors = finalize_policy(
            &out.policy_logits,
            &prep.work_action_mask,
            prep.rot_to_orig.as_deref(),
        );
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
            Terminal {
                path: smallvec::SmallVec<[usize; 32]>,
                value: f64,
            },
            Eval(Pending),
        }

        let mut items: Vec<Item> = Vec::with_capacity(outer as usize);
        let mut features_batch: Vec<Array4<f32>> = Vec::new();

        for _ in 0..outer {
            let path = select_leaf_with_vl(&mut arena, 0, cfg.ucb_c, vl, &visited);
            let leaf_idx = *path.last().unwrap();
            let leaf_data = arena.get(leaf_idx).data;

            if mechanics.is_game_over(leaf_data) {
                let v = if mechanics.winner(leaf_data).is_some() {
                    1.0
                } else {
                    0.0
                };
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
                    let priors = finalize_policy(
                        &out.policy_logits,
                        &p.work_action_mask,
                        p.rot_to_orig.as_deref(),
                    );
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
            .map(|_| EvalOutput {
                value: 0.0,
                policy_logits: vec![0.0f32; 512],
            })
            .collect();
        std::future::ready(Ok(out))
    }

    #[test]
    fn search_returns_a_legal_move_and_counts_sum_to_n() {
        let mechanics = QGameMechanics::new(5, 2, 50);
        let root = mechanics.create_initial_state();
        let cfg = MCTSConfig {
            n: Some(64),
            noise_epsilon: 0.0,
            ..MCTSConfig::default()
        };
        let bs = BatchedSearchConfig {
            leaf_parallelism: 8,
            virtual_loss: 1,
        };

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
        assert_eq!(
            unique.len(),
            children.len(),
            "children must have distinct actions"
        );

        // With the root pre-expanded, essentially every simulation lands a visit
        // on a child; allow one batch of slack for in-flight rounds.
        let total_visits: u32 = children.iter().map(|c| c.visit_count).sum();
        assert!(
            total_visits >= 64 - 8,
            "child visit sum ({total_visits}) should be ~n=64"
        );

        assert!(progress_calls.get() >= 1, "progress fired at least once");
        assert_eq!(last.get(), (64, 64), "progress ends at (n, n)");
    }

    #[test]
    fn search_is_deterministic_without_noise() {
        let mechanics = QGameMechanics::new(5, 2, 50);
        let root = mechanics.create_initial_state();
        let cfg = MCTSConfig {
            n: Some(48),
            noise_epsilon: 0.0,
            ..MCTSConfig::default()
        };
        let bs = BatchedSearchConfig {
            leaf_parallelism: 4,
            virtual_loss: 1,
        };

        let run = || {
            block_on(run_batched_search(
                &cfg,
                &bs,
                root,
                &mechanics,
                mock_eval,
                |_, _| {},
            ))
            .unwrap()
            .0
            .iter()
            .map(|c| (c.action_index, c.visit_count))
            .collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "no-noise search must be reproducible");
    }
}
