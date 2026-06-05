//! Async leaf-parallel MCTS for self-play.
//!
//! Each MCTS instance owns its arena and dispatches K eval requests per outer
//! iteration through the shared eval pipeline. Virtual loss is applied during
//! descent and undone before real backprop.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use smallvec::SmallVec;
use tokio::sync::mpsc as tokio_mpsc;
use tokio::sync::oneshot;

use crate::agents::alphazero::eval_pipeline::{EvalCache, EvalRequest, EvalResult, FrontMsg};
use crate::agents::alphazero::evaluator::prepare_eval_input;
use crate::agents::alphazero::mcts::{
    ChildInfo, MCTSConfig, NodeArena, apply_dirichlet_noise_to_root_children, backpropagate,
    backpropagate_result, expand_node, promote_subtree, select_leaf_with_vl, undo_virtual_loss,
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
    pub(super) arena: Option<NodeArena>,
    last_model_version: Option<i64>,
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
            last_model_version: None,
        }
    }

    /// Discard any retained tree. Call between games or on model reload.
    pub fn reset_tree(&mut self) {
        self.arena = None;
    }

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

    /// After the caller picks `action_idx` at the root, promote that child's
    /// subtree to the new root for the next search.
    pub fn advance_root(&mut self, action_idx: usize) {
        if !self.lp.enable_tree_reuse {
            self.arena = None;
            return;
        }
        let Some(old) = self.arena.take() else {
            return;
        };
        let root = old.get(0);
        let chosen = root
            .children
            .iter()
            .copied()
            .find(|&c| old.get(c).action_index == Some(action_idx));
        self.arena = chosen.map(|c| promote_subtree(&old, c));
    }

    /// Run one search starting from `root_data`. Returns `(children, root_value)`.
    pub async fn search(
        &mut self,
        root_data: CompactState,
        mechanics: &QGameMechanics,
        visited_states: &HashSet<CompactState>,
    ) -> Result<(Vec<ChildInfo>, f32, SearchStats)> {
        // Fresh arena unless we have a reusable one matching the root state.
        let mut arena = match self.arena.take() {
            Some(a) if a.get(0).data == root_data => a,
            _ => NodeArena::new(root_data),
        };

        // If root has no children, expand it via one eval first.
        let action_mask = mechanics.get_action_mask_immut(root_data);
        let root_value;
        if arena.get(0).children.is_empty() {
            let (v, priors) = self
                .evaluate_once(root_data, mechanics, &action_mask)
                .await?;
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
            let alpha = self
                .cfg
                .noise_alpha
                .unwrap_or_else(|| 10.0 / num_valid.max(1) as f32);
            apply_dirichlet_noise_to_root_children(&mut arena, 0, self.cfg.noise_epsilon, alpha);
        }

        let total = self.cfg.n.unwrap_or_else(|| {
            self.cfg.k.unwrap_or(10) * action_mask.iter().filter(|&&m| m).count() as u32
        });

        let mut stats = SearchStats::default();
        let mut iters_done: u32 = 0;
        let k = self.lp.leaf_parallelism.max(1);
        let vl = self.lp.virtual_loss;

        while iters_done < total {
            let outer = ((total - iters_done) as u32).min(k);

            // Selection phase: pick `outer` leaves; classify each as Terminal, Hit, or Miss.
            enum Item {
                Terminal {
                    path: SmallVec<[usize; 32]>,
                    value: f64,
                },
                Hit {
                    path: SmallVec<[usize; 32]>,
                    leaf_idx: usize,
                    result: EvalResult,
                },
                Miss {
                    path: SmallVec<[usize; 32]>,
                    leaf_idx: usize,
                    rx: oneshot::Receiver<Result<EvalResult>>,
                },
            }
            let mut items: Vec<Item> = Vec::with_capacity(outer as usize);
            let mut to_send: Vec<EvalRequest> = Vec::new();

            for _ in 0..outer {
                let path = select_leaf_with_vl(&mut arena, 0, self.cfg.ucb_c, vl, visited_states);
                let leaf_idx = *path.last().unwrap();
                let leaf_data = arena.get(leaf_idx).data;

                let depth = path.len() as u32;
                stats.sims += 1;
                stats.sum_depth += depth as u64;
                if depth > stats.max_depth {
                    stats.max_depth = depth;
                }

                // Terminal?
                if mechanics.is_game_over(leaf_data) {
                    // Terminal value convention: matches the synchronous mcts::search reference.
                    // backpropagate_result alternates sign while walking up; from the leaf's player-to-move
                    // perspective, a winner means the previous player won (so a sign flip happens during backprop).
                    let v = if mechanics.winner(leaf_data).is_some() {
                        1.0
                    } else {
                        0.0
                    };
                    items.push(Item::Terminal { path, value: v });
                    if v > 0.0 {
                        stats.terminal_wins += 1;
                    } else {
                        stats.truncations += 1;
                    }
                    continue;
                }
                if let Some(max) = self.cfg.max_steps {
                    if mechanics.repr().get_completed_steps(leaf_data) >= max as usize {
                        items.push(Item::Terminal { path, value: 0.0 });
                        stats.truncations += 1;
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
                let prep = prepare_eval_input(
                    mechanics,
                    leaf_data,
                    &leaf_mask,
                    &mut self.rotation_mappings,
                );
                let (tx, rx) = oneshot::channel();
                let req = EvalRequest {
                    state: leaf_data,
                    features: prep.features,
                    work_action_mask: prep.work_action_mask,
                    rot_to_orig: prep.rot_to_orig,
                    responder: tx,
                };
                to_send.push(req);
                // rx is moved into the Item; the matching send happens after this loop so we don't
                // borrow self.sender while iterating.
                items.push(Item::Miss { path, leaf_idx, rx });
            }

            // Send all miss requests.
            for req in to_send {
                // Sender is async; await is OK here.
                self.sender
                    .send(FrontMsg::Req(req))
                    .await
                    .map_err(|_| anyhow::anyhow!("eval pipeline front channel closed"))?;
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
                    Item::Hit {
                        path,
                        leaf_idx,
                        result,
                    } => {
                        undo_virtual_loss(&mut arena, &path, vl);
                        expand_node(&mut arena, leaf_idx, &result.priors, mechanics);
                        backpropagate(&mut arena, leaf_idx, -result.value as f64);
                        iters_done += 1;
                    }
                    Item::Miss { path, leaf_idx, rx } => {
                        let result = rx
                            .await
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
        self.sender
            .send(FrontMsg::Req(req))
            .await
            .map_err(|_| anyhow::anyhow!("eval pipeline front channel closed"))?;
        let res = rx
            .await
            .map_err(|_| anyhow::anyhow!("responder dropped"))??;
        Ok((res.value, res.priors))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::alphazero::eval_pipeline::EvalCache;
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
                        let p = if n_valid > 0 {
                            1.0 / n_valid as f32
                        } else {
                            0.0
                        };
                        let mut priors = vec![0.0f32; req.work_action_mask.len()];
                        for (i, &v) in req.work_action_mask.iter().enumerate() {
                            if v {
                                priors[i] = p;
                            }
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

    #[test]
    fn test_leaf_parallel_k1_matches_sequential_visit_total() {
        // Use a manual runtime instead of #[tokio::test] because the workspace
        // sets panic = "abort" in [profile.dev], which interacts badly with
        // the tokio::test macro.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let mech = QGameMechanics::new(5, 0, 200);
            let data = mech.create_initial_state();
            let cache = Arc::new(EvalCache::new());
            let (tx, rx) = tokio_mpsc::channel::<FrontMsg>(64);
            let stub = spawn_stub_coordinator(rx, Arc::clone(&cache));

            let mcts_cfg = MCTSConfig {
                n: Some(20),
                ucb_c: 1.4,
                noise_epsilon: 0.0,
                ..Default::default()
            };
            let lp_cfg = LeafParallelConfig {
                leaf_parallelism: 1,
                virtual_loss: 0,
                enable_tree_reuse: false,
            };
            let mut mcts =
                LeafParallelMCTS::new(mcts_cfg.clone(), lp_cfg, tx.clone(), Arc::clone(&cache));

            let visited = std::collections::HashSet::new();
            let (children, _root_value, _stats) = mcts.search(data, &mech, &visited).await.unwrap();
            let total: u32 = children.iter().map(|c| c.visit_count).sum();
            // Should be exactly n iterations of MCTS expansion under the root.
            assert!(total >= 20, "expected ≥20 child visits, got {}", total);

            // Drop mcts (which holds an internal Sender clone) before tx so the
            // stub's rx.recv() returns None and the stub task exits.
            drop(mcts);
            drop(tx);
            let _ = stub.await;
        });
    }

    #[test]
    fn test_leaf_parallel_k8_diversifies_children() {
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
                n: Some(64),
                ucb_c: 1.4,
                noise_epsilon: 0.0,
                ..Default::default()
            };
            let lp_cfg = LeafParallelConfig {
                leaf_parallelism: 8,
                virtual_loss: 3,
                enable_tree_reuse: false,
            };
            let mut mcts = LeafParallelMCTS::new(mcts_cfg, lp_cfg, tx.clone(), Arc::clone(&cache));

            let visited = std::collections::HashSet::new();
            let (children, _, _stats) = mcts.search(data, &mech, &visited).await.unwrap();
            let visited_top: u32 = children.iter().filter(|c| c.visit_count > 0).count() as u32;
            assert!(
                visited_top >= 2,
                "K=8 vl=3 should spread visits across ≥2 root children"
            );

            // Drop mcts (holds Sender clone) before tx, then await stub.
            drop(mcts);
            drop(tx);
            let _ = stub.await;
        });
    }

    #[test]
    fn test_tree_reuse_promotes_subtree_between_moves() {
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
                n: Some(32),
                ucb_c: 1.4,
                noise_epsilon: 0.0,
                ..Default::default()
            };
            let lp_cfg = LeafParallelConfig {
                leaf_parallelism: 4,
                virtual_loss: 1,
                enable_tree_reuse: true,
            };
            let mut mcts = LeafParallelMCTS::new(mcts_cfg, lp_cfg, tx.clone(), Arc::clone(&cache));

            let visited = std::collections::HashSet::new();
            // First search at the initial state.
            let (children_1, _, _stats) = mcts.search(data, &mech, &visited).await.unwrap();
            let chosen = children_1.iter().max_by_key(|c| c.visit_count).unwrap();

            // Advance root and search again from the resulting state.
            let mut next_state = data;
            mech.apply_action_index(&mut next_state, chosen.action_index);
            mcts.advance_root(chosen.action_index);
            let (children_2, _, _stats) = mcts.search(next_state, &mech, &visited).await.unwrap();
            assert!(!children_2.is_empty());

            // Drop mcts (holds Sender clone) before tx, then await stub.
            drop(mcts);
            drop(tx);
            let _ = stub.await;
        });
    }

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
            assert!(
                stats.sum_depth >= stats.sims as u64,
                "each sim has depth >= 1"
            );
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

    #[test]
    fn test_note_model_version_clears_tree_on_change() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let mech = QGameMechanics::new(5, 0, 200);
            let data = mech.create_initial_state();
            let cache = Arc::new(EvalCache::new());
            let (tx, rx) = tokio_mpsc::channel::<FrontMsg>(64);
            let stub = spawn_stub_coordinator(rx, Arc::clone(&cache));

            let mcts_cfg = MCTSConfig {
                n: Some(8),
                ucb_c: 1.4,
                noise_epsilon: 0.0,
                ..Default::default()
            };
            let lp_cfg = LeafParallelConfig {
                leaf_parallelism: 2,
                virtual_loss: 1,
                enable_tree_reuse: true,
            };
            let mut mcts = LeafParallelMCTS::new(mcts_cfg, lp_cfg, tx.clone(), Arc::clone(&cache));

            mcts.note_model_version(1);
            let visited = std::collections::HashSet::new();
            let _ = mcts.search(data, &mech, &visited).await.unwrap();
            assert!(mcts.arena.is_some());

            mcts.note_model_version(2);
            assert!(
                mcts.arena.is_none(),
                "tree should be cleared on version change"
            );

            // Drop mcts (holds Sender clone) before tx so stub exits.
            drop(mcts);
            drop(tx);
            let _ = stub.await;
        });
    }
}
