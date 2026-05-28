//! Arena-based MCTS tree search implementation.
//!
//! Uses an arena (`Vec<Node>`) with indices to avoid Rust lifetime issues with parent pointers.

use std::collections::HashSet;

use rand_distr::{Dirichlet, Distribution};
use smallvec::SmallVec;

use crate::actions::action_index_to_action;
#[cfg(test)]
use crate::actions::action_to_index;
use crate::compact::q_bit_repr::CompactState;
use crate::compact::q_game_mechanics::QGameMechanics;

use super::evaluator::Evaluator;

/// Configuration for MCTS search.
#[derive(Debug, Clone)]
pub struct MCTSConfig {
    /// Number of MCTS iterations. If Some, run exactly this many iterations.
    pub n: Option<u32>,
    /// If n is None, run k * num_valid_actions iterations.
    pub k: Option<u32>,
    /// UCB exploration constant (c_puct).
    pub ucb_c: f32,
    /// Dirichlet noise weight (0 = no noise, 1 = all noise).
    pub noise_epsilon: f32,
    /// Dirichlet alpha parameter. If None, auto-computed as 10 / num_valid_actions.
    pub noise_alpha: Option<f32>,
    /// Maximum game steps before declaring a draw.
    pub max_steps: Option<i32>,
    /// Whether to penalize visited states during MCTS selection.
    pub penalize_visited_states: bool,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            n: Some(100),
            k: None,
            ucb_c: 1.4,
            noise_epsilon: 0.25,
            noise_alpha: None,
            max_steps: None,
            penalize_visited_states: false,
        }
    }
}

/// Information about a child node for returning search results.
#[derive(Debug, Clone)]
pub struct ChildInfo {
    /// Action as [row, col, type].
    pub action: [i32; 3],
    /// Flat policy index for this action.
    pub action_index: usize,
    /// Number of times this action was visited.
    pub visit_count: u32,
}

/// A node in the MCTS tree.
#[derive(Debug)]
pub struct Node {
    /// The compact game state at this node.
    pub data: CompactState,
    /// Parent node index in the arena, None for root.
    pub parent: Option<usize>,
    /// Flat policy index for the action taken from parent to reach this node.
    pub action_index: Option<usize>,
    /// Child node indices in the arena.
    pub children: Vec<usize>,
    /// Number of times this node was visited.
    pub visit_count: u32,
    /// Sum of values backpropagated through this node.
    pub value_sum: f64,
    /// Number of wins (terminal +1 values) for tracking.
    pub wins: u32,
    /// Number of losses (terminal -1 values) for tracking.
    pub losses: u32,
    /// Prior probability from the neural network.
    pub prior: f32,
    /// Un-noised network prior; never modified after construction. Used to
    /// re-apply Dirichlet noise to new-root children after tree reuse.
    pub prior_clean: f32,
}

impl Node {
    /// Create a new root node.
    pub fn new_root(data: CompactState) -> Self {
        Self {
            data,
            parent: None,
            action_index: None,
            children: Vec::new(),
            visit_count: 0,
            value_sum: 0.0,
            wins: 0,
            losses: 0,
            prior: 1.0,
            prior_clean: 1.0,
        }
    }

    /// Create a new child node with its (already computed) state.
    pub fn new_child(parent: usize, action_index: usize, data: CompactState, prior: f32) -> Self {
        Self {
            data,
            parent: Some(parent),
            action_index: Some(action_index),
            children: Vec::new(),
            visit_count: 0,
            value_sum: 0.0,
            wins: 0,
            losses: 0,
            prior,
            prior_clean: prior,
        }
    }

    /// Check if this node should be expanded (has no children).
    pub fn should_expand(&self) -> bool {
        self.children.is_empty()
    }

    /// Compute Q-value (mean value).
    pub fn q_value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f64
        }
    }
}

/// Arena-based storage for MCTS nodes.
pub struct NodeArena {
    nodes: Vec<Node>,
}

impl NodeArena {
    /// Construct from a pre-filled `Vec<Node>` (used by `promote_subtree`).
    pub fn from_nodes(nodes: Vec<Node>) -> Self {
        Self { nodes }
    }

    /// Create a new arena with a root node.
    pub fn new(root_data: CompactState) -> Self {
        let root = Node::new_root(root_data);
        Self { nodes: vec![root] }
    }

    /// Allocate a new child node and return its index.
    pub fn alloc_child(
        &mut self,
        parent: usize,
        action_index: usize,
        data: CompactState,
        prior: f32,
    ) -> usize {
        let idx = self.nodes.len();
        self.nodes
            .push(Node::new_child(parent, action_index, data, prior));
        idx
    }

    /// Get a reference to a node.
    pub fn get(&self, idx: usize) -> &Node {
        &self.nodes[idx]
    }

    /// Get a mutable reference to a node.
    pub fn get_mut(&mut self, idx: usize) -> &mut Node {
        &mut self.nodes[idx]
    }

    /// Get the number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

/// Expand a node by creating children for all actions with non-zero prior.
pub fn expand_node(
    arena: &mut NodeArena,
    node_idx: usize,
    priors: &[f32],
    mechanics: &QGameMechanics,
) {
    let parent_data = arena.get(node_idx).data;
    let new_children: Vec<usize> = priors
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p > 1e-10)
        .map(|(action_idx, &prior)| {
            let mut child_data = parent_data;
            mechanics.apply_action_index(&mut child_data, action_idx);
            arena.alloc_child(node_idx, action_idx, child_data, prior)
        })
        .collect();
    arena.get_mut(node_idx).children.extend(new_children);
}

/// Select the best child using PUCT formula.
///
/// Returns the index of the selected child.
pub fn select_child(
    arena: &NodeArena,
    node_idx: usize,
    ucb_c: f32,
    visited_states: &HashSet<CompactState>,
) -> usize {
    let node = arena.get(node_idx);
    let parent_visits = node.visit_count.max(1) as f32;

    let mut best_idx = node.children[0];
    let mut best_ucb = f32::NEG_INFINITY;

    for &child_idx in &node.children {
        let child = arena.get(child_idx);

        // Compute UCB score using PUCT formula
        let q = child.q_value() as f32;
        let u = ucb_c * child.prior * (parent_visits.sqrt()) / (1.0 + child.visit_count as f32);
        let mut ucb = q + u;

        // Penalty when this child's state is in the visited set.
        // Set is empty when penalize_visited_states is false.
        if !visited_states.is_empty() && visited_states.contains(&child.data) {
            ucb -= 1.0;
        }

        if ucb > best_ucb {
            best_ucb = ucb;
            best_idx = child_idx;
        }
    }

    best_idx
}

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

/// Backpropagate a value up the tree.
pub fn backpropagate(arena: &mut NodeArena, node_idx: usize, mut value: f64) {
    let mut current = Some(node_idx);

    while let Some(idx) = current {
        let node = arena.get_mut(idx);
        node.visit_count += 1;
        node.value_sum += value;

        // Negate value for parent (opponent's perspective)
        value = -value;
        current = node.parent;
    }
}

/// Backpropagate a terminal result (also tracks wins/losses).
pub fn backpropagate_result(arena: &mut NodeArena, node_idx: usize, mut value: f64) {
    let mut current = Some(node_idx);

    while let Some(idx) = current {
        let node = arena.get_mut(idx);
        node.visit_count += 1;
        node.value_sum += value;

        if value > 0.5 {
            node.wins += 1;
        } else if value < -0.5 {
            node.losses += 1;
        }

        value = -value;
        current = node.parent;
    }
}

/// Apply Dirichlet noise to priors at the root.
pub fn apply_dirichlet_noise(priors: &mut [f32], epsilon: f32, alpha: f32) {
    // Count non-zero priors
    let valid_indices: Vec<usize> = priors
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p > 1e-10)
        .map(|(i, _)| i)
        .collect();

    if valid_indices.is_empty() {
        return;
    }

    // Generate Dirichlet noise
    let dirichlet = Dirichlet::new_with_size(alpha, valid_indices.len());
    if dirichlet.is_err() {
        return;
    }

    let mut rng = rand::thread_rng();
    let noise: Vec<f32> = dirichlet.unwrap().sample(&mut rng);

    // Apply noise: prior' = (1 - epsilon) * prior + epsilon * noise
    for (noise_idx, &prior_idx) in valid_indices.iter().enumerate() {
        priors[prior_idx] = (1.0 - epsilon) * priors[prior_idx] + epsilon * noise[noise_idx];
    }
}

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

/// Copy the subtree rooted at `new_root_idx` of `old_arena` into a fresh
/// `NodeArena`. The new arena's root is the copy of `new_root_idx`, with its
/// `parent` and `action_index` cleared. All descendants are copied; siblings
/// of `new_root_idx` (and their descendants) are dropped.
///
/// Statistics (`visit_count`, `value_sum`, `wins`, `losses`, `prior`,
/// `prior_clean`) carry over unchanged.
pub fn promote_subtree(old_arena: &NodeArena, new_root_idx: usize) -> NodeArena {
    let mut new_nodes: Vec<Node> = Vec::new();
    let mut idx_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

    // BFS so parents always get assigned a new index before children.
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
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

/// Run MCTS search and return child information.
pub fn search(
    config: &MCTSConfig,
    root_data: CompactState,
    mechanics: &QGameMechanics,
    evaluator: &mut dyn Evaluator,
    visited_states: &HashSet<CompactState>,
) -> anyhow::Result<(Vec<ChildInfo>, f32)> {
    let bs = mechanics.repr().board_size() as i32;
    let mut arena = NodeArena::new(root_data);

    // Evaluate root upfront to get root_value and priors.
    // Root is NOT expanded here; expansion happens inside the first loop iteration
    // so that the loop structure matches the Python implementation (where iteration 0
    // always selects root itself, expands it, and backpropagates through it).
    let action_mask = mechanics.get_action_mask_immut(root_data);
    let (root_value, root_priors) = evaluator.evaluate(root_data, mechanics, &action_mask)?;

    // Determine number of iterations
    let num_valid = action_mask.iter().filter(|&&m| m).count() as u32;
    let n_iterations = config
        .n
        .unwrap_or_else(|| config.k.unwrap_or(10) * num_valid);

    // Special case: n=0 means just use priors (expand root once without simulating)
    if n_iterations == 0 {
        expand_node(&mut arena, 0, &root_priors, mechanics);
        if config.noise_epsilon > 0.0 {
            let alpha = config.noise_alpha.unwrap_or_else(|| {
                let num_valid = action_mask.iter().filter(|&&m| m).count();
                10.0 / num_valid.max(1) as f32
            });
            apply_dirichlet_noise_to_root_children(&mut arena, 0, config.noise_epsilon, alpha);
        }
        let root = arena.get(0);
        let children = root.children.clone();

        for child_idx in children {
            let child = arena.get_mut(child_idx);
            child.visit_count = (child.prior * 1000.0) as u32;
        }
    } else {
        // Hold pre-computed root priors so the first iteration can expand root
        // without re-calling the evaluator (matching Python's inline-root-expansion
        // behaviour where iteration 0 selects root itself).
        let mut root_priors_opt: Option<Vec<f32>> = Some(root_priors);

        // Run MCTS iterations
        for _ in 0..n_iterations {
            // Selection: traverse tree to find a leaf
            let mut current_idx = 0;

            while !arena.get(current_idx).should_expand() {
                current_idx = select_child(&arena, current_idx, config.ucb_c, visited_states);
            }

            let leaf_data = arena.get(current_idx).data;

            // Check for terminal state
            if mechanics.is_game_over(leaf_data) {
                // Terminal: backpropagate result
                let value = if mechanics.winner(leaf_data).is_some() {
                    1.0
                } else {
                    0.0
                };
                backpropagate_result(&mut arena, current_idx, value);
                continue;
            }

            // Check max steps
            if let Some(max) = config.max_steps {
                if mechanics.repr().get_completed_steps(leaf_data) >= max as usize {
                    backpropagate_result(&mut arena, current_idx, 0.0);
                    continue;
                }
            }

            // Evaluate and expand. On the first iteration current_idx is always 0
            // (root) because root starts with no children; reuse the pre-computed
            // priors instead of calling the evaluator a second time.
            let (value, leaf_priors) = if let Some(pre_priors) = root_priors_opt.take() {
                (root_value, pre_priors)
            } else {
                let leaf_mask = mechanics.get_action_mask_immut(leaf_data);
                evaluator.evaluate(leaf_data, mechanics, &leaf_mask)?
            };

            expand_node(&mut arena, current_idx, &leaf_priors, mechanics);

            // Apply Dirichlet noise to root children after root expansion (first iteration)
            if config.noise_epsilon > 0.0 && current_idx == 0 {
                let alpha = config.noise_alpha.unwrap_or_else(|| {
                    let num_valid = action_mask.iter().filter(|&&m| m).count();
                    10.0 / num_valid.max(1) as f32
                });
                apply_dirichlet_noise_to_root_children(&mut arena, 0, config.noise_epsilon, alpha);
            }

            // Backpropagate negative value (from opponent's perspective)
            backpropagate(&mut arena, current_idx, -value as f64);
        }
    }

    // Collect child information from root and compute root value matching Python's
    // -(root.value_sum / root.visit_count) formula.
    let root = arena.get(0);
    let computed_root_value = if root.visit_count > 0 {
        -(root.value_sum / root.visit_count as f64) as f32
    } else {
        root_value
    };
    let children: Vec<ChildInfo> = root
        .children
        .iter()
        .map(|&child_idx| {
            let child = arena.get(child_idx);
            let ai = child
                .action_index
                .expect("child node must have action_index");
            let action = action_index_to_action(bs, ai);
            ChildInfo {
                action,
                action_index: ai,
                visit_count: child.visit_count,
            }
        })
        .collect();

    Ok((children, computed_root_value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    /// Mock evaluator that returns fixed values for testing.
    struct MockEvaluator {
        value: f32,
    }

    impl MockEvaluator {
        fn new(value: f32) -> Self {
            Self { value }
        }
    }

    impl Evaluator for MockEvaluator {
        fn evaluate(
            &mut self,
            _data: CompactState,
            _mechanics: &QGameMechanics,
            action_mask: &[bool],
        ) -> Result<(f32, Vec<f32>)> {
            // Return uniform priors over valid actions
            let num_valid = action_mask.iter().filter(|&&m| m).count();
            let prior = if num_valid > 0 {
                1.0 / num_valid as f32
            } else {
                0.0
            };
            let priors: Vec<f32> = action_mask
                .iter()
                .map(|&valid| if valid { prior } else { 0.0 })
                .collect();
            Ok((self.value, priors))
        }
    }

    fn make_mech_state() -> (QGameMechanics, CompactState) {
        let mech = QGameMechanics::new(5, 3, 200);
        let data = mech.create_initial_state();
        (mech, data)
    }

    #[test]
    fn test_node_creation() {
        let (_, data) = make_mech_state();
        let node = Node::new_root(data);

        assert_eq!(node.data, data);
        assert!(node.parent.is_none());
        assert!(node.children.is_empty());
        assert_eq!(node.visit_count, 0);
        assert_eq!(node.q_value(), 0.0);
        assert!(node.should_expand());
    }

    #[test]
    fn test_expand_node() {
        let (mech, data) = make_mech_state();
        let mut arena = NodeArena::new(data);

        // Build sparse priors aligned to the policy layout
        let total = crate::actions::policy_size(5);
        let mask = mech.get_action_mask_immut(data);
        let mut priors = vec![0.0f32; total];
        // Put non-zero prior on three valid actions
        let mut count = 0;
        for (i, &v) in mask.iter().enumerate() {
            if v {
                priors[i] = if count == 0 {
                    0.5
                } else if count == 1 {
                    0.3
                } else if count == 2 {
                    0.2
                } else {
                    0.0
                };
                count += 1;
                if count == 3 {
                    break;
                }
            }
        }

        expand_node(&mut arena, 0, &priors, &mech);

        let root = arena.get(0);
        assert_eq!(root.children.len(), 3);
    }

    #[test]
    fn test_select_child_ucb() {
        let (mech, data) = make_mech_state();
        let mut arena = NodeArena::new(data);

        // Find a couple of valid move action indices to use
        let mask = mech.get_action_mask_immut(data);
        let valid: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i) } else { None })
            .collect();
        assert!(valid.len() >= 2);

        // Manually create two children
        let mut d1 = data;
        mech.apply_action_index(&mut d1, valid[0]);
        let mut d2 = data;
        mech.apply_action_index(&mut d2, valid[1]);
        let child1 = arena.alloc_child(0, valid[0], d1, 0.5);
        let child2 = arena.alloc_child(0, valid[1], d2, 0.5);

        arena.get_mut(0).children = vec![child1, child2];
        arena.get_mut(0).visit_count = 10;

        // Child 1: many visits, low Q
        arena.get_mut(child1).visit_count = 8;
        arena.get_mut(child1).value_sum = 0.0;

        // Child 2: few visits, should be selected due to exploration bonus
        arena.get_mut(child2).visit_count = 1;
        arena.get_mut(child2).value_sum = 0.0;

        let visited = HashSet::new();
        let selected = select_child(&arena, 0, 1.4, &visited);

        assert_eq!(selected, child2);
    }

    #[test]
    fn test_backpropagate() {
        let (_, data) = make_mech_state();
        let mut arena = NodeArena::new(data);

        // Chain: root -> child -> grandchild (arbitrary action indices, data doesn't matter here)
        let child = arena.alloc_child(0, 0, data, 0.5);
        let grandchild = arena.alloc_child(child, 0, data, 0.5);

        arena.get_mut(0).children = vec![child];
        arena.get_mut(child).children = vec![grandchild];

        backpropagate(&mut arena, grandchild, 1.0);

        assert_eq!(arena.get(grandchild).visit_count, 1);
        assert!((arena.get(grandchild).value_sum - 1.0).abs() < 1e-6);

        assert_eq!(arena.get(child).visit_count, 1);
        assert!((arena.get(child).value_sum - (-1.0)).abs() < 1e-6);

        assert_eq!(arena.get(0).visit_count, 1);
        assert!((arena.get(0).value_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_backpropagate_result_tracks_wins_losses() {
        let (_, data) = make_mech_state();
        let mut arena = NodeArena::new(data);

        let child = arena.alloc_child(0, 0, data, 0.5);
        arena.get_mut(0).children = vec![child];

        backpropagate_result(&mut arena, child, 1.0);

        assert_eq!(arena.get(child).wins, 1);
        assert_eq!(arena.get(child).losses, 0);
        assert_eq!(arena.get(0).wins, 0);
        assert_eq!(arena.get(0).losses, 1);
    }

    #[test]
    fn test_visited_state_penalty() {
        let mech = QGameMechanics::new(5, 0, 200);
        let data = mech.create_initial_state();
        let mut arena = NodeArena::new(data);

        // Two valid move children
        let mask = mech.get_action_mask_immut(data);
        let valid: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i) } else { None })
            .collect();
        let mut d1 = data;
        mech.apply_action_index(&mut d1, valid[0]);
        let mut d2 = data;
        mech.apply_action_index(&mut d2, valid[1]);
        let child1 = arena.alloc_child(0, valid[0], d1, 0.5);
        let child2 = arena.alloc_child(0, valid[1], d2, 0.5);
        arena.get_mut(0).children = vec![child1, child2];
        arena.get_mut(0).visit_count = 10;

        arena.get_mut(child1).visit_count = 1;
        arena.get_mut(child2).visit_count = 1;

        // Mark child1's data as visited
        let mut visited = HashSet::new();
        visited.insert(d1);

        let selected = select_child(&arena, 0, 1.4, &visited);
        assert_eq!(selected, child2);
    }

    #[test]
    fn test_no_visited_state_penalty_when_set_is_empty() {
        let mech = QGameMechanics::new(5, 0, 200);
        let data = mech.create_initial_state();
        let mut arena = NodeArena::new(data);

        let mask = mech.get_action_mask_immut(data);
        let valid: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i) } else { None })
            .collect();
        let mut d1 = data;
        mech.apply_action_index(&mut d1, valid[0]);
        let mut d2 = data;
        mech.apply_action_index(&mut d2, valid[1]);
        let child1 = arena.alloc_child(0, valid[0], d1, 0.5);
        let child2 = arena.alloc_child(0, valid[1], d2, 0.5);

        arena.get_mut(0).children = vec![child1, child2];
        arena.get_mut(0).visit_count = 20;

        // Child1 higher Q with same visits
        arena.get_mut(child1).visit_count = 8;
        arena.get_mut(child1).value_sum = 6.0; // Q = 0.75
        arena.get_mut(child2).visit_count = 8;
        arena.get_mut(child2).value_sum = 0.0; // Q = 0.0

        // Empty visited set (simulates penalize_visited_states=false)
        let visited = HashSet::new();
        let selected = select_child(&arena, 0, 1.4, &visited);
        assert_eq!(selected, child1);

        // Now with child1's data in the visited set, child2 wins
        let mut visited_with_penalty = HashSet::new();
        visited_with_penalty.insert(d1);
        let selected = select_child(&arena, 0, 1.4, &visited_with_penalty);
        assert_eq!(selected, child2);
    }

    #[test]
    fn test_mcts_search_basic() {
        let mech = QGameMechanics::new(5, 0, 200);
        let data = mech.create_initial_state();
        let mut evaluator = MockEvaluator::new(0.0);
        let visited = HashSet::new();

        let config = MCTSConfig {
            n: Some(10),
            ucb_c: 1.4,
            noise_epsilon: 0.0,
            ..Default::default()
        };

        let result = search(&config, data, &mech, &mut evaluator, &visited);
        assert!(result.is_ok());

        let (children, _value) = result.unwrap();

        assert!(!children.is_empty());

        let total_visits: u32 = children.iter().map(|c| c.visit_count).sum();
        assert!(total_visits > 0);
    }

    #[test]
    fn test_mcts_n_zero_uses_priors() {
        let mech = QGameMechanics::new(5, 3, 200);
        let data = mech.create_initial_state();
        let mut evaluator = MockEvaluator::new(0.0);
        let visited = HashSet::new();

        let config = MCTSConfig {
            n: Some(0),
            ucb_c: 1.4,
            noise_epsilon: 0.0,
            ..Default::default()
        };

        let result = search(&config, data, &mech, &mut evaluator, &visited);
        assert!(result.is_ok());

        let (children, _) = result.unwrap();

        for child in &children {
            assert!(child.visit_count > 0);
        }
    }

    #[test]
    fn test_dirichlet_noise() {
        let mut priors = vec![0.0, 0.4, 0.0, 0.3, 0.3, 0.0];
        let original = priors.clone();

        apply_dirichlet_noise(&mut priors, 0.25, 0.5);

        let changed = priors
            .iter()
            .zip(original.iter())
            .any(|(p, o)| (p - o).abs() > 1e-6);
        assert!(changed);

        assert!(priors[0] < 1e-6);
        assert!(priors[2] < 1e-6);
        assert!(priors[5] < 1e-6);

        let sum: f32 = priors.iter().sum();
        assert!((sum - 1.0).abs() < 0.1);
    }

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
        let clean_before: Vec<f32> = arena
            .get(0)
            .children
            .iter()
            .map(|&i| arena.get(i).prior_clean)
            .collect();

        apply_dirichlet_noise_to_root_children(&mut arena, 0, 0.25, 0.5);

        // prior_clean unchanged, prior changed.
        let mut any_changed = false;
        for (offset, &child_idx) in arena.get(0).children.iter().enumerate() {
            let c = arena.get(child_idx);
            assert!(
                (c.prior_clean - clean_before[offset]).abs() < 1e-6,
                "prior_clean must not be modified"
            );
            if (c.prior - c.prior_clean).abs() > 1e-6 {
                any_changed = true;
            }
        }
        assert!(
            any_changed,
            "noise should change at least one child's prior"
        );
    }

    #[test]
    fn test_child_action_index_roundtrips_to_action() {
        let bs = 5;
        let mech = QGameMechanics::new(5, 3, 200);
        let data = mech.create_initial_state();
        let mut evaluator = MockEvaluator::new(0.0);
        let visited = HashSet::new();
        let config = MCTSConfig {
            n: Some(2),
            ucb_c: 1.4,
            noise_epsilon: 0.0,
            ..Default::default()
        };
        let (children, _) = search(&config, data, &mech, &mut evaluator, &visited).unwrap();
        for c in &children {
            assert_eq!(action_to_index(bs, &c.action), c.action_index);
        }
    }

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
        let valid: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i) } else { None })
            .collect();
        assert!(valid.len() >= 3);
        for &i in &valid[..3] {
            priors[i] = 1.0 / 3.0;
        }
        expand_node(&mut arena, 0, &priors, &mech);
        arena.get_mut(0).visit_count = 0;

        let visited = HashSet::new();
        let mut first_level_choices = std::collections::HashSet::new();
        for _ in 0..3 {
            let path: SmallVec<[usize; 32]> = select_leaf_with_vl(&mut arena, 0, 1.4, 3, &visited);
            // First-level child is path[1] (path[0] is the root).
            first_level_choices.insert(path[1]);
        }
        assert!(
            first_level_choices.len() >= 2,
            "vl should drive at least two distinct first-level child selections"
        );
    }

    #[test]
    fn test_promote_subtree_keeps_only_chosen_branch() {
        let mech = QGameMechanics::new(5, 0, 200);
        let data = mech.create_initial_state();
        let mut arena = NodeArena::new(data);

        let mask = mech.get_action_mask_immut(data);
        let valid: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i) } else { None })
            .collect();
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
        let (c1_v, c1_s) = (arena.get(c1).visit_count, arena.get(c1).value_sum);
        let (c2_v, c2_s) = (arena.get(c2).visit_count, arena.get(c2).value_sum);

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
}
