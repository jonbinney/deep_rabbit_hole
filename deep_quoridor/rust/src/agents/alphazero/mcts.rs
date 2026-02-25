//! Arena-based MCTS tree search implementation.
//!
//! Uses an arena (`Vec<Node>`) with indices to avoid Rust lifetime issues with parent pointers.

use std::collections::HashSet;

use rand_distr::{Dirichlet, Distribution};

use crate::actions::{action_index_to_action, action_to_index};
use crate::game_state::GameState;

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
    /// The game state at this node. None for lazily expanded nodes.
    pub game: Option<GameState>,
    /// Parent node index in the arena, None for root.
    pub parent: Option<usize>,
    /// Action taken from parent to reach this node.
    pub action_taken: Option<[i32; 3]>,
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
}

impl Node {
    /// Create a new root node.
    pub fn new_root(game: GameState) -> Self {
        Self {
            game: Some(game),
            parent: None,
            action_taken: None,
            children: Vec::new(),
            visit_count: 0,
            value_sum: 0.0,
            wins: 0,
            losses: 0,
            prior: 1.0,
        }
    }

    /// Create a new child node (lazy - game state computed on demand).
    pub fn new_child(parent: usize, action: [i32; 3], prior: f32) -> Self {
        Self {
            game: None,
            parent: Some(parent),
            action_taken: Some(action),
            children: Vec::new(),
            visit_count: 0,
            value_sum: 0.0,
            wins: 0,
            losses: 0,
            prior,
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
    /// Create a new arena with a root node.
    pub fn new(root_game: GameState) -> Self {
        let root = Node::new_root(root_game);
        Self { nodes: vec![root] }
    }

    /// Allocate a new child node and return its index.
    pub fn alloc_child(&mut self, parent: usize, action: [i32; 3], prior: f32) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node::new_child(parent, action, prior));
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

    /// Get the game state for a node, computing it lazily if needed.
    pub fn get_or_create_game(&mut self, idx: usize) -> &GameState {
        // First check if we already have it
        if self.nodes[idx].game.is_some() {
            return self.nodes[idx].game.as_ref().unwrap();
        }

        // Need to compute it from parent
        let parent_idx = self.nodes[idx]
            .parent
            .expect("Non-root node must have parent");
        let action = self.nodes[idx]
            .action_taken
            .expect("Child node must have action");

        // Recursively ensure parent has game state
        let parent_game = self.get_or_create_game(parent_idx).clone();

        // Apply action to get child game state
        let child_game = parent_game.clone_and_step(action);
        self.nodes[idx].game = Some(child_game);

        self.nodes[idx].game.as_ref().unwrap()
    }

    /// Get the number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

/// Expand a node by creating children for all valid actions.
pub fn expand_node(arena: &mut NodeArena, node_idx: usize, priors: &[f32], board_size: i32) {
    // Create children only for actions with non-zero prior
    for (action_idx, &prior) in priors.iter().enumerate() {
        if prior > 1e-10 {
            let action = action_index_to_action(board_size, action_idx);
            let child_idx = arena.alloc_child(node_idx, action, prior);
            arena.get_mut(node_idx).children.push(child_idx);
        }
    }
}

/// Select the best child using PUCT formula.
///
/// Returns the index of the selected child.
pub fn select_child(
    arena: &NodeArena,
    node_idx: usize,
    ucb_c: f32,
    visited_states: &HashSet<u64>,
    _board_size: i32,
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

        // Apply penalty if child state is in visited set
        if let Some(ref game) = child.game {
            if visited_states.contains(&game.get_fast_hash()) {
                ucb -= 1.0;
            }
        } else if let Some(action) = child.action_taken {
            // Compute hash for child without fully expanding game
            let parent_game = arena.get(node_idx).game.as_ref().unwrap();
            let child_game = parent_game.clone_and_step(action);
            if visited_states.contains(&child_game.get_fast_hash()) {
                ucb -= 1.0;
            }
        }

        if ucb > best_ucb {
            best_ucb = ucb;
            best_idx = child_idx;
        }
    }

    best_idx
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
        .filter(|(_, &p)| p > 1e-10)
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

/// Run MCTS search and return child information.
pub fn search<E: Evaluator>(
    config: &MCTSConfig,
    game: GameState,
    evaluator: &mut E,
    visited_states: &HashSet<u64>,
) -> anyhow::Result<(Vec<ChildInfo>, f32)> {
    let board_size = game.board_size;
    let mut arena = NodeArena::new(game.clone());

    // Evaluate root and expand
    let action_mask = game.get_action_mask();
    let (root_value, mut priors) = evaluator.evaluate(&game, &action_mask)?;

    // Apply Dirichlet noise at root if configured
    if config.noise_epsilon > 0.0 {
        let alpha = config.noise_alpha.unwrap_or_else(|| {
            let num_valid = action_mask.iter().filter(|&&m| m).count();
            10.0 / num_valid.max(1) as f32
        });
        apply_dirichlet_noise(&mut priors, config.noise_epsilon, alpha);
    }

    // Expand root
    expand_node(&mut arena, 0, &priors, board_size);

    // Determine number of iterations
    let num_valid = action_mask.iter().filter(|&&m| m).count() as u32;
    let n_iterations = config
        .n
        .unwrap_or_else(|| config.k.unwrap_or(10) * num_valid);

    // Special case: n=0 means just use priors
    if n_iterations == 0 {
        // Set visit counts proportional to priors
        let root = arena.get(0);
        let children = root.children.clone();

        for child_idx in children {
            let child = arena.get_mut(child_idx);
            child.visit_count = (child.prior * 1000.0) as u32;
        }
    } else {
        // Run MCTS iterations
        for _ in 0..n_iterations {
            // Selection: traverse tree to find a leaf
            let mut current_idx = 0;

            while !arena.get(current_idx).should_expand() {
                current_idx = select_child(
                    &arena,
                    current_idx,
                    config.ucb_c,
                    visited_states,
                    board_size,
                );
            }

            // Get/create game state for the selected node
            let leaf_game = arena.get_or_create_game(current_idx).clone();

            // Check for terminal state
            if leaf_game.is_game_over() {
                // Terminal: backpropagate result
                let value = if leaf_game.winner().is_some() {
                    1.0
                } else {
                    0.0
                };
                backpropagate_result(&mut arena, current_idx, value);
                continue;
            }

            // Check max steps
            if let Some(max) = config.max_steps {
                if leaf_game.completed_steps >= max as usize {
                    backpropagate_result(&mut arena, current_idx, 0.0);
                    continue;
                }
            }

            // Evaluate and expand
            let leaf_mask = leaf_game.get_action_mask();
            let (value, leaf_priors) = evaluator.evaluate(&leaf_game, &leaf_mask)?;

            expand_node(&mut arena, current_idx, &leaf_priors, board_size);

            // Backpropagate negative value (from opponent's perspective)
            backpropagate(&mut arena, current_idx, -value as f64);
        }
    }

    // Collect child information from root
    let root = arena.get(0);
    let children: Vec<ChildInfo> = root
        .children
        .iter()
        .map(|&child_idx| {
            let child = arena.get(child_idx);
            let action = child.action_taken.unwrap();
            let action_index = action_to_index(board_size, &action);
            ChildInfo {
                action,
                action_index,
                visit_count: child.visit_count,
            }
        })
        .collect();

    Ok((children, root_value))
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
            _state: &GameState,
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

    #[test]
    fn test_node_creation() {
        let state = GameState::new(5, 3);
        let node = Node::new_root(state);

        assert!(node.game.is_some());
        assert!(node.parent.is_none());
        assert!(node.children.is_empty());
        assert_eq!(node.visit_count, 0);
        assert_eq!(node.q_value(), 0.0);
        assert!(node.should_expand());
    }

    #[test]
    fn test_expand_node() {
        let state = GameState::new(5, 3);
        let mut arena = NodeArena::new(state);

        // Create uniform priors with some zeros
        let priors = vec![0.0, 0.5, 0.0, 0.3, 0.2];

        expand_node(&mut arena, 0, &priors, 5);

        let root = arena.get(0);
        // Should have 3 children (for non-zero priors)
        assert_eq!(root.children.len(), 3);

        // Verify children have correct priors
        let child_priors: Vec<f32> = root
            .children
            .iter()
            .map(|&idx| arena.get(idx).prior)
            .collect();
        assert!((child_priors[0] - 0.5).abs() < 1e-6);
        assert!((child_priors[1] - 0.3).abs() < 1e-6);
        assert!((child_priors[2] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_select_child_ucb() {
        let state = GameState::new(5, 3);
        let mut arena = NodeArena::new(state);

        // Manually create children with different visit counts and values
        let child1 = arena.alloc_child(0, [1, 2, 2], 0.5);
        let child2 = arena.alloc_child(0, [0, 2, 2], 0.5);

        arena.get_mut(0).children = vec![child1, child2];
        arena.get_mut(0).visit_count = 10;

        // Child 1: many visits, low Q
        arena.get_mut(child1).visit_count = 8;
        arena.get_mut(child1).value_sum = 0.0;

        // Child 2: few visits, should be selected due to exploration bonus
        arena.get_mut(child2).visit_count = 1;
        arena.get_mut(child2).value_sum = 0.0;

        let visited = HashSet::new();
        let selected = select_child(&arena, 0, 1.4, &visited, 5);

        // Child 2 should be selected (higher UCB due to fewer visits)
        assert_eq!(selected, child2);
    }

    #[test]
    fn test_backpropagate() {
        let state = GameState::new(5, 3);
        let mut arena = NodeArena::new(state);

        // Create a chain: root -> child -> grandchild
        let child = arena.alloc_child(0, [1, 2, 2], 0.5);
        let grandchild = arena.alloc_child(child, [2, 2, 2], 0.5);

        arena.get_mut(0).children = vec![child];
        arena.get_mut(child).children = vec![grandchild];

        // Backpropagate +1 from grandchild
        backpropagate(&mut arena, grandchild, 1.0);

        // Check values alternate
        assert_eq!(arena.get(grandchild).visit_count, 1);
        assert!((arena.get(grandchild).value_sum - 1.0).abs() < 1e-6);

        assert_eq!(arena.get(child).visit_count, 1);
        assert!((arena.get(child).value_sum - (-1.0)).abs() < 1e-6);

        assert_eq!(arena.get(0).visit_count, 1);
        assert!((arena.get(0).value_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_backpropagate_result_tracks_wins_losses() {
        let state = GameState::new(5, 3);
        let mut arena = NodeArena::new(state);

        let child = arena.alloc_child(0, [1, 2, 2], 0.5);
        arena.get_mut(0).children = vec![child];

        // Backpropagate a win
        backpropagate_result(&mut arena, child, 1.0);

        assert_eq!(arena.get(child).wins, 1);
        assert_eq!(arena.get(child).losses, 0);
        assert_eq!(arena.get(0).wins, 0);
        assert_eq!(arena.get(0).losses, 1);
    }

    #[test]
    fn test_visited_state_penalty() {
        let state = GameState::new(5, 0); // No walls for simpler moves
        let mut arena = NodeArena::new(state.clone());

        // Create two children
        let child1 = arena.alloc_child(0, [1, 2, 2], 0.5);
        let child2 = arena.alloc_child(0, [0, 1, 2], 0.5);

        arena.get_mut(0).children = vec![child1, child2];
        arena.get_mut(0).visit_count = 10;

        // Equal visits
        arena.get_mut(child1).visit_count = 1;
        arena.get_mut(child2).visit_count = 1;

        // Mark child1's state as visited
        let child1_game = state.clone_and_step([1, 2, 2]);
        let mut visited = HashSet::new();
        visited.insert(child1_game.get_fast_hash());

        // Child2 should be selected since child1 has penalty
        let selected = select_child(&arena, 0, 1.4, &visited, 5);
        assert_eq!(selected, child2);
    }

    #[test]
    fn test_mcts_search_basic() {
        let state = GameState::new(5, 0); // No walls for faster search
        let mut evaluator = MockEvaluator::new(0.0);
        let visited = HashSet::new();

        let config = MCTSConfig {
            n: Some(10),
            ucb_c: 1.4,
            noise_epsilon: 0.0,
            ..Default::default()
        };

        let result = search(&config, state, &mut evaluator, &visited);
        assert!(result.is_ok());

        let (children, _value) = result.unwrap();

        // Should have some children
        assert!(!children.is_empty());

        // Visit counts should be > 0
        let total_visits: u32 = children.iter().map(|c| c.visit_count).sum();
        assert!(total_visits > 0);
    }

    #[test]
    fn test_mcts_n_zero_uses_priors() {
        let state = GameState::new(5, 3);
        let mut evaluator = MockEvaluator::new(0.0);
        let visited = HashSet::new();

        let config = MCTSConfig {
            n: Some(0),
            ucb_c: 1.4,
            noise_epsilon: 0.0,
            ..Default::default()
        };

        let result = search(&config, state, &mut evaluator, &visited);
        assert!(result.is_ok());

        let (children, _) = result.unwrap();

        // Visit counts should be proportional to priors (×1000)
        for child in &children {
            assert!(child.visit_count > 0);
        }
    }

    #[test]
    fn test_dirichlet_noise() {
        let mut priors = vec![0.0, 0.4, 0.0, 0.3, 0.3, 0.0];
        let original = priors.clone();

        apply_dirichlet_noise(&mut priors, 0.25, 0.5);

        // Priors should have changed
        let changed = priors
            .iter()
            .zip(original.iter())
            .any(|(p, o)| (p - o).abs() > 1e-6);
        assert!(changed);

        // Only non-zero priors should be affected
        assert!(priors[0] < 1e-6);
        assert!(priors[2] < 1e-6);
        assert!(priors[5] < 1e-6);

        // Non-zero priors should sum to ~1
        let sum: f32 = priors.iter().sum();
        assert!((sum - 1.0).abs() < 0.1);
    }
}
