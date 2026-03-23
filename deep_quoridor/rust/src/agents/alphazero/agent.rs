//! AlphaZero agent implementation.
//!
//! Uses MCTS with neural network evaluation to select actions.

use std::collections::HashSet;

use anyhow::Result;
use rand::Rng;

use crate::agents::ActionSelector;
use crate::game_state::GameState;

use super::evaluator::OnnxEvaluator;
use super::mcts::{search, MCTSConfig};

/// Configuration for the AlphaZero agent.
#[derive(Debug, Clone)]
pub struct AlphaZeroAgentConfig {
    /// MCTS configuration.
    pub mcts: MCTSConfig,
    /// Temperature for action selection. 0 = greedy, 1 = proportional.
    pub temperature: f32,
    /// Step at which to drop temperature to 0.
    pub drop_t_on_step: Option<usize>,
    /// Whether to penalize visited states.
    pub penalize_visited_states: bool,
    /// If true, temperature=0 ties are resolved by first max child (deterministic).
    pub deterministic_tie_break: bool,
}

impl Default for AlphaZeroAgentConfig {
    fn default() -> Self {
        Self {
            mcts: MCTSConfig::default(),
            temperature: 1.0,
            drop_t_on_step: None,
            penalize_visited_states: false,
            deterministic_tie_break: false,
        }
    }
}

/// Apply temperature to visit counts and sample an action.
///
/// If temperature is 0, samples uniformly among actions with the highest visit count.
/// Otherwise, samples proportionally to visit_count^(1/T).
pub fn apply_temperature_and_sample(
    visit_counts: &[u32],
    action_indices: &[usize],
    temperature: f32,
) -> usize {
    apply_temperature_and_sample_with_mode(visit_counts, action_indices, temperature, false)
}

/// Same as `apply_temperature_and_sample` but supports deterministic tie mode.
///
/// When `deterministic_tie_break` is true and temperature is 0, the first
/// max-visit action is selected.
pub fn apply_temperature_and_sample_with_mode(
    visit_counts: &[u32],
    action_indices: &[usize],
    temperature: f32,
    deterministic_tie_break: bool,
) -> usize {
    assert!(
        !visit_counts.is_empty(),
        "apply_temperature_and_sample_with_mode requires non-empty visit_counts"
    );
    assert!(
        !action_indices.is_empty(),
        "apply_temperature_and_sample_with_mode requires non-empty action_indices"
    );
    assert_eq!(
        visit_counts.len(),
        action_indices.len(),
        "apply_temperature_and_sample_with_mode requires visit_counts and action_indices to have the same length"
    );

    if temperature == 0.0 {
        // Match Python semantics by default (sample among max-visit ties).
        // Deterministic parity mode can pick first max tie.
        let max_visits = visit_counts.iter().max().unwrap_or(&0);
        let tied_indices: Vec<usize> = visit_counts
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v == *max_visits { Some(i) } else { None })
            .collect();

        if deterministic_tie_break {
            return action_indices[tied_indices[0]];
        }

        let mut rng = rand::thread_rng();
        let pick = rng.gen_range(0..tied_indices.len());
        action_indices[tied_indices[pick]]
    } else {
        // Sample proportionally to visit_count^(1/T)
        let total: f64 = visit_counts
            .iter()
            .map(|&v| (v as f64).powf(1.0 / temperature as f64))
            .sum();

        if total <= 0.0 {
            return action_indices[0];
        }

        let probs: Vec<f64> = visit_counts
            .iter()
            .map(|&v| (v as f64).powf(1.0 / temperature as f64) / total)
            .collect();

        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;

        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return action_indices[i];
            }
        }

        action_indices[action_indices.len() - 1]
    }
}

/// AlphaZero MCTS agent.
///
/// Combines MCTS search with neural network evaluation for action selection.
pub struct AlphaZeroAgent {
    evaluator: OnnxEvaluator,
    config: AlphaZeroAgentConfig,
    visited_states: HashSet<u64>,
}

impl AlphaZeroAgent {
    /// Create a new AlphaZero agent.
    pub fn new(model_path: &str, config: AlphaZeroAgentConfig) -> Result<Self> {
        let evaluator = OnnxEvaluator::new(model_path)?;
        Ok(Self {
            evaluator,
            config,
            visited_states: HashSet::new(),
        })
    }

    /// Reset visited states between games.
    pub fn reset_game(&mut self) {
        self.visited_states.clear();
    }
}

impl ActionSelector for AlphaZeroAgent {
    fn select_action(
        &mut self,
        state: &GameState,
        action_mask: &[bool],
    ) -> Result<(usize, Vec<f32>)> {
        // Run MCTS search - only pass visited states when penalization is enabled
        let empty_visited = HashSet::new();
        let visited_ref = if self.config.penalize_visited_states {
            &self.visited_states
        } else {
            &empty_visited
        };
        let (children, _root_value) = search(
            &self.config.mcts,
            state.clone(),
            &mut self.evaluator,
            visited_ref,
        )?;

        // Extract visit counts and action indices
        let visit_counts: Vec<u32> = children.iter().map(|c| c.visit_count).collect();
        let action_indices: Vec<usize> = children.iter().map(|c| c.action_index).collect();

        // Determine effective temperature
        let temperature = if let Some(threshold) = self.config.drop_t_on_step {
            if state.completed_steps >= threshold {
                0.0
            } else {
                self.config.temperature
            }
        } else {
            self.config.temperature
        };

        // Select action using temperature
        let selected_idx = apply_temperature_and_sample_with_mode(
            &visit_counts,
            &action_indices,
            temperature,
            self.config.deterministic_tie_break,
        );

        // Build full policy vector from visit counts
        let total_visits: u32 = visit_counts.iter().sum();
        let mut policy = vec![0.0f32; action_mask.len()];
        if total_visits > 0 {
            for child in &children {
                policy[child.action_index] = child.visit_count as f32 / total_visits as f32;
            }
        }

        // Optionally add state to visited set
        if self.config.penalize_visited_states {
            // Get the hash of the resulting state after taking the action
            let action = children
                .iter()
                .find(|c| c.action_index == selected_idx)
                .map(|c| c.action)
                .unwrap_or([0, 0, 2]);
            let next_state = state.clone_and_step(action);
            self.visited_states.insert(next_state.get_fast_hash());
        }

        Ok((selected_idx, policy))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_temperature_greedy() {
        let visit_counts = vec![10, 50, 30, 5];
        let action_indices = vec![0, 1, 2, 3];

        // Greedy should always pick action 1 (highest visits)
        for _ in 0..10 {
            let selected = apply_temperature_and_sample(&visit_counts, &action_indices, 0.0);
            assert_eq!(selected, 1);
        }
    }

    #[test]
    fn test_apply_temperature_proportional() {
        let visit_counts = vec![100, 100, 100, 100];
        let action_indices = vec![0, 1, 2, 3];

        // With equal visits and T=1, should see some distribution
        let mut counts = [0u32; 4];
        for _ in 0..100 {
            let selected = apply_temperature_and_sample(&visit_counts, &action_indices, 1.0);
            counts[selected] += 1;
        }

        // All actions should be selected at least once
        for &c in &counts {
            assert!(c > 0);
        }
    }

    #[test]
    fn test_apply_temperature_high_temp_flattens() {
        let visit_counts = vec![1000, 1, 1, 1];
        let action_indices = vec![0, 1, 2, 3];

        // With very high temperature, distribution should be more uniform
        let mut counts = [0u32; 4];
        for _ in 0..400 {
            let selected = apply_temperature_and_sample(&visit_counts, &action_indices, 10.0);
            counts[selected] += 1;
        }

        // Even low-visit actions should be selected sometimes
        assert!(counts[1] > 0 || counts[2] > 0 || counts[3] > 0);
    }

    #[test]
    fn test_config_defaults() {
        let config = AlphaZeroAgentConfig::default();

        assert_eq!(config.temperature, 1.0);
        assert!(config.drop_t_on_step.is_none());
        assert!(!config.penalize_visited_states);
    }
}
