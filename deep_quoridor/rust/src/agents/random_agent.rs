//! Random agent — picks a valid action uniformly at random.

use rand::Rng;

use crate::agents::ActionSelector;
use crate::game_state::GameState;

/// An agent that selects a random valid action.
pub struct RandomAgent {
    rng: rand::rngs::ThreadRng,
}

impl RandomAgent {
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }
}

impl Default for RandomAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionSelector for RandomAgent {
    fn select_action(
        &mut self,
        _state: &GameState,
        action_mask: &[bool],
    ) -> anyhow::Result<(usize, Vec<f32>)> {
        // Collect valid action indices
        let valid_indices: Vec<usize> = action_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &valid)| if valid { Some(i) } else { None })
            .collect();

        anyhow::ensure!(!valid_indices.is_empty(), "No valid actions available");

        // Pick one uniformly at random
        let chosen = valid_indices[self.rng.gen_range(0..valid_indices.len())];

        // Build uniform policy over valid actions
        let num_valid = valid_indices.len() as f32;
        let policy: Vec<f32> = action_mask
            .iter()
            .map(|&valid| if valid { 1.0 / num_valid } else { 0.0 })
            .collect();

        Ok((chosen, policy))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_agent_picks_valid_action() {
        let mut agent = RandomAgent::new();
        let state = GameState::new(5, 3);
        let mask = vec![false, false, true, false, true, true];

        for _ in 0..50 {
            let (idx, _) = agent.select_action(&state, &mask).unwrap();
            assert!(
                mask[idx],
                "RandomAgent picked an invalid action index {}",
                idx
            );
        }
    }

    #[test]
    fn test_random_agent_policy_sums_to_one() {
        let mut agent = RandomAgent::new();
        let state = GameState::new(5, 3);
        let mask = vec![false, true, true, false, true];

        let (_, policy) = agent.select_action(&state, &mask).unwrap();

        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Invalid actions should have 0 probability
        assert_eq!(policy[0], 0.0);
        assert_eq!(policy[3], 0.0);
    }

    #[test]
    fn test_random_agent_no_valid_actions_fails() {
        let mut agent = RandomAgent::new();
        let state = GameState::new(5, 3);
        let mask = vec![false, false, false];

        let result = agent.select_action(&state, &mask);
        assert!(result.is_err());
    }
}
