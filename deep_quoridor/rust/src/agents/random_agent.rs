//! Random agent — picks a valid action uniformly at random.

use ndarray::{ArrayView1, ArrayView2};
use rand::Rng;

use crate::agents::ActionSelector;

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
        _grid: &ArrayView2<i8>,
        _player_positions: &ArrayView2<i32>,
        _walls_remaining: &ArrayView1<i32>,
        _goal_rows: &ArrayView1<i32>,
        _current_player: i32,
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
    use ndarray::{Array1, Array2};

    #[test]
    fn test_random_agent_picks_valid_action() {
        let mut agent = RandomAgent::new();
        let grid = Array2::<i8>::zeros((13, 13));
        let positions = Array2::<i32>::zeros((2, 2));
        let walls = Array1::<i32>::zeros(2);
        let goals = Array1::<i32>::zeros(2);
        let mask = vec![false, false, true, false, true, true];

        for _ in 0..50 {
            let (idx, _) = agent
                .select_action(
                    &grid.view(),
                    &positions.view(),
                    &walls.view(),
                    &goals.view(),
                    0,
                    &mask,
                )
                .unwrap();
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
        let grid = Array2::<i8>::zeros((13, 13));
        let positions = Array2::<i32>::zeros((2, 2));
        let walls = Array1::<i32>::zeros(2);
        let goals = Array1::<i32>::zeros(2);
        let mask = vec![false, true, true, false, true];

        let (_, policy) = agent
            .select_action(
                &grid.view(),
                &positions.view(),
                &walls.view(),
                &goals.view(),
                0,
                &mask,
            )
            .unwrap();

        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Invalid actions should have 0 probability
        assert_eq!(policy[0], 0.0);
        assert_eq!(policy[3], 0.0);
    }

    #[test]
    fn test_random_agent_no_valid_actions_fails() {
        let mut agent = RandomAgent::new();
        let grid = Array2::<i8>::zeros((13, 13));
        let positions = Array2::<i32>::zeros((2, 2));
        let walls = Array1::<i32>::zeros(2);
        let goals = Array1::<i32>::zeros(2);
        let mask = vec![false, false, false];

        let result = agent.select_action(
            &grid.view(),
            &positions.view(),
            &walls.view(),
            &goals.view(),
            0,
            &mask,
        );
        assert!(result.is_err());
    }
}
