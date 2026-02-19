#![allow(dead_code)]

//! Core game loop for self-play.
//!
//! This module is independent of ONNX — the agent is abstracted behind the
//! [`ActionSelector`] trait so the game runner can be tested with mock agents.

use ndarray::{Array3, ArrayView1, ArrayView2};

use crate::actions::{action_index_to_action, compute_full_action_mask, policy_size};
use crate::game_state::{apply_action, check_win, create_initial_state};
use crate::grid_helpers::grid_game_state_to_resnet_input;
use crate::rotation::{
    rotate_action_coords, rotate_goal_rows,
    rotate_grid_180, rotate_player_positions,
};

/// Trait for agents that select actions given a game state.
///
/// The provided state may already be rotated for Player 1.
pub trait ActionSelector {
    /// Select an action given the game state.
    ///
    /// Arguments are in the coordinate frame presented (possibly rotated).
    ///
    /// Returns `(action_index, policy_probabilities)` where `action_index` is
    /// a flat index into the policy vector and `policy_probabilities` is the
    /// full softmax output.
    fn select_action(
        &mut self,
        grid: &ArrayView2<i8>,
        player_positions: &ArrayView2<i32>,
        walls_remaining: &ArrayView1<i32>,
        goal_rows: &ArrayView1<i32>,
        current_player: i32,
        action_mask: &[bool],
    ) -> anyhow::Result<(usize, Vec<f32>)>;
}

/// One turn's training data, stored in "current-player-faces-downward" coords.
pub struct ReplayBufferItem {
    /// ResNet input tensor (5, M, M) — the batch dimension is squeezed out.
    pub input_array: Array3<f32>,
    /// Full softmax policy from the agent.
    pub policy: Vec<f32>,
    /// Boolean action mask (which actions were legal).
    pub action_mask: Vec<bool>,
    /// Game outcome from this player's perspective: +1 win, −1 loss, 0 draw/truncated.
    pub value: f32,
    /// Which player (0 or 1) was acting this turn.
    pub player: i32,
}

/// Result of a complete game.
pub struct GameResult {
    /// `Some(player_id)` if that player won, `None` if truncated.
    pub winner: Option<i32>,
    /// Total number of turns played.
    pub num_turns: i32,
    /// Replay buffer items for training.
    pub replay_items: Vec<ReplayBufferItem>,
}

/// Play a complete game between two agents (both use the same `ActionSelector`).
///
/// Player 0 moves first. When Player 1 is the current player, the board is
/// rotated 180° before being passed to the agent so the network always sees
/// "current player moving downward".
pub fn play_game<A: ActionSelector>(
    agent: &mut A,
    board_size: i32,
    max_walls: i32,
    max_steps: i32,
) -> anyhow::Result<GameResult> {
    let (mut grid, mut player_positions, mut walls_remaining, goal_rows) =
        create_initial_state(board_size, max_walls);

    let total_actions = policy_size(board_size);

    let mut replay_items: Vec<ReplayBufferItem> = Vec::new();
    let mut current_player: i32 = 0;
    let mut winner: Option<i32> = None;

    for step in 0..max_steps {
        // Possibly rotated copies for Player 1
        let (work_grid, work_positions, work_goals);
        if current_player == 1 {
            work_grid = rotate_grid_180(&grid.view());
            work_positions = rotate_player_positions(&player_positions.view(), board_size);
            work_goals = rotate_goal_rows(&goal_rows.view());
        } else {
            work_grid = grid.clone();
            work_positions = player_positions.clone();
            work_goals = goal_rows.clone();
        }

        // Compute action mask in the working (possibly rotated) frame
        let mut mask = vec![false; total_actions];
        compute_full_action_mask(
            &work_grid.view(),
            &work_positions.view(),
            &walls_remaining.view(),
            &work_goals.view(),
            current_player,
            &mut mask,
        );

        // Check for no valid actions (shouldn't happen in Quoridor, but be safe)
        if !mask.iter().any(|&m| m) {
            // Truncate
            break;
        }

        // Build ResNet input from the working state
        let resnet_input = grid_game_state_to_resnet_input(
            &work_grid.view(),
            &work_positions.view(),
            &walls_remaining.view(),
            current_player,
        );

        // Ask agent for action
        let (action_idx, policy) = agent.select_action(
            &work_grid.view(),
            &work_positions.view(),
            &walls_remaining.view(),
            &work_goals.view(),
            current_player,
            &mask,
        )?;

        // Store replay item (in rotated frame — the frame the model saw)
        let input_3d = resnet_input
            .index_axis(ndarray::Axis(0), 0)
            .to_owned();
        replay_items.push(ReplayBufferItem {
            input_array: input_3d,
            policy: policy.clone(),
            action_mask: mask.clone(),
            value: 0.0, // placeholder, backfilled after game ends
            player: current_player,
        });

        // Decode action index → (row, col, type) in working frame
        let action_triple = action_index_to_action(board_size, action_idx);

        // If Player 1, un-rotate action coordinates back to original frame
        let (a_row, a_col, a_type) = if current_player == 1 {
            rotate_action_coords(board_size, action_triple[0], action_triple[1], action_triple[2])
        } else {
            (action_triple[0], action_triple[1], action_triple[2])
        };

        // Apply action on the ORIGINAL game state
        let action_arr = ndarray::Array1::from(vec![a_row, a_col, a_type]);
        apply_action(
            &mut grid.view_mut(),
            &mut player_positions.view_mut(),
            &mut walls_remaining.view_mut(),
            current_player,
            &action_arr.view(),
        );

        // Check win
        if check_win(&player_positions.view(), &goal_rows.view(), current_player) {
            winner = Some(current_player);
            // Backfill values: +1 for winner, -1 for loser
            for item in replay_items.iter_mut() {
                item.value = if item.player == current_player {
                    1.0
                } else {
                    -1.0
                };
            }
            return Ok(GameResult {
                winner,
                num_turns: step + 1,
                replay_items,
            });
        }

        // Alternate players
        current_player = 1 - current_player;
    }

    // Truncated — values stay 0.0
    Ok(GameResult {
        winner,
        num_turns: max_steps,
        replay_items,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayView1, ArrayView2};

    /// A mock agent that always picks the first valid action.
    struct FirstValidAgent;

    impl ActionSelector for FirstValidAgent {
        fn select_action(
            &mut self,
            _grid: &ArrayView2<i8>,
            _player_positions: &ArrayView2<i32>,
            _walls_remaining: &ArrayView1<i32>,
            _goal_rows: &ArrayView1<i32>,
            _current_player: i32,
            action_mask: &[bool],
        ) -> anyhow::Result<(usize, Vec<f32>)> {
            let idx = action_mask
                .iter()
                .position(|&m| m)
                .expect("no valid action");
            let mut policy = vec![0.0f32; action_mask.len()];
            policy[idx] = 1.0;
            Ok((idx, policy))
        }
    }

    #[test]
    fn test_play_game_completes() {
        let mut agent = FirstValidAgent;
        let result = play_game(&mut agent, 5, 3, 200).unwrap();

        // Game should complete within 200 steps on a 5×5 board
        assert!(result.num_turns > 0);
        assert!(!result.replay_items.is_empty());
    }

    #[test]
    fn test_play_game_alternating_players() {
        let mut agent = FirstValidAgent;
        let result = play_game(&mut agent, 5, 0, 200).unwrap();

        // With 0 walls the game should end quickly via moves only
        // Players should alternate
        for (i, item) in result.replay_items.iter().enumerate() {
            assert_eq!(item.player, (i as i32) % 2);
        }
    }

    #[test]
    fn test_play_game_winner_values() {
        let mut agent = FirstValidAgent;
        let result = play_game(&mut agent, 5, 0, 200).unwrap();

        if let Some(w) = result.winner {
            for item in &result.replay_items {
                if item.player == w {
                    assert_eq!(item.value, 1.0);
                } else {
                    assert_eq!(item.value, -1.0);
                }
            }
        }
    }

    #[test]
    fn test_play_game_truncation_values() {
        let mut agent = FirstValidAgent;
        // Very short max_steps to force truncation
        let result = play_game(&mut agent, 5, 3, 2).unwrap();

        if result.winner.is_none() {
            for item in &result.replay_items {
                assert_eq!(item.value, 0.0);
            }
        }
    }

    #[test]
    fn test_replay_items_have_correct_shapes() {
        let mut agent = FirstValidAgent;
        let result = play_game(&mut agent, 5, 3, 200).unwrap();

        let grid_size = 5 * 2 + 3; // 13
        let total_actions = policy_size(5);

        for item in &result.replay_items {
            assert_eq!(item.input_array.shape(), &[5, grid_size, grid_size]);
            assert_eq!(item.policy.len(), total_actions);
            assert_eq!(item.action_mask.len(), total_actions);
        }
    }
}
