#![allow(dead_code)]

//! Core game loop for self-play.
//!
//! This module is independent of ONNX — the agent is abstracted behind the
//! [`ActionSelector`] trait so the game runner can be tested with mock agents.

use ndarray::Array3;

use crate::actions::{
    action_index_to_action, ACTION_MOVE, ACTION_WALL_HORIZONTAL, ACTION_WALL_VERTICAL,
};
use crate::agents::ActionSelector;
use crate::game_state::GameState;
use crate::grid::CELL_WALL;
use crate::grid_helpers::grid_game_state_to_resnet_input;
use crate::rotation::{
    create_rotation_mapping, remap_mask, remap_policy, rotate_goal_rows, rotate_grid_180,
    rotate_player_positions,
};

/// Format an action triple as a human-readable string.
fn format_action(_board_size: i32, row: i32, col: i32, action_type: i32) -> String {
    match action_type {
        ACTION_MOVE => format!("Move to ({}, {})", row, col),
        ACTION_WALL_HORIZONTAL => format!("Place horizontal wall at ({}, {})", row, col),
        ACTION_WALL_VERTICAL => format!("Place vertical wall at ({}, {})", row, col),
        _ => format!("Unknown action type {}", action_type),
    }
}

/// Render the board state as a human-readable string.
///
/// Shows player positions as `1` and `2`, walls as `|` (vertical) and `-`
/// (horizontal), and empty cells as `.`.
///
/// The board is always shown in the original (un-rotated) orientation.
pub fn display_board(
    grid: &ndarray::ArrayView2<i8>,
    player_positions: &ndarray::ArrayView2<i32>,
    walls_remaining: &ndarray::ArrayView1<i32>,
    board_size: i32,
) -> String {
    let mut out = String::new();
    let bs = board_size as usize;

    // Column header
    out.push_str("    ");
    for c in 0..bs {
        out.push_str(&format!(" {} ", c));
    }
    out.push('\n');

    let p0_row = player_positions[[0, 0]] as usize;
    let p0_col = player_positions[[0, 1]] as usize;
    let p1_row = player_positions[[1, 0]] as usize;
    let p1_col = player_positions[[1, 1]] as usize;

    for row in 0..bs {
        // --- cell row ---
        out.push_str(&format!("{:>3} ", row));
        for col in 0..bs {
            // cell content
            if row == p0_row && col == p0_col {
                out.push('1');
            } else if row == p1_row && col == p1_col {
                out.push('2');
            } else {
                out.push('.');
            }

            // vertical wall to the right
            if col < bs - 1 {
                // Grid coord of the gap between (row,col) and (row,col+1)
                let gr = (row * 2 + 2) as usize;
                let gc = (col * 2 + 3) as usize;
                if grid[[gr, gc]] == CELL_WALL {
                    out.push_str(" | ");
                } else {
                    out.push_str("   ");
                }
            }
        }
        // Metadata on the right of first two rows
        match row {
            0 => out.push_str(&format!("   P1 walls: {}", walls_remaining[0])),
            1 => out.push_str(&format!("   P2 walls: {}", walls_remaining[1])),
            _ => {}
        }
        out.push('\n');

        // --- horizontal wall row between this row and the next ---
        if row < bs - 1 {
            out.push_str("    ");
            for col in 0..bs {
                // Grid coord of the gap between (row,col) and (row+1,col)
                let gr = (row * 2 + 3) as usize;
                let gc = (col * 2 + 2) as usize;
                if grid[[gr, gc]] == CELL_WALL {
                    out.push('-');
                } else {
                    out.push(' ');
                }
                if col < bs - 1 {
                    out.push_str("   ");
                }
            }
            out.push('\n');
        }
    }
    out
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

/// Play a complete game between two agents.
///
/// `agent_p1` controls player 0 and `agent_p2` controls player 1.
/// Player 0 moves first. When Player 1 is the current player, the board is
/// rotated 180° before being passed to `agent_p2` so the network always sees
/// "current player moving downward".
///
/// When `trace` is `true`, each step prints whose turn it is, the action
/// chosen, and the resulting board state in the original (un-rotated)
/// orientation.
pub fn play_game(
    agent_p1: &mut dyn ActionSelector,
    agent_p2: &mut dyn ActionSelector,
    board_size: i32,
    max_walls: i32,
    max_steps: i32,
    trace: bool,
    game_length_bonus_factor: f32,
) -> anyhow::Result<GameResult> {
    let mut state = GameState::new(board_size, max_walls);

    let mut replay_items: Vec<ReplayBufferItem> = Vec::new();
    let mut winner: Option<i32> = None;

    // Precompute rotation mapping for storing player-1 replay data
    let (orig_to_rot, _) = create_rotation_mapping(board_size);

    for step in 0..max_steps {
        let current_player = state.current_player;

        // Compute action mask in the original frame
        let mask = state.get_action_mask();

        // Check for no valid actions (shouldn't happen in Quoridor, but be safe)
        if !mask.iter().any(|&m| m) {
            // Truncate
            break;
        }

        // Ask the appropriate agent for action (in original frame — evaluator
        // handles rotation internally for player 1 during MCTS)
        let agent: &mut dyn ActionSelector = if current_player == 0 {
            agent_p1
        } else {
            agent_p2
        };
        let (action_idx, policy) = agent.select_action(&state, &mask)?;

        // Store replay item in "current-player-faces-downward" frame.
        // For player 0 this is the original frame; for player 1 we rotate.
        if current_player == 1 {
            let rot_grid = rotate_grid_180(&state.grid());
            let rot_positions = rotate_player_positions(&state.player_positions(), board_size);
            let rot_goals = rotate_goal_rows(&state.goal_rows());
            let rot_state = GameState {
                grid: rot_grid,
                player_positions: rot_positions,
                walls_remaining: state.walls_remaining.clone(),
                goal_rows: rot_goals,
                current_player,
                board_size,
                completed_steps: state.completed_steps,
            };
            let resnet_input = grid_game_state_to_resnet_input(&rot_state);
            let input_3d = resnet_input.index_axis(ndarray::Axis(0), 0).to_owned();
            let rot_policy = remap_policy(&policy, &orig_to_rot);
            let rot_mask = remap_mask(&mask, &orig_to_rot);
            replay_items.push(ReplayBufferItem {
                input_array: input_3d,
                policy: rot_policy,
                action_mask: rot_mask,
                value: 0.0,
                player: current_player,
            });
        } else {
            let resnet_input = grid_game_state_to_resnet_input(&state);
            let input_3d = resnet_input.index_axis(ndarray::Axis(0), 0).to_owned();
            replay_items.push(ReplayBufferItem {
                input_array: input_3d,
                policy: policy.clone(),
                action_mask: mask.clone(),
                value: 0.0,
                player: current_player,
            });
        }

        // Decode action index → (row, col, type) in original frame
        let action_triple = action_index_to_action(board_size, action_idx);

        // Apply action on the original game state (no un-rotation needed)
        state.step(action_triple);

        if trace {
            let player_label = if current_player == 0 { "P1" } else { "P2" };
            println!(
                "--- Step {} | {} ---\n{}",
                step + 1,
                player_label,
                format_action(
                    board_size,
                    action_triple[0],
                    action_triple[1],
                    action_triple[2]
                ),
            );
            print!(
                "{}\n",
                display_board(
                    &state.grid(),
                    &state.player_positions(),
                    &state.walls_remaining(),
                    board_size
                ),
            );
        }

        // Check win (current_player already switched after step, so check previous player)
        if state.check_win(current_player) {
            winner = Some(current_player);
            let completed_steps = step + 1;
            let factor =
                1.0 + game_length_bonus_factor * (1.0 - completed_steps as f32 / max_steps as f32);
            // Backfill values: +1 for winner, -1 for loser, scaled by game length bonus
            for item in replay_items.iter_mut() {
                item.value = if item.player == current_player {
                    factor
                } else {
                    -factor
                };
            }
            return Ok(GameResult {
                winner,
                num_turns: completed_steps,
                replay_items,
            });
        }
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
    use crate::actions::policy_size;
    use crate::rotation::create_rotation_mapping;

    /// A mock agent that always picks the first valid action.
    struct FirstValidAgent;

    impl ActionSelector for FirstValidAgent {
        fn select_action(
            &mut self,
            _state: &GameState,
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
        let mut p1 = FirstValidAgent;
        let mut p2 = FirstValidAgent;
        let result = play_game(&mut p1, &mut p2, 5, 3, 200, false, 0.0).unwrap();

        // Game should complete within 200 steps on a 5×5 board
        assert!(result.num_turns > 0);
        assert!(!result.replay_items.is_empty());
    }

    #[test]
    fn test_play_game_alternating_players() {
        let mut p1 = FirstValidAgent;
        let mut p2 = FirstValidAgent;
        let result = play_game(&mut p1, &mut p2, 5, 0, 200, false, 0.0).unwrap();

        // With 0 walls the game should end quickly via moves only
        // Players should alternate
        for (i, item) in result.replay_items.iter().enumerate() {
            assert_eq!(item.player, (i as i32) % 2);
        }
    }

    #[test]
    fn test_play_game_winner_values() {
        let mut p1 = FirstValidAgent;
        let mut p2 = FirstValidAgent;
        let result = play_game(&mut p1, &mut p2, 5, 0, 200, false, 0.0).unwrap();

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
        let mut p1 = FirstValidAgent;
        let mut p2 = FirstValidAgent;
        // Very short max_steps to force truncation
        let result = play_game(&mut p1, &mut p2, 5, 3, 2, false, 0.0).unwrap();

        if result.winner.is_none() {
            for item in &result.replay_items {
                assert_eq!(item.value, 0.0);
            }
        }
    }

    #[test]
    fn test_replay_items_have_correct_shapes() {
        let mut p1 = FirstValidAgent;
        let mut p2 = FirstValidAgent;
        let result = play_game(&mut p1, &mut p2, 5, 3, 200, false, 0.0).unwrap();

        let grid_size = 5 * 2 + 3; // 13
        let total_actions = policy_size(5);

        for item in &result.replay_items {
            assert_eq!(item.input_array.shape(), &[5, grid_size, grid_size]);
            assert_eq!(item.policy.len(), total_actions);
            assert_eq!(item.action_mask.len(), total_actions);
        }
    }

    /// Find the grid position of the 1.0 in a channel of the input tensor.
    fn find_position_in_channel(input: &ndarray::Array3<f32>, channel: usize) -> Option<(usize, usize)> {
        let grid_size = input.shape()[1];
        for r in 0..grid_size {
            for c in 0..grid_size {
                if input[[channel, r, c]] == 1.0 {
                    return Some((r, c));
                }
            }
        }
        None
    }

    #[test]
    fn test_replay_player1_input_has_current_player_near_top() {
        // Play a game and check that player 1 replay items have the current
        // player (player 1) positioned near the top of the board after rotation.
        let mut p1 = FirstValidAgent;
        let mut p2 = FirstValidAgent;
        let result = play_game(&mut p1, &mut p2, 5, 0, 200, false, 0.0).unwrap();

        // On a 5x5 board, grid is 13x13. Board row 0 = grid row 2, row 4 = grid row 10.
        // "Near the top" means grid row <= 6 (board row <= 2, top half).
        for item in &result.replay_items {
            // Channel 1 = current player position
            let (row, _col) = find_position_in_channel(&item.input_array, 1)
                .expect("Should find current player position in channel 1");

            if item.player == 0 {
                // Player 0 starts at top and should generally stay in upper-ish rows at beginning
                // Just check it's a valid grid position (every even row + 2)
                assert!(row >= 2 && row <= 10, "Player 0 pos at grid row {}", row);
            } else {
                // Player 1: after rotation, should be at the top of the board (low grid row).
                // Player 1 starts at board row 4 (grid row 10), after rotation → board row 0 (grid row 2).
                // As the game progresses, player 1 moves toward their goal (board row 0 in original,
                // which is board row 4 = grid row 10 in rotated frame). So they should start near
                // grid row 2 and move down. All positions should have the player in the upper half
                // initially, but may go lower later. We just check the FIRST player 1 item.
            }
        }

        // Specifically check the first player 1 item (step 1)
        let first_p1 = result.replay_items.iter().find(|i| i.player == 1).unwrap();
        let (row, _col) = find_position_in_channel(&first_p1.input_array, 1).unwrap();
        // Player 1 starts at board (4, 2), after 180° rotation → (0, 2), grid (2, 6)
        assert_eq!(row, 2, "First player 1 item: current player should be at grid row 2 (top) after rotation, got {}", row);
    }

    #[test]
    fn test_remap_mask_matches_rotated_state_mask() {
        // Verify that remapping the original mask via the rotation mapping
        // produces the same result as computing the mask on the rotated state.
        use crate::rotation::{remap_mask, rotate_grid_180, rotate_player_positions, rotate_goal_rows};

        let board_size = 5;
        let max_walls = 2;

        // Create a state and make a move so it's player 1's turn
        let state0 = GameState::new(board_size, max_walls);
        let mask0 = state0.get_action_mask();
        let first_valid = mask0.iter().position(|&m| m).unwrap();
        let action = action_index_to_action(board_size, first_valid);
        let state1 = state0.clone_and_step(action);
        assert_eq!(state1.current_player, 1);

        // Approach 1: remap original mask
        let orig_mask = state1.get_action_mask();
        let (orig_to_rot, _) = create_rotation_mapping(board_size);
        let remapped_mask = remap_mask(&orig_mask, &orig_to_rot);

        // Approach 2: compute mask on rotated state
        let rot_grid = rotate_grid_180(&state1.grid());
        let rot_positions = rotate_player_positions(&state1.player_positions(), board_size);
        let rot_goals = rotate_goal_rows(&state1.goal_rows());
        let rot_state = GameState {
            grid: rot_grid,
            player_positions: rot_positions,
            walls_remaining: state1.walls_remaining.clone(),
            goal_rows: rot_goals,
            current_player: 1,
            board_size,
            completed_steps: state1.completed_steps,
        };
        let rotated_state_mask = rot_state.get_action_mask();

        // They should be identical
        assert_eq!(
            remapped_mask, rotated_state_mask,
            "Remapped mask differs from rotated state mask"
        );
    }

    #[test]
    fn test_replay_player1_policy_is_rotated() {
        // Verify that the policy stored for player 1 is in the rotated frame.
        // The mock agent always picks the first valid action. In original frame,
        // this is some action. In the stored replay, the policy should be
        // in the rotated frame.
        let mut p1 = FirstValidAgent;
        let mut p2 = FirstValidAgent;
        let result = play_game(&mut p1, &mut p2, 5, 0, 200, false, 0.0).unwrap();

        let (orig_to_rot, _) = create_rotation_mapping(5);

        // For each player 1 item, verify the policy is in rotated frame
        for (step, item) in result.replay_items.iter().enumerate() {
            if item.player != 1 {
                continue;
            }

            // The policy should have exactly one non-zero entry (FirstValidAgent picks one action)
            let nonzero_idx = item.policy.iter().position(|&p| p > 0.0)
                .expect("Policy should have at least one non-zero entry");

            // This index should be in the ROTATED frame.
            // Decode it as a rotated action and verify it's valid.
            let rotated_action = action_index_to_action(5, nonzero_idx);
            assert!(
                rotated_action[0] >= 0 && rotated_action[0] < 5 &&
                rotated_action[1] >= 0 && rotated_action[1] < 5,
                "Step {}: Invalid rotated action {:?} at index {}",
                step, rotated_action, nonzero_idx
            );

            // The action should be valid according to the rotated mask
            assert!(
                item.action_mask[nonzero_idx],
                "Step {}: Policy has non-zero at index {} but mask says invalid",
                step, nonzero_idx
            );
        }
    }

    #[test]
    fn test_replay_input_matches_direct_computation() {
        // For player 0: input should match grid_game_state_to_resnet_input on original state.
        // For player 1: input should match grid_game_state_to_resnet_input on rotated state.
        //
        // We can't easily reconstruct the exact game state at each step from the replay data,
        // but we can verify key properties:
        // 1. Channel 0 (walls) should have border walls in the right places
        // 2. Channel 1 and 2 should each have exactly one 1.0 (player positions)
        // 3. Channel 3 and 4 should have uniform values (walls remaining)
        let mut p1 = FirstValidAgent;
        let mut p2 = FirstValidAgent;
        let result = play_game(&mut p1, &mut p2, 5, 2, 200, false, 0.0).unwrap();

        for (step, item) in result.replay_items.iter().enumerate() {
            let input = &item.input_array;

            // Channel 1 (current player) should have exactly one 1.0
            let ch1_sum: f32 = input.slice(ndarray::s![1, .., ..]).iter().sum();
            assert!(
                (ch1_sum - 1.0).abs() < 1e-5,
                "Step {}: Channel 1 sum should be 1.0, got {}",
                step, ch1_sum
            );

            // Channel 2 (opponent) should have exactly one 1.0
            let ch2_sum: f32 = input.slice(ndarray::s![2, .., ..]).iter().sum();
            assert!(
                (ch2_sum - 1.0).abs() < 1e-5,
                "Step {}: Channel 2 sum should be 1.0, got {}",
                step, ch2_sum
            );

            // Channel 3 and 4 should be uniform
            let ch3_vals: Vec<f32> = input.slice(ndarray::s![3, .., ..]).iter().copied().collect();
            let ch3_val = ch3_vals[0];
            assert!(
                ch3_vals.iter().all(|&v| v == ch3_val),
                "Step {}: Channel 3 should be uniform",
                step
            );

            let ch4_vals: Vec<f32> = input.slice(ndarray::s![4, .., ..]).iter().copied().collect();
            let ch4_val = ch4_vals[0];
            assert!(
                ch4_vals.iter().all(|&v| v == ch4_val),
                "Step {}: Channel 4 should be uniform",
                step
            );

            // Border walls in channel 0: top-left corner should always be a wall
            assert_eq!(
                input[[0, 0, 0]], 1.0,
                "Step {}: Top-left border wall missing",
                step
            );
            assert_eq!(
                input[[0, 12, 12]], 1.0,
                "Step {}: Bottom-right border wall missing",
                step
            );
        }
    }
}
