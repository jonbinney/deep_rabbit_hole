#![allow(dead_code)]

//! Core game loop for self-play.
//!
//! This module is independent of ONNX — the agent is abstracted behind the
//! [`ActionSelector`] trait so the game runner can be tested with mock agents.

use ndarray::Array3;

use crate::actions::{
    action_index_to_action, compute_full_action_mask, policy_size, ACTION_MOVE,
    ACTION_WALL_HORIZONTAL, ACTION_WALL_VERTICAL,
};
use crate::agents::ActionSelector;
use crate::game_state::{apply_action, check_win, create_initial_state};
use crate::grid::CELL_WALL;
use crate::grid_helpers::grid_game_state_to_resnet_input;
use crate::rotation::{
    rotate_action_coords, rotate_goal_rows, rotate_grid_180, rotate_player_positions,
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

        // Ask the appropriate agent for action
        let agent: &mut dyn ActionSelector = if current_player == 0 {
            agent_p1
        } else {
            agent_p2
        };
        let (action_idx, policy) = agent.select_action(
            &work_grid.view(),
            &work_positions.view(),
            &walls_remaining.view(),
            &work_goals.view(),
            current_player,
            &mask,
        )?;

        // Store replay item (in rotated frame — the frame the model saw)
        let input_3d = resnet_input.index_axis(ndarray::Axis(0), 0).to_owned();
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
            rotate_action_coords(
                board_size,
                action_triple[0],
                action_triple[1],
                action_triple[2],
            )
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

        if trace {
            let player_label = if current_player == 0 { "P1" } else { "P2" };
            println!(
                "--- Step {} | {} ---\n{}",
                step + 1,
                player_label,
                format_action(board_size, a_row, a_col, a_type),
            );
            print!(
                "{}\n",
                display_board(
                    &grid.view(),
                    &player_positions.view(),
                    &walls_remaining.view(),
                    board_size
                ),
            );
        }

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
        let mut p1 = FirstValidAgent;
        let mut p2 = FirstValidAgent;
        let result = play_game(&mut p1, &mut p2, 5, 3, 200, false).unwrap();

        // Game should complete within 200 steps on a 5×5 board
        assert!(result.num_turns > 0);
        assert!(!result.replay_items.is_empty());
    }

    #[test]
    fn test_play_game_alternating_players() {
        let mut p1 = FirstValidAgent;
        let mut p2 = FirstValidAgent;
        let result = play_game(&mut p1, &mut p2, 5, 0, 200, false).unwrap();

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
        let result = play_game(&mut p1, &mut p2, 5, 0, 200, false).unwrap();

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
        let result = play_game(&mut p1, &mut p2, 5, 3, 2, false).unwrap();

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
        let result = play_game(&mut p1, &mut p2, 5, 3, 200, false).unwrap();

        let grid_size = 5 * 2 + 3; // 13
        let total_actions = policy_size(5);

        for item in &result.replay_items {
            assert_eq!(item.input_array.shape(), &[5, grid_size, grid_size]);
            assert_eq!(item.policy.len(), total_actions);
            assert_eq!(item.action_mask.len(), total_actions);
        }
    }
}
