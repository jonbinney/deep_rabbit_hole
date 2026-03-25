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
use crate::rotation::{build_rotated_state, create_rotation_mapping, remap_policy};

pub trait PlayGameObserver {
    fn on_state_snapshot(&mut self, step: usize, state: &GameState, action_mask: &[bool]);

    fn on_action_selected(
        &mut self,
        step: usize,
        root_value: Option<f32>,
        policy: &[f32],
        selected_action_index: usize,
    );
}

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
/// Player 0 moves first. Action selection runs in original orientation; any
/// current-player-downward rotation is handled inside evaluator codepaths.
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
    mut observer: Option<&mut dyn PlayGameObserver>,
) -> anyhow::Result<GameResult> {
    let mut state = GameState::new(board_size, max_walls);
    let (original_to_rotated, _) = create_rotation_mapping(board_size);

    let mut replay_items: Vec<ReplayBufferItem> = Vec::new();
    let mut winner: Option<i32> = None;

    let mut emitted_terminal_snapshot = false;

    for step in 0..max_steps {
        let current_player = state.current_player;

        // Match Python: run action selection in original orientation.
        let work_state = state.clone();
        let mask = work_state.get_action_mask();

        if let Some(obs) = observer.as_deref_mut() {
            obs.on_state_snapshot(step as usize, &state, &mask);
        }

        // Check for no valid actions (shouldn't happen in Quoridor, but be safe)
        if !mask.iter().any(|&m| m) {
            // Truncate
            emitted_terminal_snapshot = true;
            break;
        }

        // Build ResNet input from the working state
        let resnet_input = grid_game_state_to_resnet_input(&work_state);

        // Ask the appropriate agent for action
        let agent: &mut dyn ActionSelector = if current_player == 0 {
            agent_p1
        } else {
            agent_p2
        };
        let (action_idx, policy) = agent.select_action(&work_state, &mask)?;
        let root_value = agent
            .last_selection_trace()
            .and_then(|trace| trace.root_value);

        if let Some(obs) = observer.as_deref_mut() {
            obs.on_action_selected(step as usize, root_value, &policy, action_idx);
        }

        // Match Python storage semantics: replay is stored in current-player-downward frame.
        let (stored_input_3d, stored_policy, stored_mask) = if current_player == 1 {
            let rotated_state = build_rotated_state(&state);
            let rotated_input = grid_game_state_to_resnet_input(&rotated_state)
                .index_axis(ndarray::Axis(0), 0)
                .to_owned();
            let rotated_policy = remap_policy(&policy, &original_to_rotated);
            let rotated_mask = rotated_state.get_action_mask();
            (rotated_input, rotated_policy, rotated_mask)
        } else {
            (
                resnet_input.index_axis(ndarray::Axis(0), 0).to_owned(),
                policy.clone(),
                mask.clone(),
            )
        };

        replay_items.push(ReplayBufferItem {
            input_array: stored_input_3d,
            policy: stored_policy,
            action_mask: stored_mask,
            value: 0.0, // placeholder, backfilled after game ends
            player: current_player,
        });

        // Decode action index → (row, col, type) in working frame
        let action_triple = action_index_to_action(board_size, action_idx);

        let (a_row, a_col, a_type) = (action_triple[0], action_triple[1], action_triple[2]);

        // Apply action on the ORIGINAL game state
        state.step([a_row, a_col, a_type]);

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
    }

    if winner.is_none() && !emitted_terminal_snapshot && state.completed_steps >= max_steps as usize
    {
        let mask = state.get_action_mask();
        if let Some(obs) = observer.as_deref_mut() {
            obs.on_state_snapshot(state.completed_steps, &state, &mask);
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
        let result = play_game(&mut p1, &mut p2, 5, 3, 200, false, None).unwrap();

        // Game should complete within 200 steps on a 5×5 board
        assert!(result.num_turns > 0);
        assert!(!result.replay_items.is_empty());
    }

    #[test]
    fn test_play_game_alternating_players() {
        let mut p1 = FirstValidAgent;
        let mut p2 = FirstValidAgent;
        let result = play_game(&mut p1, &mut p2, 5, 0, 200, false, None).unwrap();

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
        let result = play_game(&mut p1, &mut p2, 5, 0, 200, false, None).unwrap();

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
        let result = play_game(&mut p1, &mut p2, 5, 3, 2, false, None).unwrap();

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
        let result = play_game(&mut p1, &mut p2, 5, 3, 200, false, None).unwrap();

        let grid_size = 5 * 2 + 3; // 13
        let total_actions = policy_size(5);

        for item in &result.replay_items {
            assert_eq!(item.input_array.shape(), &[5, grid_size, grid_size]);
            assert_eq!(item.policy.len(), total_actions);
            assert_eq!(item.action_mask.len(), total_actions);
        }
    }
}
