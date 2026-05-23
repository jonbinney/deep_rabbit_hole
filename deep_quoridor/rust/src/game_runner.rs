#![allow(dead_code)]

//! Core game loop for self-play.
//!
//! This module is independent of ONNX — the agent is abstracted behind the
//! [`ActionSelector`] trait so the game runner can be tested with mock agents.

use ndarray::Array3;

use crate::actions::{
    ACTION_MOVE, ACTION_WALL_HORIZONTAL, ACTION_WALL_VERTICAL, action_index_to_action,
};
use crate::agents::ActionSelector;
use crate::compact::q_bit_repr::CompactState;
use crate::compact::q_game_mechanics::QGameMechanics;
use crate::game_state::GameState;
use crate::grid_helpers::compact_state_to_resnet_input;
use crate::rotation::{create_rotation_mapping, remap_mask, remap_policy, rotate_compact_state};

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

/// Build a transient `GameState` from a compact state.
///
/// Used only to feed `PlayGameObserver::on_state_snapshot`, which keeps its
/// `&GameState` signature so the cross-language trace observer in
/// `python_consistency.rs` stays untouched. Allocates per call (per step); only
/// called when an observer is attached. Production self-play does not attach an
/// observer.
fn compact_to_game_state(
    mechanics: &QGameMechanics,
    data: CompactState,
    board_size: i32,
    max_walls: i32,
) -> GameState {
    let repr = mechanics.repr();
    let mut state = GameState::new(board_size, max_walls);
    state.grid = repr.to_grid(data);
    state.player_positions = repr.to_player_positions(data);
    state.walls_remaining = repr.to_walls_remaining(data);
    state.current_player = repr.get_current_player(data) as i32;
    state.completed_steps = repr.get_completed_steps(data);
    state
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

/// Play a complete game between two agents using the compact state.
///
/// `agent_p1` controls player 0 and `agent_p2` controls player 1.
/// Player 0 moves first. Action selection runs in original orientation; any
/// current-player-downward rotation is handled inside evaluator codepaths.
///
/// When `trace` is `true`, each step prints whose turn it is, the action
/// chosen, and the resulting board state via `QGameMechanics::display`.
pub fn play_game(
    agent_p1: &mut dyn ActionSelector,
    agent_p2: &mut dyn ActionSelector,
    board_size: i32,
    max_walls: i32,
    max_steps: i32,
    trace: bool,
    mut observer: Option<&mut dyn PlayGameObserver>,
) -> anyhow::Result<GameResult> {
    let mechanics =
        QGameMechanics::new(board_size as usize, max_walls as usize, max_steps as usize);
    let mut data = mechanics.create_initial_state();
    let (original_to_rotated, _) = create_rotation_mapping(board_size);

    let mut replay_items: Vec<ReplayBufferItem> = Vec::new();
    let mut winner: Option<i32> = None;

    let mut emitted_terminal_snapshot = false;

    for step in 0..max_steps {
        let current_player = mechanics.repr().get_current_player(data) as i32;
        let mask = mechanics.get_action_mask_immut(data);

        if let Some(obs) = observer.as_deref_mut() {
            let state = compact_to_game_state(&mechanics, data, board_size, max_walls);
            obs.on_state_snapshot(step as usize, &state, &mask);
        }

        // Check for no valid actions (shouldn't happen in Quoridor, but be safe)
        if !mask.iter().any(|&m| m) {
            emitted_terminal_snapshot = true;
            break;
        }

        // Build ResNet input from the working state (for current-player frame storage).
        let resnet_input = compact_state_to_resnet_input(&mechanics, data);

        // Ask the appropriate agent for action
        let agent: &mut dyn ActionSelector = if current_player == 0 {
            agent_p1
        } else {
            agent_p2
        };
        let (action_idx, policy) = agent.select_action(data, &mechanics, &mask)?;
        let root_value = agent
            .last_selection_trace()
            .and_then(|trace| trace.root_value);

        if let Some(obs) = observer.as_deref_mut() {
            obs.on_action_selected(step as usize, root_value, &policy, action_idx);
        }

        // Match Python storage semantics: replay is stored in current-player-downward frame.
        let (stored_input_3d, stored_policy, stored_mask) = if current_player == 1 {
            let rotated_data = rotate_compact_state(&mechanics, data);
            let rotated_input = compact_state_to_resnet_input(&mechanics, rotated_data)
                .index_axis(ndarray::Axis(0), 0)
                .to_owned();
            let rotated_policy = remap_policy(&policy, &original_to_rotated);
            // `QGameMechanics::goal_rows` does not flip under rotation, so the
            // mask must come from remapping the original mask, not from
            // re-validating on rotated data (which would treat walls that
            // block the rotated player's path as legal).
            let rotated_mask = remap_mask(&mask, &original_to_rotated);
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

        // Apply action on the canonical compact state
        mechanics.apply_action_index(&mut data, action_idx);

        if trace {
            let player_label = if current_player == 0 { "P1" } else { "P2" };
            let action_triple = action_index_to_action(board_size, action_idx);
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
            print!("{}\n", mechanics.display(data));
        }

        // Check win (current_player already switched after apply, so check previous player)
        if mechanics.check_win(data, current_player as usize) {
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

    let completed_steps = mechanics.repr().get_completed_steps(data);
    if winner.is_none() && !emitted_terminal_snapshot && completed_steps >= max_steps as usize {
        let mask = mechanics.get_action_mask_immut(data);
        if let Some(obs) = observer.as_deref_mut() {
            let state = compact_to_game_state(&mechanics, data, board_size, max_walls);
            obs.on_state_snapshot(completed_steps, &state, &mask);
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
            _data: CompactState,
            _mechanics: &QGameMechanics,
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

        assert!(result.num_turns > 0);
        assert!(!result.replay_items.is_empty());
    }

    #[test]
    fn test_play_game_alternating_players() {
        let mut p1 = FirstValidAgent;
        let mut p2 = FirstValidAgent;
        let result = play_game(&mut p1, &mut p2, 5, 0, 200, false, None).unwrap();

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
