/// QBitRepr-based minimax algorithm for Quoridor

use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use crate::q_game_mechanics::QGameMechanics;

pub const WINNING_REWARD: f32 = 1e6;

/// Log entry for QBitRepr-based minimax
#[derive(Clone)]
#[allow(dead_code)]
pub struct MinimaxLogEntry {
    pub data: Vec<u8>,  // Packed game state
    pub agent_player: usize,
    pub actions: Vec<(usize, usize, usize)>,  // (row, col, action_type) where type: 0=move, 1/2=wall
    pub values: Vec<f32>,
}

/// Compute distance to goal for a player using QBitRepr state
fn distance_to_goal(mechanics: &QGameMechanics, data: &[u8], player: usize) -> i32 {
    let (row, col)= mechanics.repr().get_player_position(data, player);
    let board_size = mechanics.repr().board_size();
    let goal_row = if player == 0 { 0 } else { board_size - 1 };

    // Simple BFS to find shortest path
    let mut visited = vec![false; board_size * board_size];
    let mut queue = std::collections::VecDeque::new();

    queue.push_back((row, col, 0));
    visited[row * board_size + col] = true;

    while let Some((curr_row, curr_col, dist)) = queue.pop_front() {
        if curr_row == goal_row {
            return dist;
        }

        // Try all 4 directions
        let directions = [
            (curr_row.wrapping_sub(1), curr_col),
            (curr_row + 1, curr_col),
            (curr_row, curr_col.wrapping_sub(1)),
            (curr_row, curr_col + 1),
        ];

        for (next_row, next_col) in directions {
            if next_row >= board_size || next_col >= board_size {
                continue;
            }

            let next_idx = next_row * board_size + next_col;
            if visited[next_idx] {
                continue;
            }

            // Check if wall blocks movement
            // For horizontal movement (changing col), check vertical walls
            // For vertical movement (changing row), check horizontal walls
            let wall_blocked = if curr_row == next_row {
                // Moving horizontally - check for vertical wall
                let wall_col = curr_col.min(next_col);
                let wall_row = if curr_row > 0 { curr_row - 1 } else { curr_row };
                (wall_row < board_size - 1 && wall_col < board_size - 1 &&
                    mechanics.repr().get_wall(data, wall_row, wall_col, 0)) ||
                (curr_row < board_size - 1 && wall_col < board_size - 1 &&
                    mechanics.repr().get_wall(data, curr_row, wall_col, 0))
            } else {
                // Moving vertically - check for horizontal wall
                let wall_row = curr_row.min(next_row);
                let wall_col = if curr_col > 0 { curr_col - 1 } else { curr_col };
                (wall_row < board_size - 1 && wall_col < board_size - 1 &&
                    mechanics.repr().get_wall(data, wall_row, wall_col, 1)) ||
                (wall_row < board_size - 1 && curr_col < board_size - 1 &&
                    mechanics.repr().get_wall(data, wall_row, curr_col, 1))
            };

            if wall_blocked {
                continue;
            }

            visited[next_idx] = true;
            queue.push_back((next_row, next_col, dist + 1));
        }
    }

    -1  // Unreachable
}

/// Compute heuristic for QBitRepr state
fn compute_heuristic(
    mechanics: &QGameMechanics,
    data: &[u8],
    agent_player: usize,
    heuristic: i32,
) -> f32 {
    if heuristic == 0 {
        return 0.0;
    }

    let opponent = 1 - agent_player;
    let agent_distance = distance_to_goal(mechanics, data, agent_player);
    let opponent_distance = distance_to_goal(mechanics, data, opponent);

    if agent_distance == -1 || opponent_distance == -1 {
        // If either player can't reach goal, this shouldn't happen
        return 0.0;
    }

    let distance_reward = (opponent_distance - agent_distance) as f32;
    let agent_walls = mechanics.repr().get_walls_remaining(data, agent_player);
    let opponent_walls = mechanics.repr().get_walls_remaining(data, opponent);
    let wall_reward = if agent_player == 0 {
        (agent_walls as i32 - opponent_walls as i32) as f32 / 100.0
    } else {
        (opponent_walls as i32 - agent_walls as i32) as f32 / 100.0
    };

    distance_reward + wall_reward
}

/// Sample actions for QBitRepr-based minimax
fn sample_actions(
    mechanics: &QGameMechanics,
    data: &[u8],
    branching_factor: usize,
) -> Vec<(usize, usize, usize)> {
    let mut rng = rand::thread_rng();

    // Get valid moves (type 0)
    let moves = mechanics.get_valid_moves(data);
    let mut actions: Vec<(usize, usize, usize)> = moves.into_iter()
        .map(|(row, col)| (row, col, 0))
        .collect();

    if actions.len() >= branching_factor {
        actions.shuffle(&mut rng);
        actions.truncate(branching_factor);
        return actions;
    }

    // Get valid wall placements and convert to action format
    // wall orientation 0 -> action_type 1 (vertical wall)
    // wall orientation 1 -> action_type 2 (horizontal wall)
    let mut walls = mechanics.get_valid_wall_placements(data);
    walls.shuffle(&mut rng);

    // Add walls until we reach branching factor
    let num_walls_needed = branching_factor - actions.len();
    for (row, col, orientation) in walls.into_iter().take(num_walls_needed) {
        actions.push((row, col, orientation + 1));  // Map 0->1, 1->2
    }

    actions
}

/// QBitRepr-based minimax recursive function
#[allow(clippy::too_many_arguments)]
fn minimax(
    mechanics: &QGameMechanics,
    data: &[u8],
    current_player: usize,
    agent_player: usize,
    completed_steps: usize,
    max_steps: usize,
    branching_factor: usize,
    discount_factor: f32,
    heuristic: i32,
    log_entries: Option<Arc<Mutex<Vec<MinimaxLogEntry>>>>,
) -> f32 {
    // Check win conditions
    if mechanics.check_win(data, current_player) {
        return if current_player == agent_player {
            WINNING_REWARD
        } else {
            -WINNING_REWARD
        };
    }

    let opponent = 1 - current_player;
    if mechanics.check_win(data, opponent) {
        return if opponent == agent_player {
            WINNING_REWARD
        } else {
            -WINNING_REWARD
        };
    }

    // Check max steps
    if completed_steps >= max_steps {
        return compute_heuristic(mechanics, data, agent_player, heuristic);
    }

    // Sample actions
    let actions = sample_actions(mechanics, data, branching_factor);
    if actions.is_empty() {
        // No valid actions - shouldn't happen, but return heuristic
        return compute_heuristic(mechanics, data, agent_player, heuristic);
    }

    let is_maximizing = current_player == agent_player;
    let mut best_value = if is_maximizing {
        -WINNING_REWARD * 2.0
    } else {
        WINNING_REWARD * 2.0
    };

    let mut action_values = Vec::new();

    for (row, col, action_type) in actions.iter() {
        // Copy state
        let mut new_data = data.to_vec();

        // Apply action
        if *action_type == 0 {
            // Move action
            mechanics.execute_move(&mut new_data, current_player, *row, *col);
        } else {
            // Wall action (type 1 or 2 indicates orientation)
            let orientation = *action_type - 1;
            mechanics.execute_wall_placement(&mut new_data, current_player, *row, *col, orientation);
        }

        // Switch player
        mechanics.switch_player(&mut new_data);

        // Recursive call
        let value = minimax(
            mechanics,
            &new_data,
            opponent,
            agent_player,
            completed_steps + 1,
            max_steps,
            branching_factor,
            discount_factor,
            heuristic,
            log_entries.clone(),
        );

        let discounted_value = value * discount_factor;
        action_values.push((*row, *col, *action_type, discounted_value));

        if is_maximizing {
            best_value = best_value.max(discounted_value);
        } else {
            best_value = best_value.min(discounted_value);
        }
    }

    // Log if requested
    if let Some(ref log) = log_entries {
        let (actions_vec, values_vec): (Vec<(usize, usize, usize)>, Vec<f32>) =
            action_values.into_iter()
                .map(|(r, c, t, v)| ((r, c, t), v))
                .unzip();

        let entry = MinimaxLogEntry {
            data: data.to_vec(),
            agent_player,
            actions: actions_vec,
            values: values_vec,
        };
        log.lock().unwrap().push(entry);
    }

    best_value
}

/// Evaluate actions using QBitRepr-based minimax (parallelized)
pub fn evaluate_actions(
    mechanics: &QGameMechanics,
    data: &[u8],
    max_steps: usize,
    branching_factor: usize,
    discount_factor: f32,
    heuristic: i32,
    enable_logging: bool,
) -> (Vec<(usize, usize, usize)>, Vec<f32>, Option<Vec<MinimaxLogEntry>>) {
    let current_player = mechanics.repr().get_current_player(data);

    // Sample actions
    let actions = sample_actions(mechanics, data, branching_factor);
    if actions.is_empty() {
        return (Vec::new(), Vec::new(), None);
    }

    // Create log collector if needed
    let log_collector = if enable_logging {
        Some(Arc::new(Mutex::new(Vec::new())))
    } else {
        None
    };

    // Evaluate actions in parallel
    let values: Vec<f32> = actions
        .par_iter()
        .map(|(row, col, action_type)| {
            // Copy state
            let mut new_data = data.to_vec();

            // Apply action
            if *action_type == 0 {
                mechanics.execute_move(&mut new_data, current_player, *row, *col);
            } else {
                let orientation = *action_type - 1;
                mechanics.execute_wall_placement(&mut new_data, current_player, *row, *col, orientation);
            }

            // Switch player
            mechanics.switch_player(&mut new_data);

            // Recursive evaluation
            minimax(
                mechanics,
                &new_data,
                1 - current_player,
                current_player,
                1,
                max_steps,
                branching_factor,
                discount_factor,
                heuristic,
                log_collector.clone(),
            )
        })
        .collect();

    // Extract log entries
    let log_entries = log_collector.map(|collector| {
        Arc::try_unwrap(collector)
            .ok()
            .and_then(|mutex| mutex.into_inner().ok())
            .unwrap_or_else(Vec::new)
    });

    (actions, values, log_entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_actions_basic() {
        // Create a small game
        let mechanics = QGameMechanics::new(5, 5, 100);
        let data = mechanics.create_initial_state();

        // Evaluate actions
        let (actions, values, _logs) = evaluate_actions(
            &mechanics,
            &data,
            10,  // max_steps
            5,   // branching_factor
            0.95,  // discount_factor
            1,   // heuristic
            false,  // enable_logging
        );

        // Should have some actions
        assert!(!actions.is_empty(), "Should have at least one action");
        assert_eq!(actions.len(), values.len(), "Actions and values should match");

        // Values should be finite
        for value in &values {
            assert!(value.is_finite(), "Values should be finite");
        }
    }

    #[test]
    fn test_distance_to_goal() {
        let mechanics = QGameMechanics::new(5, 5, 100);
        let data = mechanics.create_initial_state();

        // Player 0 starts at bottom, goal is top (row 0)
        let dist_p0 = distance_to_goal(&mechanics, &data, 0);
        assert!(dist_p0 > 0, "Player 0 should have distance > 0 to goal");

        // Player 1 starts at top, goal is bottom (row 4)
        let dist_p1 = distance_to_goal(&mechanics, &data, 1);
        assert!(dist_p1 > 0, "Player 1 should have distance > 0 to goal");
    }

    #[test]
    fn test_sample_actions() {
        let mechanics = QGameMechanics::new(5, 5, 100);
        let data = mechanics.create_initial_state();

        let actions = sample_actions(&mechanics, &data, 10);

        assert!(!actions.is_empty(), "Should have at least one action");
        assert!(actions.len() <= 10, "Should respect branching factor");

        // Check action types are valid
        for (row, col, action_type) in &actions {
            assert!(*row < 5, "Row should be valid");
            assert!(*col < 5, "Col should be valid");
            assert!(*action_type <= 2, "Action type should be 0, 1, or 2");
        }
    }
}
