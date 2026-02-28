/// QBitRepr-based minimax algorithm for Quoridor
use dashmap::DashMap;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::sync::Arc;

use super::q_game_mechanics::QGameMechanics;

pub const WINNING_REWARD: f32 = 1e6;

/// Transposition table entry for minimax. The state bytes are stored as the
/// DashMap key rather than in this struct to avoid duplication.
#[derive(Clone)]
#[allow(dead_code)]
pub struct TranspositionEntry {
    pub agent_player: usize,
    pub actions: Vec<(usize, usize, usize)>,
    pub values: Vec<f32>,
}

/// Compute distance to goal for a player using QBitRepr state
fn distance_to_goal(mechanics: &QGameMechanics, data: &[u8], player: usize) -> i32 {
    let (row, col) = mechanics.repr().get_player_position(data, player);
    let board_size = mechanics.repr().board_size();
    let goal_row = mechanics.get_goal_row(player);

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
                (wall_row < board_size - 1
                    && wall_col < board_size - 1
                    && mechanics.repr().get_wall(data, wall_row, wall_col, 0))
                    || (curr_row < board_size - 1
                        && wall_col < board_size - 1
                        && mechanics.repr().get_wall(data, curr_row, wall_col, 0))
            } else {
                // Moving vertically - check for horizontal wall
                let wall_row = curr_row.min(next_row);
                let wall_col = if curr_col > 0 { curr_col - 1 } else { curr_col };
                (wall_row < board_size - 1
                    && wall_col < board_size - 1
                    && mechanics.repr().get_wall(data, wall_row, wall_col, 1))
                    || (wall_row < board_size - 1
                        && curr_col < board_size - 1
                        && mechanics.repr().get_wall(data, wall_row, curr_col, 1))
            };

            if wall_blocked {
                continue;
            }

            visited[next_idx] = true;
            queue.push_back((next_row, next_col, dist + 1));
        }
    }

    -1 // Unreachable
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
    let mut actions: Vec<(usize, usize, usize)> =
        moves.into_iter().map(|(row, col)| (row, col, 2)).collect();

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
        actions.push((row, col, orientation));
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
    search_depth: usize,
    max_search_depth: usize,
    branching_factor: usize,
    discount_factor: f32,
    heuristic: i32,
    transposition_table: Arc<DashMap<Vec<u8>, TranspositionEntry>>,
) -> f32 {
    let opponent = 1 - current_player;

    // We're checking for the player that just finished their move.
    if mechanics.check_win(data, opponent) {
        return if opponent == agent_player {
            WINNING_REWARD
        } else {
            -WINNING_REWARD
        };
    }

    if mechanics.repr().get_completed_steps(data) >= mechanics.repr().max_steps() {
        return 0.0; // Tie
    }

    if search_depth >= max_search_depth {
        return compute_heuristic(mechanics, data, agent_player, heuristic);
    }

    let actions = sample_actions(mechanics, data, branching_factor);
    if actions.is_empty() {
        mechanics.print(data);
        assert!(false, "No valid actions - should never happen");
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
        if *action_type == 2 {
            mechanics.execute_move(&mut new_data, current_player, *row, *col);
        } else {
            let orientation = *action_type;
            mechanics.execute_wall_placement(
                &mut new_data,
                current_player,
                *row,
                *col,
                orientation,
            );
        }

        // Switch player
        mechanics.switch_player(&mut new_data);

        // Recursive call
        let value = minimax(
            mechanics,
            &new_data,
            opponent,
            agent_player,
            search_depth + 1,
            max_search_depth,
            branching_factor,
            discount_factor,
            heuristic,
            transposition_table.clone(),
        );

        let discounted_value = value * discount_factor;
        action_values.push((*row, *col, *action_type, discounted_value));

        if is_maximizing {
            best_value = best_value.max(discounted_value);
            if best_value == WINNING_REWARD {
                break;
            }
        } else {
            best_value = best_value.min(discounted_value);
            if best_value == -WINNING_REWARD {
                break;
            }
        }
    }

    // Store result in transposition table
    let (actions_vec, values_vec): (Vec<(usize, usize, usize)>, Vec<f32>) = action_values
        .into_iter()
        .map(|(r, c, t, v)| ((r, c, t), v))
        .unzip();

    transposition_table.insert(
        data.to_vec(),
        TranspositionEntry {
            agent_player,
            actions: actions_vec,
            values: values_vec,
        },
    );

    best_value
}

/// Evaluate actions using QBitRepr-based minimax (parallelized)
pub fn evaluate_actions(
    mechanics: &QGameMechanics,
    data: &[u8],
    max_search_depth: usize,
    branching_factor: usize,
    discount_factor: f32,
    heuristic: i32,
) -> (
    Vec<(usize, usize, usize)>,
    Vec<f32>,
    DashMap<Vec<u8>, TranspositionEntry>,
) {
    let current_player = mechanics.repr().get_current_player(data);
    let agent_player = current_player;

    // Sample actions
    let actions = sample_actions(mechanics, data, branching_factor);
    if actions.is_empty() {
        return (Vec::new(), Vec::new(), DashMap::new());
    }

    let transposition_table = Arc::new(DashMap::new());

    // Evaluate actions in parallel
    let values: Vec<f32> = actions
        .par_iter()
        .map(|(row, col, action_type)| {
            // Copy state
            let mut new_data = data.to_vec();

            // Apply action
            if *action_type == 2 {
                mechanics.execute_move(&mut new_data, current_player, *row, *col);
            } else {
                let orientation = *action_type;
                mechanics.execute_wall_placement(
                    &mut new_data,
                    current_player,
                    *row,
                    *col,
                    orientation,
                );
            }

            // Switch player
            mechanics.switch_player(&mut new_data);

            // Recursive evaluation
            minimax(
                mechanics,
                &new_data,
                1 - current_player,
                agent_player,
                1, // search_depth
                max_search_depth,
                branching_factor,
                discount_factor,
                heuristic,
                transposition_table.clone(),
            )
        })
        .collect();

    // Extract transposition table
    let result_table = Arc::try_unwrap(transposition_table)
        .ok()
        .expect("transposition_table Arc should have no other references");

    result_table.insert(
        data.to_vec(),
        TranspositionEntry {
            agent_player,
            actions: actions.clone(),
            values: values.clone(),
        },
    );

    (actions, values, result_table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_actions_basic() {
        // Create a small game
        let mechanics = QGameMechanics::new(3, 3, 10);
        let data = mechanics.create_initial_state();

        // Evaluate actions
        let (actions, values, _logs) = evaluate_actions(
            &mechanics, &data, 6,    // max_depth
            5,    // branching_factor
            0.95, // discount_factor
            1,    // heuristic
        );

        // Should have some actions
        assert!(!actions.is_empty(), "Should have at least one action");
        assert_eq!(
            actions.len(),
            values.len(),
            "Actions and values should match"
        );

        // Values should be finite
        for value in &values {
            assert!(value.is_finite(), "Values should be finite");
        }
    }

    #[test]
    fn test_evaluate_actions_win_in_one() {
        let mechanics = QGameMechanics::new(3, 0, 4);
        let mut data = mechanics.create_initial_state();

        // P1 playes something that doesn't instantly lose
        mechanics.execute_move(&mut data, 0, 0, 2);
        mechanics.switch_player(&mut data);
        mechanics.execute_move(&mut data, 1, 1, 1);
        mechanics.switch_player(&mut data);
        mechanics.execute_move(&mut data, 0, 1, 2);
        mechanics.switch_player(&mut data);

        // P2 can win in 1 more moves
        let (_, values, _) = evaluate_actions(&mechanics, &data, 1, 999, 1.0, 0);
        assert!(
            values.contains(&WINNING_REWARD),
            "Minimax failed to find viable win in one move"
        );
    }

    /// P2 should be able to win on a 3x3 board by the 4th move of the
    /// game, but not before that.
    #[test]
    fn test_evaluate_actions_search_depth() {
        let mechanics = QGameMechanics::new(3, 0, 4);
        let mut data = mechanics.create_initial_state();

        // P1 playes something that doesn't instantly lose
        mechanics.execute_move(&mut data, 0, 0, 2);
        mechanics.switch_player(&mut data);

        // P2 can win in 3 more moves
        let (_, values, _) = evaluate_actions(&mechanics, &data, 3, 999, 1.0, 0);
        assert!(
            values.contains(&WINNING_REWARD),
            "Minimax failed to find viable win"
        );

        // P2 cannot win in 2 more moves.
        let (_, values, _) = evaluate_actions(&mechanics, &data, 2, 999, 1.0, 0);
        assert!(
            !values.contains(&WINNING_REWARD),
            "Minimax foud a win where there shouldn't be one",
        );
    }

    #[test]
    fn test_distance_to_goal() {
        let mechanics = QGameMechanics::new(5, 5, 100);
        let data = mechanics.create_initial_state();
        mechanics.print(&data);

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
