use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;

use crate::actions::get_valid_move_actions;
use crate::actions::get_valid_wall_actions;
use crate::game_state::{apply_action, check_win, undo_action};
use crate::pathfinding::distance_to_row;

pub const WINNING_REWARD: f32 = 1e6;

/// Calculate Gaussian weights for wall actions based on distance to players.
fn gaussian_wall_weights(
    wall_actions: &ArrayView2<i32>,
    p1_pos: &ArrayView1<i32>,
    p2_pos: &ArrayView1<i32>,
    sigma: f32,
) -> Array1<f32> {
    let num_actions = wall_actions.nrows();
    let mut weights = Array1::<f32>::zeros(num_actions);

    for i in 0..num_actions {
        let wall_row = wall_actions[[i, 0]] as f32;
        let wall_col = wall_actions[[i, 1]] as f32;
        let wall_pos_x = wall_row + 0.5;
        let wall_pos_y = wall_col + 0.5;

        // Calculate distance to both players
        let d1 = (wall_pos_x - p1_pos[0] as f32).powi(2) + (wall_pos_y - p1_pos[1] as f32).powi(2);
        let d2 = (wall_pos_x - p2_pos[0] as f32).powi(2) + (wall_pos_y - p2_pos[1] as f32).powi(2);
        let min_dist = d1.min(d2);

        // Calculate Gaussian weight
        weights[i] = (-0.5 * min_dist / (sigma * sigma)).exp();
    }

    // Normalize weights
    let weights_sum: f32 = weights.sum();
    if weights_sum > 0.0 {
        weights = weights / weights_sum;
    } else {
        weights.fill(1.0 / num_actions as f32);
    }

    weights
}

/// Sample actions for the minimax search.
/// Returns an array of actions where each row is [row, col, action_type].
pub fn sample_actions(
    grid: &ArrayView2<i8>,
    player_positions: &ArrayView2<i32>,
    walls_remaining: &ArrayView1<i32>,
    goal_rows: &ArrayView1<i32>,
    current_player: i32,
    branching_factor: usize,
    wall_sigma: f32,
) -> Array2<i32> {
    let mut rng = rand::thread_rng();

    // Get all valid move actions
    let move_actions = get_valid_move_actions(grid, player_positions, current_player);

    // Check whether we already have enough move actions
    if move_actions.nrows() >= branching_factor {
        // Randomly select branching_factor moves
        let mut indices: Vec<usize> = (0..move_actions.nrows()).collect();
        indices.shuffle(&mut rng);
        let selected_indices = &indices[..branching_factor];

        let mut result = Array2::zeros((branching_factor, 3));
        for (i, &idx) in selected_indices.iter().enumerate() {
            result[[i, 0]] = move_actions[[idx, 0]];
            result[[i, 1]] = move_actions[[idx, 1]];
            result[[i, 2]] = move_actions[[idx, 2]];
        }
        return result;
    }

    // Calculate how many wall actions we need
    let num_wall_actions_needed = branching_factor - move_actions.nrows();

    // Get all valid wall actions
    let wall_actions = get_valid_wall_actions(grid, player_positions, walls_remaining, goal_rows, current_player);

    // Combine the actions
    if wall_actions.nrows() <= num_wall_actions_needed {
        // Use all wall actions
        let total_actions = move_actions.nrows() + wall_actions.nrows();
        let mut combined_actions = Array2::zeros((total_actions, 3));
        for i in 0..move_actions.nrows() {
            combined_actions[[i, 0]] = move_actions[[i, 0]];
            combined_actions[[i, 1]] = move_actions[[i, 1]];
            combined_actions[[i, 2]] = move_actions[[i, 2]];
        }
        for i in 0..wall_actions.nrows() {
            let idx = move_actions.nrows() + i;
            combined_actions[[idx, 0]] = wall_actions[[i, 0]];
            combined_actions[[idx, 1]] = wall_actions[[i, 1]];
            combined_actions[[idx, 2]] = wall_actions[[i, 2]];
        }
        return combined_actions;
    }

    // Sample wall actions based on distance to players
    let selected_wall_indices = if wall_sigma > 0.0 {
        let weights = gaussian_wall_weights(&wall_actions.view(), &player_positions.row(0), &player_positions.row(1), wall_sigma);

        // Sample wall actions using weights (cumulative distribution)
        let mut indices = Vec::with_capacity(num_wall_actions_needed);
        let cumulative: Vec<f32> = weights
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        for _ in 0..num_wall_actions_needed {
            let r: f32 = rng.gen();
            for (j, &cum_weight) in cumulative.iter().enumerate() {
                if r <= cum_weight {
                    indices.push(j);
                    break;
                }
            }
        }
        indices
    } else {
        // Sample wall actions uniformly
        let mut indices: Vec<usize> = (0..wall_actions.nrows()).collect();
        indices.shuffle(&mut rng);
        indices.into_iter().take(num_wall_actions_needed).collect()
    };

    // Combine move actions with sampled wall actions
    let total_actions = move_actions.nrows() + selected_wall_indices.len();
    let mut combined_actions = Array2::zeros((total_actions, 3));
    for i in 0..move_actions.nrows() {
        combined_actions[[i, 0]] = move_actions[[i, 0]];
        combined_actions[[i, 1]] = move_actions[[i, 1]];
        combined_actions[[i, 2]] = move_actions[[i, 2]];
    }
    for (i, &idx) in selected_wall_indices.iter().enumerate() {
        let combined_idx = move_actions.nrows() + i;
        combined_actions[[combined_idx, 0]] = wall_actions[[idx, 0]];
        combined_actions[[combined_idx, 1]] = wall_actions[[idx, 1]];
        combined_actions[[combined_idx, 2]] = wall_actions[[idx, 2]];
    }

    combined_actions
}

/// Evaluate a board position using a heuristic.
fn compute_heuristic_for_game_state(
    grid: &ArrayView2<i8>,
    player_positions: &ArrayView2<i32>,
    walls_remaining: &ArrayView1<i32>,
    goal_rows: &ArrayView1<i32>,
    agent_player: i32,
    heuristic: i32,
) -> f32 {
    if heuristic == 0 {
        return 0.0;
    }

    // Get distances to goals
    let opponent = 1 - agent_player;
    let agent_distance = distance_to_row(
        grid,
        player_positions[[agent_player as usize, 0]],
        player_positions[[agent_player as usize, 1]],
        goal_rows[agent_player as usize],
    );
    let opponent_distance = distance_to_row(
        grid,
        player_positions[[opponent as usize, 0]],
        player_positions[[opponent as usize, 1]],
        goal_rows[opponent as usize],
    );

    assert!(agent_distance != -1 && opponent_distance != -1);

    // Compute heuristic value based on distances and walls
    let distance_reward = (opponent_distance - agent_distance) as f32;
    let wall_reward = (walls_remaining[agent_player as usize] - walls_remaining[opponent as usize]) as f32 / 100.0;

    distance_reward + wall_reward
}

/// Minimax algorithm for evaluating a single action.
#[allow(clippy::too_many_arguments)]
fn minimax(
    action: &ArrayView1<i32>,
    grid: &mut Array2<i8>,
    player_positions: &mut Array2<i32>,
    walls_remaining: &mut Array1<i32>,
    goal_rows: &ArrayView1<i32>,
    current_player: i32,
    agent_player: i32,
    depth: i32,
    branching_factor: usize,
    wall_sigma: f32,
    discount_factor: f32,
    heuristic: i32,
) -> f32 {
    let opponent = 1 - current_player;
    let opponent_old_position = Array1::from(vec![
        player_positions[[opponent as usize, 0]],
        player_positions[[opponent as usize, 1]],
    ]);

    // Apply the action the opponent just took
    apply_action(
        &mut grid.view_mut(),
        &mut player_positions.view_mut(),
        &mut walls_remaining.view_mut(),
        opponent,
        action,
    );

    let best_value: f32;

    // Did we win?
    if check_win(&player_positions.view(), goal_rows, current_player) {
        best_value = if current_player == agent_player {
            WINNING_REWARD
        } else {
            -WINNING_REWARD
        };
    }
    // Did the opponent win?
    else if check_win(&player_positions.view(), goal_rows, opponent) {
        best_value = if current_player == agent_player {
            WINNING_REWARD
        } else {
            -WINNING_REWARD
        };
    }
    // Have we reached the maximum depth?
    else if depth == 0 {
        best_value = compute_heuristic_for_game_state(
            &grid.view(),
            &player_positions.view(),
            &walls_remaining.view(),
            goal_rows,
            agent_player,
            heuristic,
        );
    }
    // Try actions from this state
    else {
        // Sample actions to evaluate
        let next_actions = sample_actions(
            &grid.view(),
            &player_positions.view(),
            &walls_remaining.view(),
            goal_rows,
            current_player,
            branching_factor,
            wall_sigma,
        );
        assert!(next_actions.nrows() > 0, "No valid actions found");

        // Determine if maximizing or minimizing
        let is_maximizing = current_player == agent_player;
        let mut value = if is_maximizing {
            -WINNING_REWARD * 2.0
        } else {
            WINNING_REWARD * 2.0
        };

        // Evaluate all actions
        for i in 0..next_actions.nrows() {
            let next_action = next_actions.row(i);

            // Recursively evaluate position
            let eval = minimax(
                &next_action,
                grid,
                player_positions,
                walls_remaining,
                goal_rows,
                1 - current_player,
                agent_player,
                depth - 1,
                branching_factor,
                wall_sigma,
                discount_factor,
                heuristic,
            );

            // Apply discount factor
            let discounted_eval = eval * discount_factor;

            // Update best value
            if is_maximizing {
                value = value.max(discounted_eval);
            } else {
                value = value.min(discounted_eval);
            }
        }

        best_value = value;
    }

    // Undo action
    undo_action(
        &mut grid.view_mut(),
        &mut player_positions.view_mut(),
        &mut walls_remaining.view_mut(),
        opponent,
        action,
        &opponent_old_position.view(),
    );

    best_value
}

/// Evaluate all actions for the current player using the minimax algorithm (parallelized).
#[allow(clippy::too_many_arguments)]
pub fn evaluate_actions(
    grid: &ArrayView2<i8>,
    player_positions: &ArrayView2<i32>,
    walls_remaining: &ArrayView1<i32>,
    goal_rows: &ArrayView1<i32>,
    current_player: i32,
    max_depth: i32,
    branching_factor: usize,
    wall_sigma: f32,
    discount_factor: f32,
    heuristic: i32,
) -> (Array2<i32>, Array1<f32>) {
    // Sample actions to evaluate
    let actions = sample_actions(
        grid,
        player_positions,
        walls_remaining,
        goal_rows,
        current_player,
        branching_factor,
        wall_sigma,
    );
    assert!(actions.nrows() > 0, "No valid actions found");

    // Evaluate all actions in parallel
    let values: Vec<f32> = (0..actions.nrows())
        .into_par_iter()
        .map(|i| {
            // Since we run this loop in parallel, we need to copy the game state arrays for each minimax call
            let mut grid_copy = grid.to_owned();
            let mut player_positions_copy = player_positions.to_owned();
            let mut walls_remaining_copy = walls_remaining.to_owned();

            let action = actions.row(i);
            minimax(
                &action,
                &mut grid_copy,
                &mut player_positions_copy,
                &mut walls_remaining_copy,
                goal_rows,
                1 - current_player,
                current_player, // Assume we are choosing an action for the current player
                max_depth - 1,
                branching_factor,
                wall_sigma,
                discount_factor,
                heuristic,
            )
        })
        .collect();

    let values_array = Array1::from(values);
    (actions, values_array)
}
