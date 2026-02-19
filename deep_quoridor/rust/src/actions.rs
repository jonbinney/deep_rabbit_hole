#![allow(dead_code)]

use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut1};

use crate::validation::{is_move_action_valid, is_wall_action_valid, is_wall_action_valid_mut};

// Action types
pub const ACTION_WALL_VERTICAL: i32 = 0;
pub const ACTION_WALL_HORIZONTAL: i32 = 1;
pub const ACTION_MOVE: i32 = 2;

/// Compute a mask of valid move actions for the current player.
///
/// This is a direct port of compute_move_action_mask from qgrid.py.
pub fn compute_move_action_mask(
    grid: &ArrayView2<i8>,
    player_positions: &ArrayView2<i32>,
    current_player: i32,
    action_mask: &mut ArrayViewMut1<bool>,
) {
    let grid_width = grid.ncols() as i32;
    let board_size = (grid_width - 4) / 2 + 1;

    assert_eq!(action_mask.len(), (board_size * board_size) as usize);

    let player_row = player_positions[[current_player as usize, 0]];
    let player_col = player_positions[[current_player as usize, 1]];

    // Only check moves near the current player position
    let row_start = (player_row - 2).max(0);
    let row_end = (player_row + 3).min(board_size);
    let col_start = (player_col - 2).max(0);
    let col_end = (player_col + 3).min(board_size);

    for destination_row in row_start..row_end {
        for destination_col in col_start..col_end {
            if is_move_action_valid(
                grid,
                player_positions,
                current_player,
                destination_row,
                destination_col,
            ) {
                let index = (destination_row * board_size + destination_col) as usize;
                action_mask[index] = true;
            }
        }
    }
}

/// Compute a mask of valid wall actions for the current player.
///
/// This is a direct port of compute_wall_action_mask from qgrid.py.
pub fn compute_wall_action_mask(
    grid: &ArrayView2<i8>,
    player_positions: &ArrayView2<i32>,
    walls_remaining: &ArrayView1<i32>,
    goal_rows: &ArrayView1<i32>,
    current_player: i32,
    action_mask: &mut ArrayViewMut1<bool>,
) {
    let grid_width = grid.ncols() as i32;
    let board_size = (grid_width - 4) / 2 + 1;
    let wall_size = board_size - 1;

    assert_eq!(action_mask.len(), (2 * wall_size * wall_size) as usize);

    if walls_remaining[current_player as usize] <= 0 {
        return;
    }

    for wall_row in 0..wall_size {
        for wall_col in 0..wall_size {
            // Check vertical wall
            if is_wall_action_valid(
                grid,
                player_positions,
                walls_remaining,
                goal_rows,
                current_player,
                wall_row,
                wall_col,
                0, // VERTICAL
            ) {
                let index = (wall_row * wall_size + wall_col) as usize;
                action_mask[index] = true;
            }

            // Check horizontal wall
            if is_wall_action_valid(
                grid,
                player_positions,
                walls_remaining,
                goal_rows,
                current_player,
                wall_row,
                wall_col,
                1, // HORIZONTAL
            ) {
                let index = (wall_size * wall_size + wall_row * wall_size + wall_col) as usize;
                action_mask[index] = true;
            }
        }
    }
}

/// Get all valid move actions for the current player.
///
/// This is a direct port of get_valid_move_actions from qgrid.py.
/// Returns an array where each row is [row, col, ACTION_MOVE].
pub fn get_valid_move_actions(
    grid: &ArrayView2<i8>,
    player_positions: &ArrayView2<i32>,
    current_player: i32,
) -> Array2<i32> {
    let board_size = (grid.ncols() as i32 - 4) / 2 + 1;
    let mut actions = Vec::new();

    // Check all possible moves on the board
    for dest_row in 0..board_size {
        for dest_col in 0..board_size {
            if is_move_action_valid(grid, player_positions, current_player, dest_row, dest_col) {
                actions.push([dest_row, dest_col, ACTION_MOVE]);
            }
        }
    }

    // Convert to ndarray
    let count = actions.len();
    if count == 0 {
        return Array2::zeros((0, 3));
    }

    let mut result = Array2::zeros((count, 3));
    for (i, action) in actions.iter().enumerate() {
        result[[i, 0]] = action[0];
        result[[i, 1]] = action[1];
        result[[i, 2]] = action[2];
    }

    result
}

/// Get all valid wall actions for the current player.
///
/// This is a direct port of get_valid_wall_actions from qgrid.py.
/// Returns an array where each row is [row, col, action_type] with action_type 0 or 1.
pub fn get_valid_wall_actions(
    grid: &ArrayView2<i8>,
    player_positions: &ArrayView2<i32>,
    walls_remaining: &ArrayView1<i32>,
    goal_rows: &ArrayView1<i32>,
    current_player: i32,
) -> Array2<i32> {
    let board_size = (grid.ncols() as i32 - 4) / 2 + 1;
    let wall_size = board_size - 1;
    let mut actions = Vec::new();

    // Make a mutable copy of the grid for in-place validation
    let mut grid_copy = grid.to_owned();

    // Check all possible wall placements
    for wall_row in 0..wall_size {
        for wall_col in 0..wall_size {
            // Check vertical walls
            if is_wall_action_valid_mut(
                &mut grid_copy.view_mut(),
                player_positions,
                walls_remaining,
                goal_rows,
                current_player,
                wall_row,
                wall_col,
                0, // VERTICAL
            ) {
                actions.push([wall_row, wall_col, ACTION_WALL_VERTICAL]);
            }

            // Check horizontal walls
            if is_wall_action_valid_mut(
                &mut grid_copy.view_mut(),
                player_positions,
                walls_remaining,
                goal_rows,
                current_player,
                wall_row,
                wall_col,
                1, // HORIZONTAL
            ) {
                actions.push([wall_row, wall_col, ACTION_WALL_HORIZONTAL]);
            }
        }
    }

    // Convert to ndarray
    let count = actions.len();
    if count == 0 {
        return Array2::zeros((0, 3));
    }

    let mut result = Array2::zeros((count, 3));
    for (i, action) in actions.iter().enumerate() {
        result[[i, 0]] = action[0];
        result[[i, 1]] = action[1];
        result[[i, 2]] = action[2];
    }

    result
}

/// Compute the total number of actions in the policy vector for a given board size.
///
/// Layout: [moves | horizontal_walls | vertical_walls]
/// = board_size² + 2 * (board_size - 1)²
pub fn policy_size(board_size: i32) -> usize {
    let wall_size = board_size - 1;
    (board_size * board_size + 2 * wall_size * wall_size) as usize
}

/// Compute a combined action mask covering moves and walls.
///
/// Layout: [moves (board_size²) | horizontal_walls ((bs-1)²) | vertical_walls ((bs-1)²)]
/// This matches the Python ActionEncoder layout.
pub fn compute_full_action_mask(
    grid: &ArrayView2<i8>,
    player_positions: &ArrayView2<i32>,
    walls_remaining: &ArrayView1<i32>,
    goal_rows: &ArrayView1<i32>,
    current_player: i32,
    mask: &mut [bool],
) {
    let grid_width = grid.ncols() as i32;
    let board_size = (grid_width - 4) / 2 + 1;
    let wall_size = board_size - 1;
    let num_moves = (board_size * board_size) as usize;
    let num_walls = (wall_size * wall_size) as usize;

    assert_eq!(mask.len(), num_moves + 2 * num_walls);

    // Zero out the mask
    for m in mask.iter_mut() {
        *m = false;
    }

    // Fill move mask portion
    let mut move_mask = ndarray::Array1::from(vec![false; num_moves]);
    compute_move_action_mask(
        grid,
        player_positions,
        current_player,
        &mut move_mask.view_mut(),
    );
    mask[..num_moves].copy_from_slice(move_mask.as_slice().unwrap());

    // Fill wall mask portion
    let mut wall_mask = ndarray::Array1::from(vec![false; 2 * num_walls]);
    compute_wall_action_mask(
        grid,
        player_positions,
        walls_remaining,
        goal_rows,
        current_player,
        &mut wall_mask.view_mut(),
    );
    // wall_mask layout from compute_wall_action_mask: [vertical (ws²) | horizontal (ws²)]
    // But selfplay.rs maps horizontal walls first, then vertical.
    // Looking at the existing evaluate_action in selfplay.rs:
    //   horizontal: num_move_actions + row*wall_size + col
    //   vertical: num_move_actions + num_wall_actions + row*wall_size + col
    // And compute_wall_action_mask: index for vertical = row*ws+col, horizontal = ws²+row*ws+col
    // So wall_mask[0..ws²] = vertical, wall_mask[ws²..2*ws²] = horizontal
    // We want output: [moves | horizontal | vertical]
    // Actually, let me re-read the selfplay.rs mapping more carefully:
    //   horizontal wall: num_move_actions + row*wall_size + col  (offset = num_moves)
    //   vertical wall: num_move_actions + num_wall_actions + row*wall_size + col (offset = num_moves + num_walls)
    // So output layout = [moves | horizontal_walls | vertical_walls]
    // compute_wall_action_mask layout: [vertical_walls | horizontal_walls]
    // Copy horizontal walls (second half of wall_mask) to first wall section of output
    mask[num_moves..num_moves + num_walls]
        .copy_from_slice(&wall_mask.as_slice().unwrap()[num_walls..2 * num_walls]);
    // Copy vertical walls (first half of wall_mask) to second wall section of output
    mask[num_moves + num_walls..num_moves + 2 * num_walls]
        .copy_from_slice(&wall_mask.as_slice().unwrap()[..num_walls]);
}

/// Decode a flat policy index to an action triple [row, col, action_type].
///
/// Policy layout: [moves (bs²) | horizontal_walls ((bs-1)²) | vertical_walls ((bs-1)²)]
pub fn action_index_to_action(board_size: i32, index: usize) -> [i32; 3] {
    let num_moves = (board_size * board_size) as usize;
    let wall_size = board_size - 1;
    let num_walls = (wall_size * wall_size) as usize;

    if index < num_moves {
        // Move action
        let row = (index as i32) / board_size;
        let col = (index as i32) % board_size;
        [row, col, ACTION_MOVE]
    } else if index < num_moves + num_walls {
        // Horizontal wall
        let wall_idx = index - num_moves;
        let row = (wall_idx as i32) / wall_size;
        let col = (wall_idx as i32) % wall_size;
        [row, col, ACTION_WALL_HORIZONTAL]
    } else {
        // Vertical wall
        let wall_idx = index - num_moves - num_walls;
        let row = (wall_idx as i32) / wall_size;
        let col = (wall_idx as i32) % wall_size;
        [row, col, ACTION_WALL_VERTICAL]
    }
}

/// Encode an action triple [row, col, action_type] to a flat policy index.
///
/// Inverse of `action_index_to_action`.
pub fn action_to_index(board_size: i32, action: &[i32; 3]) -> usize {
    let num_moves = (board_size * board_size) as usize;
    let wall_size = board_size - 1;
    let num_walls = (wall_size * wall_size) as usize;

    match action[2] {
        ACTION_MOVE => (action[0] * board_size + action[1]) as usize,
        ACTION_WALL_HORIZONTAL => num_moves + (action[0] * wall_size + action[1]) as usize,
        ACTION_WALL_VERTICAL => {
            num_moves + num_walls + (action[0] * wall_size + action[1]) as usize
        }
        _ => panic!("Invalid action type: {}", action[2]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{CELL_FREE, CELL_WALL};
    use ndarray::{Array1, Array2};

    fn create_test_game() -> (Array2<i8>, Array2<i32>, Array1<i32>, Array1<i32>) {
        let mut grid = Array2::<i8>::from_elem((20, 20), CELL_FREE);

        // Add border walls
        for i in 0..2 {
            for j in 0..20 {
                grid[[i, j]] = CELL_WALL;
                grid[[19 - i, j]] = CELL_WALL;
                grid[[j, i]] = CELL_WALL;
                grid[[j, 19 - i]] = CELL_WALL;
            }
        }

        let mut player_positions = Array2::<i32>::zeros((2, 2));
        player_positions[[0, 0]] = 0;
        player_positions[[0, 1]] = 4;
        player_positions[[1, 0]] = 8;
        player_positions[[1, 1]] = 4;

        grid[[2, 10]] = 0; // Player 0 at (0, 4)
        grid[[18, 10]] = 1; // Player 1 at (8, 4)

        let walls_remaining = Array1::from(vec![10, 10]);
        let goal_rows = Array1::from(vec![8, 0]);

        (grid, player_positions, walls_remaining, goal_rows)
    }

    #[test]
    fn test_compute_move_action_mask() {
        let (grid, player_positions, _, _) = create_test_game();
        let board_size = 9;
        let mut action_mask = Array1::from(vec![false; (board_size * board_size) as usize]);

        compute_move_action_mask(
            &grid.view(),
            &player_positions.view(),
            0,
            &mut action_mask.view_mut(),
        );

        // Player should have some valid moves
        assert!(action_mask.iter().any(|&x| x));
    }

    #[test]
    fn test_get_valid_move_actions() {
        let (grid, player_positions, _, _) = create_test_game();

        let actions = get_valid_move_actions(&grid.view(), &player_positions.view(), 0);

        // Should have at least one valid move
        assert!(actions.nrows() > 0);
        // All actions should be move actions
        for i in 0..actions.nrows() {
            assert_eq!(actions[[i, 2]], ACTION_MOVE);
        }
    }

    #[test]
    fn test_policy_size() {
        assert_eq!(policy_size(5), 25 + 2 * 16); // 57
        assert_eq!(policy_size(9), 81 + 2 * 64); // 209
    }

    #[test]
    fn test_action_index_roundtrip() {
        for board_size in [5, 9] {
            let total = policy_size(board_size);
            for idx in 0..total {
                let action = action_index_to_action(board_size, idx);
                let recovered = action_to_index(board_size, &action);
                assert_eq!(idx, recovered, "board_size={board_size}, idx={idx}, action={action:?}");
            }
        }
    }

    #[test]
    fn test_action_index_to_action_ranges() {
        let bs = 5;
        // First bs² indices are moves
        for idx in 0..(bs * bs) as usize {
            let action = action_index_to_action(bs, idx);
            assert_eq!(action[2], ACTION_MOVE);
            assert!(action[0] >= 0 && action[0] < bs);
            assert!(action[1] >= 0 && action[1] < bs);
        }
        let num_moves = (bs * bs) as usize;
        let ws = (bs - 1) as usize;
        // Next ws² are horizontal walls
        for idx in num_moves..num_moves + ws * ws {
            let action = action_index_to_action(bs, idx);
            assert_eq!(action[2], ACTION_WALL_HORIZONTAL);
        }
        // Last ws² are vertical walls
        for idx in num_moves + ws * ws..num_moves + 2 * ws * ws {
            let action = action_index_to_action(bs, idx);
            assert_eq!(action[2], ACTION_WALL_VERTICAL);
        }
    }

    #[test]
    fn test_compute_full_action_mask_length() {
        let (grid, player_positions, walls_remaining, goal_rows) = create_test_game();
        let board_size = 9;
        let total = policy_size(board_size);
        let mut mask = vec![false; total];

        compute_full_action_mask(
            &grid.view(),
            &player_positions.view(),
            &walls_remaining.view(),
            &goal_rows.view(),
            0,
            &mut mask,
        );

        // Should have some valid actions
        assert!(mask.iter().any(|&x| x));
    }

    #[test]
    fn test_full_mask_matches_individual_masks() {
        use crate::game_state::create_initial_state;

        let (grid, player_positions, walls_remaining, goal_rows) = create_initial_state(5, 3);
        let board_size = 5;
        let total = policy_size(board_size);
        let mut full_mask = vec![false; total];

        compute_full_action_mask(
            &grid.view(),
            &player_positions.view(),
            &walls_remaining.view(),
            &goal_rows.view(),
            0,
            &mut full_mask,
        );

        // Get individual valid actions
        let move_actions = get_valid_move_actions(&grid.view(), &player_positions.view(), 0);
        let wall_actions = get_valid_wall_actions(
            &grid.view(),
            &player_positions.view(),
            &walls_remaining.view(),
            &goal_rows.view(),
            0,
        );

        // Count valid actions from full mask
        let mask_count: usize = full_mask.iter().filter(|&&x| x).count();
        let expected_count = move_actions.nrows() + wall_actions.nrows();
        assert_eq!(mask_count, expected_count, "mask has {mask_count} true entries, expected {expected_count}");

        // Verify each move action is in the mask
        for i in 0..move_actions.nrows() {
            let action = [move_actions[[i, 0]], move_actions[[i, 1]], move_actions[[i, 2]]];
            let idx = action_to_index(board_size, &action);
            assert!(full_mask[idx], "Move action {action:?} at index {idx} not in mask");
        }

        // Verify each wall action is in the mask
        for i in 0..wall_actions.nrows() {
            let action = [wall_actions[[i, 0]], wall_actions[[i, 1]], wall_actions[[i, 2]]];
            let idx = action_to_index(board_size, &action);
            assert!(full_mask[idx], "Wall action {action:?} at index {idx} not in mask");
        }
    }
}
