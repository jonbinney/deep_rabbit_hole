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
}
