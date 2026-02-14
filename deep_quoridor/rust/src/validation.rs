#![allow(dead_code)]

use ndarray::{ArrayView1, ArrayView2, ArrayViewMut2};

use crate::grid::{
    check_wall_cells, is_wall_potential_block, set_wall_cells, CELL_FREE, CELL_WALL,
};
use crate::pathfinding::distance_to_row;

/// Validate whether a move action is legal.
///
/// This is a direct port of is_move_action_valid from qgrid.py.
pub fn is_move_action_valid(
    grid: &ArrayView2<i8>,
    player_positions: &ArrayView2<i32>,
    current_player: i32,
    destination_row: i32,
    destination_col: i32,
) -> bool {
    let grid_height = grid.nrows() as i32;
    let grid_width = grid.ncols() as i32;

    let player_i = player_positions[[current_player as usize, 0]] * 2 + 2;
    let player_j = player_positions[[current_player as usize, 1]] * 2 + 2;
    let destination_i = destination_row * 2 + 2;
    let destination_j = destination_col * 2 + 2;
    let opponent = 1 - current_player;

    // Bounds checking
    if player_i < 0 || player_i >= grid_height || player_j < 0 || player_j >= grid_width {
        return false;
    }
    if destination_i < 0
        || destination_i >= grid_height
        || destination_j < 0
        || destination_j >= grid_width
    {
        return false;
    }

    if grid[[destination_i as usize, destination_j as usize]] != CELL_FREE {
        return false;
    }

    let delta_i = destination_i - player_i;
    let delta_j = destination_j - player_j;

    // Single step moves
    if delta_i == 2 && delta_j == 0 {
        // Down
        return grid[[player_i as usize + 1, player_j as usize]] == CELL_FREE;
    } else if delta_i == -2 && delta_j == 0 {
        // Up
        return grid[[player_i as usize - 1, player_j as usize]] == CELL_FREE;
    } else if delta_i == 0 && delta_j == 2 {
        // Right
        return grid[[player_i as usize, player_j as usize + 1]] == CELL_FREE;
    } else if delta_i == 0 && delta_j == -2 {
        // Left
        return grid[[player_i as usize, player_j as usize - 1]] == CELL_FREE;
    }
    // Jump over opponent (straight)
    else if delta_i == 4 && delta_j == 0 {
        // Down jump
        return grid[[player_i as usize + 1, player_j as usize]] == CELL_FREE
            && grid[[player_i as usize + 2, player_j as usize]] as i32 == opponent
            && grid[[player_i as usize + 3, player_j as usize]] == CELL_FREE;
    } else if delta_i == -4 && delta_j == 0 {
        // Up jump
        return grid[[player_i as usize - 1, player_j as usize]] == CELL_FREE
            && grid[[player_i as usize - 2, player_j as usize]] as i32 == opponent
            && grid[[player_i as usize - 3, player_j as usize]] == CELL_FREE;
    } else if delta_i == 0 && delta_j == 4 {
        // Right jump
        return grid[[player_i as usize, player_j as usize + 1]] == CELL_FREE
            && grid[[player_i as usize, player_j as usize + 2]] as i32 == opponent
            && grid[[player_i as usize, player_j as usize + 3]] == CELL_FREE;
    } else if delta_i == 0 && delta_j == -4 {
        // Left jump
        return grid[[player_i as usize, player_j as usize - 1]] == CELL_FREE
            && grid[[player_i as usize, player_j as usize - 2]] as i32 == opponent
            && grid[[player_i as usize, player_j as usize - 3]] == CELL_FREE;
    }
    // Diagonal jumps (when opponent is adjacent and there's a wall behind them)
    else if delta_i == 2 && delta_j == 2 {
        // Down-right diagonal
        (grid[[player_i as usize + 1, player_j as usize]] == CELL_FREE
            && grid[[player_i as usize + 2, player_j as usize]] as i32 == opponent
            && grid[[player_i as usize + 2, player_j as usize + 1]] == CELL_FREE
            && grid[[player_i as usize + 3, player_j as usize]] == CELL_WALL)
            || (grid[[player_i as usize, player_j as usize + 1]] == CELL_FREE
                && grid[[player_i as usize, player_j as usize + 2]] as i32 == opponent
                && grid[[player_i as usize + 1, player_j as usize + 2]] == CELL_FREE
                && grid[[player_i as usize, player_j as usize + 3]] == CELL_WALL)
    } else if delta_i == -2 && delta_j == 2 {
        // Up-right diagonal
        (grid[[player_i as usize - 1, player_j as usize]] == CELL_FREE
            && grid[[player_i as usize - 2, player_j as usize]] as i32 == opponent
            && grid[[player_i as usize - 2, player_j as usize + 1]] == CELL_FREE
            && grid[[player_i as usize - 3, player_j as usize]] == CELL_WALL)
            || (grid[[player_i as usize, player_j as usize + 1]] == CELL_FREE
                && grid[[player_i as usize, player_j as usize + 2]] as i32 == opponent
                && grid[[player_i as usize - 1, player_j as usize + 2]] == CELL_FREE
                && grid[[player_i as usize, player_j as usize + 3]] == CELL_WALL)
    } else if delta_i == 2 && delta_j == -2 {
        // Down-left diagonal
        (grid[[player_i as usize + 1, player_j as usize]] == CELL_FREE
            && grid[[player_i as usize + 2, player_j as usize]] as i32 == opponent
            && grid[[player_i as usize + 2, player_j as usize - 1]] == CELL_FREE
            && grid[[player_i as usize + 3, player_j as usize]] == CELL_WALL)
            || (grid[[player_i as usize, player_j as usize - 1]] == CELL_FREE
                && grid[[player_i as usize, player_j as usize - 2]] as i32 == opponent
                && grid[[player_i as usize + 1, player_j as usize - 2]] == CELL_FREE
                && grid[[player_i as usize, player_j as usize - 3]] == CELL_WALL)
    } else if delta_i == -2 && delta_j == -2 {
        // Up-left diagonal
        (grid[[player_i as usize - 1, player_j as usize]] == CELL_FREE
            && grid[[player_i as usize - 2, player_j as usize]] as i32 == opponent
            && grid[[player_i as usize - 2, player_j as usize - 1]] == CELL_FREE
            && grid[[player_i as usize - 3, player_j as usize]] == CELL_WALL)
            || (grid[[player_i as usize, player_j as usize - 1]] == CELL_FREE
                && grid[[player_i as usize, player_j as usize - 2]] as i32 == opponent
                && grid[[player_i as usize - 1, player_j as usize - 2]] == CELL_FREE
                && grid[[player_i as usize, player_j as usize - 3]] == CELL_WALL)
    } else {
        false
    }
}

/// Validate whether a wall placement is legal.
///
/// This is a direct port of is_wall_action_valid from qgrid.py.
pub fn is_wall_action_valid(
    grid: &ArrayView2<i8>,
    player_positions: &ArrayView2<i32>,
    walls_remaining: &ArrayView1<i32>,
    goal_rows: &ArrayView1<i32>,
    current_player: i32,
    wall_row: i32,
    wall_col: i32,
    wall_orientation: i32,
) -> bool {
    // Check if player has walls remaining
    if walls_remaining[current_player as usize] <= 0 {
        return false;
    }

    // Check if wall cells are free
    if !check_wall_cells(grid, wall_row, wall_col, wall_orientation, CELL_FREE) {
        return false;
    }

    // If wall doesn't touch existing walls at 2+ points, it can't block
    if !is_wall_potential_block(grid, wall_row, wall_col, wall_orientation) {
        return true;
    }

    // Need to check if placing the wall would block any player
    // Make a copy of the grid and temporarily place the wall
    let mut grid_copy = grid.to_owned();
    set_wall_cells(
        &mut grid_copy.view_mut(),
        wall_row,
        wall_col,
        wall_orientation,
        CELL_WALL,
    );

    // Check that all players can still reach their goal
    for player in 0..player_positions.nrows() {
        let player_row = player_positions[[player, 0]];
        let player_col = player_positions[[player, 1]];
        let goal_row = goal_rows[player];

        let dist = distance_to_row(&grid_copy.view(), player_row, player_col, goal_row);
        if dist == -1 {
            return false;
        }
    }

    true
}

/// Validate whether a wall placement is legal (mutable version).
/// This version modifies the grid in place and restores it, avoiding allocation.
pub fn is_wall_action_valid_mut(
    grid: &mut ArrayViewMut2<i8>,
    player_positions: &ArrayView2<i32>,
    walls_remaining: &ArrayView1<i32>,
    goal_rows: &ArrayView1<i32>,
    current_player: i32,
    wall_row: i32,
    wall_col: i32,
    wall_orientation: i32,
) -> bool {
    // Check if player has walls remaining
    if walls_remaining[current_player as usize] <= 0 {
        return false;
    }

    // Check if wall cells are free
    if !check_wall_cells(
        &grid.view(),
        wall_row,
        wall_col,
        wall_orientation,
        CELL_FREE,
    ) {
        return false;
    }

    // If wall doesn't touch existing walls at 2+ points, it can't block
    if !is_wall_potential_block(&grid.view(), wall_row, wall_col, wall_orientation) {
        return true;
    }

    // Place the wall temporarily
    set_wall_cells(grid, wall_row, wall_col, wall_orientation, CELL_WALL);

    // Check that all players can still reach their goal
    let mut valid = true;
    for player in 0..player_positions.nrows() {
        let player_row = player_positions[[player, 0]];
        let player_col = player_positions[[player, 1]];
        let goal_row = goal_rows[player];

        let dist = distance_to_row(&grid.view(), player_row, player_col, goal_row);
        if dist == -1 {
            valid = false;
            break;
        }
    }

    // Restore the grid
    set_wall_cells(grid, wall_row, wall_col, wall_orientation, CELL_FREE);

    valid
}

#[cfg(test)]
mod tests {
    use super::*;
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

        // Player positions (row, col) for a 9x9 board
        let mut player_positions = Array2::<i32>::zeros((2, 2));
        player_positions[[0, 0]] = 0;
        player_positions[[0, 1]] = 4;
        player_positions[[1, 0]] = 8;
        player_positions[[1, 1]] = 4;

        // Place players on grid
        grid[[2, 10]] = 0; // Player 0 at (0, 4)
        grid[[18, 10]] = 1; // Player 1 at (8, 4)

        let walls_remaining = Array1::from(vec![10, 10]);
        let goal_rows = Array1::from(vec![8, 0]);

        (grid, player_positions, walls_remaining, goal_rows)
    }

    #[test]
    fn test_is_move_action_valid_simple() {
        let (grid, player_positions, _, _) = create_test_game();

        // Player 0 can move down from (0,4) to (1,4)
        assert!(is_move_action_valid(
            &grid.view(),
            &player_positions.view(),
            0,
            1,
            4
        ));

        // Player 0 cannot move to (0, 0) - too far
        assert!(!is_move_action_valid(
            &grid.view(),
            &player_positions.view(),
            0,
            0,
            0
        ));
    }

    #[test]
    fn test_is_wall_action_valid_simple() {
        let (grid, player_positions, walls_remaining, goal_rows) = create_test_game();

        // Should be able to place a wall at (0,0)
        assert!(is_wall_action_valid(
            &grid.view(),
            &player_positions.view(),
            &walls_remaining.view(),
            &goal_rows.view(),
            0,
            0,
            0,
            0 // VERTICAL
        ));
    }
}
