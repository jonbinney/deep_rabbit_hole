// Old non-QBitRepr grid utilities, kept for backwards compatibility
#![allow(dead_code)]

use ndarray::{ArrayView2, ArrayViewMut2};

// Cell values - must match the Python constants
pub const CELL_FREE: i8 = -1;
pub const CELL_PLAYER1: i8 = 0;
pub const CELL_PLAYER2: i8 = 1;
pub const CELL_WALL: i8 = 10;

// Wall orientations
pub const WALL_ORIENTATION_VERTICAL: i32 = 0;
pub const WALL_ORIENTATION_HORIZONTAL: i32 = 1;

/// Check if the wall cells at the given position are free.
///
/// This is a direct port of are_wall_cells_free from qgrid.py.
pub fn are_wall_cells_free(
    grid: &ArrayView2<i8>,
    wall_row: i32,
    wall_col: i32,
    wall_orientation: i32,
) -> bool {
    let grid_height = grid.nrows() as i32;
    let grid_width = grid.ncols() as i32;

    if wall_orientation == WALL_ORIENTATION_VERTICAL {
        let start_i = wall_row * 2 + 2;
        let start_j = wall_col * 2 + 3;

        if start_i < 0 || start_i + 2 >= grid_height || start_j < 0 || start_j >= grid_width {
            return false;
        }

        grid[[start_i as usize, start_j as usize]] == CELL_FREE
            && grid[[start_i as usize + 1, start_j as usize]] == CELL_FREE
            && grid[[start_i as usize + 2, start_j as usize]] == CELL_FREE
    } else if wall_orientation == WALL_ORIENTATION_HORIZONTAL {
        let start_i = wall_row * 2 + 3;
        let start_j = wall_col * 2 + 2;

        if start_i < 0 || start_i >= grid_height || start_j < 0 || start_j + 2 >= grid_width {
            return false;
        }

        grid[[start_i as usize, start_j as usize]] == CELL_FREE
            && grid[[start_i as usize, start_j as usize + 1]] == CELL_FREE
            && grid[[start_i as usize, start_j as usize + 2]] == CELL_FREE
    } else {
        panic!("Invalid wall orientation: {}", wall_orientation);
    }
}

/// Set all grid cells corresponding to the wall to the given value.
///
/// This is a direct port of set_wall_cells from qgrid.py.
pub fn set_wall_cells(
    grid: &mut ArrayViewMut2<i8>,
    wall_row: i32,
    wall_col: i32,
    wall_orientation: i32,
    cell_value: i8,
) {
    let grid_height = grid.nrows() as i32;
    let grid_width = grid.ncols() as i32;

    if wall_orientation == WALL_ORIENTATION_VERTICAL {
        let start_i = wall_row * 2 + 2;
        let start_j = wall_col * 2 + 3;

        assert!(start_i >= 0 && start_i + 2 < grid_height && start_j >= 0 && start_j < grid_width);

        grid[[start_i as usize, start_j as usize]] = cell_value;
        grid[[start_i as usize + 1, start_j as usize]] = cell_value;
        grid[[start_i as usize + 2, start_j as usize]] = cell_value;
    } else {
        // HORIZONTAL
        let start_i = wall_row * 2 + 3;
        let start_j = wall_col * 2 + 2;

        assert!(start_i >= 0 && start_i < grid_height && start_j >= 0 && start_j + 2 < grid_width);

        grid[[start_i as usize, start_j as usize]] = cell_value;
        grid[[start_i as usize, start_j as usize + 1]] = cell_value;
        grid[[start_i as usize, start_j as usize + 2]] = cell_value;
    }
}

/// Check if all grid cells corresponding to the wall equal the given value.
///
/// This is a direct port of check_wall_cells from qgrid.py.
pub fn check_wall_cells(
    grid: &ArrayView2<i8>,
    wall_row: i32,
    wall_col: i32,
    wall_orientation: i32,
    cell_value: i8,
) -> bool {
    let grid_height = grid.nrows() as i32;
    let grid_width = grid.ncols() as i32;

    if wall_orientation == WALL_ORIENTATION_VERTICAL {
        let start_i = wall_row * 2 + 2;
        let start_j = wall_col * 2 + 3;

        assert!(start_i >= 0 && start_i + 2 < grid_height && start_j >= 0 && start_j < grid_width);

        grid[[start_i as usize, start_j as usize]] == cell_value
            && grid[[start_i as usize + 1, start_j as usize]] == cell_value
            && grid[[start_i as usize + 2, start_j as usize]] == cell_value
    } else {
        // HORIZONTAL
        let start_i = wall_row * 2 + 3;
        let start_j = wall_col * 2 + 2;

        assert!(start_i >= 0 && start_i < grid_height && start_j >= 0 && start_j + 2 < grid_width);

        grid[[start_i as usize, start_j as usize]] == cell_value
            && grid[[start_i as usize, start_j as usize + 1]] == cell_value
            && grid[[start_i as usize, start_j as usize + 2]] == cell_value
    }
}

/// Check if a wall placement could potentially block a player's path.
///
/// This is a direct port of is_wall_potential_block from qgrid.py.
/// Returns true if the wall touches at least 2 of its 3 adjacent cells.
pub fn is_wall_potential_block(
    grid: &ArrayView2<i8>,
    wall_row: i32,
    wall_col: i32,
    wall_orientation: i32,
) -> bool {
    let grid_height = grid.nrows() as i32;
    let grid_width = grid.ncols() as i32;

    if wall_orientation == WALL_ORIENTATION_VERTICAL {
        let start_i = wall_row * 2 + 2;
        let start_j = wall_col * 2 + 3;

        assert!(start_i >= 0 && start_i + 2 < grid_height && start_j >= 0 && start_j < grid_width);

        let mut touches = 0;

        // Check top end
        if grid[[start_i as usize - 1, start_j as usize - 1]] == CELL_WALL
            || grid[[start_i as usize - 2, start_j as usize]] == CELL_WALL
            || grid[[start_i as usize - 1, start_j as usize + 1]] == CELL_WALL
        {
            touches += 1;
        }

        // Check middle
        if grid[[start_i as usize + 1, start_j as usize - 1]] == CELL_WALL
            || grid[[start_i as usize + 1, start_j as usize + 1]] == CELL_WALL
        {
            touches += 1;
        }

        // Check bottom end
        if grid[[start_i as usize + 3, start_j as usize - 1]] == CELL_WALL
            || grid[[start_i as usize + 4, start_j as usize]] == CELL_WALL
            || grid[[start_i as usize + 3, start_j as usize + 1]] == CELL_WALL
        {
            touches += 1;
        }

        touches >= 2
    } else {
        // HORIZONTAL
        let start_i = wall_row * 2 + 3;
        let start_j = wall_col * 2 + 2;

        assert!(start_i >= 0 && start_i < grid_height && start_j >= 0 && start_j + 2 < grid_width);

        let mut touches = 0;

        // Check left end
        if grid[[start_i as usize - 1, start_j as usize - 1]] == CELL_WALL
            || grid[[start_i as usize, start_j as usize - 2]] == CELL_WALL
            || grid[[start_i as usize + 1, start_j as usize - 1]] == CELL_WALL
        {
            touches += 1;
        }

        // Check middle
        if grid[[start_i as usize - 1, start_j as usize + 1]] == CELL_WALL
            || grid[[start_i as usize + 1, start_j as usize + 1]] == CELL_WALL
        {
            touches += 1;
        }

        // Check right end
        if grid[[start_i as usize - 1, start_j as usize + 3]] == CELL_WALL
            || grid[[start_i as usize, start_j as usize + 4]] == CELL_WALL
            || grid[[start_i as usize + 1, start_j as usize + 3]] == CELL_WALL
        {
            touches += 1;
        }

        touches >= 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_test_grid() -> Array2<i8> {
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

        grid
    }

    #[test]
    fn test_are_wall_cells_free() {
        let grid = create_test_grid();

        // Should be free on empty board
        assert!(are_wall_cells_free(
            &grid.view(),
            0,
            0,
            WALL_ORIENTATION_VERTICAL
        ));
        assert!(are_wall_cells_free(
            &grid.view(),
            0,
            0,
            WALL_ORIENTATION_HORIZONTAL
        ));
    }

    #[test]
    fn test_set_and_check_wall_cells() {
        let mut grid = create_test_grid();

        // Initially should be free
        assert!(check_wall_cells(
            &grid.view(),
            0,
            0,
            WALL_ORIENTATION_VERTICAL,
            CELL_FREE
        ));

        // Set to wall
        set_wall_cells(
            &mut grid.view_mut(),
            0,
            0,
            WALL_ORIENTATION_VERTICAL,
            CELL_WALL,
        );

        // Now should be wall
        assert!(check_wall_cells(
            &grid.view(),
            0,
            0,
            WALL_ORIENTATION_VERTICAL,
            CELL_WALL
        ));
        assert!(!are_wall_cells_free(
            &grid.view(),
            0,
            0,
            WALL_ORIENTATION_VERTICAL
        ));
    }
}
