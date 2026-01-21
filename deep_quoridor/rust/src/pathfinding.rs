// Old non-QBitRepr pathfinding utilities, kept for backwards compatibility
#![allow(dead_code)]

use ndarray::{Array2, ArrayView2};
use std::collections::VecDeque;

pub use crate::grid::CELL_WALL;
#[cfg(test)]
use crate::grid::CELL_FREE;

/// Calculate the shortest distance from a position to a target row using BFS.
///
/// This is a direct port of the Numba implementation from qgrid.py.
///
/// # Arguments
/// * `grid` - The game grid (includes walls and border)
/// * `start_row` - Starting row on the Quoridor board (0-8 for standard board)
/// * `start_col` - Starting column on the Quoridor board (0-8 for standard board)
/// * `target_row` - Target row to reach
///
/// # Returns
/// The minimum number of moves to reach the target row, or -1 if unreachable
pub fn distance_to_row(
    grid: &ArrayView2<i8>,
    start_row: i32,
    start_col: i32,
    target_row: i32,
) -> i32 {
    let grid_height = grid.nrows() as i32;
    let grid_width = grid.ncols() as i32;

    // Convert board coordinates to grid coordinates
    let start_i = start_row * 2 + 2;
    let start_j = start_col * 2 + 2;
    let target_i = target_row * 2 + 2;

    // Already at target
    if target_i == start_i {
        return 0;
    }

    // Track visited positions
    let mut visited = Array2::<bool>::default((grid_height as usize, grid_width as usize));
    visited[[start_i as usize, start_j as usize]] = true;

    // BFS queue: (i, j, steps)
    let mut queue = VecDeque::new();
    queue.push_back((start_i, start_j, 0));

    while let Some((i, j, steps)) = queue.pop_front() {
        // Try all 4 directions
        // This is done explicitly (not in a loop) to match the Numba implementation
        // which unrolls the loop for better performance

        // Down
        let new_i = i + 2;
        let wall_i = i + 1;
        if new_i < grid_height
            && !visited[[new_i as usize, j as usize]]
            && grid[[wall_i as usize, j as usize]] != CELL_WALL
        {
            visited[[new_i as usize, j as usize]] = true;
            if target_i == new_i {
                return steps + 1;
            }
            queue.push_back((new_i, j, steps + 1));
        }

        // Up
        let new_i = i - 2;
        let wall_i = i - 1;
        if new_i >= 0
            && !visited[[new_i as usize, j as usize]]
            && grid[[wall_i as usize, j as usize]] != CELL_WALL
        {
            visited[[new_i as usize, j as usize]] = true;
            if target_i == new_i {
                return steps + 1;
            }
            queue.push_back((new_i, j, steps + 1));
        }

        // Right
        let new_j = j + 2;
        let wall_j = j + 1;
        if new_j < grid_width
            && !visited[[i as usize, new_j as usize]]
            && grid[[i as usize, wall_j as usize]] != CELL_WALL
        {
            visited[[i as usize, new_j as usize]] = true;
            if target_i == i {
                return steps + 1;
            }
            queue.push_back((i, new_j, steps + 1));
        }

        // Left
        let new_j = j - 2;
        let wall_j = j - 1;
        if new_j >= 0
            && !visited[[i as usize, new_j as usize]]
            && grid[[i as usize, wall_j as usize]] != CELL_WALL
        {
            visited[[i as usize, new_j as usize]] = true;
            if target_i == i {
                return steps + 1;
            }
            queue.push_back((i, new_j, steps + 1));
        }
    }

    // No path found
    -1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_to_row_simple() {
        // Create a simple 5x5 board (grid will be 12x12 with border)
        let mut grid = Array2::<i8>::from_elem((12, 12), CELL_FREE);

        // Add border walls
        for i in 0..2 {
            for j in 0..12 {
                grid[[i, j]] = CELL_WALL;
                grid[[11 - i, j]] = CELL_WALL;
                grid[[j, i]] = CELL_WALL;
                grid[[j, 11 - i]] = CELL_WALL;
            }
        }

        // No walls between positions, should be direct distance
        let dist = distance_to_row(&grid.view(), 0, 0, 4);
        assert_eq!(dist, 4);
    }

    #[test]
    fn test_distance_to_row_same_position() {
        let mut grid = Array2::<i8>::from_elem((12, 12), CELL_FREE);

        // Add border walls
        for i in 0..2 {
            for j in 0..12 {
                grid[[i, j]] = CELL_WALL;
                grid[[11 - i, j]] = CELL_WALL;
                grid[[j, i]] = CELL_WALL;
                grid[[j, 11 - i]] = CELL_WALL;
            }
        }

        let dist = distance_to_row(&grid.view(), 2, 2, 2);
        assert_eq!(dist, 0);
    }
}
