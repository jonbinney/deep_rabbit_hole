#![allow(dead_code)]

use ndarray::ArrayView2;

#[cfg(test)]
use crate::grid::CELL_FREE;

use crate::grid::CELL_WALL;

/// Calculate the shortest distance from a position to a target row using BFS.
///
/// This is an optimized version that avoids heap allocations by using fixed-size
/// arrays for the visited set and BFS queue.
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
    let grid_height = grid.nrows();
    let grid_width = grid.ncols();

    // Convert board coordinates to grid coordinates
    let start_i = (start_row * 2 + 2) as usize;
    let start_j = (start_col * 2 + 2) as usize;
    let target_i = (target_row * 2 + 2) as usize;

    // Already at target
    if target_i == start_i {
        return 0;
    }

    // Use a fixed-size visited array on the stack (supports up to 20x20 grid)
    // For a 9x9 board, grid is 20x20 (2*9 + 2 border on each side)
    const MAX_GRID_SIZE: usize = 24;
    let mut visited = [[false; MAX_GRID_SIZE]; MAX_GRID_SIZE];
    visited[start_i][start_j] = true;

    // Use a fixed-size ring buffer for BFS queue
    // Maximum positions = board_size^2 = 81 for 9x9, use 128 for safety
    const MAX_QUEUE_SIZE: usize = 128;
    let mut queue_i = [0usize; MAX_QUEUE_SIZE];
    let mut queue_j = [0usize; MAX_QUEUE_SIZE];
    let mut queue_steps = [0i32; MAX_QUEUE_SIZE];
    let mut queue_head: usize = 0;
    let mut queue_tail: usize = 0;

    // Enqueue start position
    queue_i[queue_tail] = start_i;
    queue_j[queue_tail] = start_j;
    queue_steps[queue_tail] = 0;
    queue_tail = (queue_tail + 1) % MAX_QUEUE_SIZE;

    while queue_head != queue_tail {
        let i = queue_i[queue_head];
        let j = queue_j[queue_head];
        let steps = queue_steps[queue_head];
        queue_head = (queue_head + 1) % MAX_QUEUE_SIZE;

        // Try all 4 directions with early return on finding target row

        // Down
        let new_i = i + 2;
        if new_i < grid_height && !visited[new_i][j] {
            let wall_i = i + 1;
            if grid[[wall_i, j]] != CELL_WALL {
                visited[new_i][j] = true;
                if new_i == target_i {
                    return steps + 1;
                }
                queue_i[queue_tail] = new_i;
                queue_j[queue_tail] = j;
                queue_steps[queue_tail] = steps + 1;
                queue_tail = (queue_tail + 1) % MAX_QUEUE_SIZE;
            }
        }

        // Up
        if i >= 2 {
            let new_i = i - 2;
            if !visited[new_i][j] {
                let wall_i = i - 1;
                if grid[[wall_i, j]] != CELL_WALL {
                    visited[new_i][j] = true;
                    if new_i == target_i {
                        return steps + 1;
                    }
                    queue_i[queue_tail] = new_i;
                    queue_j[queue_tail] = j;
                    queue_steps[queue_tail] = steps + 1;
                    queue_tail = (queue_tail + 1) % MAX_QUEUE_SIZE;
                }
            }
        }

        // Right
        let new_j = j + 2;
        if new_j < grid_width && !visited[i][new_j] {
            let wall_j = j + 1;
            if grid[[i, wall_j]] != CELL_WALL {
                visited[i][new_j] = true;
                if i == target_i {
                    return steps + 1;
                }
                queue_i[queue_tail] = i;
                queue_j[queue_tail] = new_j;
                queue_steps[queue_tail] = steps + 1;
                queue_tail = (queue_tail + 1) % MAX_QUEUE_SIZE;
            }
        }

        // Left
        if j >= 2 {
            let new_j = j - 2;
            if !visited[i][new_j] {
                let wall_j = j - 1;
                if grid[[i, wall_j]] != CELL_WALL {
                    visited[i][new_j] = true;
                    if i == target_i {
                        return steps + 1;
                    }
                    queue_i[queue_tail] = i;
                    queue_j[queue_tail] = new_j;
                    queue_steps[queue_tail] = steps + 1;
                    queue_tail = (queue_tail + 1) % MAX_QUEUE_SIZE;
                }
            }
        }
    }

    // No path found
    -1
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

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
