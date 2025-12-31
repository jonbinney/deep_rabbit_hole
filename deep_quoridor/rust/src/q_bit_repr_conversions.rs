/// Conversion functions between QBitRepr packed format and game state arrays.
///
/// This module contains methods for converting between the bit-packed representation
/// and the array-based game state format used by the minimax algorithm.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use crate::q_bit_repr::QBitRepr;

impl QBitRepr {
    /// Populate the packed state from game state arrays used by minimax
    ///
    /// # Arguments
    /// * `data` - The packed state data to populate
    /// * `grid` - The game grid (includes walls marked as occupied cells)
    /// * `player_positions` - 2x2 array with [row, col] for each player
    /// * `walls_remaining` - Array of walls remaining for each player
    /// * `current_player` - Which player's turn (0 or 1)
    /// * `completed_steps` - Number of steps completed so far
    pub fn from_game_state(
        &self,
        data: &mut [u8],
        grid: &ArrayView2<i8>,
        player_positions: &ArrayView2<i32>,
        walls_remaining: &ArrayView1<i32>,
        current_player: i32,
        completed_steps: i32,
    ) {
        // Set player positions
        let p1_pos = self.position_to_index(
            player_positions[[0, 0]] as usize,
            player_positions[[0, 1]] as usize,
        );
        let p2_pos = self.position_to_index(
            player_positions[[1, 0]] as usize,
            player_positions[[1, 1]] as usize,
        );
        self.set_p1_position(data, p1_pos);
        self.set_p2_position(data, p2_pos);

        // Set walls remaining
        self.set_p1_walls_remaining(data, walls_remaining[0] as usize);

        // Set current player
        self.set_current_player(data, current_player as usize);

        // Set completed steps
        self.set_completed_steps(data, completed_steps as usize);

        // Extract wall positions from grid
        // The grid uses a coordinate system where walls are between cells
        // We need to scan the grid for wall markers and convert to wall indices
        // We use check_wall_cells to verify that the cells are specifically set to CELL_WALL
        // (not just occupied by boundary walls or other things)
        use crate::grid::check_wall_cells;
        use crate::pathfinding::CELL_WALL;

        for wall_row in 0..(self.board_size() - 1) {
            for wall_col in 0..(self.board_size() - 1) {
                // Check vertical wall (orientation 0)
                if check_wall_cells(grid, wall_row as i32, wall_col as i32, 0, CELL_WALL) {
                    let wall_idx = self.wall_position_to_index(wall_row, wall_col, 0);
                    self.set_wall(data, wall_idx, true);
                }
                // Check horizontal wall (orientation 1)
                if check_wall_cells(grid, wall_row as i32, wall_col as i32, 1, CELL_WALL) {
                    let wall_idx = self.wall_position_to_index(wall_row, wall_col, 1);
                    self.set_wall(data, wall_idx, true);
                }
            }
        }
    }

    /// Extract player positions as a 2x2 array (format used by minimax)
    /// Returns [[p1_row, p1_col], [p2_row, p2_col]]
    pub fn to_player_positions(&self, data: &[u8]) -> Array2<i32> {
        let p1_index = self.get_p1_position(data);
        let p2_index = self.get_p2_position(data);
        let (p1_row, p1_col) = self.index_to_position(p1_index);
        let (p2_row, p2_col) = self.index_to_position(p2_index);

        Array2::from_shape_vec((2, 2), vec![
            p1_row as i32, p1_col as i32,
            p2_row as i32, p2_col as i32,
        ]).unwrap()
    }

    /// Extract walls remaining as a 1D array (format used by minimax)
    /// Returns [p1_walls, p2_walls]
    pub fn to_walls_remaining(&self, data: &[u8]) -> Array1<i32> {
        Array1::from_vec(vec![
            self.get_p1_walls_remaining(data) as i32,
            self.get_p2_walls_remaining(data) as i32,
        ])
    }

    /// Reconstruct the full grid with walls and player positions
    /// This creates a grid in the format used by minimax: (2*board_size + 3) x (2*board_size + 3)
    pub fn to_grid(&self, data: &[u8]) -> Array2<i8> {
        use crate::pathfinding::{CELL_FREE, CELL_PLAYER1, CELL_PLAYER2, CELL_WALL};

        let grid_size = 2 * self.board_size() + 3;
        let mut grid = Array2::from_elem((grid_size, grid_size), CELL_FREE);

        // Add boundary walls
        for i in 0..grid_size {
            grid[[0, i]] = CELL_WALL;
            grid[[1, i]] = CELL_WALL;
            grid[[grid_size - 1, i]] = CELL_WALL;
            grid[[grid_size - 2, i]] = CELL_WALL;
            grid[[i, 0]] = CELL_WALL;
            grid[[i, 1]] = CELL_WALL;
            grid[[i, grid_size - 1]] = CELL_WALL;
            grid[[i, grid_size - 2]] = CELL_WALL;
        }

        // Add placed walls
        use crate::grid::set_wall_cells;
        for wall_idx in 0..self.num_wall_positions() {
            if self.get_wall(data, wall_idx) {
                let (wall_row, wall_col, orientation) = self.wall_index_to_position(wall_idx);
                set_wall_cells(
                    &mut grid.view_mut(),
                    wall_row as i32,
                    wall_col as i32,
                    orientation as i32,
                    CELL_WALL,
                );
            }
        }

        // Add players
        let p1_index = self.get_p1_position(data);
        let p2_index = self.get_p2_position(data);
        let (p1_row, p1_col) = self.index_to_position(p1_index);
        let (p2_row, p2_col) = self.index_to_position(p2_index);

        // Convert board coordinates to grid coordinates (with padding)
        let p1_grid_row = 2 + p1_row * 2;
        let p1_grid_col = 2 + p1_col * 2;
        let p2_grid_row = 2 + p2_row * 2;
        let p2_grid_col = 2 + p2_col * 2;

        grid[[p1_grid_row, p1_grid_col]] = CELL_PLAYER1;
        grid[[p2_grid_row, p2_grid_col]] = CELL_PLAYER2;

        grid
    }
}
