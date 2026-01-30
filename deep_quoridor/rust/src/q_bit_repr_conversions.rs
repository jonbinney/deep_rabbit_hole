use crate::q_bit_repr::QBitRepr;
/// Conversion functions between QBitRepr packed format and game state arrays.
///
/// This module contains methods for converting between the bit-packed representation
/// and the array-based game state format used by the minimax algorithm.
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

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
        self.set_player_position(
            data,
            0,
            player_positions[[0, 0]] as usize,
            player_positions[[0, 1]] as usize,
        );
        self.set_player_position(
            data,
            1,
            player_positions[[1, 0]] as usize,
            player_positions[[1, 1]] as usize,
        );

        self.set_walls_remaining(data, 0, walls_remaining[0] as usize);
        self.set_walls_remaining(data, 1, walls_remaining[1] as usize);

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
        use crate::grid::CELL_WALL;

        for wall_row in 0..(self.board_size() - 1) {
            for wall_col in 0..(self.board_size() - 1) {
                // Check vertical wall (orientation 0)
                if check_wall_cells(grid, wall_row as i32, wall_col as i32, 0, CELL_WALL) {
                    self.set_wall(data, wall_row, wall_col, 0, true);
                }
                // Check horizontal wall (orientation 1)
                if check_wall_cells(grid, wall_row as i32, wall_col as i32, 1, CELL_WALL) {
                    self.set_wall(data, wall_row, wall_col, 1, true);
                }
            }
        }
    }

    /// Extract player positions as a 2x2 array (format used by minimax)
    /// Returns [[p1_row, p1_col], [p2_row, p2_col]]
    #[allow(dead_code)]
    pub fn to_player_positions(&self, data: &[u8]) -> Array2<i32> {
        let (p1_row, p1_col) = self.get_player_position(data, 0);
        let (p2_row, p2_col) = self.get_player_position(data, 1);

        Array2::from_shape_vec(
            (2, 2),
            vec![p1_row as i32, p1_col as i32, p2_row as i32, p2_col as i32],
        )
        .unwrap()
    }

    /// Extract walls remaining as a 1D array (format used by minimax)
    /// Returns [p1_walls, p2_walls]
    #[allow(dead_code)]
    pub fn to_walls_remaining(&self, data: &[u8]) -> Array1<i32> {
        Array1::from_vec(vec![
            self.get_walls_remaining(data, 0) as i32,
            self.get_walls_remaining(data, 1) as i32,
        ])
    }

    /// Reconstruct the full grid with walls and player positions
    /// This creates a grid in the format used by minimax: (2*board_size + 3) x (2*board_size + 3)
    #[allow(dead_code)]
    pub fn to_grid(&self, data: &[u8]) -> Array2<i8> {
        use crate::grid::{CELL_FREE, CELL_PLAYER1, CELL_PLAYER2, CELL_WALL};

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
            let (wall_row, wall_col, orientation) = self.wall_index_to_position(wall_idx);
            if self.get_wall(data, wall_row, wall_col, orientation) {
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
        let (p1_row, p1_col) = self.get_player_position(data, 0);
        let (p2_row, p2_col) = self.get_player_position(data, 1);

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_game_state_conversion() {
        let q = QBitRepr::new(5, 10, 100);
        let mut data = q.create_data();

        // Create test game state
        // Note: The packed format computes P2 walls from P1 walls and total walls on board,
        // so we need consistent data. With P1 having 8 walls (used 2), and 2 walls total
        // on board, P2 must have used 0 walls, so P2 has 10 walls remaining.
        let player_positions = Array2::from_shape_vec((2, 2), vec![0, 2, 4, 2]).unwrap();
        let walls_remaining = Array1::from_vec(vec![8, 10]);
        let current_player = 1;
        let completed_steps = 5;

        // Create a grid with 2 walls placed (to match P1 having used 2 walls)
        let grid_size = 2 * q.board_size() + 3;
        let mut grid = Array2::zeros((grid_size, grid_size));

        // Add boundary walls (required by grid format)
        use crate::grid::CELL_WALL;
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

        // Place 2 walls using the grid interface
        use crate::grid::set_wall_cells;
        set_wall_cells(&mut grid.view_mut(), 0, 0, 0 as i32, CELL_WALL); // Vertical wall at (0,0)
        set_wall_cells(&mut grid.view_mut(), 1, 1, 1 as i32, CELL_WALL); // Horizontal wall at (1,1)

        // Populate packed state
        q.from_game_state(
            &mut data,
            &grid.view(),
            &player_positions.view(),
            &walls_remaining.view(),
            current_player,
            completed_steps,
        );

        // Verify we can extract the same data back
        let extracted_positions = q.to_player_positions(&data);
        assert_eq!(extracted_positions[[0, 0]], 0);
        assert_eq!(extracted_positions[[0, 1]], 2);
        assert_eq!(extracted_positions[[1, 0]], 4);
        assert_eq!(extracted_positions[[1, 1]], 2);

        let extracted_walls = q.to_walls_remaining(&data);
        assert_eq!(extracted_walls[0], 8);

        // P2 walls are computed: p1_used=2, total_on_board=2, so p2_used=0, p2_remaining=10
        assert_eq!(extracted_walls[1], 10, "P2 should have 10 walls remaining");

        assert_eq!(q.get_current_player(&data), current_player as usize);
        assert_eq!(q.get_completed_steps(&data), completed_steps as usize);
    }
}
