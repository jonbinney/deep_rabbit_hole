/// Game mechanics implementation using the QBitRepr packed representation.
///
/// This provides efficient game state operations by working directly with
/// the bit-packed representation instead of converting to/from grid arrays.

use crate::q_bit_repr::QBitRepr;
use crate::pathfinding::distance_to_row;

// Wall orientations
pub const WALL_VERTICAL: usize = 0;
pub const WALL_HORIZONTAL: usize = 1;

/// Game mechanics for Quoridor using bit-packed state representation
#[derive(Clone, Debug)]
pub struct QGameMechanics {
    repr: QBitRepr,
    goal_rows: [usize; 2],  // Goal row for each player
}

impl QGameMechanics {
    /// Create a new game mechanics instance
    ///
    /// # Arguments
    /// * `board_size` - Size of the board (e.g., 9 for standard Quoridor)
    /// * `max_walls` - Maximum walls per player
    /// * `max_steps` - Maximum steps before draw
    pub fn new(board_size: usize, max_walls: usize, max_steps: usize) -> Self {
        let repr = QBitRepr::new(board_size, max_walls, max_steps);
        // Player 1 starts at bottom, aims for top (row 0)
        // Player 2 starts at top, aims for bottom (row board_size-1)
        let goal_rows = [0, board_size - 1];

        Self { repr, goal_rows }
    }

    /// Get a reference to the underlying QBitRepr
    pub fn repr(&self) -> &QBitRepr {
        &self.repr
    }

    /// Create initial game state
    pub fn create_initial_state(&self) -> Vec<u8> {
        let mut data = self.repr.create_data();
        let board_size = self.repr.board_size();

        // Player 1 starts at bottom center
        let p1_pos = self.repr.position_to_index(board_size - 1, board_size / 2);
        self.repr.set_p1_position(&mut data, p1_pos);

        // Player 2 starts at top center
        let p2_pos = self.repr.position_to_index(0, board_size / 2);
        self.repr.set_p2_position(&mut data, p2_pos);

        // Both players start with max walls
        self.repr.set_p1_walls_remaining(&mut data, self.repr.max_walls());

        // Current player is 0
        self.repr.set_current_player(&mut data, 0);

        // Steps start at 0
        self.repr.set_completed_steps(&mut data, 0);

        data
    }

    /// Check if a wall placement would be valid (no overlap)
    ///
    /// This is much faster than the grid-based approach because we just
    /// check the specific bit positions for wall overlap.
    pub fn is_wall_placement_free(&self, data: &[u8], row: usize, col: usize, orientation: usize) -> bool {
        let board_size = self.repr.board_size();

        // Check bounds
        if row >= board_size - 1 || col >= board_size - 1 {
            return false;
        }

        let wall_idx = self.repr.wall_position_to_index(row, col, orientation);

        // Check if this wall position is already occupied
        if self.repr.get_wall(data, wall_idx) {
            return false;
        }

        // Check for perpendicular wall conflict
        // A vertical wall at (r,c) conflicts with horizontal wall at (r,c)
        // A horizontal wall at (r,c) conflicts with vertical wall at (r,c)
        let perpendicular_idx = self.repr.wall_position_to_index(row, col, 1 - orientation);
        if self.repr.get_wall(data, perpendicular_idx) {
            return false;
        }

        // Check for parallel wall extensions
        if orientation == WALL_VERTICAL {
            // Check if vertical wall above conflicts (shares middle section)
            if row > 0 {
                let above_idx = self.repr.wall_position_to_index(row - 1, col, orientation);
                if self.repr.get_wall(data, above_idx) {
                    return false;
                }
            }
            // Check if vertical wall below conflicts (shares middle section)
            if row < board_size - 2 {
                let below_idx = self.repr.wall_position_to_index(row + 1, col, orientation);
                if self.repr.get_wall(data, below_idx) {
                    return false;
                }
            }
        } else {
            // WALL_HORIZONTAL
            // Check if horizontal wall to the left conflicts (shares middle section)
            if col > 0 {
                let left_idx = self.repr.wall_position_to_index(row, col - 1, orientation);
                if self.repr.get_wall(data, left_idx) {
                    return false;
                }
            }
            // Check if horizontal wall to the right conflicts (shares middle section)
            if col < board_size - 2 {
                let right_idx = self.repr.wall_position_to_index(row, col + 1, orientation);
                if self.repr.get_wall(data, right_idx) {
                    return false;
                }
            }
        }

        true
    }

    /// Place a wall (no validation - use is_wall_placement_valid first)
    #[inline]
    pub fn place_wall(&self, data: &mut [u8], row: usize, col: usize, orientation: usize) {
        let wall_idx = self.repr.wall_position_to_index(row, col, orientation);
        self.repr.set_wall(data, wall_idx, true);
    }

    /// Remove a wall
    #[inline]
    pub fn remove_wall(&self, data: &mut [u8], row: usize, col: usize, orientation: usize) {
        let wall_idx = self.repr.wall_position_to_index(row, col, orientation);
        self.repr.set_wall(data, wall_idx, false);
    }

    /// Check if a wall placement would block a player's path to their goal
    ///
    /// Returns true if the wall is valid (doesn't block path)
    pub fn is_wall_placement_valid(&self, data: &[u8], row: usize, col: usize, orientation: usize) -> bool {
        // First check if placement is physically possible
        if !self.is_wall_placement_free(data, row, col, orientation) {
            return false;
        }

        // Temporarily place the wall
        let mut temp_data = data.to_vec();
        self.place_wall(&mut temp_data, row, col, orientation);

        // Convert to grid for pathfinding check
        let grid = self.repr.to_grid(&temp_data);

        // Check both players can reach their goals
        for player in 0..2 {
            let pos_idx = if player == 0 {
                self.repr.get_p1_position(&temp_data)
            } else {
                self.repr.get_p2_position(&temp_data)
            };

            let (row, col) = self.repr.index_to_position(pos_idx);
            let goal_row = self.goal_rows[player];

            let distance = distance_to_row(&grid.view(), row as i32, col as i32, goal_row as i32);

            if distance < 0 {
                // Path is blocked
                return false;
            }
        }

        true
    }

    /// Check if a move from current position to destination is valid
    pub fn is_move_valid(
        &self,
        data: &[u8],
        player: usize,
        dest_row: usize,
        dest_col: usize,
    ) -> bool {
        let board_size = self.repr.board_size();

        // Check bounds
        if dest_row >= board_size || dest_col >= board_size {
            return false;
        }

        // Get current position
        let current_idx = if player == 0 {
            self.repr.get_p1_position(data)
        } else {
            self.repr.get_p2_position(data)
        };
        let (curr_row, curr_col) = self.repr.index_to_position(current_idx);

        // Get opponent position
        let opponent_idx = if player == 0 {
            self.repr.get_p2_position(data)
        } else {
            self.repr.get_p1_position(data)
        };
        let (opp_row, opp_col) = self.repr.index_to_position(opponent_idx);

        // Check if destination is occupied by opponent
        if dest_row == opp_row && dest_col == opp_col {
            return false;
        }

        // For now, use simple adjacency check
        // A more complete implementation would check walls and jumps
        let row_diff = (dest_row as i32 - curr_row as i32).abs();
        let col_diff = (dest_col as i32 - curr_col as i32).abs();

        // Must move exactly 1 space (or 2 for jump)
        if row_diff + col_diff > 2 {
            return false;
        }

        // For complete validation, would need to check:
        // 1. No walls blocking the path
        // 2. Valid jump over opponent
        // 3. Diagonal moves only when jumping

        // TODO: Implement full move validation with wall checking

        true
    }

    /// Execute a move action
    pub fn execute_move(&self, data: &mut [u8], player: usize, dest_row: usize, dest_col: usize) {
        let dest_idx = self.repr.position_to_index(dest_row, dest_col);

        if player == 0 {
            self.repr.set_p1_position(data, dest_idx);
        } else {
            self.repr.set_p2_position(data, dest_idx);
        }
    }

    /// Execute a wall placement action
    pub fn execute_wall_placement(
        &self,
        data: &mut [u8],
        player: usize,
        row: usize,
        col: usize,
        orientation: usize,
    ) {
        self.place_wall(data, row, col, orientation);

        // Decrement walls remaining for the player
        let current_walls = if player == 0 {
            self.repr.get_p1_walls_remaining(data)
        } else {
            self.repr.get_p2_walls_remaining(data)
        };

        if player == 0 {
            self.repr.set_p1_walls_remaining(data, current_walls.saturating_sub(1));
        }
        // Note: P2 walls are computed, so we don't set them explicitly
    }

    /// Switch to the next player
    pub fn switch_player(&self, data: &mut [u8]) {
        let current = self.repr.get_current_player(data);
        self.repr.set_current_player(data, 1 - current);

        // Increment step counter
        let steps = self.repr.get_completed_steps(data);
        self.repr.set_completed_steps(data, steps + 1);
    }

    /// Check if a player has won
    pub fn check_win(&self, data: &[u8], player: usize) -> bool {
        let pos_idx = if player == 0 {
            self.repr.get_p1_position(data)
        } else {
            self.repr.get_p2_position(data)
        };

        let (row, _col) = self.repr.index_to_position(pos_idx);
        row == self.goal_rows[player]
    }

    /// Check if the game is a draw (max steps reached)
    pub fn is_draw(&self, data: &[u8]) -> bool {
        self.repr.get_completed_steps(data) >= self.repr.max_steps()
    }

    /// Get all valid wall placements for the current player
    pub fn get_valid_wall_placements(&self, data: &[u8]) -> Vec<(usize, usize, usize)> {
        let current_player = self.repr.get_current_player(data);

        // Check if player has walls remaining
        let walls_remaining = if current_player == 0 {
            self.repr.get_p1_walls_remaining(data)
        } else {
            self.repr.get_p2_walls_remaining(data)
        };

        if walls_remaining == 0 {
            return Vec::new();
        }

        let board_size = self.repr.board_size();
        let mut valid_placements = Vec::new();

        for row in 0..board_size - 1 {
            for col in 0..board_size - 1 {
                for orientation in [WALL_VERTICAL, WALL_HORIZONTAL] {
                    if self.is_wall_placement_valid(data, row, col, orientation) {
                        valid_placements.push((row, col, orientation));
                    }
                }
            }
        }

        valid_placements
    }

    /// Get all valid moves for the current player
    pub fn get_valid_moves(&self, data: &[u8]) -> Vec<(usize, usize)> {
        let current_player = self.repr.get_current_player(data);
        let board_size = self.repr.board_size();

        let current_idx = if current_player == 0 {
            self.repr.get_p1_position(data)
        } else {
            self.repr.get_p2_position(data)
        };
        let (curr_row, curr_col) = self.repr.index_to_position(current_idx);

        let mut valid_moves = Vec::new();

        // Check all positions within 2 squares (for moves and jumps)
        for dr in -2i32..=2 {
            for dc in -2i32..=2 {
                if dr == 0 && dc == 0 {
                    continue;
                }

                let new_row = curr_row as i32 + dr;
                let new_col = curr_col as i32 + dc;

                if new_row >= 0 && new_row < board_size as i32
                    && new_col >= 0 && new_col < board_size as i32
                {
                    if self.is_move_valid(data, current_player, new_row as usize, new_col as usize) {
                        valid_moves.push((new_row as usize, new_col as usize));
                    }
                }
            }
        }

        valid_moves
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_initial_state() {
        let mechanics = QGameMechanics::new(9, 10, 100);
        let state = mechanics.create_initial_state();

        // Check player positions
        let p1_idx = mechanics.repr.get_p1_position(&state);
        let p2_idx = mechanics.repr.get_p2_position(&state);
        let (p1_row, p1_col) = mechanics.repr.index_to_position(p1_idx);
        let (p2_row, p2_col) = mechanics.repr.index_to_position(p2_idx);

        assert_eq!(p1_row, 8); // Bottom
        assert_eq!(p1_col, 4); // Center
        assert_eq!(p2_row, 0); // Top
        assert_eq!(p2_col, 4); // Center

        // Check walls
        assert_eq!(mechanics.repr.get_p1_walls_remaining(&state), 10);
        assert_eq!(mechanics.repr.get_p2_walls_remaining(&state), 10);

        // Check current player
        assert_eq!(mechanics.repr.get_current_player(&state), 0);
    }

    #[test]
    fn test_wall_placement_overlap() {
        let mechanics = QGameMechanics::new(9, 10, 100);
        let mut state = mechanics.create_initial_state();

        // Place a wall
        assert!(mechanics.is_wall_placement_free(&state, 4, 4, WALL_VERTICAL));
        mechanics.place_wall(&mut state, 4, 4, WALL_VERTICAL);

        // Same position should not be free anymore
        assert!(!mechanics.is_wall_placement_free(&state, 4, 4, WALL_VERTICAL));

        // Perpendicular position should not be free
        assert!(!mechanics.is_wall_placement_free(&state, 4, 4, WALL_HORIZONTAL));
    }

    #[test]
    fn test_wall_placement_extension() {
        let mechanics = QGameMechanics::new(9, 10, 100);
        let mut state = mechanics.create_initial_state();

        // Place a vertical wall
        mechanics.place_wall(&mut state, 4, 4, WALL_VERTICAL);

        // Adjacent vertical walls should conflict
        assert!(!mechanics.is_wall_placement_free(&state, 3, 4, WALL_VERTICAL));
        assert!(!mechanics.is_wall_placement_free(&state, 5, 4, WALL_VERTICAL));

        // Non-adjacent should be fine
        assert!(mechanics.is_wall_placement_free(&state, 2, 4, WALL_VERTICAL));
        assert!(mechanics.is_wall_placement_free(&state, 6, 4, WALL_VERTICAL));
    }

    #[test]
    fn test_win_condition() {
        let mechanics = QGameMechanics::new(9, 10, 100);
        let mut state = mechanics.create_initial_state();

        // Move player 1 to goal row
        mechanics.execute_move(&mut state, 0, 0, 4);
        assert!(mechanics.check_win(&state, 0));
        assert!(!mechanics.check_win(&state, 1));
    }

    #[test]
    fn test_execute_wall_placement() {
        let mechanics = QGameMechanics::new(9, 10, 100);
        let mut state = mechanics.create_initial_state();

        let initial_walls = mechanics.repr.get_p1_walls_remaining(&state);
        mechanics.execute_wall_placement(&mut state, 0, 4, 4, WALL_VERTICAL);

        assert_eq!(mechanics.repr.get_p1_walls_remaining(&state), initial_walls - 1);
        assert!(mechanics.repr.get_wall(&state, mechanics.repr.wall_position_to_index(4, 4, WALL_VERTICAL)));
    }

    #[test]
    fn test_switch_player() {
        let mechanics = QGameMechanics::new(9, 10, 100);
        let mut state = mechanics.create_initial_state();

        assert_eq!(mechanics.repr.get_current_player(&state), 0);
        assert_eq!(mechanics.repr.get_completed_steps(&state), 0);

        mechanics.switch_player(&mut state);

        assert_eq!(mechanics.repr.get_current_player(&state), 1);
        assert_eq!(mechanics.repr.get_completed_steps(&state), 1);
    }
}
