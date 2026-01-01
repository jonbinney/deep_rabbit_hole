/// Compact bit-packed representation accessor for game states.
///
/// This struct doesn't store the game state data itself - it only stores the
/// parameters and computed offsets needed to interpret byte arrays as packed game states.
/// The data is passed to each method, allowing flexible storage (stack or heap).

// Wall orientations
pub const WALL_VERTICAL: usize = 0;
pub const WALL_HORIZONTAL: usize = 1;

/// Calculate the number of bits needed to represent a value up to max (inclusive)
const fn bits_needed(max: usize) -> usize {
    if max == 0 {
        1
    } else {
        (usize::BITS - max.leading_zeros()) as usize
    }
}

/// Accessor for bit-packed game states.
/// Stores parameters and computed offsets, but not the actual game state data.
#[derive(Clone, Debug)]
pub struct QBitRepr {
    // Parameters
    board_size: usize,
    max_walls: usize,
    max_steps: usize,

    // Computed values
    num_wall_positions: usize,
    num_player_positions: usize,
    position_bits: usize,
    walls_remaining_bits: usize,
    steps_bits: usize,
    total_bits: usize,
    total_bytes: usize,

    // Bit offsets for each field
    walls_offset: usize,
    p1_pos_offset: usize,
    p2_pos_offset: usize,
    walls_remaining_offset: usize,
    current_player_offset: usize,
    steps_offset: usize,
}

impl QBitRepr {
    /// Create a new QBitRepr state accessor with the given parameters
    pub fn new(board_size: usize, max_walls: usize, max_steps: usize) -> Self {
        let num_wall_positions = 2 * (board_size - 1) * (board_size - 1);
        let num_player_positions = board_size * board_size;
        let position_bits = bits_needed(num_player_positions - 1);
        let walls_remaining_bits = bits_needed(max_walls);
        let steps_bits = bits_needed(max_steps);

        let walls_offset = 0;
        let p1_pos_offset = walls_offset + num_wall_positions;
        let p2_pos_offset = p1_pos_offset + position_bits;
        let walls_remaining_offset = p2_pos_offset + position_bits;
        let current_player_offset = walls_remaining_offset + walls_remaining_bits;
        let steps_offset = current_player_offset + 1;

        let total_bits = steps_offset + steps_bits;
        let total_bytes = (total_bits + 7) / 8;

        Self {
            board_size,
            max_walls,
            max_steps,
            num_wall_positions,
            num_player_positions,
            position_bits,
            walls_remaining_bits,
            steps_bits,
            total_bits,
            total_bytes,
            walls_offset,
            p1_pos_offset,
            p2_pos_offset,
            walls_remaining_offset,
            current_player_offset,
            steps_offset,
        }
    }

    /// Get the size of the packed representation in bytes
    pub fn size_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Get the number of wall positions
    pub fn num_wall_positions(&self) -> usize {
        self.num_wall_positions
    }

    /// Get the number of player positions
    pub fn num_player_positions(&self) -> usize {
        self.num_player_positions
    }

    /// Create a new byte array for storing a packed state
    pub fn create_data(&self) -> Vec<u8> {
        vec![0u8; self.total_bytes]
    }

    /// Get a bit at the specified position in the data
    #[inline]
    fn get_bit(&self, data: &[u8], bit_index: usize) -> bool {
        debug_assert!(data.len() >= self.total_bytes);
        let byte_index = bit_index / 8;
        let bit_offset = bit_index % 8;
        (data[byte_index] >> bit_offset) & 1 == 1
    }

    /// Set a bit at the specified position in the data
    #[inline]
    fn set_bit(&self, data: &mut [u8], bit_index: usize, value: bool) {
        debug_assert!(data.len() >= self.total_bytes);
        let byte_index = bit_index / 8;
        let bit_offset = bit_index % 8;
        if value {
            data[byte_index] |= 1 << bit_offset;
        } else {
            data[byte_index] &= !(1 << bit_offset);
        }
    }

    /// Get an integer value starting at bit_offset with num_bits bits
    #[inline]
    fn get_bits(&self, data: &[u8], bit_offset: usize, num_bits: usize) -> usize {
        let mut value = 0usize;
        for i in 0..num_bits {
            if self.get_bit(data, bit_offset + i) {
                value |= 1 << i;
            }
        }
        value
    }

    /// Set an integer value starting at bit_offset with num_bits bits
    #[inline]
    fn set_bits(&self, data: &mut [u8], bit_offset: usize, num_bits: usize, value: usize) {
        for i in 0..num_bits {
            self.set_bit(data, bit_offset + i, (value >> i) & 1 == 1);
        }
    }

    /// Check if a wall is present at the given position
    pub fn get_wall(&self, data: &[u8], row: usize, col: usize, orientation: usize) -> bool {
        let wall_index = self.wall_position_to_index(row, col, orientation);
        self.get_bit(data, self.walls_offset + wall_index)
    }

    /// Set a wall at the given position
    pub fn set_wall(&self, data: &mut [u8], row: usize, col: usize, orientation: usize, present: bool) {
        let wall_index = self.wall_position_to_index(row, col, orientation);
        self.set_bit(data, self.walls_offset + wall_index, present);
    }

    /// Get player 1's position (as a flat index from 0 to board_size^2 - 1)
    pub fn get_p1_position(&self, data: &[u8]) -> usize {
        self.get_bits(data, self.p1_pos_offset, self.position_bits)
    }

    /// Set player 1's position
    pub fn set_p1_position(&self, data: &mut [u8], pos: usize) {
        debug_assert!(pos < self.num_player_positions);
        self.set_bits(data, self.p1_pos_offset, self.position_bits, pos);
    }

    /// Get player 2's position (as a flat index from 0 to board_size^2 - 1)
    pub fn get_p2_position(&self, data: &[u8]) -> usize {
        self.get_bits(data, self.p2_pos_offset, self.position_bits)
    }

    /// Set player 2's position
    pub fn set_p2_position(&self, data: &mut [u8], pos: usize) {
        debug_assert!(pos < self.num_player_positions);
        self.set_bits(data, self.p2_pos_offset, self.position_bits, pos);
    }

    /// Get player 1's remaining walls
    pub fn get_p1_walls_remaining(&self, data: &[u8]) -> usize {
        self.get_bits(data, self.walls_remaining_offset, self.walls_remaining_bits)
    }

    /// Set player 1's remaining walls
    pub fn set_p1_walls_remaining(&self, data: &mut [u8], walls: usize) {
        debug_assert!(walls <= self.max_walls);
        self.set_bits(data, self.walls_remaining_offset, self.walls_remaining_bits, walls);
    }

    /// Get player 2's remaining walls (computed from total walls used)
    pub fn get_p2_walls_remaining(&self, data: &[u8]) -> usize {
        let p1_walls = self.get_p1_walls_remaining(data);
        let walls_used = self.count_walls(data);
        let p1_walls_used = self.max_walls - p1_walls;
        let p2_walls_used = walls_used.saturating_sub(p1_walls_used);
        self.max_walls.saturating_sub(p2_walls_used)
    }

    /// Get the current player (0 or 1)
    pub fn get_current_player(&self, data: &[u8]) -> usize {
        if self.get_bit(data, self.current_player_offset) { 1 } else { 0 }
    }

    /// Set the current player
    pub fn set_current_player(&self, data: &mut [u8], player: usize) {
        debug_assert!(player < 2);
        self.set_bit(data, self.current_player_offset, player == 1);
    }

    /// Get the number of completed steps
    pub fn get_completed_steps(&self, data: &[u8]) -> usize {
        self.get_bits(data, self.steps_offset, self.steps_bits)
    }

    /// Set the number of completed steps
    pub fn set_completed_steps(&self, data: &mut [u8], steps: usize) {
        debug_assert!(steps <= self.max_steps);
        self.set_bits(data, self.steps_offset, self.steps_bits, steps);
    }

    /// Count the total number of walls placed on the board
    pub fn count_walls(&self, data: &[u8]) -> usize {
        let mut count = 0;
        for i in 0..self.num_wall_positions {
            let (row, col, orientation) = self.wall_index_to_position(i);
            if self.get_wall(data, row, col, orientation) {
                count += 1;
            }
        }
        count
    }

    /// Convert a (row, col) position to a flat index
    pub fn position_to_index(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.board_size && col < self.board_size);
        row * self.board_size + col
    }

    /// Convert a flat index to (row, col) position
    pub fn index_to_position(&self, index: usize) -> (usize, usize) {
        debug_assert!(index < self.num_player_positions);
        (index / self.board_size, index % self.board_size)
    }

    /// Get board size
    pub fn board_size(&self) -> usize {
        self.board_size
    }

    /// Get max walls
    pub fn max_walls(&self) -> usize {
        self.max_walls
    }

    /// Get max steps
    pub fn max_steps(&self) -> usize {
        self.max_steps
    }

    /// Convert (row, col, orientation) to wall_index
    /// orientation: 0 = vertical, 1 = horizontal
    pub fn wall_position_to_index(&self, row: usize, col: usize, orientation: usize) -> usize {
        debug_assert!(row < self.board_size - 1);
        debug_assert!(col < self.board_size - 1);
        debug_assert!(orientation < 2);
        orientation * (self.board_size - 1) * (self.board_size - 1)
            + row * (self.board_size - 1)
            + col
    }

    /// Convert wall_index to (row, col, orientation)
    /// Returns (row, col, orientation) where orientation is 0=vertical, 1=horizontal
    pub fn wall_index_to_position(&self, wall_index: usize) -> (usize, usize, usize) {
        debug_assert!(wall_index < self.num_wall_positions);
        let positions_per_orientation = (self.board_size - 1) * (self.board_size - 1);
        let orientation = wall_index / positions_per_orientation;
        let remainder = wall_index % positions_per_orientation;
        let row = remainder / (self.board_size - 1);
        let col = remainder % (self.board_size - 1);
        (row, col, orientation)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_size_calculation() {
        let q = QBitRepr::new(5, 10, 100);
        // 5x5 board: 2*(5-1)*(5-1) = 2*4*4 = 32 wall bits
        // Positions: ceil(log2(25)) = 5 bits each, 2 players = 10 bits
        // Walls remaining: ceil(log2(10)) = 4 bits
        // Current player: 1 bit
        // Steps: ceil(log2(100)) = 7 bits
        // Total: 32 + 10 + 4 + 1 + 7 = 54 bits = 7 bytes
        assert_eq!(q.size_bytes(), 7);
    }

    #[test]
    fn test_player_positions() {
        let q = QBitRepr::new(5, 10, 100);
        let mut data = q.create_data();

        q.set_p1_position(&mut data, q.position_to_index(0, 2));
        q.set_p2_position(&mut data, q.position_to_index(4, 2));

        assert_eq!(q.get_p1_position(&data), 2);
        assert_eq!(q.get_p2_position(&data), 22);
        assert_eq!(q.index_to_position(q.get_p1_position(&data)), (0, 2));
        assert_eq!(q.index_to_position(q.get_p2_position(&data)), (4, 2));
    }

    #[test]
    fn test_walls() {
        let q = QBitRepr::new(5, 10, 100);
        let mut data = q.create_data();

        assert_eq!(q.count_walls(&data), 0);

        // Set vertical wall at (0, 0)
        q.set_wall(&mut data, 0, 0, WALL_VERTICAL, true);
        // Set vertical wall at (1, 1)
        q.set_wall(&mut data, 1, 1, WALL_VERTICAL, true);
        assert_eq!(q.count_walls(&data), 2);
        assert!(q.get_wall(&data, 0, 0, WALL_VERTICAL));
        assert!(q.get_wall(&data, 1, 1, WALL_VERTICAL));
        assert!(!q.get_wall(&data, 0, 1, WALL_VERTICAL));
    }

    #[test]
    fn test_walls_remaining() {
        let q = QBitRepr::new(5, 10, 100);
        let mut data = q.create_data();

        q.set_p1_walls_remaining(&mut data, 10);
        assert_eq!(q.get_p1_walls_remaining(&data), 10);

        // Simulate p1 placing 3 walls, p2 placing 2 walls
        q.set_wall(&mut data, 0, 0, WALL_VERTICAL, true); // vertical wall at (0, 0)
        q.set_wall(&mut data, 0, 1, WALL_VERTICAL, true); // vertical wall at (0, 1)
        q.set_wall(&mut data, 0, 2, WALL_VERTICAL, true); // vertical wall at (0, 2)
        q.set_wall(&mut data, 2, 2, WALL_VERTICAL, true); // vertical wall at (2, 2)
        q.set_wall(&mut data, 2, 3, WALL_VERTICAL, true); // vertical wall at (2, 3)
        q.set_p1_walls_remaining(&mut data, 7);

        assert_eq!(q.get_p1_walls_remaining(&data), 7);
        assert_eq!(q.get_p2_walls_remaining(&data), 8);
    }

    #[test]
    fn test_current_player() {
        let q = QBitRepr::new(5, 10, 100);
        let mut data = q.create_data();

        assert_eq!(q.get_current_player(&data), 0);

        q.set_current_player(&mut data, 1);
        assert_eq!(q.get_current_player(&data), 1);

        q.set_current_player(&mut data, 0);
        assert_eq!(q.get_current_player(&data), 0);
    }

    #[test]
    fn test_completed_steps() {
        let q = QBitRepr::new(5, 10, 100);
        let mut data = q.create_data();

        assert_eq!(q.get_completed_steps(&data), 0);

        q.set_completed_steps(&mut data, 42);
        assert_eq!(q.get_completed_steps(&data), 42);
    }

    #[test]
    fn test_stack_allocation() {
        let q = QBitRepr::new(3, 3, 64);
        let mut data = [0u8; 7]; // Stack-allocated

        q.set_p1_position(&mut data, q.position_to_index(0, 1));
        q.set_p2_position(&mut data, q.position_to_index(2, 1));
        q.set_current_player(&mut data, 1);

        assert_eq!(q.get_p1_position(&data), 1);
        assert_eq!(q.get_p2_position(&data), 7);
        assert_eq!(q.get_current_player(&data), 1);
    }

    #[test]
    fn test_wall_position_conversion() {
        let q = QBitRepr::new(5, 10, 100);

        // Test vertical wall at (1, 2)
        let idx = q.wall_position_to_index(1, 2, WALL_VERTICAL);
        let (row, col, orientation) = q.wall_index_to_position(idx);
        assert_eq!((row, col, orientation), (1, 2, WALL_VERTICAL));

        // Test horizontal wall at (3, 1)
        let idx = q.wall_position_to_index(3, 1, WALL_HORIZONTAL);
        let (row, col, orientation) = q.wall_index_to_position(idx);
        assert_eq!((row, col, orientation), (3, 1, WALL_HORIZONTAL));
    }

    #[test]
    fn test_game_state_conversion() {
        use ndarray::Array1;

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
        use crate::pathfinding::CELL_WALL;
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
        set_wall_cells(&mut grid.view_mut(), 0, 0, WALL_VERTICAL as i32, CELL_WALL); // Vertical wall at (0,0)
        set_wall_cells(&mut grid.view_mut(), 1, 1, WALL_HORIZONTAL as i32, CELL_WALL); // Horizontal wall at (1,1)

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

        // Verify the walls were stored correctly
        assert_eq!(q.count_walls(&data), 2, "Expected 2 walls on board");

        // P2 walls are computed: p1_used=2, total_on_board=2, so p2_used=0, p2_remaining=10
        assert_eq!(extracted_walls[1], 10, "P2 should have 10 walls remaining");

        assert_eq!(q.get_current_player(&data), current_player as usize);
        assert_eq!(q.get_completed_steps(&data), completed_steps as usize);
    }
}
