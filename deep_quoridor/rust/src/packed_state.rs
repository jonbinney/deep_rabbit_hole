/// Compact bit-packed representation of a Quoridor game state.
///
/// This module provides a space-efficient representation of game states using bit packing.
/// The representation is templated on board_size, max_walls, and max_steps.
///
/// Bit layout:
/// - Wall positions: 2 * (BOARD_SIZE - 1)^2 bits (1 bit per possible wall)
/// - Player 1 position: ceil(log2(BOARD_SIZE^2)) bits
/// - Player 2 position: ceil(log2(BOARD_SIZE^2)) bits
/// - Player 1 walls remaining: ceil(log2(MAX_WALLS + 1)) bits
/// - Current player: 1 bit
/// - Completed steps: ceil(log2(MAX_STEPS + 1)) bits

use std::fmt;

/// Calculate the number of bits needed to represent a value up to max (inclusive)
const fn bits_needed(max: usize) -> usize {
    if max == 0 {
        1
    } else {
        (usize::BITS - max.leading_zeros()) as usize
    }
}

/// Calculate total number of bytes needed to store the packed state
const fn bytes_needed<const BOARD_SIZE: usize, const MAX_WALLS: usize, const MAX_STEPS: usize>() -> usize {
    let num_wall_positions = 2 * (BOARD_SIZE - 1) * (BOARD_SIZE - 1);
    let num_player_positions = BOARD_SIZE * BOARD_SIZE;
    let position_bits = bits_needed(num_player_positions - 1);
    let walls_remaining_bits = bits_needed(MAX_WALLS);
    let steps_bits = bits_needed(MAX_STEPS);

    let total_bits = num_wall_positions + // walls
                     2 * position_bits +   // player positions
                     walls_remaining_bits + // p1 walls remaining
                     1 +                    // current player
                     steps_bits;            // completed steps

    (total_bits + 7) / 8 // Round up to nearest byte
}

/// Compact bit-packed representation of a Quoridor game state.
/// The SIZE parameter must equal `bytes_needed::<BOARD_SIZE, MAX_WALLS, MAX_STEPS>()`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PackedState<const BOARD_SIZE: usize, const MAX_WALLS: usize, const MAX_STEPS: usize, const SIZE: usize> {
    data: [u8; SIZE],
}

impl<const BOARD_SIZE: usize, const MAX_WALLS: usize, const MAX_STEPS: usize, const SIZE: usize>
    PackedState<BOARD_SIZE, MAX_WALLS, MAX_STEPS, SIZE>
{
    /// Number of possible wall positions (horizontal + vertical)
    const NUM_WALL_POSITIONS: usize = 2 * (BOARD_SIZE - 1) * (BOARD_SIZE - 1);

    /// Number of possible player positions
    const NUM_PLAYER_POSITIONS: usize = BOARD_SIZE * BOARD_SIZE;

    /// Bits needed for player position
    const POSITION_BITS: usize = bits_needed(Self::NUM_PLAYER_POSITIONS - 1);

    /// Bits needed for walls remaining
    const WALLS_REMAINING_BITS: usize = bits_needed(MAX_WALLS);

    /// Bits needed for completed steps
    const STEPS_BITS: usize = bits_needed(MAX_STEPS);

    /// Total bits used
    const TOTAL_BITS: usize = Self::NUM_WALL_POSITIONS +
                               2 * Self::POSITION_BITS +
                               Self::WALLS_REMAINING_BITS +
                               1 +
                               Self::STEPS_BITS;

    /// Total bytes needed
    const TOTAL_BYTES: usize = (Self::TOTAL_BITS + 7) / 8;

    /// Bit offsets for each field
    const WALLS_OFFSET: usize = 0;
    const P1_POS_OFFSET: usize = Self::WALLS_OFFSET + Self::NUM_WALL_POSITIONS;
    const P2_POS_OFFSET: usize = Self::P1_POS_OFFSET + Self::POSITION_BITS;
    const WALLS_REMAINING_OFFSET: usize = Self::P2_POS_OFFSET + Self::POSITION_BITS;
    const CURRENT_PLAYER_OFFSET: usize = Self::WALLS_REMAINING_OFFSET + Self::WALLS_REMAINING_BITS;
    const STEPS_OFFSET: usize = Self::CURRENT_PLAYER_OFFSET + 1;

    /// Create a new packed state with all fields set to zero
    pub fn new() -> Self {
        // Compile-time assertion that SIZE matches the calculated size
        const fn _assert_size_matches<const BOARD_SIZE: usize, const MAX_WALLS: usize, const MAX_STEPS: usize, const SIZE: usize>() {
            let calculated = bytes_needed::<BOARD_SIZE, MAX_WALLS, MAX_STEPS>();
            assert!(SIZE == calculated, "SIZE parameter must match calculated size");
        }
        _assert_size_matches::<BOARD_SIZE, MAX_WALLS, MAX_STEPS, SIZE>();

        Self {
            data: [0u8; SIZE],
        }
    }

    /// Get a bit at the specified position
    #[inline]
    fn get_bit(&self, bit_index: usize) -> bool {
        let byte_index = bit_index / 8;
        let bit_offset = bit_index % 8;
        (self.data[byte_index] >> bit_offset) & 1 == 1
    }

    /// Set a bit at the specified position
    #[inline]
    fn set_bit(&mut self, bit_index: usize, value: bool) {
        let byte_index = bit_index / 8;
        let bit_offset = bit_index % 8;
        if value {
            self.data[byte_index] |= 1 << bit_offset;
        } else {
            self.data[byte_index] &= !(1 << bit_offset);
        }
    }

    /// Get an integer value starting at bit_offset with num_bits bits
    #[inline]
    fn get_bits(&self, bit_offset: usize, num_bits: usize) -> usize {
        let mut value = 0usize;
        for i in 0..num_bits {
            if self.get_bit(bit_offset + i) {
                value |= 1 << i;
            }
        }
        value
    }

    /// Set an integer value starting at bit_offset with num_bits bits
    #[inline]
    fn set_bits(&mut self, bit_offset: usize, num_bits: usize, value: usize) {
        for i in 0..num_bits {
            self.set_bit(bit_offset + i, (value >> i) & 1 == 1);
        }
    }

    /// Check if a wall is present at the given position
    pub fn get_wall(&self, wall_index: usize) -> bool {
        debug_assert!(wall_index < Self::NUM_WALL_POSITIONS);
        self.get_bit(Self::WALLS_OFFSET + wall_index)
    }

    /// Set a wall at the given position
    pub fn set_wall(&mut self, wall_index: usize, present: bool) {
        debug_assert!(wall_index < Self::NUM_WALL_POSITIONS);
        self.set_bit(Self::WALLS_OFFSET + wall_index, present);
    }

    /// Get player 1's position (as a flat index from 0 to BOARD_SIZE^2 - 1)
    pub fn get_p1_position(&self) -> usize {
        self.get_bits(Self::P1_POS_OFFSET, Self::POSITION_BITS)
    }

    /// Set player 1's position
    pub fn set_p1_position(&mut self, pos: usize) {
        debug_assert!(pos < Self::NUM_PLAYER_POSITIONS);
        self.set_bits(Self::P1_POS_OFFSET, Self::POSITION_BITS, pos);
    }

    /// Get player 2's position (as a flat index from 0 to BOARD_SIZE^2 - 1)
    pub fn get_p2_position(&self) -> usize {
        self.get_bits(Self::P2_POS_OFFSET, Self::POSITION_BITS)
    }

    /// Set player 2's position
    pub fn set_p2_position(&mut self, pos: usize) {
        debug_assert!(pos < Self::NUM_PLAYER_POSITIONS);
        self.set_bits(Self::P2_POS_OFFSET, Self::POSITION_BITS, pos);
    }

    /// Get player 1's remaining walls
    pub fn get_p1_walls_remaining(&self) -> usize {
        self.get_bits(Self::WALLS_REMAINING_OFFSET, Self::WALLS_REMAINING_BITS)
    }

    /// Set player 1's remaining walls
    pub fn set_p1_walls_remaining(&mut self, walls: usize) {
        debug_assert!(walls <= MAX_WALLS);
        self.set_bits(Self::WALLS_REMAINING_OFFSET, Self::WALLS_REMAINING_BITS, walls);
    }

    /// Get player 2's remaining walls (computed from total walls used)
    pub fn get_p2_walls_remaining(&self) -> usize {
        let p1_walls = self.get_p1_walls_remaining();
        let walls_used = self.count_walls();
        let p1_walls_used = MAX_WALLS - p1_walls;
        let p2_walls_used = walls_used - p1_walls_used;
        MAX_WALLS - p2_walls_used
    }

    /// Get the current player (0 or 1)
    pub fn get_current_player(&self) -> usize {
        if self.get_bit(Self::CURRENT_PLAYER_OFFSET) { 1 } else { 0 }
    }

    /// Set the current player
    pub fn set_current_player(&mut self, player: usize) {
        debug_assert!(player < 2);
        self.set_bit(Self::CURRENT_PLAYER_OFFSET, player == 1);
    }

    /// Get the number of completed steps
    pub fn get_completed_steps(&self) -> usize {
        self.get_bits(Self::STEPS_OFFSET, Self::STEPS_BITS)
    }

    /// Set the number of completed steps
    pub fn set_completed_steps(&mut self, steps: usize) {
        debug_assert!(steps <= MAX_STEPS);
        self.set_bits(Self::STEPS_OFFSET, Self::STEPS_BITS, steps);
    }

    /// Count the total number of walls placed on the board
    pub fn count_walls(&self) -> usize {
        let mut count = 0;
        for i in 0..Self::NUM_WALL_POSITIONS {
            if self.get_wall(i) {
                count += 1;
            }
        }
        count
    }

    /// Convert a (row, col) position to a flat index
    pub fn position_to_index(row: usize, col: usize) -> usize {
        debug_assert!(row < BOARD_SIZE && col < BOARD_SIZE);
        row * BOARD_SIZE + col
    }

    /// Convert a flat index to (row, col) position
    pub fn index_to_position(index: usize) -> (usize, usize) {
        debug_assert!(index < Self::NUM_PLAYER_POSITIONS);
        (index / BOARD_SIZE, index % BOARD_SIZE)
    }

    /// Get the size of the packed representation in bytes
    pub const fn size_bytes() -> usize {
        SIZE
    }

    /// Get the underlying byte array
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

impl<const BOARD_SIZE: usize, const MAX_WALLS: usize, const MAX_STEPS: usize, const SIZE: usize> Default
    for PackedState<BOARD_SIZE, MAX_WALLS, MAX_STEPS, SIZE>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const BOARD_SIZE: usize, const MAX_WALLS: usize, const MAX_STEPS: usize, const SIZE: usize> fmt::Debug
    for PackedState<BOARD_SIZE, MAX_WALLS, MAX_STEPS, SIZE>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PackedState")
            .field("board_size", &BOARD_SIZE)
            .field("max_walls", &MAX_WALLS)
            .field("max_steps", &MAX_STEPS)
            .field("size_bytes", &SIZE)
            .field("p1_pos", &Self::index_to_position(self.get_p1_position()))
            .field("p2_pos", &Self::index_to_position(self.get_p2_position()))
            .field("p1_walls_remaining", &self.get_p1_walls_remaining())
            .field("p2_walls_remaining", &self.get_p2_walls_remaining())
            .field("current_player", &self.get_current_player())
            .field("completed_steps", &self.get_completed_steps())
            .field("walls_placed", &self.count_walls())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_calculation() {
        type State5x5 = PackedState<5, 10, 100, 7>;
        // 5x5 board: 2*(5-1)*(5-1) = 2*4*4 = 32 wall bits
        // Positions: ceil(log2(25)) = 5 bits each, 2 players = 10 bits
        // Walls remaining: ceil(log2(10)) = 4 bits
        // Current player: 1 bit
        // Steps: ceil(log2(100)) = 7 bits
        // Total: 32 + 10 + 4 + 1 + 7 = 54 bits = 7 bytes
        assert_eq!(State5x5::size_bytes(), 7);
        assert_eq!(State5x5::TOTAL_BYTES, 7);
    }

    #[test]
    fn test_player_positions() {
        type State = PackedState<5, 10, 100, 7>;
        let mut state = State::new();
        state.set_p1_position(State::position_to_index(0, 2));
        state.set_p2_position(State::position_to_index(4, 2));

        assert_eq!(state.get_p1_position(), 2);
        assert_eq!(state.get_p2_position(), 22);
        assert_eq!(State::index_to_position(state.get_p1_position()), (0, 2));
        assert_eq!(State::index_to_position(state.get_p2_position()), (4, 2));
    }

    #[test]
    fn test_walls() {
        let mut state = PackedState::<5, 10, 100, 7>::new();
        assert_eq!(state.count_walls(), 0);

        state.set_wall(0, true);
        state.set_wall(5, true);
        assert_eq!(state.count_walls(), 2);
        assert!(state.get_wall(0));
        assert!(state.get_wall(5));
        assert!(!state.get_wall(1));
    }

    #[test]
    fn test_walls_remaining() {
        let mut state = PackedState::<5, 10, 100, 7>::new();
        state.set_p1_walls_remaining(10);
        assert_eq!(state.get_p1_walls_remaining(), 10);

        // Simulate p1 placing 3 walls, p2 placing 2 walls
        state.set_wall(0, true);
        state.set_wall(1, true);
        state.set_wall(2, true);
        state.set_wall(10, true);
        state.set_wall(11, true);
        state.set_p1_walls_remaining(7);

        assert_eq!(state.get_p1_walls_remaining(), 7);
        assert_eq!(state.get_p2_walls_remaining(), 8);
    }

    #[test]
    fn test_current_player() {
        let mut state = PackedState::<5, 10, 100, 7>::new();
        assert_eq!(state.get_current_player(), 0);

        state.set_current_player(1);
        assert_eq!(state.get_current_player(), 1);

        state.set_current_player(0);
        assert_eq!(state.get_current_player(), 0);
    }

    #[test]
    fn test_completed_steps() {
        let mut state = PackedState::<5, 10, 100, 7>::new();
        assert_eq!(state.get_completed_steps(), 0);

        state.set_completed_steps(42);
        assert_eq!(state.get_completed_steps(), 42);
    }
}
