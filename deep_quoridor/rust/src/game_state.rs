#![allow(dead_code)]

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
use std::hash::{Hash, Hasher};

use crate::actions::{
    compute_full_action_mask, policy_size, ACTION_MOVE, ACTION_WALL_HORIZONTAL,
    ACTION_WALL_VERTICAL,
};
use crate::grid::{set_wall_cells, CELL_FREE, CELL_WALL};

/// A complete game state wrapper for Quoridor.
///
/// Bundles all state needed to play a game and provides convenience methods
/// for stepping, checking wins, computing action masks, and hashing.
#[derive(Clone, Debug)]
pub struct GameState {
    pub grid: Array2<i8>,
    pub player_positions: Array2<i32>,
    pub walls_remaining: Array1<i32>,
    pub goal_rows: Array1<i32>,
    pub current_player: i32,
    pub board_size: i32,
    pub completed_steps: usize,
}

impl GameState {
    /// Create a new game state with initial positions.
    pub fn new(board_size: i32, max_walls: i32) -> Self {
        let (grid, player_positions, walls_remaining, goal_rows) =
            create_initial_state(board_size, max_walls);
        Self {
            grid,
            player_positions,
            walls_remaining,
            goal_rows,
            current_player: 0,
            board_size,
            completed_steps: 0,
        }
    }

    /// Apply an action to this game state in place.
    ///
    /// Increments `completed_steps` and swaps `current_player`.
    pub fn step(&mut self, action: [i32; 3]) {
        let action_arr = Array1::from(vec![action[0], action[1], action[2]]);
        apply_action(
            &mut self.grid.view_mut(),
            &mut self.player_positions.view_mut(),
            &mut self.walls_remaining.view_mut(),
            self.current_player,
            &action_arr.view(),
        );
        self.completed_steps += 1;
        self.current_player = 1 - self.current_player;
    }

    /// Clone this state and apply an action to the clone.
    ///
    /// Useful for MCTS lazy expansion.
    pub fn clone_and_step(&self, action: [i32; 3]) -> Self {
        let mut new_state = self.clone();
        new_state.step(action);
        new_state
    }

    /// Check if the game is over (either player has won).
    pub fn is_game_over(&self) -> bool {
        self.check_win(0) || self.check_win(1)
    }

    /// Check if a specific player has won.
    pub fn check_win(&self, player: i32) -> bool {
        check_win(
            &self.player_positions.view(),
            &self.goal_rows.view(),
            player,
        )
    }

    /// Get the winner if the game is over, otherwise None.
    pub fn winner(&self) -> Option<i32> {
        if self.check_win(0) {
            Some(0)
        } else if self.check_win(1) {
            Some(1)
        } else {
            None
        }
    }

    /// Compute the action mask for the current player.
    pub fn get_action_mask(&self) -> Vec<bool> {
        let total_actions = policy_size(self.board_size);
        let mut mask = vec![false; total_actions];
        compute_full_action_mask(
            &self.grid.view(),
            &self.player_positions.view(),
            &self.walls_remaining.view(),
            &self.goal_rows.view(),
            self.current_player,
            &mut mask,
        );
        mask
    }

    /// Compute a fast hash of the game state for visited-state tracking.
    ///
    /// The hash includes: grid, positions, walls remaining, and current player.
    pub fn get_fast_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();

        // Hash grid
        for val in self.grid.iter() {
            val.hash(&mut hasher);
        }

        // Hash positions
        for val in self.player_positions.iter() {
            val.hash(&mut hasher);
        }

        // Hash walls remaining
        for val in self.walls_remaining.iter() {
            val.hash(&mut hasher);
        }

        // Hash current player
        self.current_player.hash(&mut hasher);

        hasher.finish()
    }

    /// Get the policy size for this board.
    pub fn policy_size(&self) -> usize {
        policy_size(self.board_size)
    }

    // Accessor methods returning views
    pub fn grid(&self) -> ArrayView2<i8> {
        self.grid.view()
    }

    pub fn player_positions(&self) -> ArrayView2<i32> {
        self.player_positions.view()
    }

    pub fn walls_remaining(&self) -> ArrayView1<i32> {
        self.walls_remaining.view()
    }

    pub fn goal_rows(&self) -> ArrayView1<i32> {
        self.goal_rows.view()
    }
}

/// Initialize the initial game state for a Quoridor board
///
/// Creates the initial game state for a Quoridor board with:
/// - A grid of size (board_size * 2 + 3) x (board_size * 2 + 3)
/// - Border walls around the perimeter
/// - Players positioned at top and bottom center
/// - Specified number of walls for each player
///
/// # Arguments
/// * `board_size` - The size of the board (e.g., 5 for a 5x5 board, 9 for standard Quoridor)
/// * `max_walls` - Number of walls each player starts with
///
/// # Returns
/// A tuple containing:
/// * `grid` - The game grid with border walls and player positions
/// * `player_positions` - Array of player positions [player_id, [row, col]]
/// * `walls_remaining` - Array of walls remaining for each player
/// * `goal_rows` - Array of goal rows for each player
pub fn create_initial_state(
    board_size: i32,
    max_walls: i32,
) -> (Array2<i8>, Array2<i32>, Array1<i32>, Array1<i32>) {
    let grid_size = (board_size * 2 + 3) as usize;

    let mut grid = Array2::<i8>::from_elem((grid_size, grid_size), CELL_FREE);

    // Add border walls
    for i in 0..2 {
        for j in 0..grid_size {
            grid[[i, j]] = CELL_WALL;
            grid[[grid_size - 1 - i, j]] = CELL_WALL;
            grid[[j, i]] = CELL_WALL;
            grid[[j, grid_size - 1 - i]] = CELL_WALL;
        }
    }

    let mut player_positions = Array2::<i32>::zeros((2, 2));
    let center_col = board_size / 2;

    // Player 0 starts at top center
    player_positions[[0, 0]] = 0;
    player_positions[[0, 1]] = center_col;
    // Player 1 starts at bottom center
    player_positions[[1, 0]] = board_size - 1;
    player_positions[[1, 1]] = center_col;

    // Place players on grid (grid coords are board_coords * 2 + 2)
    let p0_grid_row = (player_positions[[0, 0]] * 2 + 2) as usize;
    let p0_grid_col = (player_positions[[0, 1]] * 2 + 2) as usize;
    let p1_grid_row = (player_positions[[1, 0]] * 2 + 2) as usize;
    let p1_grid_col = (player_positions[[1, 1]] * 2 + 2) as usize;

    grid[[p0_grid_row, p0_grid_col]] = 0;
    grid[[p1_grid_row, p1_grid_col]] = 1;

    let walls_remaining = Array1::from(vec![max_walls, max_walls]);
    let goal_rows = Array1::from(vec![board_size - 1, 0]); // Player 0 wants bottom, Player 1 wants top

    (grid, player_positions, walls_remaining, goal_rows)
}

/// Check if a player has won by reaching their goal row.
///
/// This is a direct port of check_win from qgrid.py.
pub fn check_win(
    player_positions: &ArrayView2<i32>,
    goal_rows: &ArrayView1<i32>,
    player: i32,
) -> bool {
    player_positions[[player as usize, 0]] == goal_rows[player as usize]
}

/// Apply an action to the game state.
///
/// This is a direct port of apply_action from qgrid.py.
/// The action is a 3-element array: [row, col, action_type]
pub fn apply_action(
    grid: &mut ArrayViewMut2<i8>,
    player_positions: &mut ArrayViewMut2<i32>,
    walls_remaining: &mut ArrayViewMut1<i32>,
    current_player: i32,
    action: &ArrayView1<i32>,
) {
    let action_type = action[2];

    if action_type == ACTION_MOVE {
        // Move action - update player position
        let new_row = action[0];
        let new_col = action[1];
        let old_row = player_positions[[current_player as usize, 0]];
        let old_col = player_positions[[current_player as usize, 1]];

        // Update grid
        let old_i = (old_row * 2 + 2) as usize;
        let old_j = (old_col * 2 + 2) as usize;
        let new_i = (new_row * 2 + 2) as usize;
        let new_j = (new_col * 2 + 2) as usize;

        grid[[old_i, old_j]] = CELL_FREE;
        grid[[new_i, new_j]] = current_player as i8;

        // Update player positions
        player_positions[[current_player as usize, 0]] = new_row;
        player_positions[[current_player as usize, 1]] = new_col;
    } else if action_type == ACTION_WALL_VERTICAL {
        // Vertical wall action
        let wall_row = action[0];
        let wall_col = action[1];
        set_wall_cells(grid, wall_row, wall_col, 0, CELL_WALL);
        walls_remaining[current_player as usize] -= 1;
    } else if action_type == ACTION_WALL_HORIZONTAL {
        // Horizontal wall action
        let wall_row = action[0];
        let wall_col = action[1];
        set_wall_cells(grid, wall_row, wall_col, 1, CELL_WALL);
        walls_remaining[current_player as usize] -= 1;
    }
}

/// Undo a previously applied action.
///
/// This is a direct port of undo_action from qgrid.py.
/// The action is a 3-element array: [row, col, action_type]
/// The previous_position is a 2-element array: [old_row, old_col]
pub fn undo_action(
    grid: &mut ArrayViewMut2<i8>,
    player_positions: &mut ArrayViewMut2<i32>,
    walls_remaining: &mut ArrayViewMut1<i32>,
    player_that_took_action: i32,
    action: &ArrayView1<i32>,
    previous_position: &ArrayView1<i32>,
) {
    let action_type = action[2];

    if action_type == ACTION_MOVE {
        // Undo move - restore player position
        let current_row = player_positions[[player_that_took_action as usize, 0]];
        let current_col = player_positions[[player_that_took_action as usize, 1]];
        let previous_row = previous_position[0];
        let previous_col = previous_position[1];

        // Update grid
        let current_i = (current_row * 2 + 2) as usize;
        let current_j = (current_col * 2 + 2) as usize;
        let previous_i = (previous_row * 2 + 2) as usize;
        let previous_j = (previous_col * 2 + 2) as usize;

        grid[[current_i, current_j]] = CELL_FREE;
        grid[[previous_i, previous_j]] = player_that_took_action as i8;

        // Restore player position
        player_positions[[player_that_took_action as usize, 0]] = previous_row;
        player_positions[[player_that_took_action as usize, 1]] = previous_col;
    } else if action_type == ACTION_WALL_VERTICAL {
        // Undo vertical wall
        let wall_row = action[0];
        let wall_col = action[1];
        set_wall_cells(grid, wall_row, wall_col, 0, CELL_FREE);
        walls_remaining[player_that_took_action as usize] += 1;
    } else if action_type == ACTION_WALL_HORIZONTAL {
        // Undo horizontal wall
        let wall_row = action[0];
        let wall_col = action[1];
        set_wall_cells(grid, wall_row, wall_col, 1, CELL_FREE);
        walls_remaining[player_that_took_action as usize] += 1;
    }
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
    fn test_check_win() {
        let (_, player_positions, _, goal_rows) = create_test_game();

        // Players haven't won yet
        assert!(!check_win(&player_positions.view(), &goal_rows.view(), 0));
        assert!(!check_win(&player_positions.view(), &goal_rows.view(), 1));

        // Modify player 0 to be at goal
        let mut positions = player_positions.clone();
        positions[[0, 0]] = 8;
        assert!(check_win(&positions.view(), &goal_rows.view(), 0));
    }

    #[test]
    fn test_apply_and_undo_move() {
        let (mut grid, mut player_positions, mut walls_remaining, _) = create_test_game();

        let action = Array1::from(vec![1, 4, ACTION_MOVE]);
        let previous_position = Array1::from(vec![0, 4]);

        // Apply move
        apply_action(
            &mut grid.view_mut(),
            &mut player_positions.view_mut(),
            &mut walls_remaining.view_mut(),
            0,
            &action.view(),
        );

        // Check player moved
        assert_eq!(player_positions[[0, 0]], 1);
        assert_eq!(player_positions[[0, 1]], 4);

        // Undo move
        undo_action(
            &mut grid.view_mut(),
            &mut player_positions.view_mut(),
            &mut walls_remaining.view_mut(),
            0,
            &action.view(),
            &previous_position.view(),
        );

        // Check player is back
        assert_eq!(player_positions[[0, 0]], 0);
        assert_eq!(player_positions[[0, 1]], 4);
    }

    #[test]
    fn test_apply_and_undo_wall() {
        let (mut grid, mut player_positions, mut walls_remaining, _) = create_test_game();

        let action = Array1::from(vec![0, 0, ACTION_WALL_VERTICAL]);
        let previous_position = Array1::from(vec![0, 0]); // Not used for wall actions

        let initial_walls = walls_remaining[0];

        // Apply wall
        apply_action(
            &mut grid.view_mut(),
            &mut player_positions.view_mut(),
            &mut walls_remaining.view_mut(),
            0,
            &action.view(),
        );

        // Check walls decreased
        assert_eq!(walls_remaining[0], initial_walls - 1);

        // Undo wall
        undo_action(
            &mut grid.view_mut(),
            &mut player_positions.view_mut(),
            &mut walls_remaining.view_mut(),
            0,
            &action.view(),
            &previous_position.view(),
        );

        // Check walls restored
        assert_eq!(walls_remaining[0], initial_walls);
    }

    // ========== GameState wrapper tests ==========

    #[test]
    fn test_game_state_new() {
        let state = GameState::new(5, 3);

        assert_eq!(state.board_size, 5);
        assert_eq!(state.current_player, 0);
        assert_eq!(state.completed_steps, 0);
        assert_eq!(state.walls_remaining[0], 3);
        assert_eq!(state.walls_remaining[1], 3);
        assert_eq!(state.player_positions[[0, 0]], 0); // Player 0 at top
        assert_eq!(state.player_positions[[1, 0]], 4); // Player 1 at bottom
    }

    #[test]
    fn test_game_state_step() {
        let mut state = GameState::new(5, 3);

        // Player 0 starts, move down one row
        state.step([1, 2, ACTION_MOVE]);

        assert_eq!(state.player_positions[[0, 0]], 1);
        assert_eq!(state.player_positions[[0, 1]], 2);
        assert_eq!(state.current_player, 1); // Switched to player 1
        assert_eq!(state.completed_steps, 1);
    }

    #[test]
    fn test_game_state_clone_and_step() {
        let state = GameState::new(5, 3);

        let new_state = state.clone_and_step([1, 2, ACTION_MOVE]);

        // Original state unchanged
        assert_eq!(state.player_positions[[0, 0]], 0);
        assert_eq!(state.current_player, 0);
        assert_eq!(state.completed_steps, 0);

        // New state updated
        assert_eq!(new_state.player_positions[[0, 0]], 1);
        assert_eq!(new_state.current_player, 1);
        assert_eq!(new_state.completed_steps, 1);
    }

    #[test]
    fn test_game_state_check_win() {
        let mut state = GameState::new(5, 0); // No walls for faster game

        // Neither player has won initially
        assert!(!state.check_win(0));
        assert!(!state.check_win(1));
        assert!(!state.is_game_over());

        // Move player 0 to goal (row 4)
        state.player_positions[[0, 0]] = 4;
        assert!(state.check_win(0));
        assert!(!state.check_win(1));
        assert!(state.is_game_over());
        assert_eq!(state.winner(), Some(0));
    }

    #[test]
    fn test_game_state_action_mask() {
        let state = GameState::new(5, 3);
        let mask = state.get_action_mask();

        // Should have correct length
        assert_eq!(mask.len(), state.policy_size());
        assert_eq!(mask.len(), 25 + 2 * 16); // 57 for 5x5 board

        // Should have some valid actions
        assert!(mask.iter().any(|&m| m));
    }

    #[test]
    fn test_game_state_hash_stability() {
        let state1 = GameState::new(5, 3);
        let state2 = GameState::new(5, 3);

        // Same state should produce same hash
        assert_eq!(state1.get_fast_hash(), state2.get_fast_hash());
    }

    #[test]
    fn test_game_state_hash_different_states() {
        let state1 = GameState::new(5, 3);
        let state2 = state1.clone_and_step([1, 2, ACTION_MOVE]);

        // Different states should produce different hashes
        assert_ne!(state1.get_fast_hash(), state2.get_fast_hash());
    }

    #[test]
    fn test_game_state_accessors() {
        let state = GameState::new(5, 3);

        // Test that accessors return valid views
        assert_eq!(state.grid().shape(), &[13, 13]);
        assert_eq!(state.player_positions().shape(), &[2, 2]);
        assert_eq!(state.walls_remaining().len(), 2);
        assert_eq!(state.goal_rows().len(), 2);
    }
}
