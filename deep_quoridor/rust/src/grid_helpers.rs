#![allow(dead_code)]

use crate::compact::q_bit_repr::{CompactState, WALL_HORIZONTAL, WALL_VERTICAL};
use crate::compact::q_game_mechanics::QGameMechanics;
use crate::game_state::GameState;
use crate::grid::CELL_WALL;

/// Convert grid-based game state to 5-channel ResNet input format
///
/// ResNet expects input of shape (batch_size, 5, M, M) where M = board_size * 2 + 3
/// For a 5x5 board, M = 13
/// The 5 channels are:
/// 1. Walls (1 where there is a wall, 0 otherwise)
/// 2. Current player's position (1-hot encoding)
/// 3. Opponent's position (1-hot encoding)
/// 4. Current player walls remaining (same value for entire plane)
/// 5. Opponent walls remaining (same value for entire plane)
///
/// The grid is a 2D array where:
/// - Border walls are represented as 10
/// - Player 0's position is represented as 0
/// - Player 1's position is represented as 1
/// - Empty cells are represented as -1
pub fn grid_game_state_to_resnet_input(state: &GameState) -> ndarray::Array4<f32> {
    let grid = state.grid();
    let player_positions = state.player_positions();
    let walls_remaining = state.walls_remaining();
    let current_player = state.current_player;
    let grid_size = grid.ncols();
    let opponent = 1 - current_player;

    let mut input = ndarray::Array4::<f32>::zeros((1, 5, grid_size, grid_size));

    // Channel 0: Walls - extract from grid
    for i in 0..grid_size {
        for j in 0..grid_size {
            if grid[[i, j]] == CELL_WALL {
                input[[0, 0, i, j]] = 1.0;
            }
        }
    }

    // Channel 1: Current player position (1-hot encoding)
    let player_row = player_positions[[current_player as usize, 0]] as usize;
    let player_col = player_positions[[current_player as usize, 1]] as usize;
    let player_grid_row = player_row * 2 + 2;
    let player_grid_col = player_col * 2 + 2;
    input[[0, 1, player_grid_row, player_grid_col]] = 1.0;

    // Channel 2: Opponent position (1-hot encoding)
    let opponent_row = player_positions[[opponent as usize, 0]] as usize;
    let opponent_col = player_positions[[opponent as usize, 1]] as usize;
    let opponent_grid_row = opponent_row * 2 + 2;
    let opponent_grid_col = opponent_col * 2 + 2;
    input[[0, 2, opponent_grid_row, opponent_grid_col]] = 1.0;

    // Channel 3: Current player walls remaining (same value for entire plane)
    let my_walls = walls_remaining[current_player as usize] as f32;
    input.slice_mut(ndarray::s![0, 3, .., ..]).fill(my_walls);

    // Channel 4: Opponent walls remaining (same value for entire plane)
    let opp_walls = walls_remaining[opponent as usize] as f32;
    input.slice_mut(ndarray::s![0, 4, .., ..]).fill(opp_walls);

    input
}

/// Convert a compact game state to 5-channel ResNet input format.
///
/// Produces a tensor bit-identical to `grid_game_state_to_resnet_input` for an
/// equivalent `GameState`. Channels match: walls, current player pos, opponent
/// pos, current player walls broadcast, opponent walls broadcast.
pub fn compact_state_to_resnet_input(
    mechanics: &QGameMechanics,
    data: CompactState,
) -> ndarray::Array4<f32> {
    let repr = mechanics.repr();
    let bs = repr.board_size();
    let grid_size = bs * 2 + 3;
    let current_player = repr.get_current_player(data);
    let opponent = 1 - current_player;

    let mut input = ndarray::Array4::<f32>::zeros((1, 5, grid_size, grid_size));

    // Channel 0: border walls (rows/cols 0,1 and last two)
    for i in 0..grid_size {
        for j in 0..2 {
            input[[0, 0, j, i]] = 1.0;
            input[[0, 0, grid_size - 1 - j, i]] = 1.0;
            input[[0, 0, i, j]] = 1.0;
            input[[0, 0, i, grid_size - 1 - j]] = 1.0;
        }
    }
    // Channel 0: placed walls. Mirrors set_wall_cells layout:
    // Vertical at (r,c) → grid cells (r*2+2, c*2+3), (r*2+3, c*2+3), (r*2+4, c*2+3)
    // Horizontal at (r,c) → grid cells (r*2+3, c*2+2), (r*2+3, c*2+3), (r*2+3, c*2+4)
    for r in 0..bs - 1 {
        for c in 0..bs - 1 {
            if repr.get_wall(data, r, c, WALL_VERTICAL) {
                let gc = c * 2 + 3;
                for dr in 0..3 {
                    input[[0, 0, r * 2 + 2 + dr, gc]] = 1.0;
                }
            }
            if repr.get_wall(data, r, c, WALL_HORIZONTAL) {
                let gr = r * 2 + 3;
                for dc in 0..3 {
                    input[[0, 0, gr, c * 2 + 2 + dc]] = 1.0;
                }
            }
        }
    }

    // Channels 1/2: current/opponent position as 1-hot at (r*2+2, c*2+2).
    let (cr, cc) = repr.get_player_position(data, current_player);
    input[[0, 1, cr * 2 + 2, cc * 2 + 2]] = 1.0;
    let (or_, oc) = repr.get_player_position(data, opponent);
    input[[0, 2, or_ * 2 + 2, oc * 2 + 2]] = 1.0;

    // Channels 3/4: walls remaining broadcast.
    let my_w = repr.get_walls_remaining(data, current_player) as f32;
    let opp_w = repr.get_walls_remaining(data, opponent) as f32;
    input.slice_mut(ndarray::s![0, 3, .., ..]).fill(my_w);
    input.slice_mut(ndarray::s![0, 4, .., ..]).fill(opp_w);

    input
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::ACTION_MOVE;
    use crate::game_state::{GameState, create_initial_state};
    use crate::grid::{CELL_FREE, CELL_WALL, set_wall_cells};
    use ndarray::Array1;

    #[test]
    fn test_compact_resnet_matches_gamestate_resnet_initial() {
        for bs in [3, 5, 9] {
            let state = GameState::new(bs, 3);
            let mech = QGameMechanics::new(bs as usize, 3, 200);
            let data = mech.create_initial_state();
            let a = grid_game_state_to_resnet_input(&state);
            let b = compact_state_to_resnet_input(&mech, data);
            assert_eq!(a, b, "initial state mismatch bs={bs}");
        }
    }

    #[test]
    fn test_compact_resnet_matches_after_stepping() {
        // Run a small deterministic action sequence on both representations.
        // No walls → only move actions, which are trivially valid.
        let bs = 5;
        let mut state = GameState::new(bs, 0);
        let mech = QGameMechanics::new(bs as usize, 0, 200);
        let mut data = mech.create_initial_state();

        let actions: Vec<[i32; 3]> = vec![
            [1, 2, ACTION_MOVE],
            [3, 2, ACTION_MOVE],
            [1, 1, ACTION_MOVE],
            [3, 1, ACTION_MOVE],
        ];

        let a0 = grid_game_state_to_resnet_input(&state);
        let b0 = compact_state_to_resnet_input(&mech, data);
        assert_eq!(a0, b0, "step 0");

        for (i, action) in actions.iter().enumerate() {
            state.step(*action);
            let action_idx = crate::actions::action_to_index(bs, action);
            mech.apply_action_index(&mut data, action_idx);
            let a = grid_game_state_to_resnet_input(&state);
            let b = compact_state_to_resnet_input(&mech, data);
            assert_eq!(a, b, "step {}", i + 1);
        }
    }

    #[test]
    fn test_resnet_input_shape() {
        let state = GameState::new(5, 3);
        let input = grid_game_state_to_resnet_input(&state);
        // Check shape: (1, 5, 13, 13) for 5x5 board
        assert_eq!(input.shape(), &[1, 5, 13, 13]);
    }

    #[test]
    fn test_resnet_input_channel0_walls() {
        let mut state = GameState::new(5, 3);
        // Add a vertical wall at position (0, 0)
        set_wall_cells(&mut state.grid.view_mut(), 0, 0, 0, CELL_WALL);
        let input = grid_game_state_to_resnet_input(&state);

        // Channel 0 should have walls marked
        // Border walls should be present
        assert_eq!(input[[0, 0, 0, 0]], 1.0); // Top-left border
        assert_eq!(input[[0, 0, 0, 12]], 1.0); // Top-right border
        assert_eq!(input[[0, 0, 12, 0]], 1.0); // Bottom-left border
        assert_eq!(input[[0, 0, 12, 12]], 1.0); // Bottom-right border

        // Vertical wall at (0,0) should be marked at grid positions (2,3), (3,3), (4,3)
        assert_eq!(input[[0, 0, 2, 3]], 1.0);
        assert_eq!(input[[0, 0, 3, 3]], 1.0);
        assert_eq!(input[[0, 0, 4, 3]], 1.0);

        // A free cell should be 0.0
        assert_eq!(input[[0, 0, 6, 6]], 0.0);
    }

    #[test]
    fn test_resnet_input_channel1_current_player() {
        let state = GameState::new(5, 3); // current_player = 0 by default
        let input = grid_game_state_to_resnet_input(&state);

        // Channel 1 should have current player (player 0) position marked
        // Player 0 is at board position (0, 2) -> grid position (2, 6)
        assert_eq!(input[[0, 1, 2, 6]], 1.0);

        // All other positions in channel 1 should be 0.0
        assert_eq!(input[[0, 1, 0, 0]], 0.0);
        assert_eq!(input[[0, 1, 10, 6]], 0.0); // Player 1 position should not be marked
    }

    #[test]
    fn test_resnet_input_channel2_opponent() {
        let state = GameState::new(5, 3);
        let input = grid_game_state_to_resnet_input(&state);

        // Channel 2 should have opponent (player 1) position marked
        // Player 1 is at board position (4, 2) -> grid position (10, 6)
        assert_eq!(input[[0, 2, 10, 6]], 1.0);

        // All other positions in channel 2 should be 0.0
        assert_eq!(input[[0, 2, 0, 0]], 0.0);
        assert_eq!(input[[0, 2, 2, 6]], 0.0); // Current player position should not be marked
    }

    #[test]
    fn test_resnet_input_channel3_current_player_walls() {
        let state = GameState::new(5, 3);
        let input = grid_game_state_to_resnet_input(&state);

        // Channel 3 should have current player's walls remaining (3) everywhere
        assert_eq!(input[[0, 3, 0, 0]], 3.0);
        assert_eq!(input[[0, 3, 6, 6]], 3.0);
        assert_eq!(input[[0, 3, 12, 12]], 3.0);
    }

    #[test]
    fn test_resnet_input_channel4_opponent_walls() {
        let state = GameState::new(5, 3);
        let input = grid_game_state_to_resnet_input(&state);

        // Channel 4 should have opponent's walls remaining (3) everywhere
        assert_eq!(input[[0, 4, 0, 0]], 3.0);
        assert_eq!(input[[0, 4, 6, 6]], 3.0);
        assert_eq!(input[[0, 4, 12, 12]], 3.0);
    }

    #[test]
    fn test_resnet_input_player_perspective_swap() {
        let state_p0 = GameState::new(5, 3); // current_player = 0 by default
        let mut state_p1 = GameState::new(5, 3);
        state_p1.current_player = 1;

        let input_p0 = grid_game_state_to_resnet_input(&state_p0);
        let input_p1 = grid_game_state_to_resnet_input(&state_p1);

        // Channel 0 (walls) should be the same
        assert_eq!(input_p0[[0, 0, 2, 6]], input_p1[[0, 0, 2, 6]]);

        // Channel 1 for player 0 should equal channel 2 for player 1 (current player swap)
        // Player 0 position
        assert_eq!(input_p0[[0, 1, 2, 6]], 1.0);
        assert_eq!(input_p1[[0, 2, 2, 6]], 1.0);

        // Channel 2 for player 0 should equal channel 1 for player 1 (opponent swap)
        // Player 1 position
        assert_eq!(input_p0[[0, 2, 10, 6]], 1.0);
        assert_eq!(input_p1[[0, 1, 10, 6]], 1.0);
    }

    #[test]
    fn test_resnet_input_different_walls_remaining() {
        let mut state = GameState::new(5, 3);
        state.walls_remaining = Array1::from(vec![5, 1]); // Different wall counts
        let input = grid_game_state_to_resnet_input(&state);

        // Channel 3 should have current player's walls remaining (5)
        assert_eq!(input[[0, 3, 6, 6]], 5.0);

        // Channel 4 should have opponent's walls remaining (1)
        assert_eq!(input[[0, 4, 6, 6]], 1.0);
    }

    #[test]
    fn test_resnet_input_with_horizontal_wall() {
        let mut state = GameState::new(5, 3);
        // Add a horizontal wall at position (1, 1)
        set_wall_cells(&mut state.grid.view_mut(), 1, 1, 1, CELL_WALL);
        let input = grid_game_state_to_resnet_input(&state);

        // Horizontal wall at (1,1) should be marked at grid positions (5,4), (5,5), (5,6)
        assert_eq!(input[[0, 0, 5, 4]], 1.0);
        assert_eq!(input[[0, 0, 5, 5]], 1.0);
        assert_eq!(input[[0, 0, 5, 6]], 1.0);
    }

    #[test]
    fn test_resnet_input_3x3_board() {
        let state = GameState::new(3, 3);
        let input = grid_game_state_to_resnet_input(&state);

        // Check shape: (1, 5, 9, 9) for 3x3 board
        assert_eq!(input.shape(), &[1, 5, 9, 9]);

        // Player 0 at (0, 1) -> grid (2, 4)
        assert_eq!(input[[0, 1, 2, 4]], 1.0);

        // Player 1 at (2, 1) -> grid (6, 4)
        assert_eq!(input[[0, 2, 6, 4]], 1.0);
    }

    #[test]
    fn test_resnet_input_9x9_board() {
        let state = GameState::new(9, 10);
        let input = grid_game_state_to_resnet_input(&state);

        // Check shape: (1, 5, 21, 21) for 9x9 board
        assert_eq!(input.shape(), &[1, 5, 21, 21]);

        // Player 0 at (0, 4) -> grid (2, 10)
        assert_eq!(input[[0, 1, 2, 10]], 1.0);

        // Player 1 at (8, 4) -> grid (18, 10)
        assert_eq!(input[[0, 2, 18, 10]], 1.0);
    }

    // Tests for create_initial_state

    #[test]
    fn test_create_initial_state_5x5() {
        let (grid, player_positions, walls_remaining, goal_rows) = create_initial_state(5, 3);

        // Check grid size: 13x13 for 5x5 board
        assert_eq!(grid.shape(), &[13, 13]);

        // Check player positions
        assert_eq!(player_positions[[0, 0]], 0); // Player 0 at row 0
        assert_eq!(player_positions[[0, 1]], 2); // Player 0 at col 2 (center)
        assert_eq!(player_positions[[1, 0]], 4); // Player 1 at row 4
        assert_eq!(player_positions[[1, 1]], 2); // Player 1 at col 2 (center)

        // Check walls remaining
        assert_eq!(walls_remaining[0], 3);
        assert_eq!(walls_remaining[1], 3);

        // Check goal rows
        assert_eq!(goal_rows[0], 4); // Player 0 wants to reach row 4
        assert_eq!(goal_rows[1], 0); // Player 1 wants to reach row 0
    }

    #[test]
    fn test_create_initial_state_3x3() {
        let (grid, player_positions, walls_remaining, goal_rows) = create_initial_state(3, 2);

        // Check grid size: 9x9 for 3x3 board
        assert_eq!(grid.shape(), &[9, 9]);

        // Check player positions
        assert_eq!(player_positions[[0, 0]], 0); // Player 0 at row 0
        assert_eq!(player_positions[[0, 1]], 1); // Player 0 at col 1 (center of 3)
        assert_eq!(player_positions[[1, 0]], 2); // Player 1 at row 2
        assert_eq!(player_positions[[1, 1]], 1); // Player 1 at col 1 (center)

        // Check walls remaining
        assert_eq!(walls_remaining[0], 2);
        assert_eq!(walls_remaining[1], 2);

        // Check goal rows
        assert_eq!(goal_rows[0], 2);
        assert_eq!(goal_rows[1], 0);
    }

    #[test]
    fn test_create_initial_state_9x9() {
        let (grid, player_positions, walls_remaining, goal_rows) = create_initial_state(9, 10);

        // Check grid size: 21x21 for 9x9 board
        assert_eq!(grid.shape(), &[21, 21]);

        // Check player positions
        assert_eq!(player_positions[[0, 0]], 0); // Player 0 at row 0
        assert_eq!(player_positions[[0, 1]], 4); // Player 0 at col 4 (center of 9)
        assert_eq!(player_positions[[1, 0]], 8); // Player 1 at row 8
        assert_eq!(player_positions[[1, 1]], 4); // Player 1 at col 4 (center)

        // Check walls remaining
        assert_eq!(walls_remaining[0], 10);
        assert_eq!(walls_remaining[1], 10);

        // Check goal rows
        assert_eq!(goal_rows[0], 8);
        assert_eq!(goal_rows[1], 0);
    }

    #[test]
    fn test_create_initial_state_border_walls() {
        let (grid, _, _, _) = create_initial_state(5, 3);

        // Check that border walls are in place
        // Top border (rows 0-1)
        for j in 0..13 {
            assert_eq!(grid[[0, j]], CELL_WALL, "Top border row 0, col {}", j);
            assert_eq!(grid[[1, j]], CELL_WALL, "Top border row 1, col {}", j);
        }

        // Bottom border (rows 11-12)
        for j in 0..13 {
            assert_eq!(grid[[11, j]], CELL_WALL, "Bottom border row 11, col {}", j);
            assert_eq!(grid[[12, j]], CELL_WALL, "Bottom border row 12, col {}", j);
        }

        // Left border (cols 0-1)
        for i in 0..13 {
            assert_eq!(grid[[i, 0]], CELL_WALL, "Left border row {}, col 0", i);
            assert_eq!(grid[[i, 1]], CELL_WALL, "Left border row {}, col 1", i);
        }

        // Right border (cols 11-12)
        for i in 0..13 {
            assert_eq!(grid[[i, 11]], CELL_WALL, "Right border row {}, col 11", i);
            assert_eq!(grid[[i, 12]], CELL_WALL, "Right border row {}, col 12", i);
        }
    }

    #[test]
    fn test_create_initial_state_players_on_grid() {
        let (grid, player_positions, _, _) = create_initial_state(5, 3);

        // Check that players are placed on grid at correct positions
        let p0_grid_row = (player_positions[[0, 0]] * 2 + 2) as usize;
        let p0_grid_col = (player_positions[[0, 1]] * 2 + 2) as usize;
        assert_eq!(
            grid[[p0_grid_row, p0_grid_col]],
            0,
            "Player 0 should be at grid ({}, {})",
            p0_grid_row,
            p0_grid_col
        );

        let p1_grid_row = (player_positions[[1, 0]] * 2 + 2) as usize;
        let p1_grid_col = (player_positions[[1, 1]] * 2 + 2) as usize;
        assert_eq!(
            grid[[p1_grid_row, p1_grid_col]],
            1,
            "Player 1 should be at grid ({}, {})",
            p1_grid_row,
            p1_grid_col
        );
    }

    #[test]
    fn test_create_initial_state_free_cells() {
        let (grid, _, _, _) = create_initial_state(5, 3);

        // Check that interior cells (not borders or player positions) are free
        // Test a few cells in the interior
        assert_eq!(grid[[4, 4]], CELL_FREE, "Interior cell should be free");
        assert_eq!(grid[[6, 8]], CELL_FREE, "Interior cell should be free");
        assert_eq!(grid[[8, 6]], CELL_FREE, "Interior cell should be free");
    }

    #[test]
    fn test_create_initial_state_different_wall_counts() {
        let (_, _, walls_remaining1, _) = create_initial_state(5, 0);
        assert_eq!(walls_remaining1[0], 0);
        assert_eq!(walls_remaining1[1], 0);

        let (_, _, walls_remaining2, _) = create_initial_state(5, 10);
        assert_eq!(walls_remaining2[0], 10);
        assert_eq!(walls_remaining2[1], 10);

        let (_, _, walls_remaining3, _) = create_initial_state(9, 20);
        assert_eq!(walls_remaining3[0], 20);
        assert_eq!(walls_remaining3[1], 20);
    }

    #[test]
    fn test_create_initial_state_player_at_center_column() {
        // Test various board sizes to ensure players are always at center column
        for board_size in [3, 5, 7, 9].iter() {
            let (_, player_positions, _, _) = create_initial_state(*board_size, 3);
            let expected_center = board_size / 2;

            assert_eq!(
                player_positions[[0, 1]],
                expected_center,
                "Player 0 should be at center column for board size {}",
                board_size
            );
            assert_eq!(
                player_positions[[1, 1]],
                expected_center,
                "Player 1 should be at center column for board size {}",
                board_size
            );
        }
    }
}
