use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

use crate::actions::{ACTION_MOVE, ACTION_WALL_HORIZONTAL, ACTION_WALL_VERTICAL};
use crate::grid::{set_wall_cells, CELL_FREE, CELL_WALL};

/// Check if a player has won by reaching their goal row.
///
/// This is a direct port of check_win from qgrid.py.
pub fn check_win(player_positions: &ArrayView2<i32>, goal_rows: &ArrayView1<i32>, player: i32) -> bool {
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
    use ndarray::{Array1, Array2};
    use crate::grid::CELL_FREE;
    use crate::pathfinding::CELL_WALL;

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
}
