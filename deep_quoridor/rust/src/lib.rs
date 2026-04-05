#[cfg(feature = "python")]
use numpy::{
    PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
    PyReadwriteArray2,
};
#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod actions;
pub mod compact;
pub mod game_state;
pub mod grid;
pub mod grid_helpers;
mod minimax;
mod pathfinding;
pub mod rotation;
mod validation;

pub mod agents;

#[cfg(test)]
mod python_consistency;

#[cfg(feature = "binary")]
pub mod game_runner;
#[cfg(feature = "binary")]
pub mod replay_writer;
#[cfg(feature = "binary")]
pub mod selfplay_config;

/// Calculate the shortest distance from a position to a target row.
///
/// This is a direct replacement for qgrid.distance_to_row from the Numba implementation.
///
/// # Arguments
/// * `grid` - The game grid as a 2D numpy array
/// * `start_row` - Starting row on the Quoridor board (0-8 for standard board)
/// * `start_col` - Starting column on the Quoridor board (0-8 for standard board)
/// * `target_row` - Target row to reach
///
/// # Returns
/// The minimum number of moves to reach the target row, or -1 if unreachable
#[cfg(feature = "python")]
#[pyfunction]
fn distance_to_row(
    grid: PyReadonlyArray2<i8>,
    start_row: i32,
    start_col: i32,
    target_row: i32,
) -> i32 {
    let grid_view = grid.as_array();
    pathfinding::distance_to_row(&grid_view, start_row, start_col, target_row)
}

/// Check if wall cells are free.
#[cfg(feature = "python")]
#[pyfunction]
fn are_wall_cells_free(
    grid: PyReadonlyArray2<i8>,
    wall_row: i32,
    wall_col: i32,
    wall_orientation: i32,
) -> bool {
    grid::are_wall_cells_free(&grid.as_array(), wall_row, wall_col, wall_orientation)
}

/// Set wall cells to a specific value.
#[cfg(feature = "python")]
#[pyfunction]
fn set_wall_cells(
    mut grid: PyReadwriteArray2<i8>,
    wall_row: i32,
    wall_col: i32,
    wall_orientation: i32,
    cell_value: i8,
) {
    let mut grid_mut = grid.as_array_mut();
    grid::set_wall_cells(
        &mut grid_mut,
        wall_row,
        wall_col,
        wall_orientation,
        cell_value,
    );
}

/// Check if wall cells equal a specific value.
#[cfg(feature = "python")]
#[pyfunction]
fn check_wall_cells(
    grid: PyReadonlyArray2<i8>,
    wall_row: i32,
    wall_col: i32,
    wall_orientation: i32,
    cell_value: i8,
) -> bool {
    grid::check_wall_cells(
        &grid.as_array(),
        wall_row,
        wall_col,
        wall_orientation,
        cell_value,
    )
}

/// Check if a wall could potentially block a player's path.
#[cfg(feature = "python")]
#[pyfunction]
fn is_wall_potential_block(
    grid: PyReadonlyArray2<i8>,
    wall_row: i32,
    wall_col: i32,
    wall_orientation: i32,
) -> bool {
    grid::is_wall_potential_block(&grid.as_array(), wall_row, wall_col, wall_orientation)
}

/// Validate if a move action is legal.
#[cfg(feature = "python")]
#[pyfunction]
fn is_move_action_valid(
    grid: PyReadonlyArray2<i8>,
    player_positions: PyReadonlyArray2<i32>,
    current_player: i32,
    destination_row: i32,
    destination_col: i32,
) -> bool {
    validation::is_move_action_valid(
        &grid.as_array(),
        &player_positions.as_array(),
        current_player,
        destination_row,
        destination_col,
    )
}

/// Validate if a wall placement is legal.
#[cfg(feature = "python")]
#[pyfunction]
fn is_wall_action_valid(
    grid: PyReadonlyArray2<i8>,
    player_positions: PyReadonlyArray2<i32>,
    walls_remaining: PyReadonlyArray1<i32>,
    goal_rows: PyReadonlyArray1<i32>,
    current_player: i32,
    wall_row: i32,
    wall_col: i32,
    wall_orientation: i32,
) -> bool {
    validation::is_wall_action_valid(
        &grid.as_array(),
        &player_positions.as_array(),
        &walls_remaining.as_array(),
        &goal_rows.as_array(),
        current_player,
        wall_row,
        wall_col,
        wall_orientation,
    )
}

/// Compute a mask of valid move actions.
#[cfg(feature = "python")]
#[pyfunction]
fn compute_move_action_mask(
    grid: PyReadonlyArray2<i8>,
    player_positions: PyReadonlyArray2<i32>,
    current_player: i32,
    mut action_mask: PyReadwriteArray1<bool>,
) {
    let mut mask_mut = action_mask.as_array_mut();
    actions::compute_move_action_mask(
        &grid.as_array(),
        &player_positions.as_array(),
        current_player,
        &mut mask_mut,
    );
}

/// Compute a mask of valid wall actions.
#[cfg(feature = "python")]
#[pyfunction]
fn compute_wall_action_mask(
    grid: PyReadonlyArray2<i8>,
    player_positions: PyReadonlyArray2<i32>,
    walls_remaining: PyReadonlyArray1<i32>,
    goal_rows: PyReadonlyArray1<i32>,
    current_player: i32,
    mut action_mask: PyReadwriteArray1<bool>,
) {
    let mut mask_mut = action_mask.as_array_mut();
    actions::compute_wall_action_mask(
        &grid.as_array(),
        &player_positions.as_array(),
        &walls_remaining.as_array(),
        &goal_rows.as_array(),
        current_player,
        &mut mask_mut,
    );
}

/// Get all valid move actions.
#[cfg(feature = "python")]
#[pyfunction]
fn get_valid_move_actions<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<i8>,
    player_positions: PyReadonlyArray2<i32>,
    current_player: i32,
) -> Bound<'py, PyArray2<i32>> {
    let actions = actions::get_valid_move_actions(
        &grid.as_array(),
        &player_positions.as_array(),
        current_player,
    );
    PyArray2::from_owned_array_bound(py, actions)
}

/// Get all valid wall actions.
#[cfg(feature = "python")]
#[pyfunction]
fn get_valid_wall_actions<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<i8>,
    player_positions: PyReadonlyArray2<i32>,
    walls_remaining: PyReadonlyArray1<i32>,
    goal_rows: PyReadonlyArray1<i32>,
    current_player: i32,
) -> Bound<'py, PyArray2<i32>> {
    let actions = actions::get_valid_wall_actions(
        &grid.as_array(),
        &player_positions.as_array(),
        &walls_remaining.as_array(),
        &goal_rows.as_array(),
        current_player,
    );
    PyArray2::from_owned_array_bound(py, actions)
}

/// Check if a player has won.
#[cfg(feature = "python")]
#[pyfunction]
fn check_win(
    player_positions: PyReadonlyArray2<i32>,
    goal_rows: PyReadonlyArray1<i32>,
    player: i32,
) -> bool {
    game_state::check_win(&player_positions.as_array(), &goal_rows.as_array(), player)
}

/// Apply an action to the game state.
#[cfg(feature = "python")]
#[pyfunction]
fn apply_action(
    mut grid: PyReadwriteArray2<i8>,
    mut player_positions: PyReadwriteArray2<i32>,
    mut walls_remaining: PyReadwriteArray1<i32>,
    current_player: i32,
    action: PyReadonlyArray1<i32>,
) {
    let mut grid_mut = grid.as_array_mut();
    let mut positions_mut = player_positions.as_array_mut();
    let mut walls_mut = walls_remaining.as_array_mut();
    game_state::apply_action(
        &mut grid_mut,
        &mut positions_mut,
        &mut walls_mut,
        current_player,
        &action.as_array(),
    );
}

/// Undo a previously applied action.
#[cfg(feature = "python")]
#[pyfunction]
fn undo_action(
    mut grid: PyReadwriteArray2<i8>,
    mut player_positions: PyReadwriteArray2<i32>,
    mut walls_remaining: PyReadwriteArray1<i32>,
    player_that_took_action: i32,
    action: PyReadonlyArray1<i32>,
    previous_position: PyReadonlyArray1<i32>,
) {
    let mut grid_mut = grid.as_array_mut();
    let mut positions_mut = player_positions.as_array_mut();
    let mut walls_mut = walls_remaining.as_array_mut();
    game_state::undo_action(
        &mut grid_mut,
        &mut positions_mut,
        &mut walls_mut,
        player_that_took_action,
        &action.as_array(),
        &previous_position.as_array(),
    );
}

/// Evaluate all actions for the current player using the minimax algorithm.
/// This is parallelized using Rayon for better performance.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (grid, player_positions, walls_remaining, goal_rows, current_player, max_steps, branching_factor, wall_sigma, discount_factor, heuristic))]
fn evaluate_actions<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<i8>,
    player_positions: PyReadonlyArray2<i32>,
    walls_remaining: PyReadonlyArray1<i32>,
    goal_rows: PyReadonlyArray1<i32>,
    current_player: i32,
    max_steps: i32,
    branching_factor: usize,
    wall_sigma: f32,
    discount_factor: f32,
    heuristic: i32,
) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, numpy::PyArray1<f32>>)> {
    let (actions, values) = minimax::evaluate_actions(
        &grid.as_array(),
        &player_positions.as_array(),
        &walls_remaining.as_array(),
        &goal_rows.as_array(),
        current_player,
        max_steps,
        branching_factor,
        wall_sigma,
        discount_factor,
        heuristic,
    );

    Ok((
        PyArray2::from_owned_array_bound(py, actions),
        numpy::PyArray1::from_owned_array_bound(py, values),
    ))
}

/// Evaluate all actions using QBitRepr-based minimax (more efficient).
/// Takes the same inputs as evaluate_actions but converts to QBitRepr internally.
#[cfg(feature = "python")]
#[pyfunction]
fn q_evaluate_actions<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<i8>,
    player_positions: PyReadonlyArray2<i32>,
    walls_remaining: PyReadonlyArray1<i32>,
    _goal_rows: PyReadonlyArray1<i32>,
    current_player: i32,
    completed_steps: i32,
    max_search_depth: usize,
    branching_factor: usize,
    _wall_sigma: f32,
    discount_factor: f32,
    heuristic: i32,
    board_size: usize,
    max_walls: usize,
    max_steps: usize,
) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, numpy::PyArray1<f32>>)> {
    use compact::q_game_mechanics::QGameMechanics;

    let mechanics = QGameMechanics::new(board_size, max_walls, max_steps);

    // Convert game state to QBitRepr format
    let mut data = mechanics.repr().create_data();
    mechanics.repr().from_game_state(
        &mut data,
        &grid.as_array(),
        &player_positions.as_array(),
        &walls_remaining.as_array(),
        current_player,
        completed_steps,
    );

    // Evaluate actions using QBitRepr minimax
    let (actions, values, _logs) = compact::q_minimax::evaluate_actions(
        &mechanics,
        &data,
        max_search_depth,
        branching_factor,
        discount_factor,
        heuristic,
    );

    // Convert actions back to numpy format
    // Actions are (row, col, action_type) where action_type: 0=vert wall, 1=horiz wall, 2=move
    let num_actions = actions.len();
    let mut actions_array = ndarray::Array2::<i32>::zeros((num_actions, 3));
    for (i, (row, col, action_type)) in actions.iter().enumerate() {
        actions_array[[i, 0]] = *row as i32;
        actions_array[[i, 1]] = *col as i32;
        actions_array[[i, 2]] = *action_type as i32;
    }

    let values_array = ndarray::Array1::from(values);

    Ok((
        PyArray2::from_owned_array_bound(py, actions_array),
        numpy::PyArray1::from_owned_array_bound(py, values_array),
    ))
}

/// Look up a game state in a pre-computed policy database.
///
/// Enumerates all valid actions, computes each child state, queries the DB
/// for each child's value, and returns (actions, values) from the current
/// player's perspective.
#[cfg(feature = "python")]
#[pyfunction]
fn policy_db_lookup<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<i8>,
    player_positions: PyReadonlyArray2<i32>,
    walls_remaining: PyReadonlyArray1<i32>,
    current_player: i32,
    completed_steps: i32,
    board_size: usize,
    max_walls: usize,
    max_steps: usize,
    db_path: &str,
) -> PyResult<Option<(Bound<'py, PyArray2<i32>>, Bound<'py, numpy::PyArray1<i32>>)>> {
    use compact::policy_db::PolicyDb;
    use compact::q_game_mechanics::QGameMechanics;

    let mechanics = QGameMechanics::new(board_size, max_walls, max_steps);

    let mut data = mechanics.repr().create_data();
    mechanics.repr().from_game_state(
        &mut data,
        &grid.as_array(),
        &player_positions.as_array(),
        &walls_remaining.as_array(),
        current_player,
        completed_steps,
    );

    let db = PolicyDb::open(db_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to open DB: {e}")))?;

    match db
        .lookup_action_values(&mechanics, &data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("DB query error: {e}")))?
    {
        None => Ok(None),
        Some((actions, values)) => {
            let n = actions.len();
            let mut actions_array = ndarray::Array2::<i32>::zeros((n, 3));
            let mut values_array = ndarray::Array1::<i32>::zeros(n);
            for (i, &(row, col, action_type)) in actions.iter().enumerate() {
                actions_array[[i, 0]] = row as i32;
                actions_array[[i, 1]] = col as i32;
                actions_array[[i, 2]] = action_type as i32;
                values_array[i] = values[i];
            }
            Ok(Some((
                PyArray2::from_owned_array_bound(py, actions_array),
                numpy::PyArray1::from_owned_array_bound(py, values_array),
            )))
        }
    }
}

/// Convert a compact state blob to full game state arrays.
///
/// Returns (grid, player_positions, walls_remaining, old_style_walls, current_player, completed_steps).
#[cfg(feature = "python")]
#[pyfunction]
fn compact_state_to_game_state<'py>(
    py: Python<'py>,
    state: &[u8],
    board_size: usize,
    max_walls: usize,
    max_steps: usize,
) -> (
    Bound<'py, PyArray2<i8>>,
    Bound<'py, PyArray2<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray3<i8>>,
    i32,
    i32,
) {
    use compact::q_bit_repr::QBitRepr;

    let repr = QBitRepr::new(board_size, max_walls, max_steps);

    let grid = repr.to_grid(state);
    let player_positions = repr.to_player_positions(state);
    let walls_remaining = repr.to_walls_remaining(state);
    let current_player = repr.get_current_player(state) as i32;
    let completed_steps = repr.get_completed_steps(state) as i32;

    // Build old_style_walls: (board_size-1, board_size-1, 2) array where
    // [:,:,0] = vertical walls, [:,:,1] = horizontal walls.
    let b = board_size - 1;
    let mut old_style_walls = ndarray::Array3::<i8>::zeros((b, b, 2));
    for row in 0..b {
        for col in 0..b {
            if repr.get_wall(state, row, col, 0) {
                old_style_walls[[row, col, 0]] = 1;
            }
            if repr.get_wall(state, row, col, 1) {
                old_style_walls[[row, col, 1]] = 1;
            }
        }
    }

    (
        PyArray2::from_owned_array_bound(py, grid),
        PyArray2::from_owned_array_bound(py, player_positions),
        PyArray1::from_owned_array_bound(py, walls_remaining),
        PyArray3::from_owned_array_bound(py, old_style_walls),
        current_player,
        completed_steps,
    )
}

/// Return all child states reachable from a compact state.
///
/// Returns a list of (row, col, action_type, child_state_bytes) tuples where
/// action_type is 0=vertical wall, 1=horizontal wall, 2=pawn move.
#[cfg(feature = "python")]
#[pyfunction]
fn get_compact_child_states(
    state: &[u8],
    board_size: usize,
    max_walls: usize,
    max_steps: usize,
) -> Vec<(usize, usize, usize, Vec<u8>)> {
    use compact::q_game_mechanics::QGameMechanics;

    let mechanics = QGameMechanics::new(board_size, max_walls, max_steps);
    let current_player = mechanics.repr().get_current_player(state);
    let mut data = state.to_vec();

    let mut children = Vec::new();

    // Pawn moves (action_type = 2)
    let moves = mechanics.get_valid_moves(&data);
    for (row, col) in moves {
        let mut child = data.clone();
        mechanics.execute_move(&mut child, current_player, row, col);
        mechanics.switch_player(&mut child);
        children.push((row, col, 2usize, child));
    }

    // Wall placements (action_type = 0 or 1)
    let wall_placements = mechanics.get_valid_wall_placements(&mut data);
    for (row, col, orientation) in wall_placements {
        let mut child = data.clone();
        mechanics.execute_wall_placement(&mut child, current_player, row, col, orientation);
        mechanics.switch_player(&mut child);
        children.push((row, col, orientation, child));
    }

    children
}

/// A Python module implemented in Rust.
#[cfg(feature = "python")]
#[pymodule]
fn quoridor_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core functions
    m.add_function(wrap_pyfunction!(distance_to_row, m)?)?;

    // Grid functions
    m.add_function(wrap_pyfunction!(are_wall_cells_free, m)?)?;
    m.add_function(wrap_pyfunction!(set_wall_cells, m)?)?;
    m.add_function(wrap_pyfunction!(check_wall_cells, m)?)?;
    m.add_function(wrap_pyfunction!(is_wall_potential_block, m)?)?;

    // Validation functions
    m.add_function(wrap_pyfunction!(is_move_action_valid, m)?)?;
    m.add_function(wrap_pyfunction!(is_wall_action_valid, m)?)?;

    // Action functions
    m.add_function(wrap_pyfunction!(compute_move_action_mask, m)?)?;
    m.add_function(wrap_pyfunction!(compute_wall_action_mask, m)?)?;
    m.add_function(wrap_pyfunction!(get_valid_move_actions, m)?)?;
    m.add_function(wrap_pyfunction!(get_valid_wall_actions, m)?)?;

    // Game state functions
    m.add_function(wrap_pyfunction!(check_win, m)?)?;
    m.add_function(wrap_pyfunction!(apply_action, m)?)?;
    m.add_function(wrap_pyfunction!(undo_action, m)?)?;

    // Minimax evaluation
    m.add_function(wrap_pyfunction!(evaluate_actions, m)?)?;
    m.add_function(wrap_pyfunction!(q_evaluate_actions, m)?)?;

    // Policy DB lookup
    m.add_function(wrap_pyfunction!(policy_db_lookup, m)?)?;

    // Compact state utilities for training
    m.add_function(wrap_pyfunction!(compact_state_to_game_state, m)?)?;
    m.add_function(wrap_pyfunction!(get_compact_child_states, m)?)?;

    // Export constants to match qgrid.py
    m.add("CELL_FREE", grid::CELL_FREE)?;
    m.add("CELL_PLAYER1", grid::CELL_PLAYER1)?;
    m.add("CELL_PLAYER2", grid::CELL_PLAYER2)?;
    m.add("CELL_WALL", grid::CELL_WALL)?;

    // Wall orientations
    m.add("WALL_ORIENTATION_VERTICAL", 0)?;
    m.add("WALL_ORIENTATION_HORIZONTAL", 1)?;

    // Action types
    m.add("ACTION_WALL_VERTICAL", 0)?;
    m.add("ACTION_WALL_HORIZONTAL", 1)?;
    m.add("ACTION_MOVE", 2)?;

    Ok(())
}
