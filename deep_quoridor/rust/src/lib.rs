#[cfg(feature = "python")]
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod actions;
pub mod compact;
pub mod game_state;
pub mod grid;
pub mod grid_helpers;
mod minimax;
mod pathfinding;
mod validation;

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
    // Actions are (row, col, action_type) where action_type: 0=move, 1=vert wall, 2=horiz wall
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
/// The DB stores only (state, value). This function enumerates all valid
/// actions, computes each child state, queries the DB for each child's value,
/// and returns (actions, values) from the current player's perspective.
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
) -> PyResult<Option<(Bound<'py, PyArray2<i32>>, Bound<'py, numpy::PyArray1<f32>>)>> {
    use compact::q_game_mechanics::QGameMechanics;
    use rusqlite::Connection;
    use std::path::Path;

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
    mechanics.print(&data);
    println!(" ");

    let cp = current_player as usize;

    // Get all valid actions for current player
    let moves = mechanics.get_valid_moves(&mut data);
    let walls = mechanics.get_valid_wall_placements(&mut data);

    let mut actions: Vec<(u8, u8, u8)> = moves
        .into_iter()
        .map(|(r, c)| (r as u8, c as u8, 2))
        .collect();
    actions.extend(
        walls
            .into_iter()
            .map(|(r, c, t)| (r as u8, c as u8, t as u8)),
    );

    if actions.is_empty() {
        return Ok(None);
    }

    // Open the database
    let conn = Connection::open_with_flags(
        Path::new(db_path),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to open DB: {e}")))?;

    let mut stmt = conn
        .prepare_cached("SELECT value FROM policy WHERE state = ?1")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("SQL prepare: {e}")))?;

    let n = actions.len();
    let mut actions_array = ndarray::Array2::<i32>::zeros((n, 3));
    let mut values_array = ndarray::Array1::<f32>::zeros(n);
    let mut any_found = false;

    for (i, &(row, col, action_type)) in actions.iter().enumerate() {
        actions_array[[i, 0]] = row as i32;
        actions_array[[i, 1]] = col as i32;
        actions_array[[i, 2]] = action_type as i32;

        // Create child state
        let mut child_data = data.clone();
        let (r, c, t) = (row as usize, col as usize, action_type as usize);
        if action_type == 2 {
            mechanics.execute_move(&mut child_data, cp, r, c);
        } else {
            mechanics.execute_wall_placement(&mut child_data, cp, r, c, t);
        }
        mechanics.switch_player(&mut child_data);

        // Query DB for child state value
        let result: Result<i32, _> =
            stmt.query_row(rusqlite::params![child_data], |row| row.get(0));

        match result {
            Ok(child_value_p0) => {
                any_found = true;
                // DB stores values from player 0's perspective.
                // Convert to current player's perspective, then negate
                // (child is opponent's turn, so child value is opponent's).
                let child_value_cur = if current_player == 0 {
                    child_value_p0
                } else {
                    -child_value_p0
                };
                // Negate: child value is from opponent's perspective
                values_array[i] = -child_value_cur as f32;
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                // Child state not in DB; treat as unknown (0)
                values_array[i] = 0.0;
            }
            Err(e) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "DB query error: {e}"
                )));
            }
        }
    }

    if !any_found {
        return Ok(None);
    }

    Ok(Some((
        PyArray2::from_owned_array_bound(py, actions_array),
        numpy::PyArray1::from_owned_array_bound(py, values_array),
    )))
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
