#[cfg(feature = "python")]
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
#[cfg(feature = "python")]
use pyo3::prelude::*;

use rusqlite::{params, Connection};
use std::path::Path;

pub mod actions;
pub mod compact;
pub mod game_state;
pub mod grid;
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
        false, // don't enable logging
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

/// Write QBitRepr-based log entries to a SQLite database
#[allow(dead_code)]
pub fn q_log_entries_to_sqlite(
    entries: Vec<compact::q_minimax::MinimaxLogEntry>,
    filename: &str,
    board_size: usize,
    max_steps: usize,
    max_walls: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut conn = Connection::open(Path::new(filename))?;

    // Drop existing tables to avoid schema conflicts
    // This ensures we always have the correct schema for QBitRepr-based data
    conn.execute("DROP TABLE IF EXISTS policy", [])?;
    conn.execute("DROP TABLE IF EXISTS metadata", [])?;

    // Create metadata table for global parameters
    conn.execute(
        "CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value FLOAT NOT NULL
        )",
        [],
    )?;

    // Insert metadata
    conn.execute(
        "INSERT INTO metadata (key, value) VALUES ('board_size', ?1)",
        params![board_size as f32],
    )?;
    conn.execute(
        "INSERT INTO metadata (key, value) VALUES ('max_steps', ?1)",
        params![max_steps as f32],
    )?;
    conn.execute(
        "INSERT INTO metadata (key, value) VALUES ('max_walls', ?1)",
        params![max_walls as f32],
    )?;

    // Create table for policy entries
    conn.execute(
        "CREATE TABLE policy (
            id INTEGER PRIMARY KEY,
            state BLOB NOT NULL,
            agent_player INTEGER NOT NULL,
            num_actions INTEGER NOT NULL,
            actions BLOB NOT NULL,
            action_values BLOB NOT NULL
        )",
        [],
    )?;

    // Create index for fast lookups by state
    conn.execute("CREATE INDEX idx_state ON policy (state)", [])?;

    let num_entries = entries.len();

    // Insert entries in a transaction for better performance
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO policy (state, agent_player, num_actions, actions, action_values)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;

        for entry in entries {
            // State is already packed as Vec<u8>
            let state_blob = entry.data;

            // Flatten actions into a single vector: each action is (row, col, action_type)
            let actions_flat: Vec<usize> = entry
                .actions
                .into_iter()
                .flat_map(|(r, c, t)| vec![r, c, t])
                .collect();
            let actions_blob: Vec<u8> = actions_flat
                .iter()
                .flat_map(|&x| (x as u32).to_le_bytes())
                .collect();

            // Convert values to bytes
            let values_blob: Vec<u8> = entry.values.iter().flat_map(|&x| x.to_le_bytes()).collect();

            let num_actions = entry.values.len() as i32;

            stmt.execute(params![
                state_blob,
                entry.agent_player as i32,
                num_actions,
                actions_blob,
                values_blob,
            ])?;
        }
        // Explicitly drop statement before committing
        drop(stmt);
    }
    tx.commit()?;

    Ok(num_entries)
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
