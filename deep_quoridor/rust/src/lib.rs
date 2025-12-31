use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;
use std::path::Path;

use rusqlite::{Connection, params};

mod actions;
mod game_state;
mod grid;
mod minimax;
mod q_bit_repr;
mod q_bit_repr_conversions;
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
#[pyfunction]
fn set_wall_cells(
    mut grid: PyReadwriteArray2<i8>,
    wall_row: i32,
    wall_col: i32,
    wall_orientation: i32,
    cell_value: i8,
) {
    let mut grid_mut = grid.as_array_mut();
    grid::set_wall_cells(&mut grid_mut, wall_row, wall_col, wall_orientation, cell_value);
}

/// Check if wall cells equal a specific value.
#[pyfunction]
fn check_wall_cells(
    grid: PyReadonlyArray2<i8>,
    wall_row: i32,
    wall_col: i32,
    wall_orientation: i32,
    cell_value: i8,
) -> bool {
    grid::check_wall_cells(&grid.as_array(), wall_row, wall_col, wall_orientation, cell_value)
}

/// Check if a wall could potentially block a player's path.
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
#[pyfunction]
fn get_valid_move_actions<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<i8>,
    player_positions: PyReadonlyArray2<i32>,
    current_player: i32,
) -> Bound<'py, PyArray2<i32>> {
    let actions =
        actions::get_valid_move_actions(&grid.as_array(), &player_positions.as_array(), current_player);
    PyArray2::from_owned_array_bound(py, actions)
}

/// Get all valid wall actions.
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
#[pyfunction]
fn check_win(player_positions: PyReadonlyArray2<i32>, goal_rows: PyReadonlyArray1<i32>, player: i32) -> bool {
    game_state::check_win(&player_positions.as_array(), &goal_rows.as_array(), player)
}

/// Apply an action to the game state.
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
    let (actions, values, log_entries) = minimax::evaluate_actions(
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
        false, // don't enable logging of the policy
    );

    Ok((
        PyArray2::from_owned_array_bound(py, actions),
        numpy::PyArray1::from_owned_array_bound(py, values),
    ))
}

/// Write log entries to a SQLite database
fn log_entries_to_sqlite(
    entries: Vec<minimax::MinimaxLogEntry>,
    filename: &str,
    board_size: i32,
    max_steps: i32,
    max_walls: i32,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut conn = Connection::open(Path::new(filename))?;

    // Create metadata table for global parameters
    conn.execute(
        "CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value FLOAT NOT NULL
        )",
        [],
    )?;

    // Insert metadata
    conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('board_size', ?1)", params![board_size as f32])?;
    conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('max_steps', ?1)", params![max_steps as f32])?;
    conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('max_walls', ?1)", params![max_walls as f32])?;

    // Create table with columns for grid (as blob), walls remaining, completed_steps, actions (as blob), and action_values (as blob)
    // Note: grid is trimmed to exclude 2 outermost rows/cols on each side
    conn.execute(
        "CREATE TABLE IF NOT EXISTS policy (
            id INTEGER PRIMARY KEY,
            grid BLOB NOT NULL,
            current_player INTEGER NOT NULL,
            walls_p1 INTEGER NOT NULL,
            walls_p2 INTEGER NOT NULL,
            agent_player INTEGER NOT NULL,
            completed_steps INTEGER NOT NULL,
            num_actions INTEGER NOT NULL,
            actions BLOB NOT NULL,
            action_values BLOB NOT NULL
        )",
        [],
    )?;

    // Create indices for fast lookups by grid, walls, completed_steps, current_player, and agent_player
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_lookup ON policy (grid, walls_p1, walls_p2, completed_steps, current_player, agent_player)",
        [],
    )?;

    let num_entries = entries.len();

    // Insert entries in a transaction for better performance
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO policy (grid, current_player, walls_p1, walls_p2, agent_player, completed_steps, num_actions, actions, action_values)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"
        )?;

        for entry in entries {
            // Convert grid Vec<i8> to Vec<u8> for blob storage
            let grid_blob: Vec<u8> = entry.grid.iter().map(|&x| x as u8).collect();

            // Flatten actions into a single vector: each action is [row, col, type]
            let actions_flat: Vec<i32> = entry.actions.into_iter().flatten().collect();
            let actions_blob: Vec<u8> = actions_flat.iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();

            // Convert values to bytes
            let values_blob: Vec<u8> = entry.values.iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();

            let num_actions = entry.values.len() as i32;

            stmt.execute(params![
                grid_blob,
                entry.current_player,
                entry.walls_remaining[0],
                entry.walls_remaining[1],
                entry.agent_player,
                entry.completed_steps,
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

/// Create a policy database by evaluating actions and saving to a SQLite database
#[pyfunction]
#[pyo3(signature = (grid, player_positions, walls_remaining, goal_rows, current_player, max_steps, branching_factor, wall_sigma, discount_factor, heuristic, filename))]
fn create_policy_db(
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
    filename: &str,
) -> PyResult<usize> {
    // Calculate board_size from grid dimensions
    // Grid is (2*board_size + 3) x (2*board_size + 3), so board_size = (rows - 3) / 2
    let grid_array = grid.as_array();
    let grid_rows = grid_array.shape()[0] as i32;
    let board_size = (grid_rows - 3) / 2;

    // Get max_walls from walls_remaining (assumes we start with full walls)
    let walls_array = walls_remaining.as_array();
    let max_walls = walls_array[0].max(walls_array[1]);

    // Call evaluate_actions with logging enabled
    let (_actions, _values, log_entries) = minimax::evaluate_actions(
        &grid_array,
        &player_positions.as_array(),
        &walls_array,
        &goal_rows.as_array(),
        current_player,
        max_steps,
        branching_factor,
        wall_sigma,
        discount_factor,
        heuristic,
        true, // enable logging
    );

    if let Some(entries) = log_entries {
        // Write entries to SQLite database with metadata
        log_entries_to_sqlite(entries, filename, board_size, max_steps, max_walls)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to write to SQLite database: {}", e)))
    } else {
        Ok(0)
    }
}

/// A Python module implemented in Rust.
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
    m.add_function(wrap_pyfunction!(create_policy_db, m)?)?;

    // Export constants to match qgrid.py
    m.add("CELL_FREE", pathfinding::CELL_FREE)?;
    m.add("CELL_PLAYER1", pathfinding::CELL_PLAYER1)?;
    m.add("CELL_PLAYER2", pathfinding::CELL_PLAYER2)?;
    m.add("CELL_WALL", pathfinding::CELL_WALL)?;

    // Wall orientations
    m.add("WALL_ORIENTATION_VERTICAL", 0)?;
    m.add("WALL_ORIENTATION_HORIZONTAL", 1)?;

    // Action types
    m.add("ACTION_WALL_VERTICAL", 0)?;
    m.add("ACTION_WALL_HORIZONTAL", 1)?;
    m.add("ACTION_MOVE", 2)?;

    Ok(())
}
