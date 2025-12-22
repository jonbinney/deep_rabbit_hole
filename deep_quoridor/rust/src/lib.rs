use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

mod pathfinding;

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

/// A Python module implemented in Rust.
#[pymodule]
fn quoridor_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distance_to_row, m)?)?;

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
