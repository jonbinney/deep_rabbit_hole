#[cfg(feature = "python")]
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
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
        data,
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
    _board_size: usize,
    _max_walls: usize,
    _max_steps: usize,
    db_path: &str,
) -> PyResult<Option<(Bound<'py, PyArray2<i32>>, Bound<'py, numpy::PyArray1<i32>>)>> {
    use compact::policy_db::PolicyDb;

    // One-shot per-move agent path: always lazy so a single move never
    // pays the cost of loading the whole DB into RAM.
    let db = PolicyDb::open(db_path, true)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to open DB: {e}")))?;

    let mut data = db.mechanics().repr().create_data();
    db.mechanics().repr().from_game_state(
        &mut data,
        &grid.as_array(),
        &player_positions.as_array(),
        &walls_remaining.as_array(),
        current_player,
        completed_steps,
    );

    match db
        .lookup_action_values(data)
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
    state: u64,
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
    state: u64,
    board_size: usize,
    max_walls: usize,
    max_steps: usize,
) -> Vec<(usize, usize, usize, u64)> {
    use compact::q_game_mechanics::QGameMechanics;

    let mechanics = QGameMechanics::new(board_size, max_walls, max_steps);
    let current_player = mechanics.repr().get_current_player(state);
    let mut data = state;

    let mut children = Vec::new();

    // Pawn moves (action_type = 2)
    let moves = mechanics.get_valid_moves(data);
    for (row, col) in moves {
        let mut child = data;
        mechanics.execute_move(&mut child, current_player, row, col);
        mechanics.switch_player(&mut child);
        children.push((row, col, 2usize, child));
    }

    // Wall placements (action_type = 0 or 1)
    let wall_placements = mechanics.get_valid_wall_placements(&mut data);
    for (row, col, orientation) in wall_placements {
        let mut child = data;
        mechanics.execute_wall_placement(&mut child, current_player, row, col, orientation);
        mechanics.switch_player(&mut child);
        children.push((row, col, orientation, child));
    }

    children
}

/// Python wrapper around PolicyDb for database access from Python.
#[cfg(feature = "python")]
#[pyclass]
struct PyPolicyDb {
    db: compact::policy_db::PolicyDb,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPolicyDb {
    /// Open a Parquet policy DB.
    ///
    /// `lazy=False` (default) loads the entire dataset into a HashMap at
    /// open time for O(1) state lookups — best for training. `lazy=True`
    /// keeps the file on disk and walks Parquet row groups on demand —
    /// use this for DBs too large to fit in memory.
    #[new]
    #[pyo3(signature = (path, lazy = false))]
    fn new(path: &str, lazy: bool) -> PyResult<Self> {
        let db = compact::policy_db::PolicyDb::open(path, lazy)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to open DB: {e}")))?;
        Ok(Self { db })
    }

    /// Read metadata: returns (board_size, max_walls, max_steps, num_states).
    fn read_metadata(&self) -> PyResult<(usize, usize, usize, Option<usize>)> {
        self.db
            .read_metadata()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }

    /// Count total states in the policy table.
    fn count_states(&self) -> PyResult<usize> {
        self.db
            .count_states()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }

    /// Fetch (state, value) tuples by rowid. State is a u64 packed integer.
    fn fetch_states_by_rowid(&self, rowids: Vec<i64>) -> PyResult<Vec<(u64, i32)>> {
        self.db
            .fetch_states_by_rowid(&rowids)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }

    /// Look up (state, value) for the given states.
    fn lookup_values_by_state(&self, states: Vec<u64>) -> PyResult<Vec<(u64, i32)>> {
        self.db
            .lookup_values_by_state(&states)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }

    /// Look up action values for a compact state.
    ///
    /// Returns None if no valid actions, otherwise returns
    /// (actions, values) where actions is a list of (row, col, action_type)
    /// and values are from the acting player's perspective.
    fn lookup_action_values(&self, state: u64) -> PyResult<Option<(Vec<(u8, u8, u8)>, Vec<i32>)>> {
        self.db
            .lookup_action_values(state)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }

    /// Fetch a training batch in one Rust call.
    ///
    /// For each rowid:
    ///   1. Read `(state, db_value)` from the DB (db_value is P0-perspective).
    ///   2. Look up child action values to build an `mcts_policy` (uniform
    ///      over actions whose value equals the maximum, in the unrotated
    ///      action layout).
    ///   3. Build the action mask in the unrotated layout via game mechanics.
    ///   4. If the current player is P1, rotate the state and build
    ///      features from the rotated state; permute `mcts_policy` and
    ///      `action_mask` via the rotation index permutation.
    ///   5. Flip `db_value` to the acting player's perspective.
    ///
    /// Returns `(input_arrays, values, action_masks, mcts_policies, current_players)`.
    /// `input_arrays` shape depends on `nn_type`: `(N, D)` for `"mlp"`,
    /// `(N, 5, M, M)` for `"resnet"` where `M = 2*board_size+3`.
    #[pyo3(signature = (rowids, nn_type))]
    fn fetch_training_batch<'py>(
        &self,
        py: Python<'py>,
        rowids: Vec<i64>,
        nn_type: &str,
    ) -> PyResult<(
        Bound<'py, PyArrayDyn<f32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<i32>>,
    )> {
        use compact::q_features::{
            build_action_mask, build_mlp_features, build_resnet_features,
            build_rotation_permutation, child_action_index, mlp_features_len, num_actions,
            permute_into, resnet_features_len, resnet_grid_size, rotate_state, NnType,
        };

        let nn = NnType::from_str(nn_type)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        let mechanics = self.db.mechanics();
        let bs = mechanics.repr().board_size();
        let n_actions = num_actions(bs);
        let feature_len = match nn {
            NnType::Mlp => mlp_features_len(bs),
            NnType::Resnet => resnet_features_len(bs),
        };

        let rows = self
            .db
            .fetch_states_by_rowid(&rowids)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
        let n = rows.len();

        let perm = build_rotation_permutation(bs);

        // Flat output buffers — single allocation per output array.
        let mut features_buf = vec![0.0f32; n * feature_len];
        let mut values_buf = vec![0i32; n];
        let mut masks_buf = vec![0.0f32; n * n_actions];
        let mut policies_buf = vec![0.0f32; n * n_actions];
        let mut cps_buf = vec![0i32; n];

        // Scratch buffers for the per-state policy and mask in the
        // unrotated frame, before any rotation permutation.
        let mut policy_scratch = vec![0.0f32; n_actions];
        let mut mask_scratch = vec![0.0f32; n_actions];

        for (i, (state, db_value)) in rows.into_iter().enumerate() {
            let cp = mechanics.repr().get_current_player(state);

            // ---- mcts_policy (unrotated frame) ----
            let lookup = self
                .db
                .lookup_action_values(state)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?
                .ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "lookup_action_values returned None for rowid index {i}"
                    ))
                })?;
            let (actions, action_values) = lookup;
            for v in policy_scratch.iter_mut() {
                *v = 0.0;
            }
            let best_value = *action_values.iter().max().unwrap();
            let mut count = 0usize;
            for (&(r, c, t), &v) in actions.iter().zip(action_values.iter()) {
                if v == best_value {
                    let idx = child_action_index(r as usize, c as usize, t as usize, bs);
                    policy_scratch[idx] = 1.0;
                    count += 1;
                }
            }
            if count > 0 {
                let inv = 1.0 / count as f32;
                for v in policy_scratch.iter_mut() {
                    if *v > 0.0 {
                        *v = inv;
                    }
                }
            }

            // ---- action_mask (unrotated frame) ----
            build_action_mask(state, mechanics, &mut mask_scratch);

            // ---- features (rotated frame iff cp==1) ----
            let working_state = if cp == 1 {
                rotate_state(state, mechanics)
            } else {
                state
            };
            let f_off = i * feature_len;
            let f_slice = &mut features_buf[f_off..f_off + feature_len];
            match nn {
                NnType::Mlp => build_mlp_features(working_state, mechanics, f_slice),
                NnType::Resnet => build_resnet_features(working_state, mechanics, f_slice),
            };

            // ---- write policy + mask, permuting if rotated ----
            let p_off = i * n_actions;
            let m_off = i * n_actions;
            if cp == 1 {
                permute_into(
                    &policy_scratch,
                    &perm,
                    &mut policies_buf[p_off..p_off + n_actions],
                );
                permute_into(
                    &mask_scratch,
                    &perm,
                    &mut masks_buf[m_off..m_off + n_actions],
                );
            } else {
                policies_buf[p_off..p_off + n_actions].copy_from_slice(&policy_scratch);
                masks_buf[m_off..m_off + n_actions].copy_from_slice(&mask_scratch);
            }

            // ---- value (acting player's perspective) and current_player ----
            values_buf[i] = if cp == 0 { db_value } else { -db_value };
            cps_buf[i] = cp as i32;
        }

        // Wrap flat buffers as numpy arrays of the right shape.
        let features_arr = match nn {
            NnType::Mlp => ndarray::Array::from_shape_vec(
                ndarray::IxDyn(&[n, feature_len]),
                features_buf,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?,
            NnType::Resnet => {
                let m = resnet_grid_size(bs);
                ndarray::Array::from_shape_vec(ndarray::IxDyn(&[n, 6, m, m]), features_buf)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?
            }
        };
        let masks_arr = ndarray::Array2::from_shape_vec((n, n_actions), masks_buf)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
        let policies_arr = ndarray::Array2::from_shape_vec((n, n_actions), policies_buf)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
        let values_arr = ndarray::Array1::from(values_buf);
        let cps_arr = ndarray::Array1::from(cps_buf);

        Ok((
            PyArrayDyn::from_owned_array_bound(py, features_arr),
            PyArray1::from_owned_array_bound(py, values_arr),
            PyArray2::from_owned_array_bound(py, masks_arr),
            PyArray2::from_owned_array_bound(py, policies_arr),
            PyArray1::from_owned_array_bound(py, cps_arr),
        ))
    }
}

/// Return a text-art display string for a compact state.
#[cfg(feature = "python")]
#[pyfunction]
fn compact_state_display(
    state: u64,
    board_size: usize,
    max_walls: usize,
    max_steps: usize,
) -> String {
    use compact::q_bit_repr::QBitRepr;
    let repr = QBitRepr::new(board_size, max_walls, max_steps);
    repr.display(state)
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
    m.add_function(wrap_pyfunction!(compact_state_display, m)?)?;

    // PolicyDb class
    m.add_class::<PyPolicyDb>()?;

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
