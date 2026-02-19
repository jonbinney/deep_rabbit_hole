#![allow(dead_code)]

//! Board rotation utilities for self-play.
//!
//! When Player 1 is the current player, the board must be rotated 180° so
//! the neural network always sees "current player moving downward". These
//! helpers implement the rotation for the grid, player positions, goal rows,
//! action coordinates, and full policy vectors.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::actions::{
    action_index_to_action, action_to_index, policy_size, ACTION_MOVE,
};

/// Rotate a 2D grid 180° — equivalent to `np.rot90(grid, k=2)`.
///
/// `new[r][c] = old[H-1-r][W-1-c]`
pub fn rotate_grid_180(grid: &ArrayView2<i8>) -> Array2<i8> {
    let (h, w) = (grid.nrows(), grid.ncols());
    let mut out = Array2::<i8>::zeros((h, w));
    for r in 0..h {
        for c in 0..w {
            out[[r, c]] = grid[[h - 1 - r, w - 1 - c]];
        }
    }
    out
}

/// Rotate both players' board-coordinate positions 180°.
///
/// `(row, col) → (board_size-1-row, board_size-1-col)`
pub fn rotate_player_positions(positions: &ArrayView2<i32>, board_size: i32) -> Array2<i32> {
    let mut out = positions.to_owned();
    for p in 0..2 {
        out[[p, 0]] = board_size - 1 - positions[[p, 0]];
        out[[p, 1]] = board_size - 1 - positions[[p, 1]];
    }
    out
}

/// Rotate goal rows — swaps the two players' goal rows.
pub fn rotate_goal_rows(goal_rows: &ArrayView1<i32>) -> Array1<i32> {
    Array1::from(vec![goal_rows[1], goal_rows[0]])
}

/// Rotate a single action's coordinates 180°.
///
/// For move actions: `(bs-1-row, bs-1-col, MOVE)`
/// For wall actions: `(ws-1-row, ws-1-col, same_type)` where `ws = bs - 1`
pub fn rotate_action_coords(
    board_size: i32,
    row: i32,
    col: i32,
    action_type: i32,
) -> (i32, i32, i32) {
    if action_type == ACTION_MOVE {
        (board_size - 1 - row, board_size - 1 - col, ACTION_MOVE)
    } else {
        let wall_size = board_size - 1;
        let new_type = action_type; // orientation doesn't change under 180° rotation
        (wall_size - 1 - row, wall_size - 1 - col, new_type)
    }
}

/// Precompute rotation mappings from original policy indices to rotated indices.
///
/// Returns `(original_to_rotated, rotated_to_original)` — both are index arrays
/// of length `policy_size(board_size)`.
pub fn create_rotation_mapping(board_size: i32) -> (Vec<usize>, Vec<usize>) {
    let total = policy_size(board_size);
    let mut orig_to_rot = vec![0usize; total];
    let mut rot_to_orig = vec![0usize; total];

    for idx in 0..total {
        let action = action_index_to_action(board_size, idx);
        let (rr, rc, rt) = rotate_action_coords(board_size, action[0], action[1], action[2]);
        let rotated_idx = action_to_index(board_size, &[rr, rc, rt]);
        orig_to_rot[idx] = rotated_idx;
        rot_to_orig[rotated_idx] = idx;
    }

    (orig_to_rot, rot_to_orig)
}

/// Apply a rotation mapping to remap a policy vector.
///
/// `output[mapping[i]] = input[i]` for all `i`.
pub fn remap_policy(policy: &[f32], mapping: &[usize]) -> Vec<f32> {
    let mut out = vec![0.0f32; policy.len()];
    for (i, &target) in mapping.iter().enumerate() {
        out[target] = policy[i];
    }
    out
}

/// Apply a rotation mapping to remap a boolean mask.
pub fn remap_mask(mask: &[bool], mapping: &[usize]) -> Vec<bool> {
    let mut out = vec![false; mask.len()];
    for (i, &target) in mapping.iter().enumerate() {
        out[target] = mask[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::{ACTION_WALL_HORIZONTAL, ACTION_WALL_VERTICAL};
    use crate::game_state::create_initial_state;

    #[test]
    fn test_rotate_grid_180_identity() {
        // Rotating twice should give back the original
        let (grid, _, _, _) = create_initial_state(5, 3);
        let rotated = rotate_grid_180(&grid.view());
        let double_rotated = rotate_grid_180(&rotated.view());
        assert_eq!(grid, double_rotated);
    }

    #[test]
    fn test_rotate_player_positions_identity() {
        let (_, positions, _, _) = create_initial_state(5, 3);
        let board_size = 5;
        let rotated = rotate_player_positions(&positions.view(), board_size);
        let double = rotate_player_positions(&rotated.view(), board_size);
        assert_eq!(positions, double);
    }

    #[test]
    fn test_rotate_player_positions_values() {
        // Player 0 at (0, 2), Player 1 at (4, 2) on 5×5
        let (_, positions, _, _) = create_initial_state(5, 3);
        let board_size = 5;
        let rotated = rotate_player_positions(&positions.view(), board_size);
        // Player 0: (0,2) → (4,2)
        assert_eq!(rotated[[0, 0]], 4);
        assert_eq!(rotated[[0, 1]], 2);
        // Player 1: (4,2) → (0,2)
        assert_eq!(rotated[[1, 0]], 0);
        assert_eq!(rotated[[1, 1]], 2);
    }

    #[test]
    fn test_rotate_goal_rows_identity() {
        let (_, _, _, goal_rows) = create_initial_state(5, 3);
        let rotated = rotate_goal_rows(&goal_rows.view());
        let double = rotate_goal_rows(&rotated.view());
        assert_eq!(goal_rows, double);
    }

    #[test]
    fn test_rotate_goal_rows_swap() {
        let (_, _, _, goal_rows) = create_initial_state(5, 3);
        let rotated = rotate_goal_rows(&goal_rows.view());
        assert_eq!(rotated[0], goal_rows[1]);
        assert_eq!(rotated[1], goal_rows[0]);
    }

    #[test]
    fn test_rotate_action_coords_identity() {
        let bs = 5;
        // Double-rotate move
        let (r, c, t) = rotate_action_coords(bs, 1, 3, ACTION_MOVE);
        let (r2, c2, t2) = rotate_action_coords(bs, r, c, t);
        assert_eq!((r2, c2, t2), (1, 3, ACTION_MOVE));

        // Double-rotate horizontal wall
        let (r, c, t) = rotate_action_coords(bs, 2, 1, ACTION_WALL_HORIZONTAL);
        let (r2, c2, t2) = rotate_action_coords(bs, r, c, t);
        assert_eq!((r2, c2, t2), (2, 1, ACTION_WALL_HORIZONTAL));

        // Double-rotate vertical wall
        let (r, c, t) = rotate_action_coords(bs, 0, 3, ACTION_WALL_VERTICAL);
        let (r2, c2, t2) = rotate_action_coords(bs, r, c, t);
        assert_eq!((r2, c2, t2), (0, 3, ACTION_WALL_VERTICAL));
    }

    #[test]
    fn test_rotation_mapping_inverse() {
        for bs in [5, 9] {
            let (orig_to_rot, rot_to_orig) = create_rotation_mapping(bs);
            let total = policy_size(bs);
            for i in 0..total {
                assert_eq!(
                    rot_to_orig[orig_to_rot[i]], i,
                    "inverse broken at i={i} for bs={bs}"
                );
                assert_eq!(
                    orig_to_rot[rot_to_orig[i]], i,
                    "forward inverse broken at i={i} for bs={bs}"
                );
            }
        }
    }

    #[test]
    fn test_rotation_mapping_double_is_identity() {
        let bs = 5;
        let (orig_to_rot, _) = create_rotation_mapping(bs);
        let total = policy_size(bs);
        for i in 0..total {
            let once = orig_to_rot[i];
            let twice = orig_to_rot[once];
            assert_eq!(twice, i, "double rotation not identity at i={i}");
        }
    }

    #[test]
    fn test_remap_policy_roundtrip() {
        let bs = 5;
        let total = policy_size(bs);
        let (orig_to_rot, rot_to_orig) = create_rotation_mapping(bs);
        let policy: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let rotated = remap_policy(&policy, &orig_to_rot);
        let recovered = remap_policy(&rotated, &rot_to_orig);
        assert_eq!(policy, recovered);
    }
}
