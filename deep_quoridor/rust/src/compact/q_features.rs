/// Feature extraction utilities that operate directly on QBitRepr `u64`
/// states, mirroring the Python pipelines in
/// `deep_quoridor/src/agents/alphazero/{mlp,resnet}_network.py`.
///
/// The training script previously did all of this in Python — re-building a
/// `Quoridor` object per state, rotating the board, computing features and
/// the action mask, etc. The functions here let the entire batch run in
/// Rust with a single Python ↔ Rust round-trip per call.
use super::q_bit_repr::{WALL_HORIZONTAL, WALL_VERTICAL};
use super::q_game_mechanics::QGameMechanics;

/// Which feature representation to build.
pub enum NnType {
    Mlp,
    Resnet,
}

impl NnType {
    pub fn from_str(name: &str) -> Result<Self, String> {
        match name {
            "mlp" => Ok(NnType::Mlp),
            "resnet" => Ok(NnType::Resnet),
            other => Err(format!("unknown nn_type {other:?} (expected 'mlp' or 'resnet')")),
        }
    }
}

/// Number of actions in the ActionEncoder layout: `bs² + 2 * (bs-1)²`.
/// Layout: pawn moves (bs²), then vertical walls ((bs-1)²), then horizontal
/// walls ((bs-1)²).
#[inline]
pub fn num_actions(board_size: usize) -> usize {
    board_size * board_size + 2 * (board_size - 1) * (board_size - 1)
}

/// Map `(row, col, action_type)` to a flat action index, matching the Python
/// `child_action_index` in `train_policy_db_evaluator.py`. action_type is
/// 0=vertical wall, 1=horizontal wall, 2=pawn move.
#[inline]
pub fn child_action_index(row: usize, col: usize, action_type: usize, board_size: usize) -> usize {
    let wall_size = board_size - 1;
    match action_type {
        2 => row * board_size + col,
        0 => board_size * board_size + row * wall_size + col,
        1 => board_size * board_size + wall_size * wall_size + row * wall_size + col,
        _ => panic!("unknown action_type {action_type}"),
    }
}

/// Single-section index permutation: 180° flip on a `gs × gs` grid embedded
/// in a flat array starting at `offset`. Mirrors
/// `_map_original_index_to_rotated` / `_map_rotated_index_to_original` in
/// `agents/core/rotation.py`.
#[inline]
fn rotate_section_index(idx: usize, gs: usize, offset: usize) -> usize {
    let local = idx - offset;
    let row = local / gs;
    let col = local % gs;
    let rotated_row = gs - 1 - row;
    let rotated_col = gs - 1 - col;
    offset + rotated_row * gs + rotated_col
}

/// Build the index permutation that rotates a length-`num_actions` policy or
/// action-mask vector by 180°. Self-inverse: applying twice is identity.
pub fn build_rotation_permutation(board_size: usize) -> Vec<usize> {
    let bs = board_size;
    let move_count = bs * bs;
    let wall_count = (bs - 1) * (bs - 1);
    let total = move_count + 2 * wall_count;
    let mut perm = Vec::with_capacity(total);
    for idx in 0..move_count {
        perm.push(rotate_section_index(idx, bs, 0));
    }
    for idx in move_count..(move_count + wall_count) {
        perm.push(rotate_section_index(idx, bs - 1, move_count));
    }
    for idx in (move_count + wall_count)..total {
        perm.push(rotate_section_index(idx, bs - 1, move_count + wall_count));
    }
    perm
}

/// Apply `perm` to `src`, writing into `dst`. `dst[i] = src[perm[i]]`.
#[inline]
pub fn permute_into<T: Copy>(src: &[T], perm: &[usize], dst: &mut [T]) {
    debug_assert_eq!(src.len(), perm.len());
    debug_assert_eq!(dst.len(), perm.len());
    for (i, &p) in perm.iter().enumerate() {
        dst[i] = src[p];
    }
}

/// Return a u64 representing the same position from the opponent's
/// perspective: 180° flipped board, both player positions rotated, walls
/// flipped. `current_player` and `walls_remaining` are *unchanged* (matching
/// `Quoridor.rotate_board` in `quoridor.py`, which only rotates the spatial
/// state and reverses goal rows).
pub fn rotate_state(state: u64, mechanics: &QGameMechanics) -> u64 {
    let repr = mechanics.repr();
    let bs = repr.board_size();
    let mut out = repr.create_data();

    // Rotate player positions.
    for player in 0..2 {
        let (r, c) = repr.get_player_position(state, player);
        repr.set_player_position(&mut out, player, bs - 1 - r, bs - 1 - c);
    }

    // Walls remaining: copy through (rotation does not swap).
    for player in 0..2 {
        repr.set_walls_remaining(&mut out, player, repr.get_walls_remaining(state, player));
    }

    // Current player: unchanged.
    repr.set_current_player(&mut out, repr.get_current_player(state));

    // Completed steps: unchanged.
    repr.set_completed_steps(&mut out, repr.get_completed_steps(state));

    // Walls: flip each occupied slot to its (bs-2-r, bs-2-c) twin (same orientation).
    let wall_grid = bs - 1;
    for r in 0..wall_grid {
        for c in 0..wall_grid {
            for orient in [WALL_VERTICAL, WALL_HORIZONTAL] {
                if repr.get_wall(state, r, c, orient) {
                    repr.set_wall(&mut out, wall_grid - 1 - r, wall_grid - 1 - c, orient, true);
                }
            }
        }
    }

    out
}

/// MLP feature vector length. Layout (matches `MlpNetwork.game_to_input_array`):
/// `[player_board (bs²), opponent_board (bs²), walls (2*(bs-1)²), my_walls, opp_walls]`.
#[inline]
pub fn mlp_features_len(board_size: usize) -> usize {
    let bs = board_size;
    2 * bs * bs + 2 * (bs - 1) * (bs - 1) + 2
}

/// ResNet feature tensor side length: `2*bs + 3`. Total elements: `5 * M * M`.
#[inline]
pub fn resnet_grid_size(board_size: usize) -> usize {
    2 * board_size + 3
}

#[inline]
pub fn resnet_features_len(board_size: usize) -> usize {
    let m = resnet_grid_size(board_size);
    5 * m * m
}

/// Write the MLP feature vector for `state` into `out`. `out.len()` must
/// equal `mlp_features_len(board_size)`. Values are `f32` in {0.0, 1.0}
/// for boards/walls and the wall counts as `f32` in the trailing two slots.
pub fn build_mlp_features(state: u64, mechanics: &QGameMechanics, out: &mut [f32]) {
    let repr = mechanics.repr();
    let bs = repr.board_size();
    debug_assert_eq!(out.len(), mlp_features_len(bs));

    // Zero everything; we'll set the few 1.0s and trailing wall counts.
    for v in out.iter_mut() {
        *v = 0.0;
    }

    let cp = repr.get_current_player(state);
    let opp = 1 - cp;
    let (player_r, player_c) = repr.get_player_position(state, cp);
    let (opp_r, opp_c) = repr.get_player_position(state, opp);

    // Section 0: player_board[bs, bs] flattened, 1.0 at current player's pos.
    out[player_r * bs + player_c] = 1.0;

    // Section 1: opponent_board[bs, bs] flattened.
    let opp_off = bs * bs;
    out[opp_off + opp_r * bs + opp_c] = 1.0;

    // Section 2: walls. Layout matches numpy `walls.flatten()` on a
    // `(bs-1, bs-1, 2)` C-order array: row-major, with orient innermost.
    // i.e. slot `(r, c, orient)` lives at `r*(bs-1)*2 + c*2 + orient`.
    let walls_off = 2 * bs * bs;
    let wg = bs - 1;
    for r in 0..wg {
        for c in 0..wg {
            // orient 0 = vertical, orient 1 = horizontal (matches old_style_walls layout)
            let v = if repr.get_wall(state, r, c, WALL_VERTICAL) { 1.0 } else { 0.0 };
            let h = if repr.get_wall(state, r, c, WALL_HORIZONTAL) { 1.0 } else { 0.0 };
            out[walls_off + r * wg * 2 + c * 2] = v;
            out[walls_off + r * wg * 2 + c * 2 + 1] = h;
        }
    }

    // Trailing two: my walls remaining, opp walls remaining.
    let trailer = walls_off + 2 * wg * wg;
    out[trailer] = repr.get_walls_remaining(state, cp) as f32;
    out[trailer + 1] = repr.get_walls_remaining(state, opp) as f32;
}

/// Write the ResNet 5-channel feature tensor for `state` into `out`.
/// `out.len()` must equal `resnet_features_len(board_size)`. Layout in `out`
/// is C-order `(5, M, M)` with `M = 2*bs+3`. Channels match
/// `ResnetNetwork.game_to_input_array`:
///   0: walls (1.0 wherever the padded grid has a wall, including border)
///   1: current player position 1-hot at grid `(r*2+2, c*2+2)`
///   2: opponent position 1-hot
///   3: current player walls remaining (broadcast)
///   4: opponent walls remaining (broadcast)
pub fn build_resnet_features(state: u64, mechanics: &QGameMechanics, out: &mut [f32]) {
    let repr = mechanics.repr();
    let bs = repr.board_size();
    let m = resnet_grid_size(bs);
    debug_assert_eq!(out.len(), resnet_features_len(bs));

    // Zero everything first.
    for v in out.iter_mut() {
        *v = 0.0;
    }

    let plane = m * m;
    let ch = |c: usize| c * plane;

    // Channel 0: walls. Start with the border (outer 2 rows/cols on each side).
    for i in 0..m {
        for &j in &[0usize, 1, m - 2, m - 1] {
            out[ch(0) + i * m + j] = 1.0;
            out[ch(0) + j * m + i] = 1.0;
        }
    }
    // Internal walls: each placed wall paints three padded cells in the
    // grid, matching `set_wall_cells` in src/grid.rs.
    // Vertical wall at (r, c) covers grid rows {r*2+2, r*2+3, r*2+4} at col r_c+3.
    // Horizontal wall at (r, c) covers grid cols {c*2+2, c*2+3, c*2+4} at row r*2+3.
    let wg = bs - 1;
    for r in 0..wg {
        for c in 0..wg {
            if repr.get_wall(state, r, c, WALL_VERTICAL) {
                let gc = c * 2 + 3;
                for k in 0..3 {
                    let gr = r * 2 + 2 + k;
                    out[ch(0) + gr * m + gc] = 1.0;
                }
            }
            if repr.get_wall(state, r, c, WALL_HORIZONTAL) {
                let gr = r * 2 + 3;
                for k in 0..3 {
                    let gc = c * 2 + 2 + k;
                    out[ch(0) + gr * m + gc] = 1.0;
                }
            }
        }
    }

    // Channels 1 and 2: player position 1-hots.
    let cp = repr.get_current_player(state);
    let opp = 1 - cp;
    let (pr, pc) = repr.get_player_position(state, cp);
    let (or_, oc) = repr.get_player_position(state, opp);
    out[ch(1) + (pr * 2 + 2) * m + (pc * 2 + 2)] = 1.0;
    out[ch(2) + (or_ * 2 + 2) * m + (oc * 2 + 2)] = 1.0;

    // Channels 3 and 4: walls remaining (broadcast).
    let my_walls = repr.get_walls_remaining(state, cp) as f32;
    let opp_walls = repr.get_walls_remaining(state, opp) as f32;
    for v in &mut out[ch(3)..ch(4)] {
        *v = my_walls;
    }
    for v in &mut out[ch(4)..ch(5)] {
        *v = opp_walls;
    }
}

/// Write the action mask for `state` into `out` (length `num_actions(bs)`).
/// `1.0` for valid actions, `0.0` otherwise. Computed in the *unrotated*
/// frame using `mechanics`'s goal rows; callers permute via
/// `build_rotation_permutation` if working in the rotated frame.
pub fn build_action_mask(state: u64, mechanics: &QGameMechanics, out: &mut [f32]) {
    let bs = mechanics.repr().board_size();
    debug_assert_eq!(out.len(), num_actions(bs));
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let mut working = state;
    for (r, c) in mechanics.get_valid_moves(working) {
        out[child_action_index(r, c, 2, bs)] = 1.0;
    }
    for (r, c, orient) in mechanics.get_valid_wall_placements(&mut working) {
        out[child_action_index(r, c, orient, bs)] = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compact::q_minimax;
    use crate::compact::policy_db::{minimax, TranspositionTable};

    #[test]
    fn test_rotation_permutation_is_involution() {
        for bs in [3, 5, 9] {
            let perm = build_rotation_permutation(bs);
            for (i, &p) in perm.iter().enumerate() {
                assert_eq!(perm[p], i, "perm not involutive at i={i} (bs={bs})");
            }
        }
    }

    #[test]
    fn test_rotation_permutation_layout_3x3() {
        // 3x3 board: 9 moves, 4 vert walls, 4 horiz walls = 17 actions.
        let perm = build_rotation_permutation(3);
        // Move (0, 0) idx 0 ↔ move (2, 2) idx 8.
        assert_eq!(perm[0], 8);
        assert_eq!(perm[8], 0);
        // Move (1, 1) (center) is its own image.
        assert_eq!(perm[4], 4);
        // Vertical wall at (0, 0) idx 9 ↔ vertical wall at (1, 1) idx 12.
        // (wall grid is 2x2, so (0,0) <-> (1,1))
        assert_eq!(perm[9], 12);
        assert_eq!(perm[12], 9);
        // Horizontal walls section starts at 9+4=13. Same logic.
        assert_eq!(perm[13], 16);
        assert_eq!(perm[16], 13);
    }

    #[test]
    fn test_rotate_state_involution() {
        let mechanics =
            crate::compact::q_game_mechanics::QGameMechanics::new(3, 0, 8);
        let mut state = mechanics.create_initial_state();
        // Move P0 around to make the state non-symmetric under rotation,
        // since the initial state IS symmetric.
        mechanics.execute_move(&mut state, 0, 1, 0);
        mechanics.switch_player(&mut state);

        let rotated = rotate_state(state, &mechanics);
        let back = rotate_state(rotated, &mechanics);
        assert_eq!(state, back, "rotate_state should be involutive");
    }

    #[test]
    fn test_mlp_features_shape() {
        let mechanics =
            crate::compact::q_game_mechanics::QGameMechanics::new(5, 3, 50);
        let state = mechanics.create_initial_state();
        let mut out = vec![0.0f32; mlp_features_len(5)];
        build_mlp_features(state, &mechanics, &mut out);
        // Initial state: P0 at (0,2), P1 at (4,2). 5x5 board.
        // player_board[0,2] = 1.0, opponent_board[4,2] = 1.0.
        assert_eq!(out[0 * 5 + 2], 1.0);
        let opp_off = 5 * 5;
        assert_eq!(out[opp_off + 4 * 5 + 2], 1.0);
        // Trailing wall counts: 3, 3.
        let trailer = 2 * 5 * 5 + 2 * 4 * 4;
        assert_eq!(out[trailer], 3.0);
        assert_eq!(out[trailer + 1], 3.0);
    }

    #[test]
    fn test_resnet_features_shape() {
        let mechanics =
            crate::compact::q_game_mechanics::QGameMechanics::new(5, 3, 50);
        let state = mechanics.create_initial_state();
        let mut out = vec![0.0f32; resnet_features_len(5)];
        build_resnet_features(state, &mechanics, &mut out);
        let m = resnet_grid_size(5);
        assert_eq!(m, 13);
        // Channel 1: P0 at (0, 2) -> grid (2, 6).
        assert_eq!(out[1 * m * m + 2 * m + 6], 1.0);
        // Channel 2: P1 at (4, 2) -> grid (10, 6).
        assert_eq!(out[2 * m * m + 10 * m + 6], 1.0);
        // Channels 3 and 4: broadcast walls remaining = 3.
        assert_eq!(out[3 * m * m + 6 * m + 6], 3.0);
        assert_eq!(out[4 * m * m + 6 * m + 6], 3.0);
        // Channel 0: corners are border walls.
        assert_eq!(out[0 * m * m + 0 * m + 0], 1.0);
        assert_eq!(out[0 * m * m + (m - 1) * m + (m - 1)], 1.0);
    }

    #[test]
    fn test_action_mask_initial_state() {
        let mechanics =
            crate::compact::q_game_mechanics::QGameMechanics::new(3, 0, 8);
        let state = mechanics.create_initial_state();
        let mut mask = vec![0.0f32; num_actions(3)];
        build_action_mask(state, &mechanics, &mut mask);
        // P0 starts at (0, 1) with P1 at (2, 1). With no walls in this
        // game (max_walls=0), valid moves: (0, 0), (0, 2), (1, 1) — three
        // pawn moves, no wall placements.
        let move_count: f32 = mask[..9].iter().sum();
        let wall_count: f32 = mask[9..].iter().sum();
        assert_eq!(move_count, 3.0);
        assert_eq!(wall_count, 0.0);
        assert_eq!(mask[child_action_index(1, 1, 2, 3)], 1.0);
        assert_eq!(mask[child_action_index(0, 0, 2, 3)], 1.0);
        assert_eq!(mask[child_action_index(0, 2, 2, 3)], 1.0);
    }

    /// Smoke test: exercise the full per-state pipeline end-to-end.
    #[test]
    fn test_full_pipeline_3x3() {
        let mechanics =
            crate::compact::q_game_mechanics::QGameMechanics::new(3, 0, 8);
        let state = mechanics.create_initial_state();
        let cp = mechanics.repr().get_current_player(state);
        assert_eq!(cp, 0);

        // Build features for cp=0 path.
        let mut mlp = vec![0.0f32; mlp_features_len(3)];
        build_mlp_features(state, &mechanics, &mut mlp);
        let mut resnet = vec![0.0f32; resnet_features_len(3)];
        build_resnet_features(state, &mechanics, &mut resnet);
        let mut mask = vec![0.0f32; num_actions(3)];
        build_action_mask(state, &mechanics, &mut mask);

        // Now exercise the rotation path: take a state where cp=1.
        let mut state_p1 = state;
        mechanics.execute_move(&mut state_p1, 0, 1, 1);
        mechanics.switch_player(&mut state_p1);
        let cp1 = mechanics.repr().get_current_player(state_p1);
        assert_eq!(cp1, 1);

        let rotated = rotate_state(state_p1, &mechanics);
        let mut mlp_r = vec![0.0f32; mlp_features_len(3)];
        build_mlp_features(rotated, &mechanics, &mut mlp_r);
        // The current player in the rotated state is still 1, and it's at
        // its rotated position. P1 was at (2, 1) initially; on a 3x3
        // board, rotated to (0, 1).
        assert_eq!(mlp_r[0 * 3 + 1], 1.0);

        // Quiet unused-import warnings if all minimax/policy_db tests pass elsewhere.
        let _ = (q_minimax::WINNING_REWARD, TranspositionTable::new(), minimax);
    }
}
