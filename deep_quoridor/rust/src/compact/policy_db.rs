/// Minimax for building a policy database.
///
/// Uses i8 values (1 = P0 wins, 0 = tie, -1 = P1 wins)
/// with a transposition table.
use dashmap::DashMap;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use super::q_game_mechanics::QGameMechanics;

/// Maximum state size in bytes. Covers up to 9x9 boards with generous parameters.
pub const MAX_STATE_BYTES: usize = 8;

/// Fixed-size key for the transposition table, avoiding heap allocation.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StateKey {
    bytes: [u8; MAX_STATE_BYTES],
    len: u8,
}

impl StateKey {
    pub fn from_slice(data: &[u8]) -> Self {
        assert!(
            data.len() <= MAX_STATE_BYTES,
            "State size {} exceeds MAX_STATE_BYTES {}",
            data.len(),
            MAX_STATE_BYTES
        );
        let mut bytes = [0u8; MAX_STATE_BYTES];
        bytes[..data.len()].copy_from_slice(data);
        Self {
            bytes,
            len: data.len() as u8,
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }
}

impl std::hash::Hash for StateKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

/// Transposition table type alias.
pub type TranspositionTable = DashMap<StateKey, i8>;

/// Get all valid actions (moves + wall placements) for the current player.
fn get_all_actions(mechanics: &QGameMechanics, data: &mut [u8]) -> Vec<(u8, u8, u8)> {
    let moves = mechanics.get_valid_moves(data);
    let mut actions: Vec<(u8, u8, u8)> = moves
        .into_iter()
        .map(|(r, c)| (r as u8, c as u8, 2))
        .collect();
    let walls = mechanics.get_valid_wall_placements(data);
    actions.extend(
        walls
            .into_iter()
            .map(|(r, c, t)| (r as u8, c as u8, t as u8)),
    );
    actions
}

/// Minimax evaluation with transposition table.
///
/// Returns the value from player 0's (absolute) perspective:
/// - `1` = player 0 wins with best play
/// - `0` = game is a tie with best play
/// - `-1` = player 1 wins with best play
///
/// All reachable non-terminal states and their values are stored in the
/// transposition table for later export to a policy database.
pub fn minimax(
    mechanics: &QGameMechanics,
    data: &mut [u8],
    transposition_table: &TranspositionTable,
) -> i8 {
    minimax_inner(mechanics, data, transposition_table, None)
}

fn minimax_inner(
    mechanics: &QGameMechanics,
    data: &mut [u8],
    transposition_table: &TranspositionTable,
    mut rng: Option<&mut StdRng>,
) -> i8 {
    let key = StateKey::from_slice(data);
    if let Some(entry) = transposition_table.get(&key) {
        return *entry;
    }

    let current_player = mechanics.repr().get_current_player(data);
    let opponent = 1 - current_player;

    // Terminal states: return value directly without storing in transposition table.
    if mechanics.check_win(data, opponent) {
        return match opponent {
            0 => 1,
            1 => -1,
            _ => panic!("Bad player number ({})", opponent),
        };
    }
    if mechanics.repr().get_completed_steps(data) >= mechanics.repr().max_steps() {
        return 0;
    }

    // Not a terminal state. Recurse.

    let mut actions = get_all_actions(mechanics, data);
    assert!(
        !actions.is_empty(),
        "No valid actions - should never happen"
    );

    // Shuffle actions if an RNG is provided (for Lazy SMP)
    if let Some(ref mut r) = rng {
        actions.shuffle(*r);
    }

    let is_maximizing = current_player == 0;
    let mut best_value: i8 = if is_maximizing { -1 } else { 1 };

    for &(row, col, action_type) in &actions {
        let mut new_data = data.to_vec();
        let (r, c, t) = (row as usize, col as usize, action_type as usize);
        if action_type == 2 {
            mechanics.execute_move(
                &mut new_data,
                mechanics.repr().get_current_player(data),
                r,
                c,
            );
        } else {
            mechanics.execute_wall_placement(&mut new_data, current_player, r, c, t);
        }

        mechanics.switch_player(&mut new_data);

        let child_value = minimax_inner(
            mechanics,
            &mut new_data,
            transposition_table,
            rng.as_deref_mut(),
        );

        if is_maximizing {
            if child_value > best_value {
                best_value = child_value;
            }
        } else {
            if child_value < best_value {
                best_value = child_value;
            }
        }
    }

    transposition_table.insert(key, best_value);

    best_value
}

/// Lazy SMP parallel minimax. Spawns `num_threads` threads each running
/// the full minimax with randomized move ordering, all sharing the same
/// transposition table. Returns the root value.
pub fn minimax_lazy_smp(
    mechanics: &QGameMechanics,
    data: &mut [u8],
    transposition_table: &TranspositionTable,
    num_threads: usize,
) -> i8 {
    let data_snapshot = data.to_vec();
    let result = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);
        for i in 0..num_threads {
            let mut thread_data = data_snapshot.clone();
            let tt = &transposition_table;
            let mech = &mechanics;
            handles.push(s.spawn(move || {
                let mut rng = StdRng::seed_from_u64(i as u64);
                minimax_inner(mech, &mut thread_data, tt, Some(&mut rng))
            }));
        }
        let mut results = Vec::with_capacity(num_threads);
        for h in handles {
            results.push(h.join().expect("minimax thread panicked"));
        }
        results[0]
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimax_initial_state_3x3() {
        let mechanics = QGameMechanics::new(3, 0, 10);
        let mut data = mechanics.create_initial_state();
        let table = TranspositionTable::new();

        let value = minimax(&mechanics, &mut data, &table);

        assert!(value == -1, "P1 should always lose 3x3 with no walls");
    }

    #[test]
    fn test_minimax_win_in_one() {
        let mechanics = QGameMechanics::new(3, 0, 10);
        let mut data = mechanics.create_initial_state();

        mechanics.repr().set_player_position(&mut data, 0, 1, 1);
        mechanics.repr().set_player_position(&mut data, 1, 2, 0);
        mechanics.repr().set_current_player(&mut data, 0);
        mechanics.repr().set_completed_steps(&mut data, 2);

        let table = TranspositionTable::new();
        let value = minimax(&mechanics, &mut data, &table);

        assert_eq!(value, 1, "Player 0 should be able to win in one move");
    }

    #[test]
    fn test_minimax_tie_at_max_steps() {
        let mechanics = QGameMechanics::new(3, 0, 2);
        let mut data = mechanics.create_initial_state();
        mechanics.repr().set_completed_steps(&mut data, 2);

        let table = TranspositionTable::new();
        let value = minimax(&mechanics, &mut data, &table);

        assert_eq!(value, 0, "Max steps reached should be a tie");
    }

    #[test]
    fn test_transposition_table_populated() {
        let mechanics = QGameMechanics::new(3, 0, 4);
        let mut data = mechanics.create_initial_state();
        let table = TranspositionTable::new();

        minimax(&mechanics, &mut data, &table);

        assert!(!table.is_empty(), "Transposition table should have entries");
    }
}
