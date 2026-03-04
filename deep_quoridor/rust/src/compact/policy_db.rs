/// Simplified minimax for building a policy database.
///
/// Uses i8 values (1 = win, 0 = tie, -1 = loss, None = unknown)
/// with a transposition table. No alpha-beta pruning or heuristic.
use dashmap::DashMap;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use super::q_game_mechanics::QGameMechanics;

/// Transposition table entry storing only the value.
/// The state bytes are stored as the DashMap key rather than in this struct.
#[derive(Clone)]
pub struct TranspositionEntry {
    pub value: i8,
}

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

/// Minimax evaluation using negamax formulation with transposition table.
///
/// Returns the value from the current player's perspective:
/// - `Some(1)` = current player wins with best play
/// - `Some(0)` = game is a tie with best play
/// - `Some(-1)` = current player loses with best play
/// - `None` = unknown (non-terminal state at search horizon)
///
/// All reachable states and their values are stored in the
/// transposition table for later export to a policy database.
pub fn minimax(
    mechanics: &QGameMechanics,
    data: &mut [u8],
    transposition_table: &DashMap<Vec<u8>, TranspositionEntry>,
) -> Option<i8> {
    minimax_inner(mechanics, data, transposition_table, None)
}

fn minimax_inner(
    mechanics: &QGameMechanics,
    data: &mut [u8],
    transposition_table: &DashMap<Vec<u8>, TranspositionEntry>,
    mut rng: Option<&mut StdRng>,
) -> Option<i8> {
    // Check transposition table for cached result
    if let Some(entry) = transposition_table.get(data) {
        return Some(entry.value);
    }

    let current_player = mechanics.repr().get_current_player(data);
    let opponent = 1 - current_player;

    // The opponent just moved. Check if they won (current player lost).
    if mechanics.check_win(data, opponent) {
        return Some(-1);
    }

    // Draw: max steps reached
    if mechanics.repr().get_completed_steps(data) >= mechanics.repr().max_steps() {
        return Some(0);
    }

    // Get all valid actions for the current player
    let mut actions = get_all_actions(mechanics, data);
    assert!(
        !actions.is_empty(),
        "No valid actions - should never happen"
    );

    // Shuffle actions if an RNG is provided (for Lazy SMP)
    if let Some(ref mut r) = rng {
        actions.shuffle(*r);
    }

    let mut best_known: Option<i8> = None;
    let mut has_unknown = false;

    for &(row, col, action_type) in &actions {
        // Create child state
        let mut new_data = data.to_vec();
        let (r, c, t) = (row as usize, col as usize, action_type as usize);
        if action_type == 2 {
            mechanics.execute_move(&mut new_data, current_player, r, c);
        } else {
            mechanics.execute_wall_placement(&mut new_data, current_player, r, c, t);
        }
        mechanics.switch_player(&mut new_data);

        // Recurse. Child returns value from opponent's perspective; negate for ours.
        let child_value = minimax_inner(
            mechanics,
            &mut new_data,
            transposition_table,
            rng.as_deref_mut(),
        );
        let our_value = child_value.map(|v| -v);

        match our_value {
            Some(v) => {
                if best_known.is_none() || v > best_known.unwrap() {
                    best_known = Some(v);
                }
            }
            None => has_unknown = true,
        }
    }

    // Determine this node's value:
    // - If we found a guaranteed win (1), the value is 1 regardless of unknowns.
    // - If there are unknown children and no guaranteed win, the value is unknown.
    // - If all children are known, the value is the max.
    let value = if best_known == Some(1) {
        Some(1)
    } else if has_unknown {
        None
    } else {
        best_known
    };

    // Store in transposition table (use 0 for unknown)
    transposition_table.insert(
        data.to_vec(),
        TranspositionEntry {
            value: value.unwrap_or(0),
        },
    );

    value
}

/// Lazy SMP parallel minimax. Spawns `num_threads` threads each running
/// the full minimax with randomized move ordering, all sharing the same
/// transposition table. Returns the root value.
pub fn minimax_lazy_smp(
    mechanics: &QGameMechanics,
    data: &mut [u8],
    transposition_table: &DashMap<Vec<u8>, TranspositionEntry>,
    num_threads: usize,
) -> Option<i8> {
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
        // Collect all results; return thread 0's result
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
        let table = DashMap::new();

        let value = minimax(&mechanics, &mut data, &table);

        // On a 3x3 board with no walls, the game should be fully solvable
        assert!(value.is_some(), "Value should be determined");
        println!(
            "3x3 no walls: value = {:?}, table size = {}",
            value,
            table.len()
        );
    }

    #[test]
    fn test_minimax_win_in_one() {
        let mechanics = QGameMechanics::new(3, 0, 10);
        let mut data = mechanics.create_initial_state();

        // Player 0 goal is row 2, player 1 goal is row 0.
        // Place player 0 one step away from goal row.
        mechanics.repr().set_player_position(&mut data, 0, 1, 1);
        mechanics.repr().set_player_position(&mut data, 1, 2, 0);
        mechanics.repr().set_current_player(&mut data, 0);
        mechanics.repr().set_completed_steps(&mut data, 2);

        let table = DashMap::new();
        let value = minimax(&mechanics, &mut data, &table);

        assert_eq!(value, Some(1), "Player 0 should be able to win in one move");
    }

    #[test]
    fn test_minimax_tie_at_max_steps() {
        let mechanics = QGameMechanics::new(3, 0, 2);
        let mut data = mechanics.create_initial_state();
        mechanics.repr().set_completed_steps(&mut data, 2);

        let table = DashMap::new();
        let value = minimax(&mechanics, &mut data, &table);

        assert_eq!(value, Some(0), "Max steps reached should be a tie");
    }

    #[test]
    fn test_transposition_table_populated() {
        let mechanics = QGameMechanics::new(3, 0, 4);
        let mut data = mechanics.create_initial_state();
        let table = DashMap::new();

        minimax(&mechanics, &mut data, &table);

        assert!(!table.is_empty(), "Transposition table should have entries");
        println!("Table has {} entries", table.len());
    }
}
