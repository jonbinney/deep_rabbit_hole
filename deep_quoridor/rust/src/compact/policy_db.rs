/// Simplified minimax for building a policy database.
///
/// Uses i8 values (1 = win, 0 = tie, -1 = loss, None = unknown)
/// with a transposition table. No alpha-beta pruning or heuristic.
use dashmap::DashMap;

use super::q_game_mechanics::QGameMechanics;

/// Transposition table entry storing only the best action and its value.
/// The state bytes are stored as the DashMap key rather than in this struct.
#[derive(Clone)]
pub struct TranspositionEntry {
    pub best_action: (u8, u8, u8),
    pub best_value: Option<i8>,
}

/// Get all valid actions (moves + wall placements) for the current player.
fn get_all_actions(mechanics: &QGameMechanics, data: &mut [u8]) -> Vec<(u8, u8, u8)> {
    let moves = mechanics.get_valid_moves(data);
    let mut actions: Vec<(u8, u8, u8)> = moves
        .into_iter()
        .map(|(r, c)| (r as u8, c as u8, 2))
        .collect();
    let walls = mechanics.get_valid_wall_placements(data);
    actions.extend(walls.into_iter().map(|(r, c, t)| (r as u8, c as u8, t as u8)));
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
/// All reachable states and their action values are stored in the
/// transposition table for later export to a policy database.
pub fn minimax(
    mechanics: &QGameMechanics,
    data: &mut [u8],
    transposition_table: &DashMap<Vec<u8>, TranspositionEntry>,
) -> Option<i8> {
    // Check transposition table for cached result
    if let Some(entry) = transposition_table.get(data) {
        return entry.best_value;
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
    let actions = get_all_actions(mechanics, data);
    assert!(!actions.is_empty(), "No valid actions - should never happen");

    let mut best_known: Option<i8> = None;
    let mut best_action: (u8, u8, u8) = actions[0]; // default to first action
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
        let child_value = minimax(mechanics, &mut new_data, transposition_table);
        let our_value = child_value.map(|v| -v);

        match our_value {
            Some(v) => {
                if best_known.is_none() || v > best_known.unwrap() {
                    best_known = Some(v);
                    best_action = (row, col, action_type);
                }
            }
            None => has_unknown = true,
        }
    }

    // Determine this node's value:
    // - If we found a guaranteed win (1), the value is 1 regardless of unknowns.
    // - If there are unknown children and no guaranteed win, the value is unknown.
    // - If all children are known, the value is the max.
    let best_value = if best_known == Some(1) {
        Some(1)
    } else if has_unknown {
        None
    } else {
        best_known
    };

    // Store in transposition table
    transposition_table.insert(
        data.to_vec(),
        TranspositionEntry {
            best_action,
            best_value,
        },
    );

    best_value
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
