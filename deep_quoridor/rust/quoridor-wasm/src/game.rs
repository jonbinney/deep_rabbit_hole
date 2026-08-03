//! Per-game session exposed to JS. Owns mechanics + state + move history and
//! produces `StateView` snapshots.

use quoridor_rs::compact::q_bit_repr::{CompactState, WALL_HORIZONTAL, WALL_VERTICAL};
use quoridor_rs::compact::q_game_mechanics::QGameMechanics;

use crate::view::{
    EnrichedAction, StateView, WallEntry, WallOrientation, enrich_action, enrich_legal_actions,
};

pub struct WasmGame {
    mechanics: QGameMechanics,
    state: CompactState,
    board_size: i32,
    max_walls: i32,
    max_steps: i32,
    human_player: i32,
    last_action: Option<EnrichedAction>,
    move_history: Vec<u32>,
}

impl WasmGame {
    pub fn new(board_size: i32, max_walls: i32, max_steps: i32, human_player: i32) -> Self {
        let mechanics =
            QGameMechanics::new(board_size as usize, max_walls as usize, max_steps as usize);
        let state = mechanics.create_initial_state();
        Self {
            mechanics,
            state,
            board_size,
            max_walls,
            max_steps,
            human_player,
            last_action: None,
            move_history: Vec::new(),
        }
    }

    fn current_player(&self) -> i32 {
        self.mechanics.repr().get_current_player(self.state) as i32
    }

    /// A win, or hitting the step cap (draw). This is the crate's single source
    /// of truth for "game over" — `apply_action` and `run_search_js` both use it.
    pub fn is_game_over(&self) -> bool {
        self.mechanics.is_game_over(self.state)
            || self.mechanics.repr().get_completed_steps(self.state) >= self.max_steps as usize
    }

    pub fn legal_mask(&self) -> Vec<bool> {
        self.mechanics.get_action_mask_immut(self.state)
    }

    /// Apply any legal action (human or AI). Returns Err on illegal/over.
    pub fn apply_action(&mut self, action_index: u32) -> Result<(), String> {
        if self.is_game_over() {
            return Err("game is already over".into());
        }
        let mask = self.legal_mask();
        let idx = action_index as usize;
        if idx >= mask.len() || !mask[idx] {
            return Err(format!("action {action_index} is not legal"));
        }
        self.last_action = Some(enrich_action(self.board_size, idx));
        self.move_history.push(action_index);
        self.mechanics.apply_action_index(&mut self.state, idx);
        Ok(())
    }

    /// Undo the last `count` plies by replaying history from the initial state.
    pub fn undo(&mut self, count: usize) {
        let keep = self.move_history.len().saturating_sub(count);
        let replay: Vec<u32> = self.move_history[..keep].to_vec();
        self.state = self.mechanics.create_initial_state();
        self.move_history.clear();
        self.last_action = None;
        for a in replay {
            let idx = a as usize;
            self.last_action = Some(enrich_action(self.board_size, idx));
            self.move_history.push(a);
            self.mechanics.apply_action_index(&mut self.state, idx);
        }
    }

    pub fn state(&self) -> CompactState {
        self.state
    }
    pub fn mechanics(&self) -> &QGameMechanics {
        &self.mechanics
    }

    pub fn view(&self) -> StateView {
        let mask = self.legal_mask();
        let repr = self.mechanics.repr();
        let (p1r, p1c) = repr.get_player_position(self.state, 0);
        let (p2r, p2c) = repr.get_player_position(self.state, 1);
        let winner = self.mechanics.winner(self.state).map(|p| p as i32);
        StateView {
            board_size: self.board_size,
            max_walls: self.max_walls,
            max_steps: self.max_steps,
            current_player: self.current_player(),
            p1_pos: [p1r as i32, p1c as i32],
            p2_pos: [p2r as i32, p2c as i32],
            p1_walls: repr.get_walls_remaining(self.state, 0) as i32,
            p2_walls: repr.get_walls_remaining(self.state, 1) as i32,
            walls: list_walls(&self.mechanics, self.state, self.board_size),
            legal_actions: enrich_legal_actions(self.board_size, &mask),
            completed_steps: repr.get_completed_steps(self.state) as i32,
            winner,
            human_player: self.human_player,
            last_action: self.last_action.clone(),
            move_history: self.move_history.clone(),
        }
    }
}

fn list_walls(mechanics: &QGameMechanics, state: CompactState, board_size: i32) -> Vec<WallEntry> {
    let mut out = Vec::new();
    let wall_size = (board_size - 1) as usize;
    for (orientation_const, orientation) in [
        (WALL_VERTICAL, WallOrientation::V),
        (WALL_HORIZONTAL, WallOrientation::H),
    ] {
        for row in 0..wall_size {
            for col in 0..wall_size {
                if mechanics
                    .repr()
                    .get_wall(state, row, col, orientation_const)
                {
                    out.push(WallEntry {
                        row: row as i32,
                        col: col as i32,
                        orientation,
                    });
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_then_undo_restores_state() {
        let mut g = WasmGame::new(5, 2, 50, 0);
        let mask = g.legal_mask();
        let first = mask.iter().position(|&b| b).unwrap() as u32;
        g.apply_action(first).unwrap();
        assert_eq!(g.view().move_history, vec![first]);
        g.undo(1);
        let v = g.view();
        assert!(v.move_history.is_empty());
        assert!(v.last_action.is_none());
        assert_eq!(v.current_player, 0);
    }

    #[test]
    fn apply_rejects_illegal_action() {
        let mut g = WasmGame::new(5, 2, 50, 0);
        let err = g.apply_action(u32::MAX).unwrap_err();
        assert!(err.contains("not legal"));
    }
}
