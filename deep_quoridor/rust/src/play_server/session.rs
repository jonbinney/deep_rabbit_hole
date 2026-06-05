//! Per-game state (`GameSession`) and the shared registry that the HTTP
//! handlers look up by `game_id`.
//!
//! Each session owns its own `AlphaZeroAgent` so games run independently. The
//! registry holds an `Arc<Mutex<GameSession>>` per game; an HTTP handler takes
//! the outer `Mutex` briefly to look up the session and then holds the inner
//! `Mutex` for the duration of the move + AI response.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};
use rand::RngCore;

use crate::agents::ActionSelector;
use crate::agents::alphazero::agent::{AlphaZeroAgent, AlphaZeroAgentConfig};
#[cfg(test)]
use crate::agents::alphazero::evaluator::Evaluator;
use crate::agents::alphazero::mcts::MCTSConfig;
use crate::compact::q_bit_repr::{CompactState, WALL_HORIZONTAL, WALL_VERTICAL};
use crate::compact::q_game_mechanics::QGameMechanics;
use crate::play_server::config::ServerConfig;
use crate::play_server::state::{
    EnrichedAction, StateView, WallEntry, WallOrientation, enrich_action, enrich_legal_actions,
};

pub type GameId = String;

/// One running game: owns the agent and the board state.
pub struct GameSession {
    pub mechanics: QGameMechanics,
    pub state: CompactState,
    pub agent: AlphaZeroAgent,
    pub board_size: i32,
    pub max_walls: i32,
    pub max_steps: i32,
    pub human_player: i32,
    pub last_action: Option<EnrichedAction>,
    pub move_history: Vec<u32>,
}

impl GameSession {
    /// Construct an agent config matching the server's notion of "play-mode":
    /// the requested `mcts_n`, temperature 0 (argmax visits), deterministic
    /// tie-break, no Dirichlet noise, and the server's max_steps as the MCTS
    /// search cap.
    pub fn agent_config(mcts_n: u32, c_puct: f32, max_steps: i32) -> AlphaZeroAgentConfig {
        AlphaZeroAgentConfig {
            mcts: MCTSConfig {
                n: Some(mcts_n),
                k: None,
                ucb_c: c_puct,
                noise_epsilon: 0.0,
                noise_alpha: None,
                max_steps: Some(max_steps),
                penalize_visited_states: false,
            },
            temperature: 0.0,
            drop_t_on_step: None,
            penalize_visited_states: false,
            deterministic_tie_break: true,
        }
    }

    /// Build a session that loads ONNX from disk. The session's mechanics +
    /// initial state come from `cfg`.
    pub fn new_from_onnx(
        cfg: &ServerConfig,
        model_path: &std::path::Path,
        mcts_n: u32,
        human_player: i32,
    ) -> Result<Self> {
        let mechanics = QGameMechanics::new(
            cfg.board_size as usize,
            cfg.max_walls as usize,
            cfg.max_steps as usize,
        );
        let state = mechanics.create_initial_state();
        let model_str = model_path
            .to_str()
            .ok_or_else(|| anyhow!("model path is not valid UTF-8"))?;
        let agent_config = Self::agent_config(mcts_n, cfg.default_mcts_c_puct, cfg.max_steps);
        let agent =
            AlphaZeroAgent::new(model_str, agent_config).context("constructing AlphaZeroAgent")?;
        Ok(Self {
            mechanics,
            state,
            agent,
            board_size: cfg.board_size,
            max_walls: cfg.max_walls,
            max_steps: cfg.max_steps,
            human_player,
            last_action: None,
            move_history: Vec::new(),
        })
    }

    /// Test-only variant that injects a fake evaluator instead of loading ORT.
    #[cfg(test)]
    pub fn new_with_evaluator(
        board_size: i32,
        max_walls: i32,
        max_steps: i32,
        evaluator: Box<dyn Evaluator + Send>,
        mcts_n: u32,
        c_puct: f32,
        human_player: i32,
    ) -> Self {
        let mechanics =
            QGameMechanics::new(board_size as usize, max_walls as usize, max_steps as usize);
        let state = mechanics.create_initial_state();
        let agent_config = Self::agent_config(mcts_n, c_puct, max_steps);
        let agent = AlphaZeroAgent::with_evaluator(evaluator, agent_config);
        Self {
            mechanics,
            state,
            agent,
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

    fn is_game_over(&self) -> bool {
        self.mechanics.is_game_over(self.state)
            || self.mechanics.repr().get_completed_steps(self.state) >= self.max_steps as usize
    }

    fn legal_mask(&mut self) -> Vec<bool> {
        self.mechanics.get_action_mask_immut(self.state)
    }

    /// Apply one action (no matter whose turn). Records it in `last_action` +
    /// `move_history`. Returns an error if the action is illegal.
    pub fn apply_action(&mut self, action_index: u32) -> Result<()> {
        if self.is_game_over() {
            return Err(anyhow!("game is already over"));
        }
        let mask = self.legal_mask();
        let idx = action_index as usize;
        if idx >= mask.len() || !mask[idx] {
            return Err(anyhow!("action {action_index} is not legal"));
        }
        self.last_action = Some(enrich_action(self.board_size, idx));
        self.move_history.push(action_index);
        self.mechanics.apply_action_index(&mut self.state, idx);
        Ok(())
    }

    /// Run the AI for one move on the current state. Errors if it's actually
    /// the human's turn or the game is over.
    pub fn ai_step(&mut self) -> Result<()> {
        if self.is_game_over() {
            return Ok(());
        }
        if self.current_player() == self.human_player {
            return Err(anyhow!("not AI's turn"));
        }
        let mask = self.legal_mask();
        let (action_idx, _policy) = self
            .agent
            .select_action(self.state, &self.mechanics, &mask)
            .context("AI MCTS selection")?;
        self.last_action = Some(enrich_action(self.board_size, action_idx));
        self.move_history.push(action_idx as u32);
        self.mechanics
            .apply_action_index(&mut self.state, action_idx);
        Ok(())
    }

    /// Build the JSON-facing snapshot the client renders from.
    pub fn view(&mut self) -> StateView {
        let mask = self.legal_mask();
        let legal_actions = enrich_legal_actions(self.board_size, &mask);
        let repr = self.mechanics.repr();
        let (p1r, p1c) = repr.get_player_position(self.state, 0);
        let (p2r, p2c) = repr.get_player_position(self.state, 1);
        let p1w = repr.get_walls_remaining(self.state, 0) as i32;
        let p2w = repr.get_walls_remaining(self.state, 1) as i32;
        let completed_steps = repr.get_completed_steps(self.state) as i32;
        let winner = if self.mechanics.check_win(self.state, 0) {
            Some(0)
        } else if self.mechanics.check_win(self.state, 1) {
            Some(1)
        } else {
            None
        };

        StateView {
            board_size: self.board_size,
            max_walls: self.max_walls,
            max_steps: self.max_steps,
            current_player: self.current_player(),
            p1_pos: [p1r as i32, p1c as i32],
            p2_pos: [p2r as i32, p2c as i32],
            p1_walls: p1w,
            p2_walls: p2w,
            walls: list_walls(&self.mechanics, self.state, self.board_size),
            legal_actions,
            completed_steps,
            winner,
            human_player: self.human_player,
            last_action: self.last_action.clone(),
            move_history: self.move_history.clone(),
        }
    }
}

/// Iterate every potential wall slot (`(N-1)^2` for each orientation) and ask
/// the mechanics whether a wall is currently present at that location.
///
/// The `QBitRepr::get_wall` API takes `orientation: usize` where
/// `WALL_VERTICAL=0` and `WALL_HORIZONTAL=1`.
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

/// Thread-safe map `game_id -> GameSession`.
#[derive(Clone, Default)]
pub struct GameRegistry {
    inner: Arc<Mutex<HashMap<GameId, Arc<Mutex<GameSession>>>>>,
}

impl GameRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, session: GameSession) -> GameId {
        let id = new_game_id();
        self.inner
            .lock()
            .unwrap()
            .insert(id.clone(), Arc::new(Mutex::new(session)));
        id
    }

    pub fn get(&self, game_id: &str) -> Option<Arc<Mutex<GameSession>>> {
        self.inner.lock().unwrap().get(game_id).cloned()
    }
}

fn new_game_id() -> GameId {
    let mut bytes = [0u8; 4];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::alphazero::evaluator::UniformMockEvaluator;

    fn mock_session(human_player: i32) -> GameSession {
        GameSession::new_with_evaluator(
            5,
            2,
            50,
            Box::new(UniformMockEvaluator),
            8,
            1.4,
            human_player,
        )
    }

    fn register(reg: &GameRegistry, s: GameSession) -> String {
        reg.insert(s)
    }

    #[test]
    fn initial_view_has_pawns_on_home_rows_and_no_walls() {
        let mut s = mock_session(0);
        let v = s.view();
        assert_eq!(v.board_size, 5);
        assert_eq!(v.max_walls, 2);
        assert_eq!(v.current_player, 0);
        assert_eq!(v.completed_steps, 0);
        assert!(v.walls.is_empty());
        assert_eq!(v.p1_walls, 2);
        assert_eq!(v.p2_walls, 2);
        assert!(v.winner.is_none());
        assert_eq!(v.human_player, 0);
        assert!(v.last_action.is_none());
        assert!(v.move_history.is_empty());
        let has_move = v
            .legal_actions
            .iter()
            .any(|a| matches!(a, EnrichedAction::Move { .. }));
        assert!(has_move);
    }

    #[test]
    fn apply_action_records_last_action_and_advances_player() {
        let mut s = mock_session(0);
        let mask = s.legal_mask();
        let first_legal_move = mask
            .iter()
            .enumerate()
            .find(|&(_, &b)| b)
            .map(|(i, _)| i as u32)
            .expect("at least one legal action");
        s.apply_action(first_legal_move).unwrap();
        assert_eq!(s.move_history, vec![first_legal_move]);
        assert!(s.last_action.is_some());
        assert_eq!(s.current_player(), 1);
    }

    #[test]
    fn apply_action_rejects_illegal_index() {
        let mut s = mock_session(0);
        let mask = s.legal_mask();
        let illegal = mask
            .iter()
            .enumerate()
            .find(|&(_, &b)| !b)
            .map(|(i, _)| i as u32)
            .expect("at least one illegal action");
        let err = s.apply_action(illegal).unwrap_err();
        assert!(err.to_string().contains("not legal"));
    }

    #[test]
    fn ai_step_errors_when_its_human_turn() {
        let mut s = mock_session(0);
        let err = s.ai_step().unwrap_err();
        assert!(err.to_string().contains("not AI's turn"));
    }

    #[test]
    fn ai_step_runs_when_its_ai_turn() {
        let mut s = mock_session(1);
        s.ai_step().unwrap();
        assert_eq!(s.move_history.len(), 1);
        assert!(s.last_action.is_some());
        assert_eq!(s.current_player(), 1);
    }

    #[test]
    fn registry_insert_and_get_round_trip() {
        let reg = GameRegistry::new();
        let id = register(&reg, mock_session(0));
        assert!(reg.get(&id).is_some());
        assert!(reg.get("does-not-exist").is_none());
    }

    #[test]
    fn game_id_is_8_hex_chars() {
        let id = new_game_id();
        assert_eq!(id.len(), 8);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
