//! Pure-function HTTP handlers for the play server.
//!
//! Each handler takes the inputs it needs (registry + parsed JSON) and returns
//! a `Result<T, HandlerError>` where `T` is `serde::Serialize`. Task 6 wires
//! these into `tiny_http` requests and maps errors to status codes.

use serde::{Deserialize, Serialize};

use crate::play_server::config::ServerConfig;
use crate::play_server::session::{GameRegistry, GameSession};
use crate::play_server::state::StateView;

/// Error kind exposed to the HTTP layer. Each variant maps to one status code.
#[derive(Debug)]
pub enum HandlerError {
    /// 400: client sent something the server cannot honor (unknown model,
    /// illegal move, malformed JSON the handler caught itself).
    BadRequest(String),
    /// 404: `game_id` not found in the registry.
    NotFound(String),
    /// 500: internal failure (ORT load, MCTS, IO).
    Internal(String),
}

impl HandlerError {
    pub fn message(&self) -> &str {
        match self {
            HandlerError::BadRequest(m) | HandlerError::NotFound(m) | HandlerError::Internal(m) => {
                m
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ConfigView {
    pub board_size: i32,
    pub max_walls: i32,
    pub max_steps: i32,
    pub models: Vec<String>,
    pub default_mcts_n: u32,
}

#[derive(Deserialize)]
pub struct NewGameRequest {
    pub model: String,
    pub mcts_n: u32,
    pub human_player: i32,
}

#[derive(Debug, Serialize)]
pub struct NewGameResponse {
    pub game_id: String,
    pub state: StateView,
}

#[derive(Deserialize)]
pub struct MoveRequest {
    pub action_index: u32,
}

#[derive(Debug, Serialize)]
pub struct StateResponse {
    pub state: StateView,
}

/// `GET /api/config` — server tells the client what board/models are available.
pub fn get_config(cfg: &ServerConfig, default_mcts_n: u32) -> ConfigView {
    ConfigView {
        board_size: cfg.board_size,
        max_walls: cfg.max_walls,
        max_steps: cfg.max_steps,
        models: cfg.models.clone(),
        default_mcts_n,
    }
}

/// `POST /api/games` — create a new game with the chosen model + mcts_n.
pub fn create_game(
    cfg: &ServerConfig,
    registry: &GameRegistry,
    req: NewGameRequest,
) -> Result<NewGameResponse, HandlerError> {
    if req.human_player != 0 && req.human_player != 1 {
        return Err(HandlerError::BadRequest(format!(
            "human_player must be 0 or 1, got {}",
            req.human_player
        )));
    }
    let model_path = cfg
        .model_path(&req.model)
        .ok_or_else(|| HandlerError::BadRequest(format!("unknown model: {}", req.model)))?;
    let mut session = GameSession::new_from_onnx(cfg, &model_path, req.mcts_n, req.human_player)
        .map_err(|e| HandlerError::Internal(format!("failed to create session: {e:#}")))?;
    // If the AI moves first, take its move now so the initial state the client
    // renders already shows it.
    if session.human_player != session_current_player(&mut session) {
        session
            .ai_step()
            .map_err(|e| HandlerError::Internal(format!("AI opening move failed: {e:#}")))?;
    }
    let state = session.view();
    let game_id = registry.insert(session);
    Ok(NewGameResponse { game_id, state })
}

/// `GET /api/games/<game_id>` — fetch the current state of an existing game.
pub fn get_game(registry: &GameRegistry, game_id: &str) -> Result<StateResponse, HandlerError> {
    let session_arc = registry
        .get(game_id)
        .ok_or_else(|| HandlerError::NotFound(format!("game {game_id} not found")))?;
    let mut session = session_arc.lock().unwrap();
    Ok(StateResponse {
        state: session.view(),
    })
}

/// `POST /api/games/<game_id>/move` — apply the human's move, then (if it's
/// then the AI's turn) the AI's response in the same round-trip.
pub fn apply_move(
    registry: &GameRegistry,
    game_id: &str,
    req: MoveRequest,
) -> Result<StateResponse, HandlerError> {
    let session_arc = registry
        .get(game_id)
        .ok_or_else(|| HandlerError::NotFound(format!("game {game_id} not found")))?;
    let mut session = session_arc.lock().unwrap();
    session
        .apply_action(req.action_index)
        .map_err(|e| HandlerError::BadRequest(format!("{e:#}")))?;
    // If after the human move it's now the AI's turn (and the game isn't
    // over), run one AI step.
    if session_should_ai_move(&mut session) {
        session
            .ai_step()
            .map_err(|e| HandlerError::Internal(format!("AI step failed: {e:#}")))?;
    }
    Ok(StateResponse {
        state: session.view(),
    })
}

/// Helper: read current player without depending on private session API.
fn session_current_player(session: &mut GameSession) -> i32 {
    session.view().current_player
}

/// Helper: should the AI take its move right now?
fn session_should_ai_move(session: &mut GameSession) -> bool {
    let v = session.view();
    v.winner.is_none() && v.current_player != v.human_player
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::alphazero::evaluator::UniformMockEvaluator;
    use crate::play_server::session::GameSession;
    use std::path::PathBuf;

    fn fake_cfg() -> ServerConfig {
        ServerConfig {
            play_dir: PathBuf::from("/tmp/does-not-matter"),
            board_size: 5,
            max_walls: 2,
            max_steps: 50,
            default_mcts_c_puct: 1.4,
            models: vec!["a.onnx".to_string(), "b.onnx".to_string()],
        }
    }

    fn fresh_registry_with_session(human_player: i32) -> (GameRegistry, String) {
        let reg = GameRegistry::new();
        let session = GameSession::new_with_evaluator(
            5,
            2,
            50,
            Box::new(UniformMockEvaluator),
            8,
            1.4,
            human_player,
        );
        let id = reg.insert(session);
        (reg, id)
    }

    #[test]
    fn get_config_returns_models_and_dimensions() {
        let cfg = fake_cfg();
        let view = get_config(&cfg, 400);
        assert_eq!(view.board_size, 5);
        assert_eq!(view.max_walls, 2);
        assert_eq!(view.max_steps, 50);
        assert_eq!(view.models, vec!["a.onnx", "b.onnx"]);
        assert_eq!(view.default_mcts_n, 400);
    }

    #[test]
    fn create_game_rejects_unknown_model() {
        let cfg = fake_cfg();
        let reg = GameRegistry::new();
        let req = NewGameRequest {
            model: "missing.onnx".to_string(),
            mcts_n: 16,
            human_player: 0,
        };
        let err = create_game(&cfg, &reg, req).unwrap_err();
        assert!(matches!(err, HandlerError::BadRequest(_)));
        assert!(err.message().contains("unknown model"));
    }

    #[test]
    fn create_game_rejects_bad_human_player() {
        let cfg = fake_cfg();
        let reg = GameRegistry::new();
        let req = NewGameRequest {
            model: "a.onnx".to_string(),
            mcts_n: 16,
            human_player: 7,
        };
        let err = create_game(&cfg, &reg, req).unwrap_err();
        assert!(matches!(err, HandlerError::BadRequest(_)));
        assert!(err.message().contains("human_player"));
    }

    #[test]
    fn get_game_returns_state_for_known_id() {
        let (reg, id) = fresh_registry_with_session(0);
        let resp = get_game(&reg, &id).unwrap();
        assert_eq!(resp.state.board_size, 5);
        assert_eq!(resp.state.current_player, 0);
    }

    #[test]
    fn get_game_404s_for_unknown_id() {
        let reg = GameRegistry::new();
        let err = get_game(&reg, "deadbeef").unwrap_err();
        assert!(matches!(err, HandlerError::NotFound(_)));
    }

    #[test]
    fn apply_move_rejects_illegal_action() {
        let (reg, id) = fresh_registry_with_session(0);
        // Pick an action index that is definitely not legal on a fresh
        // 5x5/2-wall board: walls require board space and the initial mask
        // disallows most wall slots. Use a far-out index.
        let req = MoveRequest {
            action_index: u32::MAX,
        };
        let err = apply_move(&reg, &id, req).unwrap_err();
        assert!(matches!(err, HandlerError::BadRequest(_)));
    }

    #[test]
    fn apply_move_404s_for_unknown_id() {
        let reg = GameRegistry::new();
        let req = MoveRequest { action_index: 0 };
        let err = apply_move(&reg, "deadbeef", req).unwrap_err();
        assert!(matches!(err, HandlerError::NotFound(_)));
    }

    #[test]
    fn apply_move_legal_human_action_advances_state() {
        let (reg, id) = fresh_registry_with_session(0);
        // Find a legal action from the current view.
        let first_legal = {
            let session_arc = reg.get(&id).unwrap();
            let mut s = session_arc.lock().unwrap();
            let v = s.view();
            v.legal_actions
                .first()
                .expect("at least one legal action")
                .index()
        };
        let req = MoveRequest {
            action_index: first_legal,
        };
        let resp = apply_move(&reg, &id, req).unwrap();
        // After the human moves it becomes the AI's turn — the handler then
        // runs the AI step automatically — so move_history has at least 2.
        assert!(resp.state.move_history.len() >= 2);
    }
}
