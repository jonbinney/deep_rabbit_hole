//! JSON-facing snapshot the JS client renders from. Ported from
//! quoridor-rs `play_server::state` (which is binary-gated and not on wasm).

use serde::Serialize;

use quoridor_rs::actions::{
    action_index_to_action, ACTION_MOVE, ACTION_WALL_HORIZONTAL, ACTION_WALL_VERTICAL,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WallOrientation {
    H,
    V,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum EnrichedAction {
    Move { index: u32, to: [i32; 2] },
    Wall { index: u32, row: i32, col: i32, orientation: WallOrientation },
}

#[derive(Debug, Clone, Serialize)]
pub struct WallEntry {
    pub row: i32,
    pub col: i32,
    pub orientation: WallOrientation,
}

#[derive(Debug, Clone, Serialize)]
pub struct StateView {
    pub board_size: i32,
    pub max_walls: i32,
    pub max_steps: i32,
    pub current_player: i32,
    pub p1_pos: [i32; 2],
    pub p2_pos: [i32; 2],
    pub p1_walls: i32,
    pub p2_walls: i32,
    pub walls: Vec<WallEntry>,
    pub legal_actions: Vec<EnrichedAction>,
    pub completed_steps: i32,
    pub winner: Option<i32>,
    pub human_player: i32,
    pub last_action: Option<EnrichedAction>,
    pub move_history: Vec<u32>,
}

pub fn enrich_action(board_size: i32, index: usize) -> EnrichedAction {
    let [row, col, action_type] = action_index_to_action(board_size, index);
    match action_type {
        ACTION_WALL_VERTICAL => EnrichedAction::Wall {
            index: index as u32, row, col, orientation: WallOrientation::V,
        },
        ACTION_WALL_HORIZONTAL => EnrichedAction::Wall {
            index: index as u32, row, col, orientation: WallOrientation::H,
        },
        ACTION_MOVE => EnrichedAction::Move { index: index as u32, to: [row, col] },
        other => panic!("unexpected action type {other} for index {index}"),
    }
}

pub fn enrich_legal_actions(board_size: i32, mask: &[bool]) -> Vec<EnrichedAction> {
    mask.iter()
        .enumerate()
        .filter(|&(_, legal)| *legal)
        .map(|(i, _)| enrich_action(board_size, i))
        .collect()
}
