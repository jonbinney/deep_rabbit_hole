//! JSON-facing snapshot the JS client renders from. Ported from
//! quoridor-rs `play_server::state` (which is binary-gated and not on wasm).

use serde::Serialize;

use quoridor_rs::actions::{
    ACTION_MOVE, ACTION_WALL_HORIZONTAL, ACTION_WALL_VERTICAL, action_index_to_action,
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
    Move {
        index: u32,
        to: [i32; 2],
    },
    Wall {
        index: u32,
        row: i32,
        col: i32,
        orientation: WallOrientation,
    },
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
            index: index as u32,
            row,
            col,
            orientation: WallOrientation::V,
        },
        ACTION_WALL_HORIZONTAL => EnrichedAction::Wall {
            index: index as u32,
            row,
            col,
            orientation: WallOrientation::H,
        },
        ACTION_MOVE => EnrichedAction::Move {
            index: index as u32,
            to: [row, col],
        },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enriched_action_serializes_with_kind_tag() {
        let m = EnrichedAction::Move {
            index: 3,
            to: [4, 5],
        };
        let s = serde_json::to_string(&m).unwrap();
        assert_eq!(s, r#"{"kind":"move","index":3,"to":[4,5]}"#);

        let w = EnrichedAction::Wall {
            index: 17,
            row: 3,
            col: 2,
            orientation: WallOrientation::H,
        };
        let s = serde_json::to_string(&w).unwrap();
        assert_eq!(
            s,
            r#"{"kind":"wall","index":17,"row":3,"col":2,"orientation":"h"}"#
        );
    }

    #[test]
    fn enrich_first_vertical_then_first_horizontal_wall() {
        let n: i32 = 5;
        let nn = (n * n) as usize;
        let walls = ((n - 1) * (n - 1)) as usize;

        let v = enrich_action(n, nn);
        assert_eq!(
            v,
            EnrichedAction::Wall {
                index: nn as u32,
                row: 0,
                col: 0,
                orientation: WallOrientation::V
            }
        );

        let h = enrich_action(n, nn + walls);
        assert_eq!(
            h,
            EnrichedAction::Wall {
                index: (nn + walls) as u32,
                row: 0,
                col: 0,
                orientation: WallOrientation::H
            }
        );
    }
}
