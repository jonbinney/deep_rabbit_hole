//! Pure types and helpers for the play-server `state` JSON shape and for
//! enriching action indices with their semantic board coordinates.

use serde::Serialize;

use crate::actions::{
    ACTION_MOVE, ACTION_WALL_HORIZONTAL, ACTION_WALL_VERTICAL, action_index_to_action,
};

/// Single legal action carried over the wire. The client never needs to know
/// the action-index encoding; it just looks at `kind` and the coords.
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WallOrientation {
    H,
    V,
}

/// Snapshot of a `GameSession` for the client to render. Built by
/// `session::GameSession::view()` from the underlying `QGameMechanics` +
/// `CompactState`.
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

#[derive(Debug, Clone, Serialize)]
pub struct WallEntry {
    pub row: i32,
    pub col: i32,
    pub orientation: WallOrientation,
}

impl EnrichedAction {
    /// The bare action index common to all variants.
    pub fn index(&self) -> u32 {
        match self {
            EnrichedAction::Move { index, .. } => *index,
            EnrichedAction::Wall { index, .. } => *index,
        }
    }
}

/// Map an action index to its semantic enrichment (move dest / wall coords +
/// orientation). Matches the convention in `actions::action_index_to_action`:
/// indices < N*N are moves; the next (N-1)^2 are vertical walls; the
/// remaining (N-1)^2 are horizontal walls.
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

/// Apply `enrich_action` to every legal index in `mask`.
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
    use crate::actions::policy_size;

    #[test]
    fn enriched_action_index_returns_bare_index() {
        assert_eq!(enrich_action(5, 7).index(), 7);
        assert_eq!(enrich_action(5, 25).index(), 25); // first vertical wall on 5x5
    }

    #[test]
    fn enrich_move_action_round_trips_coords() {
        // First action on a 5x5 board: move to (0, 0).
        let a = enrich_action(5, 0);
        assert_eq!(
            a,
            EnrichedAction::Move {
                index: 0,
                to: [0, 0]
            }
        );

        // Index N*N - 1 on 5x5 = last move cell = (4, 4).
        let a = enrich_action(5, 24);
        assert_eq!(
            a,
            EnrichedAction::Move {
                index: 24,
                to: [4, 4]
            }
        );
    }

    #[test]
    fn enrich_first_vertical_then_first_horizontal_wall() {
        let n: i32 = 5;
        let nn = (n * n) as usize;
        let walls = ((n - 1) * (n - 1)) as usize;

        // First vertical wall is at index N*N.
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

        // First horizontal wall is at index N*N + (N-1)^2.
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

    #[test]
    fn enrich_legal_actions_filters_by_mask() {
        let n = 5;
        let size = policy_size(n);
        let mut mask = vec![false; size];
        mask[0] = true;
        mask[(n * n) as usize] = true; // first vertical wall

        let actions = enrich_legal_actions(n, &mask);
        assert_eq!(actions.len(), 2);
        assert!(matches!(actions[0], EnrichedAction::Move { index: 0, .. }));
        assert!(matches!(
            actions[1],
            EnrichedAction::Wall {
                orientation: WallOrientation::V,
                ..
            }
        ));
    }

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
}
