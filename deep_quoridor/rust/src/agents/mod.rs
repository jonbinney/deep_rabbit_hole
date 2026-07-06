//! Agent implementations for Quoridor self-play.
//!
//! All agents implement the [`ActionSelector`] trait.

use crate::compact::q_bit_repr::CompactState;
use crate::compact::q_game_mechanics::QGameMechanics;

#[cfg(feature = "binary")]
pub mod onnx_agent;
pub mod random_agent;

pub mod alphazero;

#[derive(Debug, Clone, Default)]
pub struct ActionSelectionTrace {
    pub root_value: Option<f32>,
}

/// Trait for agents that select actions given a compact game state.
///
/// The provided state may already be rotated for Player 1 — agents should treat
/// `data` as the canonical state in whatever frame is supplied.
pub trait ActionSelector {
    /// Select an action given the compact game state.
    ///
    /// Returns `(action_index, policy_probabilities)` where `action_index` is
    /// a flat index into the policy vector and `policy_probabilities` is the
    /// full softmax output (or a uniform/mask-based distribution for simpler agents).
    fn select_action(
        &mut self,
        data: CompactState,
        mechanics: &QGameMechanics,
        action_mask: &[bool],
    ) -> anyhow::Result<(usize, Vec<f32>)>;

    fn last_selection_trace(&self) -> Option<ActionSelectionTrace> {
        None
    }
}
