//! Agent implementations for Quoridor self-play.
//!
//! All agents implement the [`ActionSelector`] trait.

use crate::game_state::GameState;

#[cfg(feature = "binary")]
pub mod onnx_agent;
pub mod random_agent;

#[cfg(feature = "binary")]
pub mod alphazero;

/// Trait for agents that select actions given a game state.
///
/// The provided state may already be rotated for Player 1.
pub trait ActionSelector {
    /// Select an action given the game state.
    ///
    /// Arguments are in the coordinate frame presented (possibly rotated).
    ///
    /// Returns `(action_index, policy_probabilities)` where `action_index` is
    /// a flat index into the policy vector and `policy_probabilities` is the
    /// full softmax output (or a uniform/mask-based distribution for simpler agents).
    fn select_action(
        &mut self,
        state: &GameState,
        action_mask: &[bool],
    ) -> anyhow::Result<(usize, Vec<f32>)>;
}
