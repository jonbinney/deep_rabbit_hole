//! Evaluator trait and ONNX implementation for MCTS.

use anyhow::Result;

use crate::game_state::GameState;

/// Trait for evaluating game positions.
///
/// Returns `(value_for_current_player, masked_softmax_priors)`.
pub trait Evaluator {
    fn evaluate(&self, state: &GameState, action_mask: &[bool]) -> Result<(f32, Vec<f32>)>;
}
