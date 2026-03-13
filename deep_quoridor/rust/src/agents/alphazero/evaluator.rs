//! Evaluator trait and ONNX implementation for MCTS.

use anyhow::{Context, Result};
use ort::session::Session;

use crate::agents::onnx_agent::softmax;
use crate::game_state::GameState;
use crate::grid_helpers::grid_game_state_to_resnet_input;

/// Trait for evaluating game positions.
///
/// Returns `(value_for_current_player, masked_softmax_priors)`.
pub trait Evaluator {
    fn evaluate(&self, state: &GameState, action_mask: &[bool]) -> Result<(f32, Vec<f32>)>;
}

/// ONNX-based evaluator for MCTS.
///
/// Loads a neural network model and uses it to evaluate positions,
/// returning both a value estimate and policy priors.
pub struct OnnxEvaluator {
    session: Session,
}

impl OnnxEvaluator {
    /// Create a new evaluator from an ONNX model file.
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;
        Ok(Self { session })
    }
}

impl Evaluator for OnnxEvaluator {
    fn evaluate(&self, state: &GameState, action_mask: &[bool]) -> Result<(f32, Vec<f32>)> {
        // Build ResNet input tensor
        let resnet_input = grid_game_state_to_resnet_input(state);

        // Convert to flat vec for ORT
        let shape = resnet_input.shape().to_vec();
        let data: Vec<f32> = resnet_input.iter().copied().collect();
        let input_value = ort::value::Value::from_array((shape.as_slice(), data))
            .context("Failed to create ONNX input value")?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs!["input" => input_value])
            .context("Failed to run ONNX inference")?;

        // Extract value
        let value_tensor = outputs["value"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract value")?;
        let value = value_tensor.1[0];

        // Extract policy logits and apply mask
        let policy_logits = outputs["policy_logits"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract policy logits")?;

        // Apply masked softmax to get priors
        let priors = masked_softmax(policy_logits.1, action_mask);

        Ok((value, priors))
    }
}

/// Apply masked softmax to logits.
///
/// Invalid actions (where mask is false) get ~0 probability.
pub fn masked_softmax(logits: &[f32], mask: &[bool]) -> Vec<f32> {
    let masked: Vec<f32> = logits
        .iter()
        .zip(mask.iter())
        .map(|(&l, &valid)| if valid { l } else { -1e32 })
        .collect();
    softmax(&masked)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masked_softmax_valid_only() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = vec![false, true, false, true, false];

        let probs = masked_softmax(&logits, &mask);

        // Invalid actions should have ~0 probability
        assert!(probs[0] < 1e-10);
        assert!(probs[2] < 1e-10);
        assert!(probs[4] < 1e-10);

        // Valid actions should have non-zero probability
        assert!(probs[1] > 0.0);
        assert!(probs[3] > 0.0);

        // Sum of valid probabilities should be ~1
        let valid_sum: f32 = probs[1] + probs[3];
        assert!((valid_sum - 1.0).abs() < 1e-5);

        // Higher logit should have higher probability
        assert!(probs[3] > probs[1]);
    }

    #[test]
    fn test_masked_softmax_all_valid() {
        let logits = vec![1.0, 2.0, 3.0];
        let mask = vec![true, true, true];

        let probs = masked_softmax(&logits, &mask);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_masked_softmax_single_valid() {
        let logits = vec![1.0, 2.0, 3.0];
        let mask = vec![false, true, false];

        let probs = masked_softmax(&logits, &mask);

        // Single valid action should get probability ~1
        assert!((probs[1] - 1.0).abs() < 1e-5);
        assert!(probs[0] < 1e-10);
        assert!(probs[2] < 1e-10);
    }
}
