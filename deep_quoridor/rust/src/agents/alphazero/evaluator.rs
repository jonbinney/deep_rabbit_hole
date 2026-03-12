//! Evaluator trait and ONNX implementation for MCTS.

use anyhow::{Context, Result};
use ort::session::Session;

use crate::agents::onnx_agent::softmax;
use crate::game_state::GameState;
use crate::grid_helpers::grid_game_state_to_resnet_input;
use crate::rotation::{
    create_rotation_mapping, remap_mask, rotate_goal_rows, rotate_grid_180,
    rotate_player_positions,
};

/// Trait for evaluating game positions.
///
/// Returns `(value_for_current_player, masked_softmax_priors)`.
pub trait Evaluator {
    fn evaluate(&mut self, state: &GameState, action_mask: &[bool]) -> Result<(f32, Vec<f32>)>;
}

/// ONNX-based evaluator for MCTS.
///
/// Loads a neural network model and uses it to evaluate positions,
/// returning both a value estimate and policy priors.
pub struct OnnxEvaluator {
    session: Session,
    /// Cached rotation mapping: rotated_to_original index array.
    /// Lazily initialized on first player-1 evaluation.
    rot_to_orig: Option<Vec<usize>>,
    /// Cached rotation mapping: original_to_rotated index array.
    orig_to_rot: Option<Vec<usize>>,
}

impl OnnxEvaluator {
    /// Create a new evaluator from an ONNX model file.
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;
        Ok(Self {
            session,
            rot_to_orig: None,
            orig_to_rot: None,
        })
    }

    /// Get or initialize the rotation mappings for the given board size.
    fn get_rotation_mappings(&mut self, board_size: i32) -> (&[usize], &[usize]) {
        if self.orig_to_rot.is_none() {
            let (orig_to_rot, rot_to_orig) = create_rotation_mapping(board_size);
            self.orig_to_rot = Some(orig_to_rot);
            self.rot_to_orig = Some(rot_to_orig);
        }
        (
            self.orig_to_rot.as_ref().unwrap(),
            self.rot_to_orig.as_ref().unwrap(),
        )
    }
}

impl Evaluator for OnnxEvaluator {
    fn evaluate(&mut self, state: &GameState, action_mask: &[bool]) -> Result<(f32, Vec<f32>)> {
        let needs_rotation = state.current_player == 1;
        let board_size = state.board_size;

        // When Player 1 is the current player, rotate the board 180° so the
        // network always sees "current player moving downward" (same as Python).
        let (eval_state, rotated_mask);
        if needs_rotation {
            let work_grid = rotate_grid_180(&state.grid());
            let work_positions = rotate_player_positions(&state.player_positions(), board_size);
            let work_goals = rotate_goal_rows(&state.goal_rows());
            let rotated_state = GameState {
                grid: work_grid,
                player_positions: work_positions,
                walls_remaining: state.walls_remaining.clone(),
                goal_rows: work_goals,
                current_player: state.current_player,
                board_size,
                completed_steps: state.completed_steps,
            };

            let (orig_to_rot, _) = self.get_rotation_mappings(board_size);
            rotated_mask = remap_mask(action_mask, orig_to_rot);
            eval_state = Some(rotated_state);
        } else {
            eval_state = None;
            rotated_mask = Vec::new(); // unused
        }

        let actual_state = eval_state.as_ref().unwrap_or(state);
        let actual_mask = if needs_rotation {
            &rotated_mask
        } else {
            action_mask
        };

        // Build ResNet input tensor
        let resnet_input = grid_game_state_to_resnet_input(actual_state);

        // Convert to flat vec for ORT
        let shape = resnet_input.shape().to_vec();
        let data: Vec<f32> = resnet_input.iter().copied().collect();
        let input_value = ort::value::Value::from_array((shape.as_slice(), data))
            .context("Failed to create ONNX input value")?;

        // Run inference and extract results into owned data so the borrow
        // on self.session is released before we call get_rotation_mappings.
        let (value, priors_in_eval_frame) = {
            let outputs = self
                .session
                .run(ort::inputs!["input" => input_value])
                .context("Failed to run ONNX inference")?;

            let value_tensor = outputs["value"]
                .try_extract_tensor::<f32>()
                .context("Failed to extract value")?;
            let value = value_tensor.1[0];

            let policy_logits = outputs["policy_logits"]
                .try_extract_tensor::<f32>()
                .context("Failed to extract policy logits")?;

            let priors = masked_softmax(policy_logits.1, actual_mask);
            (value, priors)
        };

        // If rotated, remap policy back to original action space
        let priors = if needs_rotation {
            let (_, rot_to_orig) = self.get_rotation_mappings(board_size);
            let mut remapped = vec![0.0f32; priors_in_eval_frame.len()];
            for (i, &target) in rot_to_orig.iter().enumerate() {
                remapped[target] = priors_in_eval_frame[i];
            }
            remapped
        } else {
            priors_in_eval_frame
        };

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
