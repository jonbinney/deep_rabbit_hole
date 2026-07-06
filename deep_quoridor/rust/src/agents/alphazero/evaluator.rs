//! Evaluator trait and implementations for MCTS.

use std::collections::HashMap;

#[cfg(feature = "binary")]
use anyhow::Context;
use anyhow::Result;
use ndarray::Array4;
#[cfg(feature = "binary")]
use ort::session::Session;

use crate::compact::q_bit_repr::CompactState;
use crate::compact::q_game_mechanics::QGameMechanics;
use crate::grid_helpers::compact_state_to_resnet_input;
use crate::rotation::{create_rotation_mapping, remap_mask, remap_policy, rotate_compact_state};

/// Trait for evaluating game positions.
///
/// Returns `(value_for_current_player, masked_softmax_priors)`.
pub trait Evaluator {
    fn evaluate(
        &mut self,
        data: CompactState,
        mechanics: &QGameMechanics,
        action_mask: &[bool],
    ) -> Result<(f32, Vec<f32>)>;
}

/// Result of `prepare_eval_input`: features ready for inference plus the
/// rotated mask and (if rotation was applied) the inverse mapping to restore
/// the original action-space ordering on the output policy.
pub struct PreparedEvalInput {
    pub features: Array4<f32>,
    pub work_action_mask: Vec<bool>,
    pub rot_to_orig: Option<Vec<usize>>,
}

/// Apply the player-1 rotation (if needed) and build the ResNet input tensor.
///
/// `rotation_mappings` is a per-evaluator cache of (orig_to_rot, rot_to_orig)
/// keyed by board size — caching avoids recomputing the mapping on every call.
pub fn prepare_eval_input(
    mechanics: &QGameMechanics,
    data: CompactState,
    action_mask: &[bool],
    rotation_mappings: &mut HashMap<i32, (Vec<usize>, Vec<usize>)>,
) -> PreparedEvalInput {
    let bs = mechanics.repr().board_size() as i32;
    let current_player = mechanics.repr().get_current_player(data);

    let mappings = rotation_mappings
        .entry(bs)
        .or_insert_with(|| create_rotation_mapping(bs));
    let (orig_to_rot, rot_to_orig) = (&mappings.0, &mappings.1);

    let (work_data, work_action_mask, rot_to_orig_out) = if current_player == 1 {
        let rotated_data = rotate_compact_state(mechanics, data);
        let rotated_mask = remap_mask(action_mask, orig_to_rot);
        (rotated_data, rotated_mask, Some(rot_to_orig.clone()))
    } else {
        (data, action_mask.to_vec(), None)
    };

    let features = compact_state_to_resnet_input(mechanics, work_data);

    PreparedEvalInput {
        features,
        work_action_mask,
        rot_to_orig: rot_to_orig_out,
    }
}

/// Convert raw `policy_logits` from the network into masked-softmax priors in
/// the original (un-rotated) action space.
pub fn finalize_policy(
    policy_logits: &[f32],
    work_action_mask: &[bool],
    rot_to_orig: Option<&[usize]>,
) -> Vec<f32> {
    let priors_work = masked_softmax(policy_logits, work_action_mask);
    match rot_to_orig {
        Some(map) => remap_policy(&priors_work, map),
        None => priors_work,
    }
}

/// ONNX-based evaluator for MCTS.
///
/// Loads a neural network model and uses it to evaluate positions,
/// returning both a value estimate and policy priors.
#[cfg(feature = "binary")]
pub struct OnnxEvaluator {
    session: Session,
    rotation_mappings_by_board_size: HashMap<i32, (Vec<usize>, Vec<usize>)>,
}

/// Deterministic evaluator for cross-language consistency tests.
///
/// Returns value=0.0 and a uniform prior over valid actions.
pub struct UniformMockEvaluator;

#[cfg(feature = "binary")]
impl OnnxEvaluator {
    /// Create a new evaluator from an ONNX model file.
    pub fn new(model_path: &str) -> Result<Self> {
        // Fail fast when built with `gpu` (which enables ort/load-dynamic) but
        // ORT_DYLIB_PATH isn't set: in that case `Session::builder()` deadlocks
        // on a dynamic-loader futex instead of erroring out. Same guard pattern
        // as eval_pipeline::load_session.
        #[cfg(feature = "gpu")]
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            anyhow::bail!(
                "gpu feature is enabled but ORT_DYLIB_PATH is not set. Point it at the \
                 onnxruntime-gpu shared library (.../onnxruntime/capi/libonnxruntime.so.<version>) \
                 and put the CUDA and cuDNN lib directories on LD_LIBRARY_PATH."
            );
        }
        // Pin ORT intra-op threads to 1: ORT defaults to all CPU cores, and
        // parallel CPU sessions (e.g. concurrent tests on CI) oversubscribe its
        // threadpool and intermittently deadlock. For production GPU inference
        // this is moot.
        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("Failed to set intra-op thread count: {e}"))?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;
        Ok(Self {
            session,
            rotation_mappings_by_board_size: HashMap::new(),
        })
    }
}

#[cfg(feature = "binary")]
impl Evaluator for OnnxEvaluator {
    fn evaluate(
        &mut self,
        data: CompactState,
        mechanics: &QGameMechanics,
        action_mask: &[bool],
    ) -> Result<(f32, Vec<f32>)> {
        let prepared = prepare_eval_input(
            mechanics,
            data,
            action_mask,
            &mut self.rotation_mappings_by_board_size,
        );

        // Convert features to flat vec for ORT
        let shape = prepared.features.shape().to_vec();
        let input_data: Vec<f32> = prepared.features.iter().copied().collect();
        let input_value = ort::value::Value::from_array((shape.as_slice(), input_data))
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

        // Extract policy logits and apply mask + un-rotation
        let policy_logits = outputs["policy_logits"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract policy logits")?;

        let priors = finalize_policy(
            policy_logits.1,
            &prepared.work_action_mask,
            prepared.rot_to_orig.as_deref(),
        );

        Ok((value, priors))
    }
}

impl Evaluator for UniformMockEvaluator {
    fn evaluate(
        &mut self,
        _data: CompactState,
        _mechanics: &QGameMechanics,
        action_mask: &[bool],
    ) -> Result<(f32, Vec<f32>)> {
        let valid_count = action_mask.iter().filter(|&&valid| valid).count();
        let mut priors = vec![0.0f32; action_mask.len()];
        if valid_count > 0 {
            let p = 1.0f32 / valid_count as f32;
            for (i, &valid) in action_mask.iter().enumerate() {
                if valid {
                    priors[i] = p;
                }
            }
        }
        Ok((0.0, priors))
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

/// Numerically-stable softmax over a slice.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp_values.iter().sum();
    exp_values.iter().map(|&x| x / sum).collect()
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
