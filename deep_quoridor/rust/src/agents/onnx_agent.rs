//! ONNX model inference agent.
//!
//! This module is only available behind the `binary` feature flag.

use anyhow::{Context, Result};
use ort::session::Session;

use crate::agents::ActionSelector;
use crate::compact::q_bit_repr::CompactState;
use crate::compact::q_game_mechanics::QGameMechanics;
use crate::grid_helpers::compact_state_to_resnet_input;

pub use crate::agents::alphazero::evaluator::softmax;

/// An agent that uses an ONNX model to select actions (greedy argmax).
pub struct OnnxAgent {
    session: Session,
}

impl OnnxAgent {
    /// Load an ONNX model from the given path.
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
        // Pin ORT intra-op threads to 1. Without this, ORT defaults to all
        // available CPU cores; multiple concurrent sessions (e.g. parallel
        // tests on CPU CI) oversubscribe its threadpool and intermittently
        // deadlock. For production GPU inference intra-op threading is moot.
        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("Failed to set intra-op thread count: {e}"))?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;
        Ok(Self { session })
    }
}

impl ActionSelector for OnnxAgent {
    fn select_action(
        &mut self,
        data: CompactState,
        mechanics: &QGameMechanics,
        action_mask: &[bool],
    ) -> Result<(usize, Vec<f32>)> {
        // Build ResNet input tensor
        let resnet_input = compact_state_to_resnet_input(mechanics, data);

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

        // Extract policy logits and compute softmax
        let policy_logits = outputs["policy_logits"]
            .try_extract_tensor::<f32>()
            .context("Failed to extract policy logits")?;
        let probs = softmax(policy_logits.1);

        // Greedy: pick the valid action with the highest probability
        let mut best_idx = 0;
        let mut best_prob = f32::NEG_INFINITY;
        for (i, (&p, &valid)) in probs.iter().zip(action_mask.iter()).enumerate() {
            if valid && p > best_prob {
                best_prob = p;
                best_idx = i;
            }
        }

        Ok((best_idx, probs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Sum should be ~1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Higher logit → higher prob
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_uniform() {
        let logits = vec![0.0, 0.0, 0.0, 0.0];
        let probs = softmax(&logits);
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_large_values() {
        // Should not overflow
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
    }
}
