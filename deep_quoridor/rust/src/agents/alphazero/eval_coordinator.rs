//! Eval coordinator: owns the ONNX session and batches inference requests
//! from worker threads.
//!
//! Workers submit `EvalRequest`s (which already carry their own rotated
//! features and mask) through a shared `SyncSender`. The coordinator collects
//! up to `eval_batch_size` requests, bounded by `eval_max_wait_ms` from the
//! arrival of the first request in the batch, then runs one batched ONNX
//! inference. Per-request results (un-rotated, masked priors + value) are
//! inserted into a shared `DashMap` cache (subject to `eval_cache_max_size`)
//! and sent back via each request's oneshot responder.
//!
//! The coordinator also listens for control messages: `ReloadModel(path)`
//! swaps in a new session and clears the cache; `Shutdown` exits the loop.

use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use dashmap::DashMap;
use ndarray::{Array4, Axis};
use ort::session::Session;

use crate::agents::alphazero::evaluator::finalize_policy;
use crate::compact::q_bit_repr::CompactState;

/// Shared eval cache: maps a (compact) game state to its (value, masked priors).
pub type EvalCache = DashMap<CompactState, EvalResult>;

#[derive(Clone, Debug)]
pub struct EvalResult {
    pub value: f32,
    pub priors: Vec<f32>,
}

/// A single eval request from a worker thread.
///
/// The worker has already done rotation and built the input tensor; the
/// coordinator just stacks and runs them through the network.
pub struct EvalRequest {
    pub state: CompactState,
    pub features: Array4<f32>,
    pub work_action_mask: Vec<bool>,
    pub rot_to_orig: Option<Vec<usize>>,
    pub responder: SyncSender<Result<EvalResult>>,
}

/// Control messages handled by the coordinator outside of the request channel.
pub enum CtrlMsg {
    /// Drop the current session, load a new ONNX model, clear the cache.
    ReloadModel(String),
    /// Stop the coordinator loop.
    Shutdown,
}

#[derive(Debug, Clone, Copy)]
pub struct CoordinatorConfig {
    pub eval_batch_size: usize,
    pub eval_max_wait_ms: u64,
    /// 0 means "do not cache" (lookups still cheap, inserts skipped).
    pub eval_cache_max_size: usize,
}

/// Open an ONNX `Session` from a file path.
pub fn load_session(model_path: &str) -> Result<Session> {
    Session::builder()
        .context("Failed to create ONNX session builder")?
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to load ONNX model from {}", model_path))
}

/// Main coordinator loop. Returns when `Shutdown` is received or the request
/// channel is closed and the control channel is also closed.
pub fn run_coordinator(
    mut session: Session,
    cache: Arc<EvalCache>,
    config: CoordinatorConfig,
    req_recv: Receiver<EvalRequest>,
    ctrl_recv: Receiver<CtrlMsg>,
) {
    let batch_size = config.eval_batch_size.max(1);
    let max_wait = Duration::from_millis(config.eval_max_wait_ms);
    let cache_max = config.eval_cache_max_size;
    // Periodic poll interval when there are no requests, to check for ctrl msgs.
    let idle_poll = Duration::from_millis(50);

    loop {
        // Drain any pending control messages first so reloads/shutdown happen
        // promptly even when requests are arriving steadily.
        loop {
            match ctrl_recv.try_recv() {
                Ok(CtrlMsg::ReloadModel(path)) => match load_session(&path) {
                    Ok(new_session) => {
                        session = new_session;
                        cache.clear();
                        eprintln!("eval-coordinator: loaded model {} (cache cleared)", path);
                    }
                    Err(e) => {
                        eprintln!("eval-coordinator: failed to load model {}: {:#}", path, e);
                    }
                },
                Ok(CtrlMsg::Shutdown) => return,
                Err(_) => break,
            }
        }

        // Block on first request with a short timeout so the ctrl channel can
        // still be polled while idle.
        let first = match req_recv.recv_timeout(idle_poll) {
            Ok(req) => req,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => {
                // Request side closed; drain remaining ctrl then exit.
                if matches!(ctrl_recv.try_recv(), Ok(CtrlMsg::Shutdown)) {
                    return;
                }
                return;
            }
        };

        let mut batch: Vec<EvalRequest> = Vec::with_capacity(batch_size);
        batch.push(first);

        // Try to fill the batch up to `batch_size`, bounded by `max_wait`
        // measured from the first request's arrival.
        let deadline = Instant::now() + max_wait;
        while batch.len() < batch_size {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match req_recv.recv_timeout(remaining) {
                Ok(req) => batch.push(req),
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }

        process_batch(&mut session, &cache, cache_max, batch);
    }
}

fn process_batch(
    session: &mut Session,
    cache: &EvalCache,
    cache_max: usize,
    batch: Vec<EvalRequest>,
) {
    if batch.is_empty() {
        return;
    }

    // Stack features into a single (B, 5, M, M) tensor.
    let views: Vec<_> = batch.iter().map(|r| r.features.view()).collect();
    let stacked = match ndarray::concatenate(Axis(0), &views) {
        Ok(arr) => arr,
        Err(e) => {
            let msg = format!("Failed to concatenate features: {}", e);
            for req in batch {
                let _ = req.responder.send(Err(anyhow::anyhow!(msg.clone())));
            }
            return;
        }
    };

    let shape = stacked.shape().to_vec();
    let input_data: Vec<f32> = stacked.iter().copied().collect();
    let batch_len = batch.len();

    let input_value = match ort::value::Value::from_array((shape.as_slice(), input_data)) {
        Ok(v) => v,
        Err(e) => {
            let msg = format!("Failed to build ONNX input: {}", e);
            for req in batch {
                let _ = req.responder.send(Err(anyhow::anyhow!(msg.clone())));
            }
            return;
        }
    };

    let outputs = match session.run(ort::inputs!["input" => input_value]) {
        Ok(o) => o,
        Err(e) => {
            let msg = format!("Failed to run ONNX inference: {}", e);
            for req in batch {
                let _ = req.responder.send(Err(anyhow::anyhow!(msg.clone())));
            }
            return;
        }
    };

    let value_tensor = match outputs["value"].try_extract_tensor::<f32>() {
        Ok(t) => t,
        Err(e) => {
            let msg = format!("Failed to extract value tensor: {}", e);
            for req in batch {
                let _ = req.responder.send(Err(anyhow::anyhow!(msg.clone())));
            }
            return;
        }
    };
    let policy_tensor = match outputs["policy_logits"].try_extract_tensor::<f32>() {
        Ok(t) => t,
        Err(e) => {
            let msg = format!("Failed to extract policy logits: {}", e);
            for req in batch {
                let _ = req.responder.send(Err(anyhow::anyhow!(msg.clone())));
            }
            return;
        }
    };

    let values: &[f32] = value_tensor.1;
    let policy: &[f32] = policy_tensor.1;
    // Policy logits are laid out as (B, policy_size); derive policy_size from
    // the total length and the batch size.
    let policy_size = policy.len() / batch_len;

    for (i, req) in batch.into_iter().enumerate() {
        let value = values[i];
        let logits = &policy[i * policy_size..(i + 1) * policy_size];
        let priors = finalize_policy(logits, &req.work_action_mask, req.rot_to_orig.as_deref());
        let result = EvalResult { value, priors };
        if cache_max > 0 && cache.len() < cache_max {
            cache.insert(req.state, result.clone());
        }
        let _ = req.responder.send(Ok(result));
    }
}
