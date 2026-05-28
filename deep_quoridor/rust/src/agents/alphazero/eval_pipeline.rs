//! Pipelined eval coordinator: Batcher → Inference → Post-process.
//!
//! Game tasks send `EvalRequest`s over a tokio `mpsc`. A Batcher OS thread
//! pulls them (via `blocking_recv`), assembles batches up to
//! `eval_batch_size` (deadline-bounded by `eval_max_wait_ms` from the first
//! request), stacks features, and hands `BatchPayload` to an Inference OS
//! thread that owns the ORT session. Inference forwards `BatchOutputs` to a
//! Post-process OS thread that runs `finalize_policy` in parallel via rayon,
//! inserts to the cache, and fires each request's tokio oneshot.
//!
//! Control messages (`Reload(path)`, `Shutdown`) ride the front mpsc as
//! enum variants so ordering with batches is preserved.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, sync_channel};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use dashmap::DashMap;
use ndarray::{Array4, Axis};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use rayon::prelude::*;
use tokio::sync::mpsc as tokio_mpsc;
use tokio::sync::oneshot;

use crate::agents::alphazero::evaluator::finalize_policy;
use crate::compact::q_bit_repr::CompactState;

/// One-time sentinel for cache-saturation warning.
static FIRST_FULL: AtomicBool = AtomicBool::new(false);

/// Atomic performance counters for the eval pipeline.
#[derive(Default)]
pub struct PipelineCounters {
    pub gpu_ns: AtomicU64,
    pub batcher_wait_ns: AtomicU64,
    pub postprocess_ns: AtomicU64,
    pub batches: AtomicU64,
    pub items: AtomicU64,
}

/// Shared eval cache: maps a (compact) state to its (value, masked priors).
pub type EvalCache = DashMap<CompactState, EvalResult>;

#[derive(Clone, Debug)]
pub struct EvalResult {
    pub value: f32,
    pub priors: Vec<f32>,
}

/// A single eval request from a game task. The request owns its rotated
/// features + work_action_mask + un-rotation map; the coordinator just stacks
/// and runs them.
pub struct EvalRequest {
    pub state: CompactState,
    pub features: Array4<f32>,
    pub work_action_mask: Vec<bool>,
    pub rot_to_orig: Option<Vec<usize>>,
    pub responder: oneshot::Sender<Result<EvalResult>>,
}

/// Front-channel message: a regular request or a control directive.
pub enum FrontMsg {
    Req(EvalRequest),
    Reload(String),
    Shutdown,
}

#[derive(Debug, Clone, Copy)]
pub struct CoordinatorConfig {
    pub eval_batch_size: usize,
    pub eval_max_wait_ms: u64,
    /// 0 disables caching (lookups still cheap, inserts skipped).
    pub eval_cache_max_size: usize,
}

/// Internal pipeline messages between stages.
struct BatchPayload {
    stacked: Array4<f32>,
    reqs: Vec<EvalRequest>,
}

enum InferenceIn {
    Batch(BatchPayload),
    Reload(String),
    Shutdown,
}

struct BatchOutputs {
    values: Vec<f32>,
    policy: Vec<f32>,
    policy_size: usize,
    reqs: Vec<EvalRequest>,
}

enum PostIn {
    Outputs(BatchOutputs),
    Reload, // forwarded marker, post-process doesn't need the path
    Shutdown,
}

/// Open an ONNX `Session` from a file path.
///
/// Built with the `gpu` feature, this registers the CUDA execution provider
/// with a CPU fallback, so a host without a working CUDA stack still runs.
/// Without the feature the session uses the CPU execution provider only.
pub fn load_session(model_path: &str) -> Result<Session> {
    #[cfg(feature = "gpu")]
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        anyhow::bail!(
            "gpu feature is enabled but ORT_DYLIB_PATH is not set. Point it at the \
             onnxruntime-gpu shared library (…/onnxruntime/capi/libonnxruntime.so.<version>) \
             and put the CUDA and cuDNN lib directories on LD_LIBRARY_PATH."
        );
    }

    // NOTE: `with_optimization_level`/`with_execution_providers` return
    // `Result<Self, ort::Error<SessionBuilder>>`; that error carries the builder
    // back for recovery and is not `Send`/`Sync`, so anyhow's `.context()` does
    // not apply. Convert through the error's `Display` instead.
    let builder = Session::builder()
        .context("Failed to create ONNX session builder")?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("Failed to set graph optimization level: {e}"))?
        // Pin ORT intra-op threads to 1: ORT defaults to all CPU cores, and
        // parallel CPU sessions (e.g. concurrent tests on CI) oversubscribe
        // its threadpool and intermittently deadlock. For production GPU
        // inference this is moot.
        .with_intra_threads(1)
        .map_err(|e| anyhow::anyhow!("Failed to set intra-op thread count: {e}"))?;

    #[cfg(feature = "gpu")]
    let builder = builder
        .with_execution_providers([
            ort::execution_providers::CUDAExecutionProvider::default().build(),
            ort::execution_providers::CPUExecutionProvider::default().build(),
        ])
        .map_err(|e| anyhow::anyhow!("Failed to register CUDA/CPU execution providers: {e}"))?;

    // `commit_from_file` takes `&mut self` in this ort version, so the final
    // builder must be a mutable binding (works for both gpu and non-gpu paths).
    let mut builder = builder;
    builder
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to load ONNX model from {}", model_path))
}

/// Spawn the three-stage pipeline. Returns join handles for graceful shutdown.
pub struct CoordinatorHandles {
    pub batcher: thread::JoinHandle<()>,
    pub inference: thread::JoinHandle<()>,
    pub post: thread::JoinHandle<()>,
}

pub fn spawn_coordinator(
    initial_session: Session,
    cache: Arc<EvalCache>,
    config: CoordinatorConfig,
    front_rx: tokio_mpsc::Receiver<FrontMsg>,
    counters: Arc<PipelineCounters>,
) -> CoordinatorHandles {
    let (inf_tx, inf_rx) = sync_channel::<InferenceIn>(1);
    let (post_tx, post_rx) = sync_channel::<PostIn>(1);

    let batcher = thread::Builder::new()
        .name("eval-batcher".to_string())
        .spawn({
            let counters = Arc::clone(&counters);
            move || run_batcher(front_rx, inf_tx, config, counters)
        })
        .expect("spawn batcher");
    let inference = thread::Builder::new()
        .name("eval-inference".to_string())
        .spawn({
            let cache = Arc::clone(&cache);
            let counters = Arc::clone(&counters);
            move || run_inference(initial_session, cache, inf_rx, post_tx, counters)
        })
        .expect("spawn inference");
    let post = thread::Builder::new()
        .name("eval-post".to_string())
        .spawn({
            let counters = Arc::clone(&counters);
            move || run_postprocess(cache, config.eval_cache_max_size, post_rx, counters)
        })
        .expect("spawn post");

    CoordinatorHandles {
        batcher,
        inference,
        post,
    }
}

fn run_batcher(
    mut front_rx: tokio_mpsc::Receiver<FrontMsg>,
    inf_tx: std::sync::mpsc::SyncSender<InferenceIn>,
    config: CoordinatorConfig,
    counters: Arc<PipelineCounters>,
) {
    let batch_size = config.eval_batch_size.max(1);
    let max_wait = Duration::from_millis(config.eval_max_wait_ms);

    loop {
        // Block on first message; time how long we wait.
        let t0 = Instant::now();
        let first = match front_rx.blocking_recv() {
            Some(m) => m,
            None => break, // channel closed; drain done
        };
        counters
            .batcher_wait_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let first_req = match first {
            FrontMsg::Req(r) => r,
            FrontMsg::Reload(p) => {
                let _ = inf_tx.send(InferenceIn::Reload(p));
                continue;
            }
            FrontMsg::Shutdown => {
                let _ = inf_tx.send(InferenceIn::Shutdown);
                break;
            }
        };

        let mut reqs: Vec<EvalRequest> = Vec::with_capacity(batch_size);
        reqs.push(first_req);

        // Fill batch up to `batch_size`, deadline = first arrival + max_wait.
        // Note: max_wait = 0 (eval_max_wait_ms=0) effectively forces batch-size-1
        // because the deadline expires before the first try_recv runs. This is the
        // intended "ship immediately" behavior.
        let deadline = Instant::now() + max_wait;
        while reqs.len() < batch_size {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            // Use try_recv-with-yield pattern: non-blocking try, sleep briefly,
            // re-check. tokio mpsc's blocking_recv has no deadline variant we can use here.
            match front_rx.try_recv() {
                Ok(FrontMsg::Req(r)) => reqs.push(r),
                Ok(FrontMsg::Reload(p)) => {
                    // flush current batch first, then forward reload
                    flush_batch(&inf_tx, std::mem::take(&mut reqs));
                    let _ = inf_tx.send(InferenceIn::Reload(p));
                    break;
                }
                Ok(FrontMsg::Shutdown) => {
                    flush_batch(&inf_tx, std::mem::take(&mut reqs));
                    let _ = inf_tx.send(InferenceIn::Shutdown);
                    return;
                }
                Err(tokio_mpsc::error::TryRecvError::Empty) => {
                    thread::sleep(Duration::from_micros(50));
                }
                Err(tokio_mpsc::error::TryRecvError::Disconnected) => break,
            }
        }

        if !reqs.is_empty() {
            flush_batch(&inf_tx, reqs);
        }
    }
}

fn flush_batch(inf_tx: &std::sync::mpsc::SyncSender<InferenceIn>, reqs: Vec<EvalRequest>) {
    if reqs.is_empty() {
        return;
    }
    let views: Vec<_> = reqs.iter().map(|r| r.features.view()).collect();
    let stacked = match ndarray::concatenate(Axis(0), &views) {
        Ok(a) => a,
        Err(e) => {
            let msg = format!("Failed to concatenate features: {}", e);
            for r in reqs {
                let _ = r.responder.send(Err(anyhow::anyhow!(msg.clone())));
            }
            return;
        }
    };
    let payload = BatchPayload { stacked, reqs };
    let _ = inf_tx.send(InferenceIn::Batch(payload));
}

fn run_inference(
    mut session: Session,
    cache: Arc<EvalCache>,
    inf_rx: Receiver<InferenceIn>,
    post_tx: std::sync::mpsc::SyncSender<PostIn>,
    counters: Arc<PipelineCounters>,
) {
    while let Ok(msg) = inf_rx.recv() {
        match msg {
            InferenceIn::Batch(BatchPayload { stacked, reqs }) => {
                let shape = stacked.shape().to_vec();
                let input_data: Vec<f32> = stacked.iter().copied().collect();
                let batch_len = reqs.len();
                let input_value =
                    match ort::value::Value::from_array((shape.as_slice(), input_data)) {
                        Ok(v) => v,
                        Err(e) => {
                            let msg = format!("Failed to build ONNX input: {}", e);
                            for r in reqs {
                                let _ = r.responder.send(Err(anyhow::anyhow!(msg.clone())));
                            }
                            continue;
                        }
                    };
                let gpu_t0 = Instant::now();
                let outputs = match session.run(ort::inputs!["input" => input_value]) {
                    Ok(o) => o,
                    Err(e) => {
                        let msg = format!("Failed to run ONNX inference: {}", e);
                        for r in reqs {
                            let _ = r.responder.send(Err(anyhow::anyhow!(msg.clone())));
                        }
                        continue;
                    }
                };
                counters
                    .gpu_ns
                    .fetch_add(gpu_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                counters.batches.fetch_add(1, Ordering::Relaxed);
                counters
                    .items
                    .fetch_add(batch_len as u64, Ordering::Relaxed);
                let value_tensor = match outputs["value"].try_extract_tensor::<f32>() {
                    Ok(t) => t,
                    Err(e) => {
                        let msg = format!("Failed to extract value tensor: {}", e);
                        for r in reqs {
                            let _ = r.responder.send(Err(anyhow::anyhow!(msg.clone())));
                        }
                        continue;
                    }
                };
                let policy_tensor = match outputs["policy_logits"].try_extract_tensor::<f32>() {
                    Ok(t) => t,
                    Err(e) => {
                        let msg = format!("Failed to extract policy logits: {}", e);
                        for r in reqs {
                            let _ = r.responder.send(Err(anyhow::anyhow!(msg.clone())));
                        }
                        continue;
                    }
                };
                let values: Vec<f32> = value_tensor.1.to_vec();
                let policy: Vec<f32> = policy_tensor.1.to_vec();
                debug_assert!(
                    batch_len > 0,
                    "Batch should never be empty here (flush_batch guards is_empty)"
                );
                let policy_size = policy.len() / batch_len;
                let outputs = BatchOutputs {
                    values,
                    policy,
                    policy_size,
                    reqs,
                };
                let _ = post_tx.send(PostIn::Outputs(outputs));
            }
            InferenceIn::Reload(path) => match load_session(&path) {
                Ok(s) => {
                    session = s;
                    cache.clear();
                    eprintln!("eval-pipeline: loaded model {} (cache cleared)", path);
                    let _ = post_tx.send(PostIn::Reload);
                }
                Err(e) => eprintln!("eval-pipeline: failed to load model {}: {:#}", path, e),
            },
            InferenceIn::Shutdown => {
                let _ = post_tx.send(PostIn::Shutdown);
                return;
            }
        }
    }
}

fn run_postprocess(
    cache: Arc<EvalCache>,
    cache_max: usize,
    post_rx: Receiver<PostIn>,
    counters: Arc<PipelineCounters>,
) {
    while let Ok(msg) = post_rx.recv() {
        match msg {
            PostIn::Outputs(out) => {
                let BatchOutputs {
                    values,
                    policy,
                    policy_size,
                    reqs,
                } = out;
                // Parallel finalize over the request batch.
                let post_t0 = Instant::now();
                let finalized: Vec<EvalResult> = reqs
                    .par_iter()
                    .enumerate()
                    .map(|(i, req)| {
                        let logits = &policy[i * policy_size..(i + 1) * policy_size];
                        let priors = finalize_policy(
                            logits,
                            &req.work_action_mask,
                            req.rot_to_orig.as_deref(),
                        );
                        EvalResult {
                            value: values[i],
                            priors,
                        }
                    })
                    .collect();
                counters
                    .postprocess_ns
                    .fetch_add(post_t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                // Insert into cache (serial — DashMap is sharded internally, parallel
                // inserts have contention; serial is fine here).
                for (req, res) in reqs.iter().zip(finalized.iter()) {
                    if cache_max > 0
                        && cache.len() >= cache_max
                        && !FIRST_FULL.swap(true, Ordering::Relaxed)
                    {
                        eprintln!(
                            "eval-pipeline: cache reached cap of {} entries — further inserts will be skipped (by design)",
                            cache_max
                        );
                    }
                    if cache_max > 0 && cache.len() < cache_max {
                        cache.insert(req.state, res.clone());
                    }
                }
                // Fire oneshots.
                for (req, res) in reqs.into_iter().zip(finalized.into_iter()) {
                    let _ = req.responder.send(Ok(res));
                }
            }
            PostIn::Reload => {
                // nothing to do; cache clear and session swap already happened in run_inference
                continue;
            }
            PostIn::Shutdown => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;
    use tokio::sync::oneshot;

    fn make_request(
        policy_size: usize,
        board_size: i32,
    ) -> (EvalRequest, oneshot::Receiver<Result<EvalResult>>) {
        let m = (board_size * 2 + 3) as usize;
        let features = Array4::<f32>::zeros((1, 5, m, m));
        let mask = vec![true; policy_size];
        let (tx, rx) = oneshot::channel();
        let req = EvalRequest {
            state: CompactState::default(),
            features,
            work_action_mask: mask,
            rot_to_orig: None,
            responder: tx,
        };
        (req, rx)
    }

    #[test]
    fn test_postprocess_parallel_matches_serial_finalize() {
        // Synthesise a batch and feed it directly through run_postprocess via a
        // hand-driven channel pair; then compare against a serial reference.
        let cache = Arc::new(EvalCache::new());
        // Buffer of 2: holds both the Outputs message and the Shutdown sentinel so
        // both can be enqueued before run_postprocess starts draining the channel.
        let (post_tx, post_rx) = sync_channel::<PostIn>(2);

        let batch_len = 8usize;
        let policy_size = 10usize;
        let mut reqs = Vec::with_capacity(batch_len);
        let mut rxs = Vec::with_capacity(batch_len);
        for _ in 0..batch_len {
            let (r, rx) = make_request(policy_size, 5);
            reqs.push(r);
            rxs.push(rx);
        }
        let values: Vec<f32> = (0..batch_len).map(|i| i as f32 * 0.1 - 0.4).collect();
        let policy: Vec<f32> = (0..batch_len * policy_size)
            .map(|i| (i as f32).sin())
            .collect();

        // Expected from serial loop using finalize_policy directly.
        let mut expected: Vec<Vec<f32>> = Vec::with_capacity(batch_len);
        for i in 0..batch_len {
            let logits = &policy[i * policy_size..(i + 1) * policy_size];
            expected.push(finalize_policy(logits, &vec![true; policy_size], None));
        }

        post_tx
            .send(PostIn::Outputs(BatchOutputs {
                values: values.clone(),
                policy: policy.clone(),
                policy_size,
                reqs,
            }))
            .unwrap();
        post_tx.send(PostIn::Shutdown).unwrap();

        // Drive the post stage on this thread.
        run_postprocess(
            Arc::clone(&cache),
            1024,
            post_rx,
            Arc::new(PipelineCounters::default()),
        );

        for (i, rx) in rxs.into_iter().enumerate() {
            let res = rx.blocking_recv().unwrap().unwrap();
            assert!((res.value - values[i]).abs() < 1e-6);
            for (p, e) in res.priors.iter().zip(expected[i].iter()) {
                assert!((p - e).abs() < 1e-6);
            }
        }
    }
}
