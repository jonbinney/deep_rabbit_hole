#![cfg(feature = "gpu")]
//! Numerics check: CUDA-EP inference must match CPU-EP inference for the same
//! model and input. Confirms moving inference to the GPU did not change the
//! computed value/policy.
//!
//! Run: cargo test --release --features binary,gpu --test cuda_consistency -- --nocapture

use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;

const MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../runs/jdb-performance-base/models/checkpoints/model_0.onnx"
);

fn run_batch(session: &mut Session, shape: &[usize], data: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
    let input = Value::from_array((shape.to_vec(), data)).expect("build input value");
    let outputs = session
        .run(ort::inputs!["input" => input])
        .expect("run inference");
    let value = outputs["value"]
        .try_extract_tensor::<f32>()
        .expect("extract value")
        .1
        .to_vec();
    let policy = outputs["policy_logits"]
        .try_extract_tensor::<f32>()
        .expect("extract policy_logits")
        .1
        .to_vec();
    (value, policy)
}

#[test]
fn cuda_matches_cpu() {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("SKIP cuda_matches_cpu: model not found at {MODEL}");
        return;
    }

    let mut cpu = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .unwrap()
        .commit_from_file(MODEL)
        .unwrap();

    let mut cuda = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_execution_providers([
            CUDAExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ])
        .unwrap()
        .commit_from_file(MODEL)
        .unwrap();

    // Deterministic pseudo-random batch of 8, shape [8, 5, 21, 21].
    let n = 8usize;
    let shape = [n, 5usize, 21, 21];
    let len: usize = shape.iter().product();
    let data: Vec<f32> = (0..len)
        .map(|i| ((i.wrapping_mul(2654435761)) % 1000) as f32 / 1000.0)
        .collect();

    let (v_cpu, p_cpu) = run_batch(&mut cpu, &shape, data.clone());
    let (v_cuda, p_cuda) = run_batch(&mut cuda, &shape, data);

    assert_eq!(v_cpu.len(), v_cuda.len(), "value length mismatch");
    assert_eq!(p_cpu.len(), p_cuda.len(), "policy length mismatch");

    let max_v = v_cpu
        .iter()
        .zip(&v_cuda)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_p = p_cpu
        .iter()
        .zip(&p_cuda)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("max value diff = {max_v:.2e}, max policy_logit diff = {max_p:.2e}");

    assert!(max_v < 1e-3, "value diff too large: {max_v}");
    assert!(max_p < 1e-2, "policy_logit diff too large: {max_p}");
}
