# GPU Self-Play Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the AlphaZero resnet inference in Rust self-play off the CPU and onto the RTX 5080 (Blackwell, sm_120) via the ONNX Runtime CUDA execution provider, with a clean CPU fallback.

**Architecture:** A single chokepoint — `load_session()` in `rust/src/agents/alphazero/eval_pipeline.rs` — builds the ORT `Session` used by the inference thread (and by the model-`Reload` path). We add a `gpu` Cargo feature that turns on `ort/cuda` and, behind that feature, register the CUDA EP with a CPU fallback. Nothing else in the batcher → inference → post pipeline changes. Precision stays fp32. A throwaway build-and-benchmark step de-risks sm_120 support before we rely on it; a numerics test confirms the GPU output matches CPU.

**Tech Stack:** Rust (edition 2024), `ort` 2.0.0-rc.11 (ONNX Runtime bindings), CUDA 13 + cuDNN 9 runtime (already on disk via the venv's torch cu130 wheels), RTX 5080.

**Spec:** `docs/superpowers/specs/2026-05-23-gpu-self-play-eval-design.md`

---

## Reference: measured CPU baseline (from brainstorming)

These are the numbers the GPU build must beat. Captured on this machine with the target config and `model_0.onnx`:

- Throughput: **~7,600 NN evals/sec**, **0 games completed in 95 s**.
- `nvidia-smi`: GPU utilization **7–9%** (idle), no `selfplay` compute process on the GPU.
- `selfplay` CPU: **~1100%** (≈11 of 12 worker threads).
- `--profile-counters`: `gpu_busy≈97%` (this counts inference-*thread* busy time, i.e. CPU work — it is NOT GPU usage), `batch_wait_ms=0.0`, `avg_batch≈480`.

## Reference: exact commands used in brainstorming (reuse verbatim)

Paths are absolute so they work from any directory.

**Config:** `/home/jbinney/ws/deep_rabbit_hole/experiments/2026_05_23_jon_b9w10_performance/config.yaml`
**Model:** `/home/jbinney/ws/deep_rabbit_hole/deep_quoridor/runs/jdb-performance-base/models/checkpoints/model_0.onnx`
(input `[batch,5,21,21]`, outputs `value` and `policy_logits[209]` — matches the 9×9/10-wall config)

The release `selfplay` binary lives at `deep_quoridor/rust/target/release/selfplay`.

---

## Task 1: Add `gpu` Cargo feature and register CUDA EP in `load_session`

**Files:**
- Modify: `deep_quoridor/rust/Cargo.toml` (the `[features]` table, around line 45-48)
- Modify: `deep_quoridor/rust/src/agents/alphazero/eval_pipeline.rs:104-110` (`load_session`)

- [ ] **Step 1: Add the `gpu` feature to `Cargo.toml`**

In `deep_quoridor/rust/Cargo.toml`, the current `[features]` table is:

```toml
[features]
default = ["python"]
python = ["pyo3", "numpy"]
binary = ["clap", "ort", "serde_yaml", "ndarray-npy", "zip", "rand_distr", "tokio", "futures"]
```

Change it to add a `gpu` feature that enables the `ort` CUDA execution provider:

```toml
[features]
default = ["python"]
python = ["pyo3", "numpy"]
binary = ["clap", "ort", "serde_yaml", "ndarray-npy", "zip", "rand_distr", "tokio", "futures"]
gpu = ["ort/cuda"]
```

(`gpu` is built alongside `binary`, e.g. `--features binary,gpu`. The `ort/cuda` syntax turns on the `cuda` feature of the `ort` crate, which also pulls `ort` in.)

- [ ] **Step 2: Rewrite `load_session` to register the CUDA EP behind the feature**

Replace the current body (`eval_pipeline.rs:104-110`):

```rust
/// Open an ONNX `Session` from a file path.
pub fn load_session(model_path: &str) -> Result<Session> {
    Session::builder()
        .context("Failed to create ONNX session builder")?
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to load ONNX model from {}", model_path))
}
```

with:

```rust
/// Open an ONNX `Session` from a file path.
///
/// Built with the `gpu` feature, this registers the CUDA execution provider
/// with a CPU fallback, so a host without a working CUDA stack still runs.
/// Without the feature the session uses the CPU execution provider only.
pub fn load_session(model_path: &str) -> Result<Session> {
    let builder = Session::builder()
        .context("Failed to create ONNX session builder")?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .context("Failed to set graph optimization level")?;

    #[cfg(feature = "gpu")]
    let builder = builder
        .with_execution_providers([
            ort::execution_providers::CUDAExecutionProvider::default().build(),
            ort::execution_providers::CPUExecutionProvider::default().build(),
        ])
        .context("Failed to register CUDA/CPU execution providers")?;

    builder
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to load ONNX model from {}", model_path))
}
```

Notes for the implementer:
- The execution-provider types are referenced by full path inside the `#[cfg(feature = "gpu")]` block on purpose — that avoids unused-import warnings on the CPU-only build.
- `with_optimization_level` and `with_execution_providers` both return `Result<Self>` in `ort` 2.0.0-rc.11 (verified on docs.rs), hence the `.context(...)?`.
- If `ort::execution_providers::CUDAExecutionProvider` / `CPUExecutionProvider` does not resolve, the items are feature-gated; confirm the exact path on docs.rs for `ort` 2.0.0-rc.11 with the `cuda` feature, then adjust. `.default().build()` returns an `ExecutionProviderDispatch`.

- [ ] **Step 3: Add the `GraphOptimizationLevel` import**

At the top of `eval_pipeline.rs`, next to the existing `use ort::session::Session;` (line 23), add:

```rust
use ort::session::builder::GraphOptimizationLevel;
```

- [ ] **Step 4: Confirm the CPU-only build still compiles (fallback safety)**

This proves the change is transparent to CPU-only hosts/CI. No CUDA download happens here.

Run:
```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo build --release --features binary --bin selfplay
```
Expected: builds successfully, no errors. (Warnings about the unused `gpu` branch are not expected because it is `#[cfg]`-gated.)

- [ ] **Step 5: Commit the functional change**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/rust/Cargo.toml deep_quoridor/rust/src/agents/alphazero/eval_pipeline.rs
git commit -m "vibe: register CUDA EP for self-play inference

Self-play resnet inference ran entirely on CPU because load_session
registered no execution provider. Add a gpu Cargo feature (ort/cuda)
and, behind it, register the CUDA execution provider with a CPU
fallback plus Level3 graph optimization. CPU-only builds are
unaffected."
```

---

## Task 2: Prove GPU inference runs on sm_120 (go/no-go gate)

This is the de-risking step. It builds with the `gpu` feature (which downloads the ORT CUDA binaries) and confirms the RTX 5080 is actually used. **Do not proceed past this task until the GPU is confirmed active.**

**Files:** none modified (build + run only), unless Path B is needed (see below).

- [ ] **Step 1: Build the GPU binary (downloads ORT CUDA binaries — needs network)**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo build --release --features binary,gpu --bin selfplay
```
Expected: builds successfully. If the ORT CUDA binary download or link fails, jump to **Path B** below.

- [ ] **Step 2: Run a short benchmark with `nvidia-smi` polling**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor
OUT=$(mktemp -d)
rust/target/release/selfplay \
    --config /home/jbinney/ws/deep_rabbit_hole/experiments/2026_05_23_jon_b9w10_performance/config.yaml \
    --model-path /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/runs/jdb-performance-base/models/checkpoints/model_0.onnx \
    --output-dir "$OUT" --num-games 100000 --profile-counters > "$OUT/run.log" 2>&1 &
PID=$!
for i in 1 2 3 4 5 6; do
  sleep 5
  echo "--- sample $i ---"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
done
echo "=== GPU compute processes ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
kill $PID 2>/dev/null; wait $PID 2>/dev/null
echo "=== profile counters ==="
grep '\[pipe\]' "$OUT/run.log" | tail -6
rm -rf "$OUT"
```

Expected (GPU active): GPU utilization well above the 7–9% idle baseline; `selfplay` appears under "GPU compute processes"; `[pipe]` lines show `items` per 5 s materially higher than the CPU baseline (~38,000/5 s ≈ 7,600/s).

- [ ] **Step 3: Decision gate**

- **GPU active** (util up, `selfplay` listed as a compute process) → success. Proceed to Task 3.
- **Still CPU** (util ~9%, `selfplay` absent from compute processes). Check `run.log` for an ORT message about CUDA EP registration failing (missing/mismatched CUDA or cuDNN libs). Proceed to **Path B**.

### Path B (contingency): `load-dynamic` against `onnxruntime-gpu`

Use only if Path A's bundled CUDA binaries do not build or do not run on sm_120. This links ORT dynamically against an `onnxruntime-gpu` build that supports CUDA 13 + cuDNN 9 + sm_120 (the same lib generation torch already proves works on this GPU).

- [ ] **B1: Install a GPU-capable ONNX Runtime into the venv**

```bash
/home/jbinney/ws/deep_rabbit_hole/.venv/bin/pip install onnxruntime-gpu
/home/jbinney/ws/deep_rabbit_hole/.venv/bin/python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
```
Expected: `CUDAExecutionProvider` is listed. Pick a version new enough for Blackwell sm_120 if the default is too old.

- [ ] **B2: Locate the dynamic library**

```bash
find /home/jbinney/ws/deep_rabbit_hole/.venv -name 'libonnxruntime.so*'
```
Note the path (typically `.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.<ver>`).

- [ ] **B3: Switch the `gpu` feature to dynamic loading**

In `deep_quoridor/rust/Cargo.toml`, change:
```toml
gpu = ["ort/cuda"]
```
to:
```toml
gpu = ["ort/cuda", "ort/load-dynamic"]
```

- [ ] **B4: Rebuild and run with the dylib + CUDA libs on the loader path**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo build --release --features binary,gpu --bin selfplay
export ORT_DYLIB_PATH=<path from B2>
export LD_LIBRARY_PATH=/home/jbinney/ws/deep_rabbit_hole/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:/home/jbinney/ws/deep_rabbit_hole/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```
Then re-run Step 2's benchmark (the `ORT_DYLIB_PATH` and `LD_LIBRARY_PATH` exports must be set in the same shell). Confirm the GPU goes active.

- [ ] **B5: Record the required runtime env in the spec/results**

If Path B is used, the `selfplay` process needs `ORT_DYLIB_PATH` and `LD_LIBRARY_PATH` set. Note this in the results doc (Task 4) and confirm `train_v2.py` passes the environment through to the `selfplay` subprocess (check how it spawns the binary; add the env vars there if needed). Commit the Cargo.toml change:

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/rust/Cargo.toml
git commit -m "vibe: load ORT dynamically for sm_120 CUDA EP"
```

---

## Task 3: Numerics consistency test (CUDA output ≈ CPU output)

Confirms we changed *where* the net runs, not *what* it computes. (GPU usage itself is confirmed by Task 2's `nvidia-smi`; this test only checks the numbers match.)

**Files:**
- Create: `deep_quoridor/rust/tests/cuda_consistency.rs`

- [ ] **Step 1: Write the test**

Create `deep_quoridor/rust/tests/cuda_consistency.rs`:

```rust
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

fn run_batch(session: &Session, shape: &[usize], data: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
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

    let cpu = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .unwrap()
        .commit_from_file(MODEL)
        .unwrap();

    let cuda = Session::builder()
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

    let (v_cpu, p_cpu) = run_batch(&cpu, &shape, data.clone());
    let (v_cuda, p_cuda) = run_batch(&cuda, &shape, data);

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
```

Notes:
- fp32 CUDA vs fp32 CPU differs only by accumulation order; tolerances (1e-3 on value, 1e-2 on pre-softmax logits) are generous but tight enough to catch a real divergence.
- The test skips cleanly if the model file is absent, so it does not break a checkout without `runs/`.
- If Path B was chosen in Task 2, run this test in a shell with `ORT_DYLIB_PATH` and `LD_LIBRARY_PATH` exported.

- [ ] **Step 2: Run the test**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust && cargo test --release --features binary,gpu --test cuda_consistency -- --nocapture
```
Expected: PASS, with a printed line like `max value diff = ...e-04, max policy_logit diff = ...e-03`.

- [ ] **Step 3: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/rust/tests/cuda_consistency.rs
git commit -m "vibe: test CUDA inference matches CPU output"
```

---

## Task 4: Full before/after benchmark and results writeup

Captures the win against the baseline and the next-bottleneck signal, and produces a markdown summary for the PR (per AGENTS.md).

**Files:**
- Create: `docs/superpowers/plans/2026-05-23-gpu-self-play-eval-results.md`

- [ ] **Step 1: Run the full 75 s GPU benchmark**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor
OUT=$(mktemp -d); LOG="$OUT/run.log"
timeout 95 rust/target/release/selfplay \
    --config /home/jbinney/ws/deep_rabbit_hole/experiments/2026_05_23_jon_b9w10_performance/config.yaml \
    --model-path /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/runs/jdb-performance-base/models/checkpoints/model_0.onnx \
    --output-dir "$OUT" --num-games 100000 --profile-counters > "$LOG" 2>&1 || true
echo "=== profile counters ==="; grep '\[pipe\]' "$LOG" | tail -12
echo "=== progress lines ==="; grep 'games/s' "$LOG" | tail -3
echo "=== games completed (npz) ==="; find "$OUT" -maxdepth 1 -name '*.npz' | wc -l
rm -rf "$OUT"
```
(If Path B: export `ORT_DYLIB_PATH` and `LD_LIBRARY_PATH` first.)

Record: steady-state `items`/5 s (→ evals/sec), games completed, and `batch_wait_ms`.

- [ ] **Step 2: Write the results doc**

Create `docs/superpowers/plans/2026-05-23-gpu-self-play-eval-results.md` with:
- The CPU baseline (from this plan's Reference section: ~7,600 evals/sec, 0 games in 95 s, GPU 7–9%).
- The GPU numbers from Step 1 (evals/sec, games completed, GPU utilization from Task 2).
- The speedup factor.
- Which integration path was used (A: bundled `ort/cuda`, or B: `load-dynamic` + `onnxruntime-gpu`), and any required runtime env (`ORT_DYLIB_PATH`, `LD_LIBRARY_PATH`) if Path B.
- The observed `batch_wait_ms`: if it climbed above ~0, note that CPU-side MCTS (incl. Quoridor wall-legality pathfinding) is now the next bottleneck — input for the deferred Approach B / sim-count / concurrency decisions.

- [ ] **Step 3: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add docs/superpowers/plans/2026-05-23-gpu-self-play-eval-results.md
git commit -m "vibe: document GPU self-play eval benchmark results"
```

---

## Task 5: Formatting commit (per AGENTS.md)

AGENTS.md requires functional and formatting changes in separate commits, and `cargo fmt` before committing Rust changes.

- [ ] **Step 1: Format and verify**

```bash
cd /home/jbinney/ws/deep_rabbit_hole/deep_quoridor/rust
cargo fmt
cargo fmt --check && echo "FMT CLEAN"
cargo build --release --features binary,gpu --bin selfplay
```
Expected: `FMT CLEAN`, build succeeds.

- [ ] **Step 2: Commit only if formatting produced changes**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git status --short deep_quoridor/rust
# If there are modified .rs files from fmt:
git add -u deep_quoridor/rust
git commit -m "vibe: cargo fmt"
```
If `git status` shows no changes, skip the commit.

---

## Self-Review (completed during plan authoring)

- **Spec coverage:** root cause (Task 1 reference + Task 1 fix), CUDA-EP-with-CPU-fallback (Task 1), sm_120 spike with Path A/B (Task 2), CPU-fallback build still works (Task 1 Step 4), numerics match (Task 3), re-profile + before/after + batch_wait_ms note (Task 4), fp32/deferred items (kept out of scope) — all mapped.
- **Placeholders:** none — all code and commands are concrete.
- **Type consistency:** `load_session` signature unchanged (`fn(&str) -> Result<Session>`); `GraphOptimizationLevel`, `CUDAExecutionProvider`, `CPUExecutionProvider`, `with_execution_providers`, `with_optimization_level`, `commit_from_file` used consistently and match the docs.rs-verified `ort` 2.0.0-rc.11 API; the `gpu` feature name is used identically in Cargo.toml and every build/test command.
