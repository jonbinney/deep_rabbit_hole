# GPU self-play eval — design

**Date:** 2026-05-23
**Branch:** `jdb/b9w10-performance`
**Status:** approved (design), pending implementation plan

## Problem

Rust self-play (`train_v2.py` → `rust/target/release/selfplay`) is fast enough for
5×5 / 2-wall Quoridor but unusably slow for the target configuration, 9×9 / 10-wall
(`experiments/2026_05_23_jon_b9w10_performance/config.yaml`).

### Root cause (measured, not assumed)

Profiling on this machine (RTX 5080, 24-core CPU) with the target config and
`runs/jdb-performance-base/models/checkpoints/model_0.onnx`:

- `--profile-counters` reported `gpu_busy≈97%`, ~7,600 NN evals/sec, **0 games
  completed in 95 s**.
- `nvidia-smi` during the run: **GPU utilization 7–9%**, no `selfplay` compute
  process on the GPU. `selfplay` CPU usage ≈ **1100%** (≈11 of 12 worker threads).

The resnet inference runs **entirely on the CPU**; the RTX 5080 is idle. The
`gpu_busy` counter is misnamed — `gpu_ns` measures how busy the *inference thread*
is, which here is CPU work.

The cause is in `load_session()` (`rust/src/agents/alphazero/eval_pipeline.rs:105`):

```rust
Session::builder()?.commit_from_file(model_path)
```

No execution provider is registered, and the `ort` dependency
(`rust/Cargo.toml:35`) has no `cuda`/`tensorrt` feature. In `ort` 2.0 that defaults
to the **CPU execution provider**. 5×5 "works" only because its games are short
enough that CPU inference keeps up; 9×9 exposes it.

At ~7,600 evals/sec shared across 128 concurrent games, a single 1000-sim move
takes ~16 s of wall-clock, so a full 9×9 game runs tens of minutes.

## Goal & success criteria

Move resnet inference onto the RTX 5080 with a clean CPU fallback. No change to
MCTS, batching, or game logic in this phase.

Success is met when, on the same benchmark (target config, `model_0.onnx`, ~75 s):

1. `nvidia-smi` shows the `selfplay` process consuming the GPU (utilization well
   above the current 7–9% idle baseline).
2. `--profile-counters` throughput rises to a multiple of today's ~7,600 evals/sec.
3. Games actually complete within the benchmark window (today: 0).
4. Inference numerics still match the CPU path on sample inputs (we changed *where*
   the net runs, not *what* it computes).

## Environment (constraints the design must respect)

- GPU: RTX 5080, compute capability **12.0 (sm_120, Blackwell)**.
- Driver 580.159.03 (CUDA 13 capable).
- PyTorch already runs on this GPU (`torch 2.12.0+cu130`, `cuda.is_available()=True`,
  cap (12, 0)) — the CUDA-13 / sm_120 hardware+driver path is already proven.
- The venv ships **CUDA 13 runtime** (`libcudart.so.13`, `libcublas.so.13`) and
  **cuDNN 9** (`libcudnn*.so.9`) via the torch cu130 wheels. This is a known-good
  sm_120 lib generation already on disk.
- No system ONNX Runtime; no `onnxruntime` in the venv (only `onnx`).

## Architecture / where the change lands

Single chokepoint: `load_session()` in `eval_pipeline.rs`. Changes:

- Add a Cargo feature to the `ort` dependency that pulls in CUDA support, gated so
  the binary still builds CPU-only when the feature is off.
- In `load_session`, register a CUDA execution provider **with CPU fallback** (a host
  without the GPU stack still runs), and set graph optimization to Level 3.
- Everything downstream is untouched: the 3-stage batcher → inference → post pipeline,
  the `Reload` path (so retrained models also load onto the GPU), and the single
  dedicated inference thread (`run_inference`) all stay as-is. CUDA EP from one
  dedicated thread is fine.

Precision stays **fp32** in this phase to match training numerics and avoid accuracy
questions. fp16/TensorRT is deferred.

## Phases

### Phase 0 — sm_120 de-risking spike (throwaway)

The plan hinges on ONNX Runtime actually executing on sm_120. Before touching
`load_session`, get a single batched forward pass of `model_0.onnx` running on the
GPU and confirm correct output + GPU utilization. The spike resolves the integration
choice:

- **Path A (try first):** `ort`'s bundled `cuda` feature binaries. Simplest if they
  support sm_120 and match the available CUDA runtime.
- **Path B (fallback):** `ort`'s `load-dynamic`, pointed via `ORT_DYLIB_PATH` at a
  recent `onnxruntime-gpu` that supports CUDA 13 + cuDNN 9 + sm_120 — reusing the
  same lib generation torch already proves works here.

Only after the spike picks a working combination do we wire it into `load_session`.

### Phase 1 — wire CUDA EP into `load_session`

Apply the working combination from Phase 0: register the CUDA EP with CPU fallback
and Level-3 graph optimization, behind the Cargo feature. Rebuild the release binary.

### Phase 2 — verify & measure

Re-run the exact benchmark from today (`--profile-counters`, same config/model,
~75 s) with `nvidia-smi` polling. Record games/sec and evals/sec before/after, and
confirm inference numerics match the CPU path on a few inputs (cache /
`python_consistency`-style check). Note where `batch_wait_ms` lands.

## Explicitly out of scope (deferred — measure first, then decide)

- fp16 / TensorRT execution provider (Approach B).
- MCTS sim-count tuning (`mcts_n` 1000 → lower).
- Concurrency / batch-fill changes (`games_per_process`, `leaf_parallelism`,
  `eval_batch_size`).

We expect `batch_wait_ms` to rise above 0 once eval is fast (CPU MCTS — including
Quoridor wall-legality pathfinding — becoming the next bottleneck). That is a
follow-up decision, informed by Phase 2 numbers, not part of this phase.

## Risks & fallbacks

- **sm_120 unsupported by available ORT builds.** Mitigated by Phase 0 spike and the
  Path A / Path B split. The torch cu130 + cuDNN 9 stack already runs on this GPU, so
  a matching ORT build is expected to exist.
- **CUDA/cuDNN runtime version mismatch** (ORT built for a different CUDA major than
  the `.so.13` libs on disk). Path B addresses this by selecting an ORT build aligned
  to the on-disk CUDA 13 / cuDNN 9 libs.
- **CPU-only hosts / CI.** The CUDA path is feature-gated and falls back to CPU, so
  CPU-only builds and runs are unaffected.
