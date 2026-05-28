# GPU Self-Play Eval — Benchmark Results

**Date:** 2026-05-23  
**Branch:** `jdb/b9w10-performance`  
**GPU:** RTX 5080 (Blackwell, sm_120)

---

## Summary

Moving AlphaZero resnet inference from the CPU to the RTX 5080 via the
ONNX Runtime CUDA execution provider delivers approximately a **5× throughput
increase**: from ~7,600 NN evals/sec on CPU to ~38,000 evals/sec on GPU.
The GPU is confirmed active (nvidia-smi shows `selfplay` as a compute process
consuming ~2,565 MiB VRAM; GPU utilization reaches 20–31% in steady state
vs 7–9% idle during the CPU run). A one-time ~30–40 s ORT kernel
JIT-compilation warm-up occurs on the first run for sm_120; subsequent
runs start at full throughput immediately if ORT caches the compiled kernels.

---

## Root Cause Recap

The original `load_session` in
`deep_quoridor/rust/src/agents/alphazero/eval_pipeline.rs` created an ORT
`Session` with no explicit execution provider registration. ORT therefore
defaulted to the CPU execution provider, even though the binary was linked
against ORT libraries. The fix adds a `gpu` Cargo feature (`ort/cuda` +
`ort/load-dynamic`) and registers the CUDA EP (with CPU fallback) inside a
`#[cfg(feature = "gpu")]` block.

---

## Before vs After

| Metric                     | CPU baseline         | GPU (measured)          |
|----------------------------|----------------------|-------------------------|
| NN evals/sec (steady state)| ~7,600               | ~38,000                 |
| Speedup                    | 1×                   | **~5×**                 |
| Games completed (95–130 s) | 0                    | 1 (same; games are long)|
| GPU utilization (nvidia-smi)| 7–9% (idle)         | 20–31%                  |
| GPU memory (`selfplay`)    | 0 MiB (no process)   | ~2,565 MiB              |
| avg_batch (items/batch)    | ~480                 | ~500–510                |
| batch_wait_ms              | 0.0                  | 0.0                     |
| post_ms                    | ~0.4                 | ~0.3                    |
| CPU usage (`selfplay`)     | ~1100% (≈11 threads) | not separately measured |

### Raw [pipe] lines (GPU run, 24 × 5-second intervals)

```
[pipe] batches=388 items=71743  avg_batch=184.9 gpu_busy=95.0% batch_wait_ms=0.0 post_ms=0.3  ← warmup
[pipe] batches=371 items=155084 avg_batch=418.0 gpu_busy=90.1% batch_wait_ms=0.0 post_ms=0.5
[pipe] batches=365 items=173250 avg_batch=474.7 gpu_busy=89.2% batch_wait_ms=0.0 post_ms=0.6
[pipe] batches=364 items=168360 avg_batch=462.5 gpu_busy=89.0% batch_wait_ms=0.0 post_ms=0.5
[pipe] batches=359 items=170907 avg_batch=476.1 gpu_busy=89.3% batch_wait_ms=0.0 post_ms=0.4
[pipe] batches=365 items=171959 avg_batch=471.1 gpu_busy=89.4% batch_wait_ms=0.0 post_ms=0.4
[pipe] batches=362 items=172419 avg_batch=476.3 gpu_busy=89.4% batch_wait_ms=0.0 post_ms=0.4
[pipe] batches=377 items=188261 avg_batch=499.4 gpu_busy=87.8% batch_wait_ms=0.0 post_ms=0.4
[pipe] batches=363 items=178241 avg_batch=491.0 gpu_busy=88.1% batch_wait_ms=0.0 post_ms=0.4
[pipe] batches=371 items=185253 avg_batch=499.3 gpu_busy=88.2% batch_wait_ms=0.0 post_ms=0.4
[pipe] batches=362 items=180005 avg_batch=497.3 gpu_busy=88.2% batch_wait_ms=0.0 post_ms=0.4
[pipe] batches=363 items=181848 avg_batch=501.0 gpu_busy=87.8% batch_wait_ms=0.0 post_ms=0.3
[pipe] batches=383 items=192450 avg_batch=502.5 gpu_busy=87.3% batch_wait_ms=0.0 post_ms=0.3
[pipe] batches=386 items=194493 avg_batch=503.9 gpu_busy=87.5% batch_wait_ms=0.0 post_ms=0.3
[pipe] batches=382 items=193645 avg_batch=506.9 gpu_busy=87.0% batch_wait_ms=0.0 post_ms=0.3
[pipe] batches=390 items=198187 avg_batch=508.2 gpu_busy=86.7% batch_wait_ms=0.0 post_ms=0.3
[pipe] batches=389 items=197145 avg_batch=506.8 gpu_busy=87.4% batch_wait_ms=0.0 post_ms=0.3
[pipe] batches=388 items=197398 avg_batch=508.8 gpu_busy=86.3% batch_wait_ms=0.0 post_ms=0.3
[pipe] batches=381 items=193213 avg_batch=507.1 gpu_busy=86.5% batch_wait_ms=0.0 post_ms=0.3
[pipe] batches=381 items=194432 avg_batch=510.3 gpu_busy=86.6% batch_wait_ms=0.0 post_ms=0.3
[pipe] batches=394 items=198616 avg_batch=504.1 gpu_busy=87.0% batch_wait_ms=0.0 post_ms=0.3
[pipe] batches=389 items=196730 avg_batch=505.7 gpu_busy=87.2% batch_wait_ms=0.0 post_ms=0.3
[pipe] batches=379 items=191540 avg_batch=505.4 gpu_busy=86.7% batch_wait_ms=0.0 post_ms=0.3
[pipe] batches=391 items=198934 avg_batch=508.8 gpu_busy=87.7% batch_wait_ms=0.0 post_ms=0.3
```

Warmup ended around t=35–40 s (items rising from 71k → 155k → 173k per
interval; GPU utilization correspondingly jumping from ~17% to ~22%).
Steady state begins around interval 8 (t=40 s).

**Note on `gpu_busy%`:** This field measures inference-thread busy time (CPU
time in the inference loop), not GPU hardware utilization. At 87–89% it
reflects the inference thread being kept busy dispatching batches. Real GPU
utilization (20–31%) is lower because GPU kernel execution is fast and the
thread spends some time on CPU-side marshalling.

---

## How GPU Was Enabled

**Path A (bundled ORT CUDA binaries) — failed:** Cargo feature `ort/cuda`
pulls prebuilt ONNX Runtime CUDA binaries built against CUDA 12 (they look for
`libcudart.so.12`), which is mismatched against this host's CUDA 13 runtime
(installed via the `torch` cu130 wheels); the bundled CUDA provider also could
not find a compatible cuDNN on the library path. The session built but fell
back silently to CPU.

**Path B (load-dynamic) — succeeded:** Feature flags set to
`gpu = ["ort/cuda", "ort/load-dynamic"]` in `Cargo.toml`. The binary is
linked without embedded ORT; it loads `libonnxruntime.so` at runtime via
`ORT_DYLIB_PATH`. The `onnxruntime-gpu==1.26.0` wheel (installed into the
project venv) supplies a library built against CUDA 13 / cuDNN 9 / sm_120
that supports the CUDA execution provider on Blackwell.

---

## Required Runtime Environment

These two environment variables must be exported before running `selfplay`
(or any process that spawns it):

```bash
export ORT_DYLIB_PATH=/home/jbinney/ws/deep_rabbit_hole/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.26.0
export LD_LIBRARY_PATH=/home/jbinney/ws/deep_rabbit_hole/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:/home/jbinney/ws/deep_rabbit_hole/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

**IMPORTANT — `train_v2.py` subprocess inheritance:**
`train_v2.py` launches `selfplay` via `subprocess.Popen(cmd)` without an
explicit `env=` argument, so the subprocess inherits the parent shell's
environment. GPU inference will only work if the user exports the two
variables above before starting training. Recommended follow-up: update
`train_v2.py` to inject `ORT_DYLIB_PATH` and `LD_LIBRARY_PATH` explicitly
when spawning `selfplay`, so the GPU path is automatic and not
shell-state-dependent.

**New runtime dependency:** `onnxruntime-gpu==1.26.0` should be added to
`requirements.txt` (or the project's pip install instructions). It is not
currently listed.

---

## Warmup Note

On the first invocation after a system or GPU reset, ORT JIT-compiles CUDA
kernels for sm_120 (Blackwell). This takes approximately 30–40 s during
which throughput is lower (~14,000–35,000 evals/sec) before reaching steady
state. Subsequent runs may start closer to full throughput if ORT caches the
compiled kernels. Plan for this warm-up period when timing training cycles
or running short benchmarks.

---

## Next Bottleneck / Deferred Work

**What the counters say.** Three readings have to be taken together:

- `gpu_busy ≈ 87–89%` — this counter measures the **inference thread's** busy
  fraction (time spent inside `session.run()`), *not* GPU hardware use. It is
  near-saturated.
- `batch_wait_ms = 0.0` — the batcher's blocking receive never starves; there
  is always a request queued.
- `nvidia-smi` GPU utilization is only **20–31%**.

If CPU-side MCTS were the binding constraint, the inference thread would idle
waiting for batches (`gpu_busy` low) and the batcher would starve
(`batch_wait_ms` high). Neither is true. So the bottleneck is the **single
synchronous inference stage**, and it is **launch/transfer-overhead-bound, not
GPU-compute-bound**: this is a small, shallow resnet (6 blocks × 32 ch, 21×21)
run at ~500-item batches, so each forward pass is a long sequence of tiny CUDA
kernels whose per-launch latency dominates while the GPU sits mostly idle
between launches. That is exactly why `session.run()` occupies ~88% of wall
time while the GPU hardware is only ~25% busy.

Batch size is the lever that matters: ~505 of a configured max of 1024 means
each batch amortizes its fixed per-launch overhead over only half the items it
could. Larger batches would raise throughput at roughly constant GPU time.

**Is MCTS co-limiting?** Possibly, but the data doesn't show it as the primary
constraint. The clean way to find out is to raise concurrency and watch
`batch_wait_ms`: if it stays ~0 and throughput scales, the eval stage was the
limit; if `batch_wait_ms` climbs, MCTS (incl. Quoridor wall-legality
pathfinding) has become the limit. Decide based on that measurement, not
assumption.

**Deferred levers (from spec), in likely-impact order:**

1. **More concurrency / larger batches:** Increase `games_per_process`
   (currently 128) and/or `leaf_parallelism` so more requests are in flight per
   eval cycle, pushing `avg_batch` toward 1024 and amortizing per-launch
   overhead. Cost: GPU memory and CPU threads. This directly targets the
   measured bottleneck and is the first thing to try.

2. **fp16 / TensorRT EP:** The installed `onnxruntime-gpu==1.26.0` includes
   `TensorrtExecutionProvider`. TensorRT fuses the resnet into far fewer
   kernels (cutting launch count) and fp16 halves transfer size — both attack
   the launch/transfer overhead identified above. Strong candidate.

3. **MCTS sim-count tuning:** Reducing simulations per move lowers total eval
   demand per game (faster games) at some cost to data quality; a per-phase
   schedule is an option.

4. **Quoridor pathfinding optimisation:** Faster wall-legality checks
   (incremental BFS state, lookup tables). Only worth it if step 1's
   concurrency experiment shows `batch_wait_ms` rising — i.e. only once MCTS is
   demonstrably the limit.
