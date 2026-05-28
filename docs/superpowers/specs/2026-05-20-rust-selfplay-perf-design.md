# Rust Self-Play Performance Redesign

## Goal

Improve the throughput of the Rust self-play binary used by `train_v2.py` so
that a single host saturates both CPU and GPU on a representative training
workload. Target: ≥4× games/sec on an NVIDIA RTX 5080 + 12-core / 24-thread
CPU at `mcts_n=1000`.

## Current bottleneck

The existing design (commit `fc12e54`) has one OS thread per concurrent game,
each running MCTS iterations sequentially and blocking on a single eval per
iteration. With the user's current production config — 12 threads × 128
games_per_thread = **1536 OS threads** on 12 physical cores — observed
utilization is roughly 800% CPU and 15% GPU, i.e. both resources are mostly
idle.

The system runs in lockstep:

1. ~1536 worker threads all block on the eval coordinator.
2. The coordinator concatenates inputs, runs one GPU batch, then *serially*
   softmaxes and un-rotates all ~1536 results.
3. Workers wake, do one quick MCTS step, block again.

Both the GPU (during steps 1, 2-stack, 2-softmax, 3) and the CPU (during step
2-GPU) sit idle, because the round-trip is fully serialized. Adding more
threads makes context-switch overhead worse without helping throughput.

## Approach

Three-layer redesign, all landing together:

- **Per-game leaf-parallel MCTS with virtual loss** — each game's MCTS issues
  K eval requests per outer iteration instead of one, decoupling in-flight
  eval count from worker thread count.
- **Async worker model** — replace OS-thread-per-game with one tokio task per
  concurrent game, so game count can far exceed hardware thread count and
  scheduler overhead drops.
- **Pipelined coordinator** — split the eval coordinator into Batcher →
  Inference → Post-process stages on dedicated OS threads, so GPU compute
  overlaps with batch assembly and parallel softmax/un-rotation.

Additional algorithmic and infrastructural improvements ride along:

- **Tree reuse across moves** within a single game (avoids re-evaluating a
  subtree the previous search already explored).
- **Real eval cache cap** raised from 0 to 100K entries; no eviction policy
  beyond "stop inserting at cap" — relying on model-reload to clear.
- **Parallel `finalize_policy`** via rayon at the post-processor.

The Python side (`train_v2.py`) is unchanged. Each Rust subprocess still runs
independently; the design only changes what happens inside one subprocess.

## High-level architecture

A Rust self-play process becomes a tokio runtime with three roles:

1. **Game tasks.** One `tokio::spawn`'d future per concurrent game. Each task
   owns its MCTS tree. When MCTS needs evals, the future `.await`s on K
   `tokio::sync::oneshot` receivers.
2. **Eval coordinator.** A pipeline of three dedicated OS threads (not async)
   that own the ORT session. Receives `EvalRequest`s from game tasks over a
   bounded `tokio::sync::mpsc`, batches them, runs inference, posts results
   back via the per-request oneshot.
3. **Main task.** Sets up the runtime, polls `latest.yaml` for model reloads,
   watches the shutdown sentinel, joins everything on exit.

```
game task 1 ─┐
game task 2 ─┤      tokio          chan(1)          chan(1)         tokio oneshot
   …         ├──► mpsc ─► [Batcher] ──► [Inference: ORT] ──► [Post-process] ──► game task
game task N ─┘                                                       │
                                                                     └─► shared EvalCache
                            ▲
                            └── tokio mpsc ── ctrl (Reload/Shutdown) from main task
```

Tokio worker-thread count defaults to hardware threads (24). Games-per-process
is independent and can be 100–500.

## Per-game MCTS detail

Each game task runs a leaf-parallel MCTS per move.

### Outer loop

Total iterations per search is still the configured `mcts_n`. Each outer step
consumes up to K iterations:

```
while iters_done < mcts_n:
    batch = min(K, mcts_n - iters_done)
    pending = []
    for _ in 0..batch:
        path, leaf_idx = select_with_vl(&mut tree, vl)
        if leaf is terminal:
            undo_vl(path); backprop_terminal(path, value); iters_done += 1
            continue
        result = cache.get(leaf_state)
        if hit:  pending.push(Hit  { path, leaf_idx, result })
        else:    pending.push(Miss { path, leaf_idx, future = submit_eval(...) })
    try_join_all(misses).await                              # GPU round-trip
    for item in pending (in selection order):
        (value, priors) = item.result
        undo_vl(item.path)
        expand_node(item.leaf_idx, priors)
        backprop_real(item.path, value)
        iters_done += 1
```

### Virtual loss

During descent, at each node entered (including the leaf) apply
`visit_count += vl, value_sum -= vl`. When the result comes back, walk the
same path and `visit_count -= vl, value_sum += vl` before doing real
backprop. The path is stored as a `SmallVec<[usize; 32]>` — chosen to
cover typical Quoridor MCTS tree depth without spilling to the heap.

### Cache hit short-circuit

The game task checks the shared `EvalCache` *before* sending an
`EvalRequest`. On hit, the leaf is recorded with its cached value and never
crosses the channel. The K-leaf selection loop mixes hits and misses freely
and only awaits the miss futures. We accept the possibility that two games
submit the same state in the brief window between lookup and batch dispatch
— the coordinator will evaluate it twice. No dedup machinery is added.

### Tree reuse across moves

After the agent selects action A at the root, A's child subtree is promoted
to the new root. The kept subtree is copied into a fresh arena (siblings and
their descendants are dropped). Inherited visit counts and value sums are
kept. Dirichlet noise is re-applied to the new root's priors at the start of
the next search.

### Pristine priors

Each node stores `priors_clean` (un-noised). Root Dirichlet noise is applied
to a working copy used only for child UCB scoring during the current search;
the stored priors are never overwritten. This makes re-noising the new root
after tree-reuse straightforward.

### Model reloads mid-game

Each game records the model version it started with. When the game observes
that the coordinator's version has advanced, it discards its tree at the
next move boundary and starts fresh — the existing tree's value estimates
were produced by the old network.

## Coordinator detail

Three dedicated OS threads connected by bounded `std::sync::mpsc::sync_channel`s
of capacity 1.

### Batcher

Owns the tokio `Receiver` for `EvalRequest`s; uses `blocking_recv()` from its
OS thread. Collects up to `eval_batch_size` requests, deadline-bounded by
`eval_max_wait_ms` from the first request's arrival. Stacks features into a
single `(B, C, M, M)` tensor and forwards `BatchPayload { stacked, reqs }`
downstream. Control messages (`Reload(path)`, `Shutdown`) are forwarded
in-band as enum variants on the same channel so ordering with batches is
preserved.

### Inference

Owns the `ort::Session`. Loops on `recv()`:

- `Batch`: call `session.run(...)` synchronously. GPU/CPU pipelining comes
  for free because while this thread is on the GPU, the batcher is filling
  the next batch and the post-processor is finalizing the previous one.
- `Reload(path)`: drop the current session, load the new one, clear the
  shared cache, forward the marker downstream.
- `Shutdown`: exit.

### Post-processor

Receives `(reqs, raw_value_tensor, raw_policy_tensor)`. Runs
`finalize_policy` (mask + softmax + un-rotation) in parallel via
`rayon::par_iter` — this is the work that's serial today and slow at batch
sizes above ~1000. Inserts results into the shared `EvalCache`. Fires each
request's tokio oneshot.

### Backpressure

Capacity-1 inter-stage channels make the system self-regulating: if any
stage is the slowest, the others block on it. Profiling tells us which
stage to widen.

### Cache

Same `DashMap<CompactState, EvalResult>` as today, same
model-reload-clears-it behavior. Default cap raised from 0 to 100K. Keep
"stop inserting at cap" — no eviction. Tree reuse handles within-game
repetition; the shared cache is mainly for cross-game opening positions.

## Config schema

```yaml
self_play:
  num_processes: int                # unchanged (Python-side subprocess count)
  games_per_process: int            # NEW (replaces threads_per_process × games_per_thread)
  leaf_parallelism: int = 16        # NEW: K, in-flight evals per game per outer iter
  virtual_loss: int = 3             # NEW: vl magnitude (AlphaGo Zero default)
  enable_tree_reuse: bool = true    # NEW
  mcts_worker_threads: int | null   # NEW: runtime worker threads, null = hw threads
  eval_batch_size: int = 2048       # unchanged
  eval_max_wait_ms: int = 0         # unchanged
  eval_cache_max_size: int = 100000 # default raised from 0
  alphazero, program, rust_selfplay_binary: unchanged
```

`mcts_worker_threads` is named to be runtime-agnostic so the config doesn't
change if we ever swap tokio for something else.

`threads_per_process` and `games_per_thread` are deleted outright from both
the Python `SelfPlayConfig` (`src/v2/config.py`) and the Rust
`SelfPlayWorkerConfig` (`rust/src/selfplay_config.rs`). No
backward-compatibility shims — the rename happened a commit ago and there
are no production runs that need migration.

The Rust `selfplay` binary CLI mirrors the new YAML:
`--games-per-process`, `--leaf-parallelism`, `--virtual-loss`,
`--mcts-worker-threads`, `--no-tree-reuse`. Experiment YAMLs under
`deep_quoridor/experiments/` are updated in the same commit.

## Testing strategy

### Unit tests (no GPU required; use `UniformMockEvaluator`)

- Virtual-loss apply/undo on a hand-built tree: stats return exactly to
  baseline after undo + real backprop.
- Leaf-parallel selection diversification: with K=8 and vl=3, the 8 selected
  paths span ≥2 distinct first-level children when priors are reasonably
  spread.
- Tree reuse: promote child A; verify kept subtree's node count and root
  statistics; verify other branches dropped; verify Dirichlet noise
  re-applied to the new root using stored pristine priors.
- Cache hit short-circuits eval submission: mock the request sender, assert
  send count.
- Coordinator stages tested individually: batcher accumulation under
  `eval_max_wait_ms`; post-processor `finalize_policy` output equals the
  serial version for the same inputs.

### Integration tests

- End-to-end self-play game with K=1, vl=0, tree reuse off → identical
  results to sequential MCTS modulo RNG.
- End-to-end self-play game with K=16, vl=3, tree reuse on → games complete
  and per-move stats look sane.

### Cross-language consistency

Existing `python_consistency.rs` tests should pass unchanged — evaluator
and rotation code paths are not modified.

### Performance check

A script that runs the Rust self-play binary against a fixed model for 60
seconds and reports games/sec. Run before/after on the reference hardware
(RTX 5080 + 12c/24t). Target ≥4× improvement.

The binary gains a `--profile-counters` flag that periodically logs:
GPU-busy % (fraction of time the Inference thread is inside
`session.run`), batcher wait time (time the batcher is blocked on its
input channel), post-processor parallel time. This tells us which stage
limits throughput in the new design.

## Out of scope

- LRU or other eviction policies on the shared cache.
- Coordinator-side eval-request deduplication.
- "Virtual mean" (Cazenave) as an alternative to virtual loss.
- Multi-threading inside a single game's MCTS tree.
- FP16 inference, CUDA EP-specific tuning, multi-GPU.

These remain candidates if the first round of improvements does not hit
the throughput target.
