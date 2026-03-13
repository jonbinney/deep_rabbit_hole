# Parallel Self-Play and NN Evaluation Caching — Results

## Summary

Implemented multi-threaded parallel game execution and LRU-based NN
evaluation caching for the Rust selfplay binary, matching the
parallelism already present in the Python `self_play.py`.

## What Changed

### Evaluator Trait (`evaluator.rs`)
- Changed `Evaluator::evaluate(&mut self, ...)` → `evaluate(&self, ...)`
  to allow shared access across threads.
- `OnnxEvaluator` now wraps `ort::Session` in a `Mutex` (required
  because `Session::run()` takes `&mut self`).

### AlphaZeroAgent (`agent.rs`)
- Changed from owning an `OnnxEvaluator` to holding
  `Arc<dyn Evaluator + Send + Sync>`, enabling multiple agents to
  share a single evaluator.

### CachingEvaluator (`evaluator.rs`)
- New struct wrapping `Mutex<Session>` + `Mutex<LruCache<u64, ...>>`.
- LRU cache keyed by `GameState::get_fast_hash()`. Default capacity
  200,000 entries, configurable via CLI.
- `reload_model()` swaps the ONNX session and clears the cache (used
  by continuous mode for hot-reload).
- The `lru` crate (v0.12) was added behind the `binary` feature flag.

### `play_games_parallel` (`game_runner.rs`)
- New function using `std::thread::scope` to run games across N threads.
- Each thread creates its own `AlphaZeroAgent` pair with independent
  `visited_states`, sharing the evaluator via `Arc`.
- Work is divided evenly: `ceil(num_games / threads)` per thread.
- Feature-gated behind `#[cfg(feature = "binary")]`.
- Two integration tests using a `MockEvaluator`.

### Selfplay Binary (`selfplay.rs`)
- New CLI flags: `--parallel-games` (default 1), `--max-cache-size`
  (default 200,000).
- **Batch mode:** When `parallel_games > 1` and using AlphaZero
  self-play (not raw ONNX, no P2 override), uses `CachingEvaluator` +
  `play_games_parallel`. Falls back to sequential otherwise.
- **Continuous mode:** Parallel path uses `CachingEvaluator` with
  `reload_model()` for hot-reload. Plays games in batches of
  `parallel_games`, checking for shutdown/model updates between
  batches. Sequential path preserved as fallback.

### Backlog (`backlog.md`)
- Removed "NN Evaluation Caching" (now implemented).
- Renamed "Batch Search" → "Batched NN Inference" with updated
  description focusing on collecting MCTS leaves across trees into a
  single GPU forward pass.

## Commits

| Hash | Description |
|------|-------------|
| `6148b4f` | Add plan for parallel games and cache |
| `bf0cc0e` | Change Evaluator trait to `&self` |
| `5e20367` | Refactor AlphaZeroAgent to borrow evaluator |
| `0d95964` | Add CachingEvaluator with LRU cache |
| `2757302` | Add play_games_parallel with threading |
| `3cdac5f` | Wire parallel games and caching in selfplay |
| `7370c37` | Update backlog after cache/parallel work |

## Testing

- 90 tests pass with default features (no ONNX).
- 127 tests pass with `--features binary` (includes 2 new parallel tests).
- `cargo fmt --check` clean throughout.

## Usage

```bash
# Sequential (unchanged behavior)
selfplay --config ci.yaml --model-path model.onnx --output-dir /tmp/out --num-games 100

# 4 threads with caching
selfplay --config ci.yaml --model-path model.onnx --output-dir /tmp/out \
  --num-games 100 --parallel-games 4

# Custom cache size
selfplay --config ci.yaml --model-path model.onnx --output-dir /tmp/out \
  --num-games 100 --parallel-games 4 --max-cache-size 500000

# Continuous mode with parallelism
selfplay --config ci.yaml --output-dir /tmp/out --continuous \
  --latest-model-yaml /tmp/run/models/latest.yaml \
  --shutdown-file /tmp/run/.shutdown --parallel-games 4
```
