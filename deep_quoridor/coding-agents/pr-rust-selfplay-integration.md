# PR: Integrate Rust Self-Play with Python Training Pipeline

## Summary

This PR connects the Rust self-play binary with the Python v2 training pipeline, enabling the trainer to use fast Rust-based self-play instead of Python workers. The integration is activated with a single `--rust-selfplay` flag.

## Changes

### 1. Switch Python to npz format (`5e68672`)
- `alphazero.py`: Changed `end_game_batch_and_save_replay_buffers` from pickle to npz with Rust-compatible field names (`input_arrays`, `policies`, `action_masks`, `values`, `players`)
- `trainer.py`: Updated replay reading from `.pkl` to `.npz`, with conversion to per-sample dicts at read time

### 2. Add `--model-version` CLI parameter (`6d060bb`)
- `selfplay.rs`: Added `--model-version` CLI arg to stamp replay buffer metadata with the correct model version

### 3. Add continuous mode with model hot-reload (`903f8d7`)
- `selfplay.rs`: Added `--continuous`, `--latest-model-yaml`, `--shutdown-file` CLI args
- `selfplay_config.rs`: Added `LatestModelYaml` struct and `load_latest_model()` function
- Rust binary polls `latest.yaml` between games, reloads ONNX model on version change, exits on shutdown sentinel file

### 4. Add `--rust-selfplay` flag to Python trainer (`182003f`)
- `train_v2.py`: Added `--rust-selfplay` arg that spawns Rust self-play processes in continuous mode
- `config.py`: Added `rust_selfplay_binary` field to `SelfPlayConfig`
- Auto-enables ONNX export when Rust self-play is active

### 5. Fix race conditions + ResNet CI config (`8e6c94e`)
- `trainer.py`: Moved `LatestModel.write()` after ONNX export to prevent consumers from reading yaml before model files exist
- `selfplay.rs`: Added wait-for-ONNX-file logic in continuous mode
- `selfplay_config.rs`: Added `NetworkType` enum and config parsing for `alphazero.network.type`
- Added `ci-resnet.yaml` for testing with ResNet network type
- Added MLP input support to Rust backlog

## Usage

```bash
# Build the Rust binary
cd deep_quoridor/rust && cargo build --release --features binary --bin selfplay

# Run training with Rust self-play
cd deep_quoridor/src && python train_v2.py ../experiments/ci-resnet.yaml --rust-selfplay
```

## Testing

- All 118 Rust unit tests pass
- End-to-end test verified with ResNet config (`ci-resnet.yaml`):
  - Rust workers load initial ONNX model, play games, produce npz replay buffers
  - Python trainer consumes buffers, trains models, exports ONNX
  - Rust workers detect model hot-reloads (v0→v1→v2)
  - Graceful shutdown via sentinel file works correctly

## Deferred

- MLP network input support in Rust (tracked in `deep_quoridor/rust/backlog.md`)
