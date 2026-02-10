#!/bin/bash
# Run the Quoridor ONNX inference binary

cargo build --release --bin infer_quoridor
cargo run --release --bin infer_quoridor
