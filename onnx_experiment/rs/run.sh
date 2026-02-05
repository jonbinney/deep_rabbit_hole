#!/bin/bash
# Run the Rust ONNX inference with proper library paths for CUDA support

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$SCRIPT_DIR/target/release"

export LD_LIBRARY_PATH="$RELEASE_DIR:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"

"$RELEASE_DIR/infer" "$@"
