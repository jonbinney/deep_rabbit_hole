#!/usr/bin/env bash
# Run the Rust self-play binary in batch mode for a fixed wall-clock window
# and report games/sec. Pass a YAML config and an ONNX model path.
set -euo pipefail

CONFIG="${1:?usage: bench_rust_selfplay.sh CONFIG_YAML MODEL_ONNX [DURATION_SECONDS]}"
MODEL="${2:?usage: bench_rust_selfplay.sh CONFIG_YAML MODEL_ONNX [DURATION_SECONDS]}"
DURATION="${3:-60}"

BIN="deep_quoridor/rust/target/release/selfplay"
if [[ ! -x "$BIN" ]]; then
    echo "Building selfplay release binary..."
    (cd deep_quoridor/rust && cargo build --release --features binary --bin selfplay)
fi

OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

echo "Benchmarking for ${DURATION}s..."
timeout "$DURATION" "$BIN" \
    --config "$CONFIG" \
    --model-path "$MODEL" \
    --output-dir "$OUT" \
    --num-games 100000 \
    --profile-counters \
    || true

COUNT=$(find "$OUT" -maxdepth 1 -name '*.npz' | wc -l)
echo "Games completed in ${DURATION}s: $COUNT"
echo "Rate: $(python3 -c "print(${COUNT} / ${DURATION})") games/sec"
