#!/bin/bash
# Local test script for Rust CI workflow
# This runs the same checks that GitHub Actions will run

set -e  # Exit on any error

echo "================================"
echo "Testing Rust CI Workflow Locally"
echo "================================"
echo ""

cd "$(dirname "$0")/../deep_quoridor/rust"

echo "✓ Working directory: $(pwd)"
echo ""

echo "[1/6] Checking code formatting..."
cargo fmt --all -- --check
echo "✓ Formatting check passed"
echo ""

echo "[2/6] Running clippy (linter)..."
cargo clippy --all-targets --all-features
echo "✓ Clippy check passed"
echo ""

echo "[3/6] Building in debug mode..."
cargo build --verbose
echo "✓ Debug build passed"
echo ""

echo "[4/6] Running tests..."
cargo test --verbose
echo "✓ All tests passed"
echo ""

echo "[5/6] Running tests with all features..."
cargo test --all-features --verbose
echo "✓ All-features tests passed"
echo ""

echo "[6/6] Building in release mode..."
cargo build --release --verbose
echo "✓ Release build passed"
echo ""

echo "================================"
echo "✓ All CI checks passed locally!"
echo "================================"
echo ""
echo "Your code is ready to push. The GitHub Actions workflow should pass."
