# Quoridor Rust Implementation

High-performance Rust implementation of Quoridor game logic with Python bindings via PyO3.

## Quick Start

### Prerequisites

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin
```

### Build and Install

```bash
# Development build (faster compilation, slower runtime)
maturin develop

# Release build (recommended - much faster runtime)
maturin develop --release
```

### Test

```bash
# Run Rust unit tests
cargo test

# Run Python equivalence tests (compares with qgrid.py)
python ../test/test_rust_python_equivalence.py

# Or from the project root:
# cd ..
# python test/test_rust_python_equivalence.py
```

## Usage

Once built, you can use `quoridor_rs` as a drop-in replacement for `qgrid`:

```python
import numpy as np
import quoridor_rs

# Use exactly like qgrid
grid = np.array(...)  # Your game grid
distance = quoridor_rs.distance_to_row(grid, start_row=0, start_col=4, target_row=8)
```

## Project Structure

```
rust/
├── Cargo.toml                          # Rust dependencies
├── pyproject.toml                      # Python project metadata
├── src/
│   ├── lib.rs                          # PyO3 bindings
│   └── pathfinding.rs                  # BFS algorithms
├── README.md                           # This file
└── MIGRATION_PLAN.md                   # Full migration guide

../test/
└── test_rust_python_equivalence.py     # Integration tests
```

## Status

**Phase 1 Complete**: ✓
- [x] Project setup with PyO3 and maturin
- [x] `distance_to_row` implementation
- [x] Python bindings
- [x] Test suite

**Next Steps**: See [MIGRATION_PLAN.md](MIGRATION_PLAN.md) for the complete roadmap.

## Performance

The Rust implementation aims to match or exceed Numba performance. Run the benchmark:

```bash
python test_rust_python_equivalence.py
```

## Development

### Run tests
```bash
cargo test              # Rust unit tests
cargo test -- --nocapture  # With output
```

### Build release binary
```bash
maturin build --release
```

### Format code
```bash
cargo fmt
```

### Lint
```bash
cargo clippy
```

## Documentation

- [MIGRATION_PLAN.md](MIGRATION_PLAN.md) - Complete migration strategy and roadmap

## Benefits

- **Fast**: Comparable or better than Numba
- **Safe**: Rust's type system catches bugs at compile time
- **Portable**: Single wheel file, works everywhere
- **Maintainable**: Better tooling than Numba
- **No JIT warmup**: Instant startup, unlike Numba
