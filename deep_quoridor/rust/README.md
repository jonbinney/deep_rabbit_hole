# quoridor-rs

Rust implementation of Quoridor game logic and minimax evaluation, with Python bindings via PyO3.

## Prerequisites

- Rust toolchain (install via [rustup](https://rustup.rs/))
- Python 3.8+
- maturin: `pip install maturin` (also included in deep_quoridor/requirements/txt)

## Build & Test (Rust only)

```bash
cargo build          # Debug build
cargo build --release # Release build
cargo test           # Run tests
```

## Install Python Extension

Source your python virtual environment, then use these commands.

```bash
# Development (editable install, debug build)
maturin develop

# Development (editable install, release build)
maturin develop --release
```

## Usage (Python)

```python
import quoridor_rs

# Key functions
quoridor_rs.evaluate_actions(grid, positions, walls, goals, player, max_steps, ...)
quoridor_rs.distance_to_row(grid, start_row, start_col, target_row)
quoridor_rs.apply_action(grid, positions, walls, player, action)

# Constants
quoridor_rs.CELL_FREE, quoridor_rs.CELL_WALL
quoridor_rs.WALL_ORIENTATION_VERTICAL, quoridor_rs.WALL_ORIENTATION_HORIZONTAL
```
