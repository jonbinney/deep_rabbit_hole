# Quoridor Rust Migration Plan

## Overview

This document outlines the strategy for migrating the Quoridor game implementation from Python/Numba to Rust while maintaining full backward compatibility with existing Python code.

## Goals

1. **Performance**: Match or exceed Numba performance
2. **Compatibility**: Drop-in replacement for existing `qgrid` module
3. **Maintainability**: Clean, idiomatic Rust code with good test coverage
4. **Ease of deployment**: Single wheel file, no Numba dependency

## Architecture

### Current Structure
```
deep_quoridor/src/
├── quoridor.py       # High-level game logic (Board, Quoridor classes)
├── qgrid.py          # Performance-critical functions (Numba-optimized)
└── agents/
    └── simple.py     # Agent implementations using qgrid
```

### Target Structure
```
deep_quoridor/
├── src/
│   ├── quoridor.py       # Keep mostly unchanged
│   └── agents/
│       └── simple.py     # Keep unchanged (imports quoridor_rs instead of qgrid)
├── test/
│   └── test_rust_python_equivalence.py  # Equivalence tests
└── rust/
    ├── Cargo.toml
    ├── pyproject.toml
    └── src/
        ├── lib.rs            # PyO3 module definition
        ├── pathfinding.rs    # BFS for distance calculations (✓ DONE)
        ├── validation.rs     # Move/wall validation logic
        ├── actions.rs        # Action masks and application
        └── grid.rs           # Grid utilities
```

## Migration Phases

### Phase 1: Setup & Proof of Concept (✓ COMPLETED)

**Status**: ✓ Done

**What was done**:
- [x] Set up Rust project with PyO3 and maturin
- [x] Created `Cargo.toml` with dependencies
- [x] Implemented `distance_to_row` in Rust
- [x] Created Python bindings via PyO3
- [x] Created test script to verify equivalence

**How to build and test**:
```bash
cd deep_quoridor/rust

# Install maturin if not already installed
pip install maturin

# Build and install the Rust module in development mode
maturin develop --release

# Run equivalence tests
python ../test/test_rust_python_equivalence.py
```

### Phase 2: Core Validation Functions

**Estimated effort**: 1-2 days

Functions to port from `qgrid.py`:
1. `are_wall_cells_free()` - Check if wall placement is unobstructed
2. `is_wall_potential_block()` - Check if wall could block a path
3. `set_wall_cells()` / `check_wall_cells()` - Grid manipulation utilities
4. `is_move_action_valid()` - Validate move actions
5. `is_wall_action_valid()` - Validate wall placement actions

**Implementation strategy**:
- Create `src/validation.rs` module
- Port each function with identical logic to Numba version
- Add unit tests in Rust
- Expose via PyO3 in `lib.rs`
- Add tests to `../test/test_rust_python_equivalence.py`

### Phase 3: Action Generation

**Estimated effort**: 1-2 days

Functions to port:
1. `compute_move_action_mask()` - Generate valid move actions
2. `compute_wall_action_mask()` - Generate valid wall actions
3. `get_valid_move_actions()` - Get array of valid move actions
4. `get_valid_wall_actions()` - Get array of valid wall actions

**Implementation strategy**:
- Create `src/actions.rs` module
- Use `ndarray` for efficient array operations
- Consider using `rayon` for parallelization if beneficial
- Add comprehensive tests

### Phase 4: Game State Manipulation

**Estimated effort**: 0.5 days

Functions to port:
1. `apply_action()` - Apply action to game state
2. `undo_action()` - Undo previously applied action
3. `check_win()` - Check if player has won

**Implementation strategy**:
- Create `src/game_state.rs` module
- These are simpler functions, should be straightforward

### Phase 5: Integration & Testing

**Estimated effort**: 1-2 days

**Tasks**:
1. Update `quoridor.py` to use `quoridor_rs` instead of `qgrid`
   - Add import fallback: try `quoridor_rs`, fall back to `qgrid`
   - This allows gradual migration
2. Run full test suite to ensure no regressions
3. Benchmark performance against Numba version
4. Update `SimpleAgent` to use Rust implementation
5. Run agent tournaments to verify correctness

### Phase 6: Deployment & Documentation

**Estimated effort**: 1 day

**Tasks**:
1. Set up CI/CD to build wheels for multiple platforms
2. Update `requirements.txt` to include `quoridor-rs`
3. Document build process
4. Create migration guide for other agents
5. Add performance benchmarks to documentation

## Function Mapping

Complete mapping of functions from `qgrid.py` to Rust modules:

| qgrid.py Function | Rust Module | Priority | Status |
|-------------------|-------------|----------|--------|
| `distance_to_row` | `pathfinding.rs` | HIGH | ✓ Done |
| `are_wall_cells_free` | `validation.rs` | HIGH | Todo |
| `is_wall_potential_block` | `validation.rs` | HIGH | Todo |
| `set_wall_cells` | `grid.rs` | HIGH | Todo |
| `check_wall_cells` | `grid.rs` | HIGH | Todo |
| `is_move_action_valid` | `validation.rs` | HIGH | Todo |
| `is_wall_action_valid` | `validation.rs` | HIGH | Todo |
| `compute_move_action_mask` | `actions.rs` | MEDIUM | Todo |
| `compute_wall_action_mask` | `actions.rs` | MEDIUM | Todo |
| `get_valid_move_actions` | `actions.rs` | MEDIUM | Todo |
| `get_valid_wall_actions` | `actions.rs` | MEDIUM | Todo |
| `apply_action` | `game_state.rs` | LOW | Todo |
| `undo_action` | `game_state.rs` | LOW | Todo |
| `check_win` | `game_state.rs` | LOW | Todo |

## Constants to Export

All constants from `qgrid.py` (already done in Phase 1):
- `CELL_FREE = -1`
- `CELL_PLAYER1 = 0`
- `CELL_PLAYER2 = 1`
- `CELL_WALL = 10`
- `WALL_ORIENTATION_VERTICAL = 0`
- `WALL_ORIENTATION_HORIZONTAL = 1`
- `ACTION_WALL_VERTICAL = 0`
- `ACTION_WALL_HORIZONTAL = 1`
- `ACTION_MOVE = 2`

## Testing Strategy

### Unit Tests
- Each Rust function has unit tests in its module
- Use `#[cfg(test)]` blocks for Rust-only tests

### Integration Tests
- `../test/test_rust_python_equivalence.py` verifies identical behavior
- Run on extensive test cases including:
  - Empty boards
  - Boards with walls
  - Edge cases (corners, borders)
  - Complex game states

### Performance Tests
- Benchmark each function against Numba version
- Target: Match or exceed Numba performance
- Measure: Time per function call, memory usage

### Regression Tests
- Run full agent test suite
- Play games between Rust and Numba versions
- Verify game outcomes are deterministic and identical

## Performance Optimizations

1. **Compiler Optimizations**:
   - Release mode with `opt-level = 3`
   - Link-time optimization (LTO)
   - Single codegen unit for better optimization

2. **Algorithm Optimizations**:
   - Use `VecDeque` for BFS (better than `Vec`)
   - Inline hot functions with `#[inline]`
   - Use array indexing instead of iterators where beneficial

3. **Parallelization** (if needed):
   - Use `rayon` for parallel action evaluation
   - Consider in Phase 3 for action mask computation

4. **Memory Optimization**:
   - Reuse allocations where possible
   - Use stack allocation for small arrays

## Backward Compatibility

### During Migration
```python
# In quoridor.py
try:
    import quoridor_rs as qgrid
except ImportError:
    import qgrid  # Fallback to Numba version
```

### After Migration
- Keep `qgrid.py` for reference
- Update all imports to use `quoridor_rs`
- Remove Numba dependency from `requirements.txt`

## Building and Distribution

### Development
```bash
cd rust
maturin develop --release
```

### Production
```bash
cd rust
maturin build --release
pip install target/wheels/quoridor_rs-*.whl
```

### CI/CD
Use GitHub Actions to build wheels for:
- Linux (x86_64, aarch64)
- macOS (x86_64, arm64)
- Windows (x86_64)

## Next Steps

1. **Build the proof of concept**:
   ```bash
   cd deep_quoridor/rust
   pip install maturin
   maturin develop --release
   python ../test/test_rust_python_equivalence.py
   ```

2. **Review results**: Check that tests pass and performance is comparable

3. **Start Phase 2**: Port validation functions
   - Begin with `are_wall_cells_free` (simplest)
   - Then `is_move_action_valid` (most frequently called)
   - Then `is_wall_action_valid` (most complex)

4. **Iterate**: Port functions one at a time, test thoroughly

## Benefits of This Approach

1. **Incremental**: Can migrate function by function
2. **Safe**: Fallback to Numba if Rust build fails
3. **Testable**: Easy to verify correctness at each step
4. **Performant**: Rust should match or exceed Numba
5. **Maintainable**: Cleaner code than Numba, better tooling

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Performance regression | Benchmark thoroughly, use profiling tools |
| Subtle behavioral differences | Extensive testing, fuzzing |
| Build complexity | Document clearly, use CI/CD |
| Platform-specific issues | Test on all target platforms |
| Increased binary size | Acceptable trade-off for benefits |

## Success Criteria

- [ ] All `qgrid` functions ported to Rust
- [ ] 100% pass rate on equivalence tests
- [ ] Performance within 10% of Numba (ideally better)
- [ ] All agent tests pass
- [ ] Clean build on Linux, macOS, Windows
- [ ] Documentation complete

## Timeline Estimate

- Phase 1: ✓ Completed
- Phase 2: 1-2 days
- Phase 3: 1-2 days
- Phase 4: 0.5 days
- Phase 5: 1-2 days
- Phase 6: 1 day

**Total**: 5-8 days of focused work

## Questions?

For questions or issues during migration:
1. Check `../test/test_rust_python_equivalence.py` output
2. Review Rust compiler errors carefully
3. Compare Rust implementation line-by-line with `qgrid.py`
4. Use `cargo test` to run Rust unit tests
5. Profile with `cargo flamegraph` if performance issues arise
