# Rust AlphaZero MCTS — Backlog

Future improvements and optimizations for the Rust AlphaZero implementation.

---

## Batch Search

**Status:** Not started  
**Priority:** High  
**Effort:** Large

Currently, each MCTS iteration evaluates a single position. Batch search would:
- Run multiple MCTS trees in parallel (for multiple games)
- Collect leaf positions across all trees
- Batch NN inference for all leaves at once
- Significantly improve GPU utilization

### Implementation Notes
- `MCTSBatch` struct managing multiple `NodeArena` instances
- Collect `(tree_idx, leaf_idx, game_state)` tuples during selection
- Single batched `Evaluator::evaluate_batch()` call
- Distribute results back to respective trees

---

## NN Evaluation Caching

**Status:** Not started  
**Priority:** Medium  
**Effort:** Medium

Cache neural network evaluation results to avoid redundant inference.

### Implementation Notes
- LRU cache keyed by `GameState::get_fast_hash()`
- Matching Python's `NNEvaluator` cache behavior
- Consider cache size based on available memory
- Invalidation strategy: per-game or fixed-size LRU

---

## Undo-Action Tree Traversal

**Status:** Not started  
**Priority:** Low  
**Effort:** Medium

Instead of cloning game states for each MCTS expansion, use `undo_action` to traverse the tree more efficiently.

### Implementation Notes
- Single mutable `GameState` for tree traversal
- Apply actions on descent, undo on ascent
- Requires careful tracking of action history
- May complicate parallelization
- Measure actual performance gain vs. cloning overhead

---

## Transposition Table

**Status:** Not started  
**Priority:** Low  
**Effort:** Large

Recognize when different action sequences lead to the same game state (transpositions).

### Implementation Notes
- HashMap of `state_hash -> node_idx`
- Merge statistics when same state reached via different paths
- Careful handling of parent pointers (DAG instead of tree)
- May improve value estimates in games with many transpositions

---

## Root Parallelization

**Status:** Not started  
**Priority:** Medium  
**Effort:** Medium

Run multiple independent MCTS searches from the same root and aggregate results.

### Implementation Notes
- Simple parallelization: N independent searches
- Aggregate visit counts across all searches
- No inter-thread communication during search
- Good for multi-core CPUs without GPU batching

---

## Virtual Loss

**Status:** Not started  
**Priority:** Medium  
**Effort:** Small

Add virtual loss for multi-threaded MCTS to encourage exploration diversity.

### Implementation Notes
- During descent, temporarily subtract a "virtual loss" from selected nodes
- Other threads less likely to select same path
- Remove virtual loss after backpropagation
- Commonly set to 1.0 or tunable

---

## Progressive Widening

**Status:** Not started  
**Priority:** Low  
**Effort:** Medium

Gradually expand children as visit count increases, rather than expanding all at once.

### Implementation Notes
- Only expand `k * sqrt(N)` children initially
- Add more children as parent visits increase
- May help in games with very large action spaces
- Quoridor action space is manageable, so lower priority

---

## MCTS-specific Optimizations

**Status:** Not started  
**Priority:** Low  
**Effort:** Various

Miscellaneous optimizations:
- **FPU (First Play Urgency):** Use a prior value for unexplored nodes instead of Q=0
- **Temperature decay:** Gradual temperature reduction instead of hard cutoff
- **Move ordering:** Sort children by prior during selection for better cache locality
- **RAVE/AMAF:** All-moves-as-first heuristic (less applicable to Quoridor)
