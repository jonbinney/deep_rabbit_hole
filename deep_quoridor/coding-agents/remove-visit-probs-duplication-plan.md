I'm using AGENTS.md

# Remove Visit-Prob Duplication Plan

## Goal
Eliminate duplicated MCTS visit-count normalization logic in AlphaZero agent code.

## Steps
1. Add one internal helper that computes normalized visit probabilities and performs the zero-visit guard.
2. Reuse this helper from both policy-vector construction and action selection.
3. Run diagnostics for the modified file.
4. Update results markdown with this follow-up cleanup.
