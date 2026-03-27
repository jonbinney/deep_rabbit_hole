I'm using AGENTS.md

# Manual Update Validate/Commit/PR Plan

## Goal
Validate recent manual edits still behave correctly, then commit, push, and open a PR against `main`.

## Steps
1. Identify tests that cover `alphazero.py` and `selfplay_real_model_reference.py` changes.
2. Run targeted tests and checks in the project virtualenv.
3. If tests pass, commit staged changes with required commit message format.
4. Push branch and create PR to `main` with concise summary.

## Notes
- Keep commit history review-friendly.
- Do not include unrelated changes.
