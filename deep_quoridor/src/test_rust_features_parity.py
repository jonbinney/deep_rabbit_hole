"""Parity test: PyPolicyDb.fetch_training_batch (Rust) must produce the
same per-sample features, action mask, mcts_policy, value, and current
player as the legacy Python pipeline (compact_state_to_game + rotate +
network.game_to_input_array + build_policy_from_action_values).

Run:
    cd deep_quoridor && pytest src/test_rust_features_parity.py -v

This requires the `quoridor_rs` Python extension to be rebuilt
(e.g., `maturin develop --release` from `deep_quoridor/rust`).
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import quoridor_rs
import torch

sys.path.insert(0, str(Path(__file__).parent))

from agents.alphazero.alphazero import AlphaZeroParams  # noqa: E402
from agents.alphazero.nn_evaluator import NNConfig, NNEvaluator  # noqa: E402
from quoridor import ActionEncoder  # noqa: E402

# Reuse the helpers from the training script so we exercise the exact same
# logic the Rust path was meant to replace.
from train_policy_db_evaluator import (  # noqa: E402
    build_policy_from_action_values,
    compact_state_to_game,
)


def _make_small_db(tmp_path):
    """Build a 3x3, 0-walls, 8-steps policy DB for parity testing."""
    repo_root = Path(__file__).parent.parent.parent
    binary = repo_root / "deep_quoridor" / "rust" / "target" / "release" / "create_policy_db"
    if not binary.exists():
        pytest.skip(
            f"create_policy_db binary not found at {binary} — build with `cargo build --release --features binary --bins`"
        )
    out_path = tmp_path / "parity.parquet"
    subprocess.run(
        [
            str(binary),
            "--board-size",
            "3",
            "--max-walls",
            "0",
            "--max-steps",
            "8",
            "--output",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    return str(out_path)


@pytest.mark.parametrize("nn_type", ["mlp", "resnet"])
def test_fetch_training_batch_matches_python_pipeline(tmp_path, nn_type):
    db_path = _make_small_db(tmp_path)
    db = quoridor_rs.PyPolicyDb(db_path, lazy=False)
    board_size, max_walls, max_steps, num_states = db.read_metadata()

    # Build an evaluator just to access the Python feature/rotation helpers.
    az_params = AlphaZeroParams()
    az_params.nn_type = nn_type
    nn_config = NNConfig.from_alphazero_params(az_params)
    if nn_config.resnet is not None:
        nn_config.resnet.max_steps = max_steps
    evaluator = NNEvaluator(
        ActionEncoder(board_size),
        torch.device("cpu"),
        nn_config,
        max_cache_size=100,
    )

    # Pull every state in the DB through both pipelines.
    rowids = list(range(1, num_states + 1))
    inputs_rust, values_rust, masks_rust, policies_rust, cps_rust = db.fetch_training_batch(rowids, nn_type)
    rows = db.fetch_states_by_rowid(rowids)
    num_actions = evaluator.action_encoder.num_actions

    py_inputs = []
    py_values = []
    py_masks = []
    py_policies = []
    py_cps = []
    for state, db_value in rows:
        game = compact_state_to_game(state, board_size, max_walls, max_steps)
        cp = int(game.current_player)
        if cp == 1:
            db_value = -db_value
        mcts_policy = build_policy_from_action_values(db, state, board_size, num_actions)
        rotated_game, is_rotated = evaluator.rotate_if_needed_to_point_downwards(game)
        input_array = evaluator.game_to_input_array(rotated_game)
        action_mask = rotated_game.get_action_mask().astype(np.float32)
        if is_rotated:
            mcts_policy = evaluator.rotate_policy_from_original(mcts_policy)
        py_inputs.append(input_array)
        py_values.append(db_value)
        py_masks.append(action_mask)
        py_policies.append(mcts_policy)
        py_cps.append(cp)

    py_inputs = np.stack(py_inputs)
    py_values = np.array(py_values, dtype=np.int32)
    py_masks = np.stack(py_masks).astype(np.float32)
    py_policies = np.stack(py_policies).astype(np.float32)
    py_cps = np.array(py_cps, dtype=np.int32)

    # Strict equality on input arrays (both pipelines do the same arithmetic
    # on the same integer state representation; no rounding involved).
    assert inputs_rust.shape == py_inputs.shape
    np.testing.assert_array_equal(inputs_rust, py_inputs)
    np.testing.assert_array_equal(masks_rust, py_masks)
    np.testing.assert_allclose(policies_rust, py_policies, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(values_rust, py_values)
    np.testing.assert_array_equal(cps_rust, py_cps)
