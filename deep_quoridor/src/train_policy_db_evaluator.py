"""Train a neural network evaluator from a policy DB.

The model approximates the minimax lookup: given a compact game state,
predict a value in [-1, 1] (P0-absolute: 1=P0 wins, -1=P1 wins, 0=draw).

Uses NNEvaluator (and its underlying network) so that the resulting model
is compatible with the rest of the AlphaZero infrastructure.

Two metrics are logged during training:
  - MSE loss on a held-out test set
  - Move accuracy: fraction of test states where the model picks the same
    best child state as the DB
"""

import argparse
import random
import sqlite3
import sys
from dataclasses import asdict

import numpy as np
import quoridor_rs
import torch
from agents.alphazero.alphazero import AlphaZeroParams
from agents.alphazero.nn_evaluator import NNConfig, NNEvaluator
from quoridor import ActionEncoder, Board, Player, Quoridor
from utils.subargs import parse_subargs

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def read_metadata(conn):
    rows = conn.execute("SELECT key, value FROM metadata").fetchall()
    meta = {k: v for k, v in rows}
    board_size = int(meta["board_size"])
    max_walls = int(meta["max_walls"])
    max_steps = int(meta["max_steps"])
    num_states = int(meta["num_states"]) if "num_states" in meta else None
    return board_size, max_walls, max_steps, num_states


def compact_state_to_game(state_bytes, board_size, max_walls, max_steps):
    """Convert compact state bytes to a Quoridor game object."""
    grid, player_positions, walls_remaining, old_style_walls, current_player, _completed_steps = (
        quoridor_rs.compact_state_to_game_state(state_bytes, board_size, max_walls, max_steps)
    )
    board = Board.from_arrays(
        board_size,
        max_walls,
        np.asarray(grid),
        np.asarray(player_positions),
        np.asarray(walls_remaining),
        np.asarray(old_style_walls),
    )
    return Quoridor(board, Player(current_player))


def child_action_index(row, col, action_type, board_size):
    """Map (row, col, action_type) from get_compact_child_states to an ActionEncoder index."""
    wall_size = board_size - 1
    if action_type == 2:  # pawn move
        return row * board_size + col
    elif action_type == 0:  # vertical wall
        return board_size ** 2 + row * wall_size + col
    elif action_type == 1:  # horizontal wall
        return board_size ** 2 + wall_size ** 2 + row * wall_size + col
    raise ValueError(f"Unknown action_type {action_type}")


def build_policy_from_children(conn, state_bytes, current_player, board_size, max_walls, max_steps, num_actions):
    """Build a policy vector from DB child state values.

    Finds the optimal child value (max for P0, min for P1) and assigns
    uniform probability across all actions that achieve that value.
    All other actions get zero probability.

    Returns the policy array, or None if some children are missing from the DB.
    """
    children = quoridor_rs.get_compact_child_states(state_bytes, board_size, max_walls, max_steps)
    if not children:
        return None

    child_blobs = [bytes(c[3]) for c in children]
    placeholders = ",".join("?" * len(child_blobs))
    child_rows = conn.execute(
        f"SELECT state, value FROM policy WHERE state IN ({placeholders})",
        child_blobs,
    ).fetchall()
    if len(child_rows) != len(child_blobs):
        return None  # Some children not in DB.
    child_db_vals = {bytes(s): v for s, v in child_rows}

    # Find optimal value from current player's perspective.
    child_values = [child_db_vals[bytes(c[3])] for c in children]
    best_value = max(child_values) if current_player == 0 else min(child_values)

    # Assign equal probability to all actions achieving the best value.
    policy = np.zeros(num_actions, dtype=np.float32)
    for (row, col, action_type, child_state) in children:
        if child_db_vals[bytes(child_state)] == best_value:
            idx = child_action_index(row, col, action_type, board_size)
            policy[idx] = 1.0
    policy /= policy.sum()
    return policy


def fetch_batch(conn, ids, evaluator, board_size, max_walls, max_steps):
    """Fetch rows by rowid from the policy table and compute features on the fly.

    Returns a list of sample dicts compatible with NNEvaluator.compute_losses.
    Each dict has keys: input_array, value, action_mask, mcts_policy, current_player.

    mcts_policy is derived from the DB values of child states: uniform over
    actions that lead to the optimal child value (max for P0, min for P1),
    zero for all others. States whose children are not all present in the
    DB are skipped.
    """
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT state, value FROM policy WHERE rowid IN ({placeholders})",
        ids,
    ).fetchall()
    num_actions = evaluator.action_encoder.num_actions
    samples = []
    for state, value in rows:
        state_bytes = bytes(state)
        game = compact_state_to_game(state_bytes, board_size, max_walls, max_steps)
        current_player = int(game.current_player)

        mcts_policy = build_policy_from_children(
            conn, state_bytes, current_player, board_size, max_walls, max_steps, num_actions,
        )
        if mcts_policy is None:
            continue

        action_mask = game.get_action_mask().astype(np.float32)
        samples.append(
            {
                "input_array": evaluator.game_to_input_array(game),
                "value": value,
                "action_mask": action_mask,
                "mcts_policy": mcts_policy,
                "current_player": current_player,
            }
        )
    return samples


# ---------------------------------------------------------------------------
# Accuracy metric
# ---------------------------------------------------------------------------


def compute_accuracy(test_ids, conn, evaluator, board_size, max_walls, max_steps, num_samples, test_player=None):
    """Fraction of sampled states where model picks the same best child as DB.

    If test_player is 0 or 1, only states where the current player matches are counted.
    """
    sample_ids = random.sample(test_ids, min(num_samples, len(test_ids)))

    # Fetch the state blobs and values for the sampled test IDs.
    placeholders = ",".join("?" * len(sample_ids))
    rows = conn.execute(
        f"SELECT state, value FROM policy WHERE rowid IN ({placeholders})",
        sample_ids,
    ).fetchall()

    correct = 0
    total = 0
    evaluator.network.eval()
    device = evaluator.device
    with torch.no_grad():
        for state_blob, _value in rows:
            state_bytes = bytes(state_blob)
            game = compact_state_to_game(state_bytes, board_size, max_walls, max_steps)
            current_player = int(game.current_player)

            # Filter by player if requested.
            if test_player is not None and current_player != test_player:
                continue

            children = quoridor_rs.get_compact_child_states(state_bytes, board_size, max_walls, max_steps)
            if not children:
                continue

            child_blobs = [bytes(c[3]) for c in children]

            # Look up DB values for all children via the state index.
            child_placeholders = ",".join("?" * len(child_blobs))
            child_rows = conn.execute(
                f"SELECT state, value FROM policy WHERE state IN ({child_placeholders})",
                child_blobs,
            ).fetchall()
            if len(child_rows) != len(child_blobs):
                continue  # Some children not in DB (e.g. terminal states).
            child_db_vals = {bytes(s): v for s, v in child_rows}
            db_vals = [child_db_vals[cb] for cb in child_blobs]

            # Model predictions for children.
            child_features = []
            for cb in child_blobs:
                child_game = compact_state_to_game(cb, board_size, max_walls, max_steps)
                child_features.append(evaluator.game_to_input_array(child_game))
            x = torch.tensor(np.stack(child_features), dtype=torch.float32, device=device)
            _, model_vals = evaluator.network(x)
            model_vals = model_vals.squeeze(-1).cpu().numpy()

            # The model's pick is correct if it leads to a child with the best DB value.
            best_db_val = max(db_vals) if current_player == 0 else min(db_vals)
            model_pick = np.argmax(model_vals) if current_player == 0 else np.argmin(model_vals)
            if db_vals[model_pick] == best_db_val:
                correct += 1
            total += 1

    return correct / total if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MAX_TEST_SIZE = 10000


def parse_args():
    p = argparse.ArgumentParser(description="Train a neural network evaluator from a policy DB.")
    p.add_argument("db_path", help="Path to .sqlite policy DB")
    p.add_argument(
        "-p",
        "--params",
        type=str,
        default="",
        help="AlphaZero params in subargs form (e.g. nn_type=mlp,learning_rate=0.001)",
    )
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--num-steps", type=int, default=10000)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--accuracy-states", type=int, default=200)
    p.add_argument("--output", default="evaluator.pt")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument(
        "--test-player",
        type=int,
        default=None,
        choices=[1, 2],
        help="Only evaluate test loss and accuracy on positions where this player is to move (1 or 2)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    az_params = parse_subargs(args.params, AlphaZeroParams)
    nn_config = NNConfig.from_alphazero_params(az_params)
    print(f"AlphaZero params: {az_params}")

    # ------------------------------------------------------------------
    # Open DB and read metadata
    # ------------------------------------------------------------------
    conn = sqlite3.connect(args.db_path)
    board_size, max_walls, max_steps, num_states = read_metadata(conn)
    print(f"Board size: {board_size}, max_walls: {max_walls}, max_steps: {max_steps}")

    if num_states is None:
        # Fallback for DBs created before autoincrement ID was added.
        num_states = conn.execute("SELECT COUNT(*) FROM policy").fetchone()[0]
        print(f"(num_states not in metadata, counted {num_states} rows)")
    print(f"Policy DB contains {num_states} states")

    # Convert 1-based player arg to 0-based internal representation.
    test_player = None
    if args.test_player is not None:
        test_player = args.test_player - 1
        print(f"Filtering test set to player {args.test_player} (internal: {test_player})")

    # ------------------------------------------------------------------
    # Train/test split by ID (IDs are 1-based, contiguous)
    # ------------------------------------------------------------------
    test_size = min(max(1, int(num_states * args.test_fraction)), MAX_TEST_SIZE)
    test_id_set = set(random.sample(range(1, num_states + 1), test_size))
    test_ids = sorted(test_id_set)
    print(f"Train size: ~{num_states - test_size}, test size: {len(test_ids)}")

    # ------------------------------------------------------------------
    # Create NNEvaluator and set up optimizer
    # ------------------------------------------------------------------
    action_encoder = ActionEncoder(board_size)
    evaluator = NNEvaluator(action_encoder, device, nn_config, max_cache_size=0)
    evaluator.train_prepare(az_params.learning_rate, az_params.batch_size, args.num_steps, az_params.weight_decay)

    # ------------------------------------------------------------------
    # Pre-compute test samples (test set is small / capped)
    # ------------------------------------------------------------------
    print("Computing test features...")
    test_samples = fetch_batch(conn, test_ids, evaluator, board_size, max_walls, max_steps)

    # Filter test set by player if requested.
    if test_player is not None:
        test_samples = [s for s in test_samples if s["current_player"] == test_player]
        print(f"Filtered test set to {len(test_samples)} states for player {args.test_player}")

    feature_dim = test_samples[0]["input_array"].shape[0]
    print(f"Feature dim: {feature_dim}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_test_loss = float("inf")
    batch_size = az_params.batch_size

    for step in range(1, args.num_steps + 1):
        # Oversample to account for test ID removal, player filtering, and states
        # dropped by build_policy_from_children (missing children in DB).
        oversample = 4 if test_player is None else 8
        batch_ids = random.sample(range(1, num_states + 1), min(batch_size * oversample, num_states))
        batch_ids = [i for i in batch_ids if i not in test_id_set]
        batch_samples = fetch_batch(conn, batch_ids, evaluator, board_size, max_walls, max_steps)
        if test_player is not None:
            batch_samples = [s for s in batch_samples if s["current_player"] == test_player]
        batch_samples = batch_samples[:batch_size]
        _, train_value_loss, train_total_loss = evaluator.train_iteration_v2(batch_samples)

        if step % args.log_interval == 0 or step == 1:
            evaluator.network.eval()
            with torch.no_grad():
                _, test_value_loss, _ = evaluator.compute_losses(test_samples)
            test_loss = test_value_loss.item()
            evaluator.network.train()

            acc = compute_accuracy(
                test_ids,
                conn,
                evaluator,
                board_size,
                max_walls,
                max_steps,
                args.accuracy_states,
                test_player=test_player,
            )

            print(
                f"step {step:6d} | train_loss {train_value_loss:.4f} | test_loss {test_loss:.4f} | accuracy {acc:.3f}"
            )
            sys.stdout.flush()

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(
                    {
                        "network_state_dict": evaluator.network.state_dict(),
                        "board_size": board_size,
                        "max_walls": max_walls,
                        "max_steps": max_steps,
                        "params": asdict(az_params),
                    },
                    args.output,
                )

    conn.close()
    print(f"Training complete. Best test loss: {best_test_loss:.4f}. Model saved to {args.output}")


if __name__ == "__main__":
    main()
