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
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import quoridor_rs
from agents.alphazero.nn_evaluator import NNConfig, NNEvaluator
from agents.alphazero.resnet_network import ResnetConfig
from quoridor import ActionEncoder
from utils.subargs import SubargsBase, parse_subargs


# ---------------------------------------------------------------------------
# NNConfig subargs
# ---------------------------------------------------------------------------


@dataclass
class NNConfigParams(SubargsBase):
    """Flat parameter class for NNConfig (for use with --nn subargs)."""

    type: str = "mlp"
    resnet_num_blocks: int = 5
    resnet_num_channels: int = 32
    mask_training_predictions: bool = False


def nn_config_from_params(params: NNConfigParams) -> NNConfig:
    config = NNConfig(type=params.type, mask_training_predictions=params.mask_training_predictions)
    if params.type == "resnet":
        config.resnet = ResnetConfig(
            num_blocks=params.resnet_num_blocks,
            num_channels=params.resnet_num_channels,
        )
    return config


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def read_metadata(conn):
    rows = conn.execute("SELECT key, value FROM metadata").fetchall()
    meta = {k: int(v) for k, v in rows}
    return meta["board_size"], meta["max_walls"], meta["max_steps"]


def compact_state_to_game_input(state_bytes, board_size, max_walls, max_steps):
    """Convert a compact state to the same feature vector as MLPNetwork.game_to_input_array.

    Output layout (matches game_to_input_array):
      [current_player_board (board_size²),
       opponent_board       (board_size²),
       walls                ((board_size-1)² * 2),
       my_walls_remaining   (1),
       opp_walls_remaining  (1)]
    """
    feats = quoridor_rs.compact_state_to_features(state_bytes, board_size, max_walls, max_steps)

    b = board_size - 1
    num_wall = 2 * b * b
    num_cells = board_size * board_size

    # compact_state_to_features wall layout: (orientation, row, col) order
    # game_to_input_array wall layout: (row, col, orientation) order — reshape to match
    wall_bits = feats[:num_wall].reshape(2, b, b).transpose(1, 2, 0).flatten()

    p0_board = feats[num_wall : num_wall + num_cells]
    p1_board = feats[num_wall + num_cells : num_wall + 2 * num_cells]
    p0_walls_norm = feats[num_wall + 2 * num_cells]
    p1_walls_norm = feats[num_wall + 2 * num_cells + 1]
    current_player = int(feats[-1])

    p0_walls = p0_walls_norm * max_walls
    p1_walls = p1_walls_norm * max_walls

    if current_player == 0:
        my_board, opp_board = p0_board, p1_board
        my_walls_count, opp_walls_count = p0_walls, p1_walls
    else:
        my_board, opp_board = p1_board, p0_board
        my_walls_count, opp_walls_count = p1_walls, p0_walls

    return np.concatenate([my_board, opp_board, wall_bits, [my_walls_count, opp_walls_count]])


def build_feature_matrix(state_blobs, board_size, max_walls, max_steps):
    features = [
        compact_state_to_game_input(s, board_size, max_walls, max_steps)
        for s in state_blobs
    ]
    return np.stack(features)  # (N, feature_dim) float32


# ---------------------------------------------------------------------------
# Accuracy metric
# ---------------------------------------------------------------------------


def compute_accuracy(test_blobs, db_dict, evaluator, board_size, max_walls, max_steps, num_samples):
    """Fraction of sampled states where model picks the same best child as DB."""
    sample = random.sample(test_blobs, min(num_samples, len(test_blobs)))
    correct = 0
    total = 0
    evaluator.network.eval()
    device = evaluator.device
    with torch.no_grad():
        for state_bytes in sample:
            children = quoridor_rs.get_compact_child_states(
                bytes(state_bytes), board_size, max_walls, max_steps
            )
            if not children:
                continue

            child_blobs = [c[3] for c in children]
            db_vals = []
            all_found = True
            for cb in child_blobs:
                cb_bytes = bytes(cb)
                if cb_bytes in db_dict:
                    db_vals.append(db_dict[cb_bytes])
                else:
                    all_found = False
                    break
            if not all_found:
                continue

            child_features = np.stack(
                [
                    compact_state_to_game_input(bytes(cb), board_size, max_walls, max_steps)
                    for cb in child_blobs
                ]
            )
            x = torch.tensor(child_features, dtype=torch.float32, device=device)
            _, model_vals = evaluator.network(x)
            model_vals = model_vals.squeeze(-1).cpu().numpy()

            # Determine current player from compact features (last element)
            current_player = int(
                quoridor_rs.compact_state_to_features(state_bytes, board_size, max_walls, max_steps)[-1]
            )
            pick = np.argmin if current_player == 1 else np.argmax
            if pick(db_vals) == pick(model_vals):
                correct += 1
            total += 1

    return correct / total if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Train a neural network evaluator from a policy DB.")
    p.add_argument("db_path", help="Path to .sqlite policy DB")
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-steps", type=int, default=10000)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--accuracy-states", type=int, default=200)
    p.add_argument("--output", default="evaluator.pt")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument(
        "--nn",
        default="",
        help=(
            "NNEvaluator network parameters as comma-separated key=value pairs. "
            f"Fields: {', '.join(NNConfigParams.fields())}. "
            "Example: --nn type=resnet,resnet_num_blocks=5,resnet_num_channels=32"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    nn_params = parse_subargs(args.nn, NNConfigParams)
    nn_config = nn_config_from_params(nn_params)
    print(f"Network config: {nn_params}")

    # ------------------------------------------------------------------
    # Load DB
    # ------------------------------------------------------------------
    conn = sqlite3.connect(args.db_path)
    board_size, max_walls, max_steps = read_metadata(conn)
    print(f"Board size: {board_size}, max_walls: {max_walls}, max_steps: {max_steps}")

    rows = conn.execute("SELECT state, value FROM policy").fetchall()
    conn.close()
    print(f"Loaded {len(rows)} entries from DB")

    state_blobs = [r[0] for r in rows]
    raw_values = np.array([r[1] for r in rows], dtype=np.float32)
    db_dict = {bytes(s): float(v) for s, v in rows}

    # ------------------------------------------------------------------
    # Feature matrix
    # ------------------------------------------------------------------
    print("Building feature matrix...")
    features = build_feature_matrix(
        [bytes(s) for s in state_blobs], board_size, max_walls, max_steps
    )
    feature_dim = features.shape[1]
    print(f"Feature dim: {feature_dim}, dataset size: {len(features)}")

    # ------------------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------------------
    N = len(features)
    indices = np.arange(N)
    np.random.shuffle(indices)
    test_size = max(1, int(N * args.test_fraction))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    X_train = torch.tensor(features[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(raw_values[train_idx], dtype=torch.float32, device=device)
    X_test = torch.tensor(features[test_idx], dtype=torch.float32, device=device)
    y_test = torch.tensor(raw_values[test_idx], dtype=torch.float32, device=device)
    test_blobs = [bytes(state_blobs[i]) for i in test_idx]

    print(f"Train size: {len(train_idx)}, test size: {len(test_idx)}")

    # ------------------------------------------------------------------
    # Create NNEvaluator and set up optimizer
    # ------------------------------------------------------------------
    action_encoder = ActionEncoder(board_size)
    evaluator = NNEvaluator(action_encoder, device, nn_config, max_cache_size=0)
    evaluator.train_prepare(args.learning_rate, args.batch_size, args.num_steps, args.weight_decay)
    loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_test_loss = float("inf")
    train_size = len(train_idx)

    for step in range(1, args.num_steps + 1):
        evaluator.network.train()
        batch_idx = torch.randint(0, train_size, (args.batch_size,), device=device)
        x_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        _, preds = evaluator.network(x_batch)
        preds = preds.squeeze(-1)
        loss = loss_fn(preds, y_batch)

        evaluator.optimizer.zero_grad()
        loss.backward()
        evaluator.optimizer.step()

        if step % args.log_interval == 0 or step == 1:
            evaluator.network.eval()
            with torch.no_grad():
                _, test_preds = evaluator.network(X_test)
                test_loss = loss_fn(test_preds.squeeze(-1), y_test).item()

            acc = compute_accuracy(
                test_blobs,
                db_dict,
                evaluator,
                board_size,
                max_walls,
                max_steps,
                args.accuracy_states,
            )

            print(
                f"step {step:6d} | train_loss {loss.item():.4f} | "
                f"test_loss {test_loss:.4f} | accuracy {acc:.3f}"
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
                        "nn_config": args.nn,
                    },
                    args.output,
                )

    print(f"Training complete. Best test loss: {best_test_loss:.4f}. Model saved to {args.output}")


if __name__ == "__main__":
    main()
