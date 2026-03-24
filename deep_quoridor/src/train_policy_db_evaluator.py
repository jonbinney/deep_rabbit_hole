"""Train a neural network evaluator from a policy DB.

The model approximates the minimax lookup: given a compact game state,
predict a value in [-1, 1] (P0-absolute: 1=P0 wins, -1=P1 wins, 0=draw).

Two metrics are logged during training:
  - MSE loss on a held-out test set
  - Move accuracy: fraction of test states where the model picks the same
    best child state as the DB
"""

import argparse
import random
import sqlite3
import sys

import numpy as np
import torch
import torch.nn as nn

import quoridor_rs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class EvaluatorMLP(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def read_metadata(conn):
    rows = conn.execute("SELECT key, value FROM metadata").fetchall()
    meta = {k: int(v) for k, v in rows}
    return meta["board_size"], meta["max_walls"], meta["max_steps"]


def build_feature_matrix(state_blobs, board_size, max_walls, max_steps):
    features = [
        quoridor_rs.compact_state_to_features(s, board_size, max_walls, max_steps)
        for s in state_blobs
    ]
    return np.stack(features)  # (N, feature_dim) float32


# ---------------------------------------------------------------------------
# Accuracy metric
# ---------------------------------------------------------------------------


def compute_accuracy(test_blobs, db_dict, model, board_size, max_walls, max_steps, num_samples, device):
    """Fraction of sampled states where model picks the same best child as DB."""
    sample = random.sample(test_blobs, min(num_samples, len(test_blobs)))
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for state_bytes in sample:
            children = quoridor_rs.get_compact_child_states(
                bytes(state_bytes), board_size, max_walls, max_steps
            )
            if not children:
                continue

            child_blobs = [c[3] for c in children]

            # DB values (P0-absolute)
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

            # Model values
            child_features = np.stack([
                quoridor_rs.compact_state_to_features(bytes(cb), board_size, max_walls, max_steps)
                for cb in child_blobs
            ])
            x = torch.tensor(child_features, dtype=torch.float32, device=device)
            model_vals = model(x).squeeze(-1).cpu().numpy()

            # Determine best child: P0 maximises P0-absolute; P1 minimises it.
            parent_feats = quoridor_rs.compact_state_to_features(
                state_bytes, board_size, max_walls, max_steps
            )
            current_player = int(parent_feats[-1])
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
    p.add_argument("--num-steps", type=int, default=10000)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--accuracy-states", type=int, default=200)
    p.add_argument("--output", default="evaluator.pt")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument("--hidden-dim", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

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

    # Build full lookup dict for accuracy metric (bytes -> float)
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
    # Model, optimizer, loss
    # ------------------------------------------------------------------
    model = EvaluatorMLP(feature_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_test_loss = float("inf")
    train_size = len(train_idx)

    for step in range(1, args.num_steps + 1):
        model.train()
        batch_idx = torch.randint(0, train_size, (args.batch_size,), device=device)
        x_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        preds = model(x_batch).squeeze(-1)
        loss = loss_fn(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test).squeeze(-1)
                test_loss = loss_fn(test_preds, y_test).item()

            acc = compute_accuracy(
                test_blobs, db_dict, model, board_size, max_walls, max_steps,
                args.accuracy_states, device
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
                        "model_state_dict": model.state_dict(),
                        "feature_dim": feature_dim,
                        "hidden_dim": args.hidden_dim,
                        "board_size": board_size,
                        "max_walls": max_walls,
                        "max_steps": max_steps,
                    },
                    args.output,
                )

    print(f"Training complete. Best test loss: {best_test_loss:.4f}. Model saved to {args.output}")


if __name__ == "__main__":
    main()
