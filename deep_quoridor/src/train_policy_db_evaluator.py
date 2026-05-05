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
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import quoridor_rs
import torch
import wandb
from agents.alphazero.alphazero import AlphaZeroParams
from agents.alphazero.nn_evaluator import NNConfig, NNEvaluator
from quoridor import ActionEncoder, Board, Player, Quoridor
from utils.subargs import parse_subargs
from utils.timer import Timer, timer

DEBUG = False

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def policy_to_str(policy, action_encoder):
    """Print non-zero entries of a policy array as human-readable actions."""
    nonzero = np.nonzero(policy)[0]
    parts = []
    for idx in nonzero:
        action = action_encoder.index_to_action(idx)
        parts.append(f"{action}: {policy[idx]:.3f}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@timer("compact_state_to_game")
def compact_state_to_game(state, board_size, max_walls, max_steps):
    """Convert a compact state (Python int) to a Quoridor game object."""
    grid, player_positions, walls_remaining, old_style_walls, current_player, completed_steps = (
        quoridor_rs.compact_state_to_game_state(state, board_size, max_walls, max_steps)
    )
    board = Board.from_arrays(
        board_size,
        max_walls,
        np.asarray(grid),
        np.asarray(player_positions),
        np.asarray(walls_remaining),
        np.asarray(old_style_walls),
    )
    return Quoridor(board, Player(current_player), completed_steps=completed_steps)


def child_action_index(row, col, action_type, board_size):
    """Map (row, col, action_type) from get_compact_child_states to an ActionEncoder index."""
    wall_size = board_size - 1
    if action_type == 2:  # pawn move
        return row * board_size + col
    elif action_type == 0:  # vertical wall
        return board_size**2 + row * wall_size + col
    elif action_type == 1:  # horizontal wall
        return board_size**2 + wall_size**2 + row * wall_size + col
    raise ValueError(f"Unknown action_type {action_type}")


@timer("build_policy_from_action_values")
def build_policy_from_action_values(db, state, board_size, num_actions):
    """Build a policy vector from DB action values.

    Uses lookup_action_values which correctly handles terminal child states.
    Assigns uniform probability across all actions with the best value
    (values are from acting player's perspective, so best = max).

    Returns the policy array, or None if no valid actions.
    """
    result = db.lookup_action_values(state)
    assert result is not None

    actions, values = result
    best_value = max(values)

    policy = np.zeros(num_actions, dtype=np.float32)
    for (row, col, action_type), value in zip(actions, values):
        if value == best_value:
            idx = child_action_index(row, col, action_type, board_size)
            policy[idx] = 1.0
    policy /= policy.sum()
    return policy


@timer("fetch_batch")
def fetch_batch(db, ids, nn_type):
    """Fetch a training batch from the policy DB in one Rust call.

    Returns a dict of pre-stacked NumPy arrays (one row per requested rowid):
      input_arrays:    shape depends on nn_type (MLP: (N, D); ResNet: (N, 5, M, M))
      values:          (N,) int32, acting-player perspective
      action_masks:    (N, num_actions) float32
      mcts_policies:   (N, num_actions) float32, uniform over best-valued actions
      current_players: (N,) int32, 0 or 1 in unrotated frame
    """
    inputs, values, masks, policies, cps = db.fetch_training_batch(ids, nn_type)
    return {
        "input_arrays": inputs,
        "values": values,
        "action_masks": masks,
        "mcts_policies": policies,
        "current_players": cps,
    }


def _filter_by_player(batch, test_player):
    """Keep only rows where current_player matches test_player. No-op if test_player is None."""
    if test_player is None:
        return batch
    keep = batch["current_players"] == test_player
    return {k: v[keep] for k, v in batch.items()}


def _slice_batch(batch, start, stop):
    return {k: v[start:stop] for k, v in batch.items()}


def _concat_batches(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


# ---------------------------------------------------------------------------
# Test metrics
# ---------------------------------------------------------------------------


@timer("compute_test_metrics_batched")
def compute_test_metrics_batched(
    test_ids, db, evaluator: NNEvaluator, batch_size, *, nn_type, test_player=None
):
    """Iterate `test_ids` in chunks, emitting batches of exactly `batch_size`
    samples (post player filter). Only the final batch may be smaller.

    When `test_player` filtering drops samples below `batch_size`, we keep
    fetching more `test_ids` and refilling a buffer until the batch is full
    or `test_ids` is exhausted.

    Returns (policy_loss, value_loss, total_loss, accuracy) as floats.
    """
    from agents.alphazero.nn_evaluator import INVALID_ACTION_VALUE

    total_pol = total_val = total_tot = 0.0
    correct = total = 0

    n_test_ids = len(test_ids)
    ids_idx = 0
    buffer = None  # dict of np arrays, or None when empty

    def buffer_size(buf):
        return 0 if buf is None else buf["values"].shape[0]

    evaluator.network.eval()
    with torch.no_grad():
        while ids_idx < n_test_ids or buffer_size(buffer) > 0:
            while buffer_size(buffer) < batch_size and ids_idx < n_test_ids:
                end = min(ids_idx + batch_size, n_test_ids)
                next_ids = test_ids[ids_idx:end]
                ids_idx = end
                new_batch = fetch_batch(db, next_ids, nn_type)
                new_batch = _filter_by_player(new_batch, test_player)
                buffer = new_batch if buffer is None else _concat_batches(buffer, new_batch)

            n = min(batch_size, buffer_size(buffer))
            if n == 0:
                break
            front = _slice_batch(buffer, 0, n)
            buffer = _slice_batch(buffer, n, buffer_size(buffer))

            print(f"Evaluating test batch {total + 1}-{total + n} of ~{n_test_ids}")
            pol, val, tot = evaluator.compute_losses_batched(
                front["input_arrays"],
                front["values"],
                front["action_masks"],
                front["mcts_policies"],
            )
            total_pol += pol.item() * n
            total_val += val.item() * n
            total_tot += tot.item() * n

            # Accuracy: forward pass on the same inputs, mask logits, argmax,
            # compare against the (uniform-over-best) mcts_policy. A pick
            # whose mcts_policy entry equals the maximum is "correct".
            inputs = torch.from_numpy(front["input_arrays"]).to(evaluator.device)
            pred_logits, _ = evaluator.network(inputs)
            masks_t = torch.from_numpy(front["action_masks"]).to(evaluator.device)
            if evaluator.config.mask_training_predictions:
                pred_logits = pred_logits * masks_t + INVALID_ACTION_VALUE * (1 - masks_t)
            picks = pred_logits.argmax(dim=1).cpu().numpy()
            policies = front["mcts_policies"]
            best_probs = policies.max(axis=1)
            for i, pick in enumerate(picks):
                if policies[i, pick] == best_probs[i]:
                    correct += 1
                total += 1

            Timer.log_totals()
    evaluator.network.train()

    assert total > 0, "No test samples found (check test_player filter?)"
    return total_pol / total, total_val / total, total_tot / total, correct / total


# ---------------------------------------------------------------------------
# DB source resolution
# ---------------------------------------------------------------------------


def resolve_db_path(db_path: str) -> str:
    """If `db_path` is `wandb:<entity>/<project>/<name>:<alias>`, fetch the
    artifact and return the local path to the `.parquet` file inside it.
    Otherwise return `db_path` unchanged. wandb's API caches downloads in
    `~/.cache/wandb/` so repeat invocations are local-disk cheap.
    """
    if not db_path.startswith("wandb:"):
        return db_path
    artifact_ref = db_path[len("wandb:"):]
    print(f"Fetching wandb artifact: {artifact_ref}")
    api = wandb.Api()
    artifact = api.artifact(artifact_ref, type="policy_db")
    download_dir = artifact.download()
    parquet_files = list(Path(download_dir).glob("*.parquet"))
    if len(parquet_files) != 1:
        raise RuntimeError(
            f"Expected exactly one .parquet in artifact {artifact_ref}, "
            f"found {parquet_files}"
        )
    print(f"Using downloaded DB: {parquet_files[0]}")
    return str(parquet_files[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MAX_TEST_SIZE = 10000


def parse_args():
    p = argparse.ArgumentParser(description="Train a neural network evaluator from a policy DB.")
    p.add_argument(
        "db_path",
        help=(
            "Path to a .parquet policy DB, or "
            "'wandb:<entity>/<project>/<name>:<alias>' to fetch from a wandb artifact"
        ),
    )
    p.add_argument(
        "-p",
        "--params",
        type=str,
        default="",
        help="AlphaZero params in subargs form (e.g. nn_type=mlp,learning_rate=0.001)",
    )
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument(
        "--test-batch-size",
        type=int,
        default=256,
        help="Per-batch sample count for test evaluation (only the last batch may be smaller)",
    )
    p.add_argument("--num-steps", type=int, default=10000)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--output", default="evaluator.pt")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument(
        "--test-player",
        type=int,
        default=None,
        choices=[1, 2],
        help="Only evaluate test loss and accuracy on positions where this player is to move (1 or 2)",
    )
    p.add_argument(
        "--exclude-test-set",
        action="store_true",
        default=False,
        help="Exclude test set IDs from training batches (default: allow overlap)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable verbose debug output",
    )
    p.add_argument(
        "-w",
        "--wandb",
        nargs="?",
        const="",
        default=None,
        type=str,
        help="Enable wandb logging. Optionally pass project name (default: policydb_evaluator)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    global DEBUG
    DEBUG = args.debug

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    az_params = parse_subargs(args.params, AlphaZeroParams)
    nn_config = NNConfig.from_alphazero_params(az_params)
    print(f"AlphaZero params: {az_params}")

    # ------------------------------------------------------------------
    # Wandb
    # ------------------------------------------------------------------
    use_wandb = args.wandb is not None
    if use_wandb:
        wandb_project = args.wandb if args.wandb else "policydb_evaluator"
        wandb_run = wandb.init(
            project=wandb_project,
            config={
                "db_path": os.path.basename(args.db_path),
                "num_steps": args.num_steps,
                "test_fraction": args.test_fraction,
                "test_batch_size": args.test_batch_size,
                "test_player": args.test_player,
                "exclude_test_set": args.exclude_test_set,
                "output": args.output,
                **asdict(az_params),
            },
        )
        Timer.set_wandb_run(wandb_run)
        wandb.define_metric("step", hidden=True)
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("test/*", step_metric="step")

    # ------------------------------------------------------------------
    # Open DB and read metadata
    # ------------------------------------------------------------------
    db_path = resolve_db_path(args.db_path)
    db = quoridor_rs.PyPolicyDb(db_path, lazy=False)
    board_size, max_walls, max_steps, num_states = db.read_metadata()
    print(f"Board size: {board_size}, max_walls: {max_walls}, max_steps: {max_steps}")

    if num_states is None:
        # Fallback for DBs created before autoincrement ID was added.
        num_states = db.count_states()
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
    print(f"Train size: ~{num_states - test_size}, test size: {len(test_ids)}, test batch size: {args.test_batch_size}")

    # ------------------------------------------------------------------
    # Create NNEvaluator and set up optimizer
    # ------------------------------------------------------------------
    action_encoder = ActionEncoder(board_size)
    evaluator = NNEvaluator(action_encoder, device, nn_config, max_cache_size=100000)
    evaluator.train_prepare(az_params.learning_rate, az_params.batch_size, args.num_steps, az_params.weight_decay)

    # ------------------------------------------------------------------
    # Probe one sample for feature_dim
    # ------------------------------------------------------------------
    nn_type = nn_config.type
    probe = fetch_batch(db, [test_ids[0]], nn_type)
    feature_dim = probe["input_arrays"].shape[1:]
    print(f"NN type: {nn_type}, feature shape per sample: {feature_dim}")

    if use_wandb:
        wandb.config.update(
            {
                "board_size": board_size,
                "max_walls": max_walls,
                "max_steps": max_steps,
                "num_states": num_states,
                "train_size": num_states - test_size,
                "test_size": len(test_ids),
                "feature_shape": list(feature_dim),
            }
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_test_loss = float("inf")
    batch_size = az_params.batch_size

    learning_rate = az_params.learning_rate
    if use_wandb:
        wandb.log(
            {
                "learning_rate": learning_rate,
                "step": 0,
            }
        )

    for step in range(1, args.num_steps + 1):
        if step % 100000 == 0:
            learning_rate = learning_rate / 2.0
            print(f"Lowering learning rate to {learning_rate}")
            evaluator.train_prepare(learning_rate, az_params.batch_size, args.num_steps, az_params.weight_decay)
            if use_wandb:
                wandb.log(
                    {
                        "learning_rate": learning_rate,
                        "step": step,
                    }
                )

        # Oversample to account for player filtering and states dropped by
        # build_policy_from_children (missing children in DB).
        oversample = 4 if test_player is None else 8
        batch_ids = random.sample(range(1, num_states + 1), min(batch_size * oversample, num_states))
        if args.exclude_test_set:
            batch_ids = [i for i in batch_ids if i not in test_id_set]
        batch = fetch_batch(db, batch_ids, nn_type)
        batch = _filter_by_player(batch, test_player)
        batch = _slice_batch(batch, 0, batch_size)
        train_policy_loss, train_value_loss, train_total_loss = evaluator.train_iteration_batched(
            batch["input_arrays"],
            batch["values"],
            batch["action_masks"],
            batch["mcts_policies"],
        )

        if use_wandb:
            wandb.log(
                {
                    "train/value_loss": train_value_loss,
                    "train/policy_loss": train_policy_loss,
                    "train/total_loss": train_total_loss,
                    "step": step,
                }
            )

        if step % args.log_interval == 0 or step == 1:
            test_policy_loss, test_value_loss, test_total_loss, acc = compute_test_metrics_batched(
                test_ids,
                db,
                evaluator,
                args.test_batch_size,
                nn_type=nn_type,
                test_player=test_player,
            )

            print(
                f"step {step:6d} | "
                f"train: pol={train_policy_loss:.4f} val={train_value_loss:.4f} tot={train_total_loss:.4f} | "
                f"test: pol={test_policy_loss:.4f} val={test_value_loss:.4f} tot={test_total_loss:.4f} | "
                f"accuracy {acc:.3f}"
            )
            sys.stdout.flush()

            if use_wandb:
                wandb.log(
                    {
                        "test/value_loss": test_value_loss,
                        "test/policy_loss": test_policy_loss,
                        "test/total_loss": test_total_loss,
                        "test/accuracy": acc,
                        "step": step,
                    }
                )

            if test_total_loss < best_test_loss:
                best_test_loss = test_total_loss
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

        Timer.log_cumulative("step", step)

    print(f"Training complete. Best test loss: {best_test_loss:.4f}. Model saved to {args.output}")

    if use_wandb:
        wandb.summary["best_test_loss"] = best_test_loss
        wandb.finish()


if __name__ == "__main__":
    main()
