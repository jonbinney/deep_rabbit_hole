"""Standalone CLI script: runs deterministic MCTS game and prints step trace.

Usage:
  python mcts_game_reference.py <src_dir> <board_size> <max_walls> <max_steps> <mcts_n>

Trace format per step:

  G,<step>,<grid_hex>
  P,<step>,<p0r>,<p0c>,<p1r>,<p1c>
  W,<step>,<w0>,<w1>
  C,<step>,<current_player>
  M,<step>,<bitmask>
  T,<step>,<tensor_hex>
  RM,<step>,<bitmask>           (only when current player is Player.TWO)
  RT,<step>,<tensor_hex>        (only when current player is Player.TWO)
  V,<step>,<value_hex>          (root value from MCTS, float32 bytes as hex)
  Q,<step>,<policy_hex>         (full policy vector, float32 bytes as hex)
  A,<step>,<action_idx>         (selected action index in rotated/current-player frame)

A final snapshot is emitted after the last move, without V/Q/A tags.
"""

import sys
from pathlib import Path

import numpy as np
import torch

src_dir = Path(sys.argv[1])
board_size = int(sys.argv[2])
max_walls = int(sys.argv[3])
max_steps = int(sys.argv[4])
mcts_n = int(sys.argv[5])

sys.path.insert(0, str(src_dir))

from agents.alphazero.mcts import MCTS
from agents.alphazero.resnet_network import ResnetConfig, ResnetNetwork
from quoridor import ActionEncoder, Board, Player, Quoridor


class UniformMockNNEvaluator:
    """Deterministic evaluator: value 0 and uniform priors over valid actions."""

    def evaluate_batch(self, games: list[Quoridor]):
        values = []
        priors = []
        for game in games:
            mask = np.asarray(game.get_action_mask(), dtype=np.float32)
            total = float(mask.sum())
            if total > 0:
                prior = mask / total
            else:
                prior = np.zeros_like(mask, dtype=np.float32)
            values.append(np.float32(0.0))
            priors.append(prior.astype(np.float32))
        return values, priors


def grid_to_hex(grid):
    return grid.astype("int8").tobytes().hex()


def tensor_to_hex(arr):
    return arr.astype("float32").tobytes().hex()


def float32_to_hex(value):
    return np.asarray([value], dtype=np.float32).tobytes().hex()


def emit_snapshot(game, step, net):
    grid = game.board._grid
    p0 = game.board.get_player_position(Player.ONE)
    p1 = game.board.get_player_position(Player.TWO)
    w0 = int(game.board._walls_remaining[Player.ONE])
    w1 = int(game.board._walls_remaining[Player.TWO])
    cp = int(game.get_current_player())

    print(f"G,{step},{grid_to_hex(grid)}")
    print(f"P,{step},{int(p0[0])},{int(p0[1])},{int(p1[0])},{int(p1[1])}")
    print(f"W,{step},{w0},{w1}")
    print(f"C,{step},{cp}")

    mask = game.get_action_mask()
    mask_str = "".join("1" if x else "0" for x in mask)
    print(f"M,{step},{mask_str}")

    tensor = net.game_to_input_array(game)
    print(f"T,{step},{tensor_to_hex(tensor)}")

    if cp == 1:
        rotated = game.create_new()
        rotated.rotate_board()
        rmask = rotated.get_action_mask()
        rmask_str = "".join("1" if x else "0" for x in rmask)
        print(f"RM,{step},{rmask_str}")
        rtensor = net.game_to_input_array(rotated)
        print(f"RT,{step},{tensor_to_hex(rtensor)}")


encoder = ActionEncoder(board_size)
dummy_device = torch.device("cpu")
net = ResnetNetwork(encoder, dummy_device, ResnetConfig(num_blocks=1, num_channels=1))

evaluator = UniformMockNNEvaluator()
mcts = MCTS(
    n=mcts_n,
    k=None,
    ucb_c=1.4,
    noise_epsilon=0.0,
    noise_alpha=1.0,
    max_steps=max_steps,
    evaluator=evaluator,
    visited_states=set(),
)

game = Quoridor(Board(board_size, max_walls))
step = 0

while True:
    emit_snapshot(game, step, net)

    if game.is_game_over() or (max_steps >= 0 and game.completed_steps >= max_steps):
        break

    root_children, root_value = mcts.search(game)
    visit_counts = np.array([child.visit_count for child in root_children], dtype=np.int64)
    total_visits = int(visit_counts.sum())
    if total_visits <= 0:
        raise RuntimeError("No nodes visited during MCTS")

    policy = np.zeros(encoder.num_actions, dtype=np.float32)
    for child in root_children:
        action_index = encoder.action_to_index(child.action_taken)
        policy[action_index] = np.float32(child.visit_count / total_visits)

    max_visits = int(visit_counts.max())
    best_child = next(child for child in root_children if child.visit_count == max_visits)
    action_idx = encoder.action_to_index(best_child.action_taken)

    print(f"V,{step},{float32_to_hex(np.float32(root_value))}")
    print(f"Q,{step},{policy.astype(np.float32).tobytes().hex()}")
    print(f"A,{step},{action_idx}")

    game.step(best_child.action_taken)
    step += 1
