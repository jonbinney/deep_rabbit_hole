"""Deterministic real-model self-play reference trace generator.

This script reuses the Python AlphaZero self-play codepath with a real `.pt`
model, emits per-step traces, and writes one replay game (`.npz` + `.yaml`).

Usage:
    python selfplay_real_model_reference.py \
        <src_dir> <board_size> <max_walls> <max_steps> <mcts_n> <pt_model> <output_dir> \
        [deterministic_tie_break:0|1]

Trace format:
    CFG,<board_size>,<max_walls>,<max_steps>,<mcts_n>
    G,<step>,<grid_hex>
    P,<step>,<p0r>,<p0c>,<p1r>,<p1c>
    W,<step>,<w0>,<w1>
    C,<step>,<current_player>
    M,<step>,<bitmask>
    T,<step>,<tensor_hex>
    RM,<step>,<bitmask>
    RT,<step>,<tensor_hex>
    V,<step>,<value_hex>
    Q,<step>,<policy_hex>
    A,<step>,<action_idx>
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

# Keep inference on CPU for reproducible cross-language parity.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

src_dir = Path(".")


def configure_imports(base_src_dir: Path):
    sys.path.insert(0, str(base_src_dir))

    import quoridor_env
    from agents.alphazero import AlphaZeroAgent, AlphaZeroParams
    from quoridor import Player, construct_game_from_observation

    return quoridor_env, AlphaZeroAgent, AlphaZeroParams, Player, construct_game_from_observation


def set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def grid_to_hex(grid):
    return grid.astype("int8").tobytes().hex()


def tensor_to_hex(arr):
    return arr.astype("float32").tobytes().hex()


def float32_to_hex(value):
    return np.asarray([value], dtype=np.float32).tobytes().hex()


def build_policy_vector(root_children, action_encoder):
    visit_counts = np.array([child.visit_count for child in root_children], dtype=np.int64)
    total_visits = int(visit_counts.sum())
    if total_visits <= 0:
        raise RuntimeError("No nodes visited during MCTS")

    policy = np.zeros(action_encoder.num_actions, dtype=np.float32)
    for child in root_children:
        idx = action_encoder.action_to_index(child.action_taken)
        policy[idx] = np.float32(child.visit_count / total_visits)
    return policy


def emit_snapshot(agent, game, step, player_enum):
    grid = game.board.get_grid()
    p0 = game.board.get_player_position(player_enum.ONE)
    p1 = game.board.get_player_position(player_enum.TWO)
    w0 = int(game.board.get_walls_remaining(player_enum.ONE))
    w1 = int(game.board.get_walls_remaining(player_enum.TWO))
    cp = int(game.get_current_player())

    print(f"G,{step},{grid_to_hex(grid)}")
    print(f"P,{step},{int(p0[0])},{int(p0[1])},{int(p1[0])},{int(p1[1])}")
    print(f"W,{step},{w0},{w1}")
    print(f"C,{step},{cp}")

    mask = game.get_action_mask()
    mask_str = "".join("1" if x else "0" for x in mask)
    print(f"M,{step},{mask_str}")

    tensor = agent.evaluator.game_to_input_array(game)
    print(f"T,{step},{tensor_to_hex(tensor)}")

    if cp == 1:
        rotated = game.create_new()
        rotated.rotate_board()
        rmask = rotated.get_action_mask()
        rmask_str = "".join("1" if x else "0" for x in rmask)
        print(f"RM,{step},{rmask_str}")
        rtensor = agent.evaluator.game_to_input_array(rotated)
        print(f"RT,{step},{tensor_to_hex(rtensor)}")


def run_trace_and_write_game(
    board_size: int,
    max_walls: int,
    max_steps: int,
    mcts_n: int,
    pt_model: Path,
    output_dir: Path,
    deterministic_tie_break: bool,
):
    quoridor_env, AlphaZeroAgent, AlphaZeroParams, Player, construct_game_from_observation = configure_imports(src_dir)

    params = AlphaZeroParams(
        training_mode=True,
        model_filename=str(pt_model),
        mcts_n=mcts_n,
        mcts_k=None,
        mcts_ucb_c=1.4,
        mcts_noise_epsilon=0.0,
        mcts_noise_alpha=1.0,
        temperature=0.0,
        drop_t_on_step=0,
        nn_type="resnet",
        deterministic_tie_break=deterministic_tie_break,
    )

    agent = AlphaZeroAgent(board_size=board_size, max_walls=max_walls, max_steps=max_steps, params=params)

    captured = {"children_batch": None, "root_values": None}
    original_search_batch = agent.mcts.search_batch

    def wrapped_search_batch(games):
        children_batch, root_values = original_search_batch(games)
        captured["children_batch"] = children_batch
        captured["root_values"] = root_values
        return children_batch, root_values

    agent.mcts.search_batch = wrapped_search_batch

    env = quoridor_env.env(board_size=board_size, max_walls=max_walls, max_steps=max_steps)
    env.reset()
    agent.start_game_batch([env])

    print(f"CFG,{board_size},{max_walls},{max_steps},{mcts_n}")

    step = 0
    while True:
        observation, _, termination, truncation, _ = env.last()
        game, _, _ = construct_game_from_observation(observation["observation"])

        emit_snapshot(agent, game, step, Player)

        if termination or truncation or game.is_game_over() or (max_steps >= 0 and game.completed_steps >= max_steps):
            break

        action_batch = agent.get_action_batch([(0, observation)])
        if len(action_batch) != 1:
            raise RuntimeError("Expected exactly one action from get_action_batch")
        chosen_idx = int(action_batch[0][1])

        children_batch = captured["children_batch"]
        root_values = captured["root_values"]
        if children_batch is None or root_values is None:
            raise RuntimeError("Expected wrapped MCTS search_batch to capture root outputs")
        if len(children_batch) != 1 or len(root_values) != 1:
            raise RuntimeError("Expected one captured root output for single-game selfplay")

        root_children = children_batch[0]
        root_value = float(root_values[0])
        policy = build_policy_vector(root_children, game.action_encoder)

        print(f"V,{step},{float32_to_hex(np.float32(root_value))}")
        print(f"Q,{step},{policy.astype(np.float32).tobytes().hex()}")
        print(f"A,{step},{chosen_idx}")

        env.step(chosen_idx)
        step += 1

    ready_dir = output_dir / "ready"
    tmp_dir = ready_dir / "tmp"
    ready_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    agent.end_game_batch_and_save_replay_buffers(tmp_dir, ready_dir, model_version=0)


def parse_cli_args(argv: list[str]):
    if len(argv) not in (8, 9):
        raise SystemExit(
            "usage: python selfplay_real_model_reference.py "
            "<src_dir> <board_size> <max_walls> <max_steps> <mcts_n> <pt_model> <output_dir> "
            "[deterministic_tie_break:0|1]"
        )

    deterministic_tie_break = False
    if len(argv) == 9:
        deterministic_tie_break = argv[8].strip().lower() in ("1", "true", "yes", "on")

    return {
        "src_dir": Path(argv[1]),
        "board_size": int(argv[2]),
        "max_walls": int(argv[3]),
        "max_steps": int(argv[4]),
        "mcts_n": int(argv[5]),
        "pt_model": Path(argv[6]),
        "output_dir": Path(argv[7]),
        "deterministic_tie_break": deterministic_tie_break,
    }


def main(argv: list[str]):
    parsed = parse_cli_args(argv)
    global src_dir
    src_dir = parsed["src_dir"]

    pt_model = parsed["pt_model"]
    if not pt_model.exists():
        raise FileNotFoundError(f"PT model fixture not found: {pt_model}")

    set_deterministic(42)

    run_trace_and_write_game(
        board_size=parsed["board_size"],
        max_walls=parsed["max_walls"],
        max_steps=parsed["max_steps"],
        mcts_n=parsed["mcts_n"],
        pt_model=pt_model,
        output_dir=parsed["output_dir"],
        deterministic_tie_break=parsed["deterministic_tie_break"],
    )


if __name__ == "__main__":
    main(sys.argv)
