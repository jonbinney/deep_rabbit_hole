"""Standalone CLI script: emits deterministic MCTS traces and explains trace files.

Usage:
    python mcts_game_reference.py <src_dir> <board_size> <max_walls> <max_steps> <mcts_n>
    python mcts_game_reference.py --explain-trace <src_dir> <trace_path>

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

import sys
from pathlib import Path

import numpy as np
import torch


src_dir = Path(".")


def configure_imports(base_src_dir: Path):
    sys.path.insert(0, str(base_src_dir))

    from agents.alphazero.mcts import MCTS
    from agents.alphazero.resnet_network import ResnetConfig, ResnetNetwork
    from quoridor import ActionEncoder, Board, Player, Quoridor

    return MCTS, ResnetConfig, ResnetNetwork, ActionEncoder, Board, Player, Quoridor


class UniformMockNNEvaluator:
    """Deterministic evaluator: value 0 and uniform priors over valid actions."""

    def evaluate_batch(self, games):
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


def hex_to_float32(hex_value: str) -> float:
    return float(np.frombuffer(bytes.fromhex(hex_value), dtype=np.float32)[0])


def decode_policy_hex(policy_hex: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(policy_hex), dtype=np.float32)


def format_action(action) -> str:
    return repr(action)


def build_policy_vector(root_children, encoder):
    visit_counts = np.array([child.visit_count for child in root_children], dtype=np.int64)
    total_visits = int(visit_counts.sum())
    if total_visits <= 0:
        raise RuntimeError("No nodes visited during MCTS")

    policy = np.zeros(encoder.num_actions, dtype=np.float32)
    for child in root_children:
        action_index = encoder.action_to_index(child.action_taken)
        policy[action_index] = np.float32(child.visit_count / total_visits)
    return policy, visit_counts, total_visits


def select_best_child(root_children, encoder):
    visit_counts = np.array([child.visit_count for child in root_children], dtype=np.int64)
    max_visits = int(visit_counts.max())
    best_child = next(child for child in root_children if child.visit_count == max_visits)
    action_idx = encoder.action_to_index(best_child.action_taken)
    return best_child, action_idx


def emit_snapshot(game, step, net, player_enum):
    grid = game.board._grid
    p0 = game.board.get_player_position(player_enum.ONE)
    p1 = game.board.get_player_position(player_enum.TWO)
    w0 = int(game.board._walls_remaining[player_enum.ONE])
    w1 = int(game.board._walls_remaining[player_enum.TWO])
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


def create_runner(board_size: int, max_walls: int, max_steps: int, mcts_n: int):
    MCTS, ResnetConfig, ResnetNetwork, ActionEncoder, Board, Player, Quoridor = configure_imports(src_dir)
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
    return {
        "encoder": encoder,
        "net": net,
        "mcts": mcts,
        "game": game,
        "player_enum": Player,
    }


def emit_trace(board_size: int, max_walls: int, max_steps: int, mcts_n: int):
    runner = create_runner(board_size, max_walls, max_steps, mcts_n)
    encoder = runner["encoder"]
    net = runner["net"]
    mcts = runner["mcts"]
    game = runner["game"]
    player_enum = runner["player_enum"]
    step = 0

    print(f"CFG,{board_size},{max_walls},{max_steps},{mcts_n}")

    while True:
        emit_snapshot(game, step, net, player_enum)

        if game.is_game_over() or (max_steps >= 0 and game.completed_steps >= max_steps):
            break

        root_children, root_value = mcts.search(game)
        policy, _, _ = build_policy_vector(root_children, encoder)
        best_child, action_idx = select_best_child(root_children, encoder)

        print(f"V,{step},{float32_to_hex(np.float32(root_value))}")
        print(f"Q,{step},{policy.astype(np.float32).tobytes().hex()}")
        print(f"A,{step},{action_idx}")

        game.step(best_child.action_taken)
        step += 1


def parse_trace_text(trace_text: str):
    metadata = None
    steps = {}

    for raw_line in trace_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(",")
        tag = parts[0]

        if tag == "CFG":
            if len(parts) != 5:
                raise ValueError(f"invalid CFG line: {line}")
            metadata = {
                "board_size": int(parts[1]),
                "max_walls": int(parts[2]),
                "max_steps": int(parts[3]),
                "mcts_n": int(parts[4]),
            }
            continue

        if tag not in {"G", "P", "W", "C", "M", "T", "RM", "RT", "V", "Q", "A"}:
            continue

        if len(parts) < 3:
            continue

        step = int(parts[1])
        payload = ",".join(parts[2:])
        steps.setdefault(step, {})[tag] = payload

    if metadata is None:
        raise ValueError("trace missing CFG line")

    return metadata, steps


def infer_total_visits(policy: np.ndarray, max_expected: int) -> int | None:
    positive = policy[policy > 0]
    if positive.size == 0:
        return None

    search_limit = max(max_expected, 1)
    for total in range(1, search_limit + 1):
        scaled = positive * total
        rounded = np.round(scaled)
        if np.allclose(scaled, rounded, atol=1e-5) and np.all(rounded >= 1):
            return total
    return None


def explain_trace_steps(metadata, steps, trace_path: Path):
    _, _, _, ActionEncoder, _, _, _ = configure_imports(src_dir)
    encoder = ActionEncoder(metadata["board_size"])

    print(f"trace={trace_path}")
    print(
        f"config=B{metadata['board_size']}W{metadata['max_walls']} max_steps={metadata['max_steps']} mcts_n={metadata['mcts_n']}"
    )

    for step in sorted(steps):
        entry = steps[step]
        if "Q" not in entry:
            continue

        current_player = int(entry["C"])
        positions = [int(x) for x in entry["P"].split(",")]
        walls = [int(x) for x in entry["W"].split(",")]
        root_value = hex_to_float32(entry["V"])
        policy = decode_policy_hex(entry["Q"])
        action_index = int(entry["A"])
        inferred_total = infer_total_visits(policy, metadata["mcts_n"])

        print(
            f"step={step} current_player={current_player} p0=({positions[0]},{positions[1]}) "
            f"p1=({positions[2]},{positions[3]}) walls=({walls[0]},{walls[1]}) root_value={root_value:.6f}"
        )
        print(f"selected idx={action_index} action={format_action(encoder.index_to_action(action_index))}")
        if inferred_total is None:
            print("inferred_total_visits=unknown")
        else:
            print(f"inferred_total_visits={inferred_total}")

        explained_children = []
        for idx, probability in enumerate(policy):
            if probability <= 0:
                continue
            if inferred_total is None:
                visits_text = "?"
            else:
                visits_text = str(int(round(float(probability) * inferred_total)))
            explained_children.append(
                (
                    -float(probability),
                    idx,
                    visits_text,
                    float(probability),
                    format_action(encoder.index_to_action(idx)),
                )
            )

        for _, idx, visits_text, probability, action_text in sorted(explained_children)[:12]:
            print(
                f"child idx={idx} visits~={visits_text} prob={probability:.8f} action={action_text}"
            )


def explain_trace_file(trace_path: Path):
    trace_text = trace_path.read_text()
    metadata, steps = parse_trace_text(trace_text)
    explain_trace_steps(metadata, steps, trace_path)


def parse_cli_args(argv: list[str]):
    if len(argv) >= 4 and argv[1] == "--explain-trace":
        return {
            "mode": "explain-trace",
            "src_dir": Path(argv[2]),
            "trace_path": Path(argv[3]),
        }

    if len(argv) >= 6:
        return {
            "mode": "emit-trace",
            "src_dir": Path(argv[1]),
            "board_size": int(argv[2]),
            "max_walls": int(argv[3]),
            "max_steps": int(argv[4]),
            "mcts_n": int(argv[5]),
        }

    raise SystemExit(
        "usage: python mcts_game_reference.py <src_dir> <board_size> <max_walls> <max_steps> <mcts_n> | --explain-trace <src_dir> <trace_path>"
    )


def main(argv: list[str]):
    parsed = parse_cli_args(argv)
    global src_dir
    src_dir = parsed["src_dir"]

    if parsed["mode"] == "explain-trace":
        explain_trace_file(parsed["trace_path"])
        return

    emit_trace(
        parsed["board_size"],
        parsed["max_walls"],
        parsed["max_steps"],
        parsed["mcts_n"],
    )


if __name__ == "__main__":
    main(sys.argv)
