import argparse
import random
from glob import glob
from pathlib import Path
from typing import Optional

import gymnasium.utils.seeding
import numpy as np
import torch
import yaml


def resolve_path(dir: str, filename: Optional[str] = None) -> Path:
    path = Path(dir)
    if not path.is_absolute():
        # Update this if this file is moved
        path = Path(__file__).resolve().parents[3] / path

    return path / filename if filename else path


def set_deterministic(seed=42):
    """Sets all random seeds for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gymnasium.utils.seeding.np_random(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def my_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def yargs(parser: argparse.ArgumentParser, default_path: str):
    """
    This function enhances the command line argument parsing by allowing arguments to be
    specified in YAML configuration files. It supports both single file and directory inputs,
    where multiple YAML configurations can be defined.

    Args:
        parser (argparse.ArgumentParser): The argument parser to augment with YAML support
        default_path (str): The default path to look for YAML configuration files

    Returns:
        dict: A dictionary mapping configuration IDs to their parsed arguments. If no YAML
            configuration is used, returns a dictionary with an empty string key and the
            parsed command line arguments as value.

    The function adds two optional arguments to the parser:
        -yp/--yargs_path: Path to the YAML configuration file or directory (if none, the default_path will be used)
        -yi/--yargs_ids: List of configuration IDs to run (if None, runs all configs)

    Example yaml file:
        benchmark_B5W3:
            args: -t 100 -N 5 -W 3 --players greedy greedy:p_random=0.1,nick=greedy-ish
        benchmark_B9W10:
            args: -t 100 -N 9 -W 10 --players greedy greedy:p_random=0.1,nick=greedy-ish

    Example calls from the command line:
        # Execute all the entries in all the yaml files in default_path
        python myapp.py -yp

        # Execute all the entries in the file {default_path}/agent_benchmark.yaml
        python myapp.py -yp agent_benchmark.yaml

        # Execute the entries benchmark_B5W3 benchmark_B9W10 (should be found in the files in the default_path)
        python myapp.py -yi benchmark_B5W3 benchmark_B9W10

    """
    parser.add_argument(
        "-yp",
        "--yargs-path",
        nargs="?",
        default=None,
        const="",
        help=f"File name to read the yaml configuration, relative to {default_path}",
    )
    parser.add_argument(
        "-yi",
        "--yargs-ids",
        nargs="+",
        default=None,
        help="List of ids to run",
    )

    args = parser.parse_args()
    if args.yargs_path is None and args.yargs_ids is None:
        # Not using yargs
        return {"": args}

    if args.yargs_path is None:
        path = Path(default_path)
    else:
        path = Path(default_path) / args.yargs_path

    if path.is_dir():
        files = glob(str(path / "*.yaml"))
    else:
        files = [str(path)]

    args_dict = {}
    for file in files:
        with open(file, "r") as f:
            file_dict = yaml.load(f, Loader=yaml.FullLoader)
            for k, v in file_dict.items():
                if args.yargs_ids is None or k in args.yargs_ids:
                    args_dict[k] = parser.parse_args(v["args"].split(" "))

    return args_dict


def compute_elo(results: list["GameResult"], k=32, initial_rating=1500, initial_elos={}) -> dict[str, float]:  # type: ignore # noqa: F821
    ratings = initial_elos
    random.shuffle(results)

    for result in results:
        if result.player1 not in ratings:
            ratings[result.player1] = initial_rating

        if result.player2 not in ratings:
            ratings[result.player2] = initial_rating

        delta = (ratings[result.player1] - ratings[result.player2]) / 400.0
        e1 = 1 / (1 + 10**-delta)
        e2 = 1 / (1 + 10**delta)

        # If the game was truncated because it was too long, they both will count as losing
        # so their rating decreases
        p1_won = 1 if result.winner == result.player1 else 0
        p2_won = 1 if result.winner == result.player2 else 0

        ratings[result.player1] += k * (p1_won - e1)
        ratings[result.player2] += k * (p2_won - e2)

    return ratings


def get_opponent_player_id(player_id: str) -> str:
    """
    Returns the opponent player id given the player id.
    """
    if player_id == "player_1":
        return "player_0"
    elif player_id == "player_0":
        return "player_1"
    else:
        raise ValueError(f"Invalid player id: {player_id}")


def cnn_output_size_per_channel(input_size, padding, kernel_size=3, stride=1):
    return (input_size + 2 * padding - kernel_size) // stride + 1
