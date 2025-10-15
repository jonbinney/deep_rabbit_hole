import argparse

from agents import AgentRegistry
from metrics import Metrics
from prettytable import PrettyTable


def player_with_params(arg):
    if not AgentRegistry.is_valid_encoded_name(arg):
        raise argparse.ArgumentTypeError(f"Invalid player name: {arg}")
    return arg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Quoridor - Run Metrics")
    parser.add_argument("-N", "--board_size", type=int, default=9, help="Board Size")
    parser.add_argument("-W", "--max_walls", type=int, default=10, help="Max walls per player")
    parser.add_argument(
        "-p",
        "--players",
        nargs="+",
        type=player_with_params,
        required=True,
        help="List of players to run the metrics.  The metrics for each player are computed independently",
    )
    parser.add_argument(
        "-b",
        "--benchmarks",
        nargs="+",
        type=str,
        default=["random", "simple"],
        help=f"List of players to benchmark against. Can include parameters in parentheses. Allowed types {AgentRegistry.names()}",
    )
    parser.add_argument(
        "-bt",
        "--benchmarks_t",
        type=int,
        default=10,
        help="How many time to play against each opponent during benchmarks",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max number of turns before game is called a tie (pass -1 for no limit)",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes")

    args = parser.parse_args()
    m = Metrics(args.board_size, args.max_walls, args.benchmarks, args.benchmarks_t, args.max_steps, args.num_workers)
    table = PrettyTable()
    table.field_names = ["Agent", "Elo", "Relative Elo", "Win %", "Dumb Score"]

    for player in args.players:
        print(f"Computing metrics for {player}")
        _, _, relative_elo, win_perc, p1_win_percentages, p2_win_percentages, absolute_elo, dumb_score = m.compute(
            player
        )
        table.add_row([player, absolute_elo, relative_elo, win_perc, dumb_score])

    print(table)
