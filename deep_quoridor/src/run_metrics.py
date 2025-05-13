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
    args = parser.parse_args()
    m = Metrics(args.board_size, args.max_walls)
    table = PrettyTable()
    table.field_names = ["Agent", "Elo", "Relative Elo", "Win %"]

    for player in args.players:
        print(f"Computing metrics for {player}")
        _, _, relative_elo, win_perc, absolute_elo = m.compute(player)
        table.add_row([player, absolute_elo, relative_elo, win_perc])

    print(table)
