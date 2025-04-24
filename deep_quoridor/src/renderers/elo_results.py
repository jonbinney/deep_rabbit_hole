from arena import GameResult
from prettytable import PrettyTable
from utils import compute_elo

from renderers import Renderer


class EloResultsRenderer(Renderer):
    def end_arena(self, game, results: list[GameResult]):
        ratings = compute_elo(results)
        table = PrettyTable()
        table.field_names = ["Player", "Elo Rating"]

        sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
        for player, rating in sorted_ratings:
            table.add_row([player, f"{int(rating)}"])

        print(table)
