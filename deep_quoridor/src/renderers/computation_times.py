from arena import GameResult
from prettytable import PrettyTable

from renderers import Renderer


class ComputationTimesRenderer(Renderer):
    def end_arena(self, game, results: list[GameResult]):
        player_moves = {}
        for result in results:
            for player in [result.player1, result.player2]:
                if player not in player_moves:
                    player_moves[player] = []

                player_moves[player].extend([move for move in result.moves if move.player == player])

        table = PrettyTable()
        table.field_names = ["Player", "Average Time (ms)", "Min Time (ms)", "Max Time (ms)"]

        for player, moves in player_moves.items():
            num_moves = len(moves)
            computation_times_ms = [m.computation_time * 1000.0 for m in moves]
            table.add_row(
                [
                    player,
                    f"{sum(computation_times_ms) / num_moves:.3f}",
                    f"{min(computation_times_ms):.3f}",
                    f"{max(computation_times_ms):.3f}",
                ]
            )

        print(table)
