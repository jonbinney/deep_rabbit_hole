import numpy as np
from arena import GameResult
from prettytable import PrettyTable

from renderers import Renderer


class ArenaResultsRenderer(Renderer):
    def end_arena(self, game, results: list[GameResult]):
        def perc(wins: int, played: int) -> str:
            if played == 0:
                return "-"

            return f"{wins / played * 100.0:.0f}%"

        print("Arena stats")
        print("===========")
        print("Board size: ", game.board_size)
        print("Max walls: ", game.max_walls)
        print("Step rewards: ", game.step_rewards)
        print("Total number of games: ", len(results), "\n")

        all_players = sorted(set([r.player1 for r in results]) | set([r.player2 for r in results]))
        players = {player: i for i, player in enumerate(all_players)}

        table = PrettyTable()
        table.field_names = ["P1 \\ P2"] + list(players) + ["Total"]

        N = len(players)
        wins = np.zeros((N + 1, N + 1))
        games = np.zeros((N + 1, N + 1))

        for r in results:
            games[players[r.player1], players[r.player2]] += 1
            if r.winner == r.player1:
                wins[players[r.player1], players[r.player2]] += 1

        # Get the totals per row and column
        wins[-1, :] = np.sum(wins, axis=0)
        wins[:, -1] = np.sum(wins, axis=1)
        games[-1, :] = np.sum(games, axis=0)
        games[:, -1] = np.sum(games, axis=1)

        # Hacky way of adding an extra column and row for the totals
        players["Total"] = max(players.values()) + 1

        # Set the results in the PrettyTable
        for player1, i1 in players.items():
            row = [player1]
            for _, i2 in players.items():
                row.append(perc(wins[i1, i2], games[i1, i2]))

            table.add_row(row)

            # Before the last row add this separation, since the last row is the total
            if i1 == len(players) - 2:
                table.add_row(["======" for _ in range(len(players) + 1)])

        print(table)
