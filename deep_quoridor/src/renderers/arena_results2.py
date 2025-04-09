from collections import defaultdict

from prettytable import PrettyTable

from arena import GameResult
from renderers import Renderer


class ArenaResults2Renderer(Renderer):
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

        games = defaultdict(lambda: 0)
        wins = defaultdict(lambda: 0)

        for r in results:
            pair1 = (f"P1 {r.player1}", r.player2)
            pair2 = (f"P2 {r.player2}", r.player1)
            games[pair1] += 1
            games[pair2] += 1
            if r.winner == r.player1:
                wins[pair1] += 1
            else:
                wins[pair2] += 1

        table = PrettyTable()
        table.field_names = ["Player"] + list(players) + ["Total"]

        for player in players:
            for pos_player in [f"P1 {player}", f"P2 {player}"]:
                row_wins = [wins[(pos_player, opponent)] for opponent in players]
                row_games = [games[(pos_player, opponent)] for opponent in players]
                row_wins.append(sum(row_wins))
                row_games.append(sum(row_games))

                table.add_row([pos_player] + [perc(w, g) for w, g in zip(row_wins, row_games)])

        print(table)
        return
