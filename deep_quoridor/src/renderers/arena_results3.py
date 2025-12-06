from collections import defaultdict

from arena import GameResult
from prettytable import PrettyTable

from renderers import Renderer


class ArenaResults3Renderer(Renderer):
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

        games = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})

        for r in results:
            matchup = (r.player1, r.player2)
            if r.winner == r.player1:
                games[matchup]["wins"] += 1
            elif r.winner == r.player2:
                games[matchup]["losses"] += 1
            elif r.winner == "tie":
                games[matchup]["ties"] += 1
            else:
                raise ValueError(f"Invalid winner in result: {r.winner}")

        table = PrettyTable()
        table.field_names = ["Matchup", "Wins", "Losses", "Ties"]

        for player1 in players:
            table.add_row([f"P1 {player1}", "", "", ""])

            for player2 in players:
                matchup = (player1, player2)
                if matchup not in games:
                    continue

                num_games = sum([n for n in games[matchup].values()])
                win_perc = perc(games[matchup]["wins"], num_games)
                loss_perc = perc(games[matchup]["losses"], num_games)
                tie_perc = perc(games[matchup]["ties"], num_games)

                table.add_row([f"  vs {player2}", win_perc, loss_perc, tie_perc])

            table.add_row(["", "", "", ""])

        print(table)
        return
