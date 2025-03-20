from arena import ArenaPlugin, Agent, GameResult
import curses
from prettytable import PrettyTable
from collections import defaultdict
import numpy as np


class Renderer(ArenaPlugin):
    """
    Base class for all renderers, which take a game and render it in some way.
    """

    renderers = {}

    def __init_subclass__(cls, **kwargs):
        friendly_name = cls.__name__.replace("Renderer", "").lower()
        Renderer.renderers[friendly_name] = cls

    @staticmethod
    def create(friendly_name: str):
        return Renderer.renderers[friendly_name]()

    @staticmethod
    def names():
        return list(Renderer.renderers.keys())


class NoneRenderer(Renderer):
    pass


class ResultsRenderer(Renderer):
    def start_game(self, game, agent1: Agent, agent2: Agent):
        self.match = f"{agent1.name()} vs {agent2.name()}"

    def end_game(self, game, result: GameResult):
        print(f"{result.game_id}: {self.match} - {result.winner} won in {result.steps} steps")

    def end_arena(self, game, results: list[GameResult]):
        def perc(wins: int, played: int) -> str:
            if played == 0:
                return "-"

            return f"{wins / played * 100.0:.0f}%"

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


class TextRenderer(ResultsRenderer):
    def start_game(self, game, agent1: Agent, agent2: Agent):
        super().start_game(game, agent1, agent2)
        print("Initial Board State:")
        print(game.render())

    def action(self, game, step, agent, action):
        print(f"\nStep {step + 1}: {agent} takes action {action}")
        print(f"Rewards: {game.rewards}")
        print(game.render())


class CursesRenderer(Renderer):
    def __init__(self, napms=100):
        super().__init__()
        self.napms = napms

    def start_arena(self, game):
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.start_color()

    def end_arena(self, game, results: list[GameResult]):
        curses.nocbreak()
        curses.echo()
        curses.endwin()

    def action(self, game, step, agent, action):
        board = game.render()
        self.stdscr.clear()
        self.stdscr.addstr(2, 0, board)
        self.stdscr.refresh()
        curses.napms(self.napms)
