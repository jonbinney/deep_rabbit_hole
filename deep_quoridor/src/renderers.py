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


class TextRenderer(ResultsRenderer):
    def start_game(self, game, agent1: Agent, agent2: Agent):
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
