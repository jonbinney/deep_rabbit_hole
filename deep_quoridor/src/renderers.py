from arena import ArenaPlugin, Agent, GameResult
from typing import Optional
import curses


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
        print(f"{self.match}: {result.winner} won in {result.steps} steps")

    def end_arena(self, game, results: list[GameResult]):
        won = {}
        won_as_p1 = {}
        for r in results:
            for player in [r.player1, r.player2]:
                if player not in won:
                    won[player] = 0
                    won_as_p1[player] = 0

            if r.winner == r.player1:
                won[r.player1] += 1
                won_as_p1[r.player1] += 1
            else:
                won[r.player2] += 1

        print("\n")
        print("Player          | Wins   | Wins P1 ")
        print("===================================")

        for player, wins in sorted(won.items(), key=lambda x: x[1], reverse=True):
            print(f"{player:15s} | {wins:6d} | {won_as_p1[player]:6d} ")


class TextRenderer(ResultsRenderer):
    def start_game(self, game, agent1: Agent, agent2: Agent):
        print("Initial Board State:")
        print(game.render())

    def end_game(self, game, result: GameResult):
        print(f"\nGame Over! {result.winner} wins after {result.steps} steps.")

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
