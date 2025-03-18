from arena import ArenaPlugin, Agent
from typing import Optional
import curses


class ResultsRenderer(ArenaPlugin):
    def start_game(self, game, agent1: Optional[Agent] = None, agent2: Optional[Agent] = None):
        self.match = f"{agent1.__class__.__name__} vs {agent2.__class__.__name__}"

    def end_game(self, game, step):
        print(f"Game over! {self.match}: {game.winner()} won in {step} steps")


class TextRenderer(ArenaPlugin):
    def start_game(self, game, agent1: Optional[Agent] = None, agent2: Optional[Agent] = None):
        print("Initial Board State:")
        print(game.render())

    def end_game(self, game, step):
        print(f"\nGame Over! {game.winner()} wins after {step} steps.")

    def action(self, game, step, agent, action):
        print(f"\nStep {step + 1}: {agent} takes action {action}")
        print(f"Rewards: {game.rewards}")
        print(game.render())


class CursesRenderer(ArenaPlugin):
    def __init__(self, napms=100):
        super().__init__()
        self.napms = napms

    def start_arena(self, game):
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.start_color()

    def end_arena(self, game):
        curses.nocbreak()
        curses.echo()
        curses.endwin()

    def action(self, game, step, agent, action):
        board = game.render()
        self.stdscr.clear()
        self.stdscr.addstr(2, 0, board)
        self.stdscr.refresh()
        curses.napms(self.napms)
