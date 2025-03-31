import curses

from agents import Agent
from arena import GameResult

from renderers import Renderer


class CursesBoardRenderer(Renderer):
    def __init__(self, napms=100):
        super().__init__()
        self.napms = napms

    def start_arena(self, game, total_games: int):
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.start_color()

    def end_arena(self, game, results: list[GameResult]):
        curses.nocbreak()
        curses.echo()
        curses.endwin()

    def start_game(self, game, agent1: Agent, agent2: Agent):
        self.match = f"{agent1.name()} vs {agent2.name()}"

    def after_action(self, game, step, agent, action):
        board = game.render()
        self.stdscr.erase()
        self.stdscr.addstr(0, 0, f"Game: {self.match} - Step {step + 1}: {agent} takes action {action}")
        self.stdscr.addstr(2, 0, board)
        self.stdscr.refresh()
        curses.napms(self.napms)
