import time

from agents import Agent
from arena import GameResult

from renderers import Renderer


class MatchResultsRenderer(Renderer):
    def start_game(self, game, agent1: Agent, agent2: Agent):
        self.match = f"{agent1.name()} vs {agent2.name()}"
        self.init_time = time.time()

    def end_game(self, game, result: GameResult):
        total_time = round((time.time() - self.init_time) * 1000)
        print(
            f"{result.game_id}: {self.match} - {result.winner} won in {result.steps:3d} steps,  time: {total_time:5d}ms"
        )
