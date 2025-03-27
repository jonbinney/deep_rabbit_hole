from agents import Agent
from arena import GameResult

from renderers import Renderer


class MatchResultsRenderer(Renderer):
    def start_game(self, game, agent1: Agent, agent2: Agent):
        self.match = f"{agent1.name()} vs {agent2.name()}"

    def end_game(self, game, result: GameResult):
        print(f"{result.game_id}: {self.match} - {result.winner} won in {result.steps} steps")
