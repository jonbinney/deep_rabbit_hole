from renderers import Renderer
from arena import GameResult
from agents import Agent


class TextBoardRenderer(Renderer):
    def start_game(self, game, agent1: Agent, agent2: Agent):
        self.match = f"{agent1.name()} vs {agent2.name()}"
        print("Initial Board State:")
        print(game.render())

    def end_game(self, game, result: GameResult):
        print(f"{result.game_id}: {self.match} - {result.winner} won in {result.steps} steps")

    def action(self, game, step, agent, action):
        print(f"\nStep {step + 1}: {agent} takes action {action}")
        print(f"Rewards: {game.rewards}")
        print(game.render())
