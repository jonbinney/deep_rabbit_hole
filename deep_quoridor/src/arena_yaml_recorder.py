import yaml

from arena_utils import ArenaPlugin, GameResult
from agents import Agent


class ArenaYAMLRecorder(ArenaPlugin):
    """
    Saves metadata about the game and all the actions in a YAML file
    """

    def __init__(self, filename: str):
        self.games = {}
        self.filename = filename

    def start_arena(self, game, total_games: int):
        self.games = {}

    def start_game(self, game, agent1: Agent, agent2: Agent):
        self.actions = []

    def action(self, game, step, agent, action):
        self.actions.append(action)

    def end_game(self, game, result: GameResult):
        self.games[result.game_id] = {
            "player1": result.player1,
            "player2": result.player2,
            "winner": result.winner,
            "steps": result.steps,
            "time_ms": result.time_ms,
            "actions": self.actions,
        }

    def end_arena(self, game, results: list[GameResult]):
        output = {
            "config": {"board_size": game.board_size, "max_walls": game.max_walls, "step_rewards": game.step_rewards},
            "games": self.games,
        }
        with open(self.filename, "w") as file:
            file.write(yaml.dump(output, sort_keys=False))

    @staticmethod
    def load_recorded_arena_data(filename: str) -> dict:
        with open(filename, "r") as file:
            return yaml.load(file, Loader=yaml.FullLoader)
