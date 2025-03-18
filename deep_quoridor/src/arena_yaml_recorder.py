from arena import ArenaPlugin, Agent
from typing import Optional
import time
import yaml


class ArenaYAMLRecorder(ArenaPlugin):
    """
    Saves metadata about the game and all the actions in a YAML file
    """

    def __init__(self, filename: str):
        self.data = {}
        self.game_n = 0
        self.start_time = None
        self.filename = filename

    def start_game(self, game, agent1: Optional[Agent] = None, agent2: Optional[Agent] = None):
        self.actions = []
        self.start_time = time.time()
        self.player1 = agent1.__class__.__name__
        self.player2 = agent2.__class__.__name__
        self.game_n += 1

    def action(self, game, step, agent, action):
        self.actions.append(action)

    def end_game(self, game, steps):
        assert self.start_time, "Tried to call end_game before calling start_game"
        self.data[f"game{self.game_n}"] = {
            "player1": self.player1,
            "player2": self.player2,
            "winner": game.winner(),
            "steps": steps,
            "time_ms": int((time.time() - self.start_time) * 1000),
            "actions": self.actions,
        }

    def end_arena(self, game):
        with open(self.filename, "w") as file:
            file.write(yaml.dump(self.data, sort_keys=False))
