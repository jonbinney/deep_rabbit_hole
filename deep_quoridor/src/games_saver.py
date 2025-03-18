import yaml
import time


class GamesSaver:
    def __init__(self, filename: str):
        self.data = {}
        self.game_n = 0
        self.start_time = None
        self.filename = filename

    def start_game(self, player1: str, player2: str):
        self.actions = []
        self.start_time = time.time()
        self.player1 = player1
        self.player2 = player2
        self.game_n += 1

    def add_action(self, action: int):
        assert self.start_time, "Tried to call add_action before calling start_game"
        self.actions.append(action)

    def end_game(self, winner: str):
        assert self.start_time, "Tried to call end_game before calling start_game"
        self.data[f"game{self.game_n}"] = {
            "player1": self.player1,
            "player2": self.player2,
            "winner": winner,
            "time_ms": int((time.time() - self.start_time) * 1000),
            "actions": self.actions,
        }

    def save(self):
        with open(self.filename, "w") as file:
            file.write(yaml.dump(self.data, sort_keys=False))
