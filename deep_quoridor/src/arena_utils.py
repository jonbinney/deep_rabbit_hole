from dataclasses import dataclass

from agents import Agent


@dataclass
class GameResult:
    player1: str
    player2: str
    winner: str
    steps: int
    time_ms: int
    game_id: str


class ArenaPlugin:
    """
    Base class for all arena plugins.
    The plug in can override any combinantion of the methods below in order to provide additional functionality.
    """

    def start_game(self, game, agent1: Agent, agent2: Agent):
        pass

    def end_game(self, game, result: GameResult):
        pass

    def start_arena(self, game, total_games: int):
        pass

    def end_arena(self, game, results: list[GameResult]):
        pass

    def action(self, game, step, agent, action):
        pass


class CompositeArenaPlugin:
    """
    Allows to combine multiple plugins into a single one, calling them sequentially for each method.

    """

    def __init__(self, plugins: list[ArenaPlugin]):
        """
        For the sake of convenience, the plugin list is allowed to be empty, in which case the plugin will be a no-op.
        """
        self.plugins = plugins

    def start_game(self, game, agent1: Agent, agent2: Agent):
        [plugin.start_game(game, agent1, agent2) for plugin in self.plugins]

    def end_game(self, game, result: GameResult):
        [plugin.end_game(game, result) for plugin in self.plugins]

    def start_arena(self, game, total_games: int):
        [plugin.start_arena(game, total_games) for plugin in self.plugins]

    def end_arena(self, game, results: list[GameResult]):
        [plugin.end_arena(game, results) for plugin in self.plugins]

    def action(self, game, step, agent, action):
        [plugin.action(game, step, agent, action) for plugin in self.plugins]
