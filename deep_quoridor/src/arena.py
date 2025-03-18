from typing import Optional
from quoridor_env import env
from agents import Agent


class ArenaPlugin:
    """
    Base class for all arena plugins.
    The plug in can override any combinantion of the methods below in order to provide additional functionality.
    """

    def start_game(self, game, agent1: Optional[Agent] = None, agent2: Optional[Agent] = None):
        pass

    def end_game(self, game, step):
        pass

    def start_arena(self, game):
        pass

    def end_arena(self, game):
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

    def start_game(self, game, agent1: Optional[Agent] = None, agent2: Optional[Agent] = None):
        [plugin.start_game(game, agent1, agent2) for plugin in self.plugins]

    def end_game(self, game, step):
        [plugin.end_game(game, step) for plugin in self.plugins]

    def start_arena(self, game):
        [plugin.start_arena(game) for plugin in self.plugins]

    def end_arena(self, game):
        [plugin.end_arena(game) for plugin in self.plugins]

    def action(self, game, step, agent, action):
        [plugin.action(game, step, agent, action) for plugin in self.plugins]


class Arena:
    def __init__(
        self,
        board_size: int = 9,
        max_walls: int = 10,
        step_rewards: bool = False,
        renderer: Optional[ArenaPlugin] = None,
        saver: Optional[ArenaPlugin] = None,
    ):
        self.board_size = board_size
        self.max_walls = max_walls
        self.step_rewards = step_rewards
        self.game = env(board_size=board_size, max_walls=max_walls, step_rewards=step_rewards)

        plugins = [p for p in [renderer, saver] if p is not None]
        self.plugins = CompositeArenaPlugin(plugins)

    def _play_game(self, agent1: Agent, agent2: Agent):
        self.game.reset()

        agents = {
            "player_0": agent1,
            "player_1": agent2,
        }

        self.plugins.start_game(self.game, agent1, agent2)

        for step, agent in enumerate(self.game.agent_iter()):
            _, _, termination, truncation, _ = self.game.last()
            if termination or truncation:
                action = None
                self.plugins.end_game(self.game, step)
                break

            action = int(agents[agent].get_action(self.game))
            self.game.step(action)

            self.plugins.action(self.game, step, agent, action)

        self.game.close()

    def play_game(self, agent1: Agent, agent2: Agent):
        self.plugins.start_arena(self.game)
        self._play_game(agent1, agent2)
        self.plugins.end_arena(self.game)
