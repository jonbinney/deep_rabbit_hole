import os

from agents.core import AbstractTrainableAgent
from arena_utils import ArenaPlugin


class SaveModelEveryNEpisodesPlugin(ArenaPlugin):
    def __init__(
        self, update_every: int, path: str, board_size: int, max_walls: int, agents: list[AbstractTrainableAgent]
    ):
        self.agents = agents
        self.update_every = update_every
        self.path = path
        self.episode_count = 0
        self.board_size = board_size
        self.max_walls = max_walls

    def end_game(self, game, result):
        if self.episode_count % self.update_every == 0 and self.episode_count > 0:
            self._save_models(f"_episode_{self.episode_count}")
        self.episode_count += 1

    def end_arena(self, game, results):
        self._save_models("final")

    def _save_models(self, suffix: str):
        for agent in self.agents:
            agent_name = agent.name()
            filename = agent.resolve_filename(suffix)
            save_file = os.path.join(self.path, filename)
            agent.save_model(save_file)
            print(f"{agent_name} Model saved to {save_file}")
