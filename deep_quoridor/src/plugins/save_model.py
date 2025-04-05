from agents.core import AbstractTrainableAgent
from arena_utils import ArenaPlugin
from utils.misc import resolve_path


class SaveModelEveryNEpisodesPlugin(ArenaPlugin):
    def __init__(
        self,
        update_every: int,
        board_size: int,
        max_walls: int,
        save_final: bool = True,
    ):
        self.update_every = update_every
        self.episode_count = 0
        self.board_size = board_size
        self.max_walls = max_walls
        self.save_final = save_final

    def start_game(self, game, agent1, agent2):
        self.agents = [agent1, agent2]

    def end_game(self, game, result):
        if self.episode_count % self.update_every == 0 and self.episode_count > 0:
            self._save_models(f"episode_{self.episode_count}")
        self.episode_count += 1

    def end_arena(self, game, results):
        if self.save_final:
            self._save_models("final")

    def _save_models(self, suffix: str):
        for agent in self.agents:
            if not isinstance(agent, AbstractTrainableAgent) or not agent.training_mode:
                continue
            agent_name = agent.name()
            save_file = resolve_path(agent.params.model_dir, agent.resolve_filename(suffix))
            agent.save_model(save_file)
            print(f"{agent_name} Model saved to {save_file}")
