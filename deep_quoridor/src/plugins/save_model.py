from collections import deque
from typing import Any, Callable, Optional

from agents.core.trainable_agent import TrainableAgent
from arena_utils import ArenaPlugin
from utils import resolve_path


class SaveModelEveryNEpisodesPlugin(ArenaPlugin):
    def __init__(
        self,
        update_every: int,
        board_size: int,
        max_walls: int,
        save_final: bool = True,
        run_id: str = "",
        after_save: Optional[Callable[[str], Any]] = None,
        trigger_metrics: Optional[tuple[int, int]] = None,  # wins, last_episodes
    ):
        self.update_every = update_every
        self.episode_count = 0
        self.board_size = board_size
        self.max_walls = max_walls
        self.save_final = save_final
        self.run_id = run_id
        self.after_save = after_save

        if trigger_metrics:
            self.trigger_metrics_n_wins = trigger_metrics[0]
            self.win_history = deque(maxlen=trigger_metrics[1])
        else:
            self.trigger_metrics_n_wins = 0
            self.win_history = None

    def start_game(self, game, agent1, agent2):
        self.agent = None
        if isinstance(agent1, TrainableAgent) and agent1.is_training():
            self.agent = agent1

        if isinstance(agent2, TrainableAgent) and agent2.is_training():
            if self.agent:
                raise ValueError(
                    "SaveModelEveryNEpisodesPlugin can only be used with 1 training agent, but 2 are present."
                )
            self.agent = agent2

        if self.agent is None:
            raise ValueError("SaveModelEveryNEpisodesPlugin requires an agent being trained to be used")

    def end_game(self, game, result):
        assert self.agent
        self.episode_count += 1
        trigger_save = False
        if self.win_history is not None:
            won = 1 if result.winner == self.agent.name() else 0
            self.win_history.append(won)
            n_won = sum(list(self.win_history))

            if n_won >= self.trigger_metrics_n_wins:
                print(
                    f"Won {n_won} out of {len(self.win_history)} games.  Saving model for episode {self.episode_count}"
                )
                self.win_history.clear()
                trigger_save = True

        if trigger_save or self.episode_count % self.update_every == 0:
            self._save_models(f"{self.run_id}_episode_{self.episode_count}")

    def end_arena(self, game, results):
        if self.save_final:
            self._save_models(f"{self.run_id}_final")

    def _save_models(self, suffix: str):
        assert self.agent
        save_file = resolve_path(self.agent.params.model_dir, self.agent.resolve_filename(suffix))
        self.agent.save_model(save_file)
        print(f"{self.agent.name()} Model saved to {save_file}")

        if self.after_save:
            self.after_save(str(save_file))
