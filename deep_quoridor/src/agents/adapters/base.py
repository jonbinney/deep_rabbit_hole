from typing import Any, Tuple

from agents.core.trainable_agent import TrainableAgent


class BaseTrainableAgentAdapter(TrainableAgent):
    def __init__(self, agent: TrainableAgent):
        self.agent = agent

    def _agent(self) -> TrainableAgent:
        return self.agent

    def is_trainable(self) -> bool:
        return self._agent().is_trainable()

    def start_game(self, game: Any, player_id: int) -> None:
        self.board_size = game.board_size
        self.player_id = player_id
        self._agent().start_game(game, player_id)

    def end_game(self, game: Any) -> None:
        self._agent().end_game(game)

    def handle_step_outcome(
        self,
        observation_before_action,
        opponent_observation_after_action,
        observation_after_action,
        reward,
        action,
        done=False,
    ):
        return self._agent().handle_step_outcome(
            observation_before_action,
            opponent_observation_after_action,
            observation_after_action,
            reward,
            action,
            done,
        )

    def handle_opponent_step_outcome(
        self,
        opponent_observation_before_action,
        my_observation_after_opponent_action,
        opponent_observation_after_action,
        opponent_reward,
        opponent_action,
        done,
    ):
        """Handle the opponent's step outcome."""
        return self._agent().handle_opponent_step_outcome(
            opponent_observation_before_action,
            my_observation_after_opponent_action,
            opponent_observation_after_action,
            opponent_reward,
            opponent_action,
            done,
        )

    def compute_loss_and_reward(self, length: int) -> Tuple[float, float]:
        return self._agent().compute_loss_and_reward(length)

    def model_hyperparameters(self) -> dict:
        return self._agent().model_hyperparameters()

    def get_action(self, observation, action_mask) -> int:
        return self._agent().get_action(observation, action_mask)

    def version(self) -> str:
        return self._agent().version()

    def model_id(self) -> str:
        return self._agent().model_id()

    def model_name(self) -> str:
        return self._agent().model_name()

    def wandb_local_filename(self, artifact: Any) -> str:
        return self._agent().wandb_local_filename(artifact)

    def resolve_filename(self, suffix: str) -> str:
        return self._agent().resolve_filename(suffix)

    def save_model(self, path: str) -> None:
        self._agent().save_model(path)

    def load_model(self, path: str) -> None:
        self._agent().load_model(path)

    def get_opponent_player_id(self, player_id):
        """Get the opponent player ID."""
        return "player_1" if self.player_id == "player_0" else "player_0"
