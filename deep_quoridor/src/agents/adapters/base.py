from typing import Any, Tuple

from agents.core.trainable_agent import TrainableAgent


class BaseTrainableAgentAdapter(TrainableAgent):
    def __init__(self, agent: TrainableAgent):
        self.agent = agent

    def is_trainable(self) -> bool:
        return self.agent.is_trainable()

    def start_game(self, game: Any, player_id: int) -> None:
        self.agent.start_game(game, player_id)

    def end_game(self, game: Any) -> None:
        self.agent.end_game(game)

    def handle_opponent_step_outcome(self, observation_before_action: Any, action: Any, game: Any) -> None:
        self.agent.handle_opponent_step_outcome(observation_before_action, action, game)

    def handle_step_outcome(self, observation_before_action: Any, action: Any, game: Any) -> None:
        self.agent.handle_step_outcome(observation_before_action, action, game)

    def compute_loss_and_reward(self, length: int) -> Tuple[float, float]:
        return self.agent.compute_loss_and_reward(length)

    def model_hyperparameters(self) -> dict:
        return self.agent.model_hyperparameters()

    def get_action(self, game: Any) -> Any:
        return self.agent.get_action(game)

    def version(self) -> str:
        return self.agent.version()

    def model_id(self) -> str:
        return self.agent.model_id()

    def model_name(self) -> str:
        return self.agent.model_name()

    def wandb_local_filename(self, artifact: Any) -> str:
        return self.agent.wandb_local_filename(artifact)

    def resolve_filename(self, suffix: str) -> str:
        return self.agent.resolve_filename(suffix)

    def save_model(self, path: str) -> None:
        self.agent.save_model(path)

    def load_model(self, path: str) -> None:
        self.agent.load_model(path)
