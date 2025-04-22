from typing import Any

from agents.adapters.base import BaseTrainableAgentAdapter
from agents.core import rotation
from environment.rotate_wrapper import RotateWrapper


class RotateAdapter(BaseTrainableAgentAdapter):
    def _rotate_if_needed(self, observation, action, game, player_id):
        if player_id == "player_1":
            rotated_obs = rotation.rotate_observation(observation)
            rotated_game = RotateWrapper(game)
            rotated_action = rotation.convert_original_action_index_to_rotated(self.board_size, action)
            return rotated_obs, rotated_action, rotated_game
        else:
            return observation, action, game

    def handle_opponent_step_outcome(self, observation_before_action: Any, action: Any, game: Any) -> None:
        rotated_obs, rotated_action, rotated_game = self._rotate_if_needed(
            observation_before_action, action, game, self.get_opponent_id()
        )
        return self._agent().handle_opponent_step_outcome(rotated_obs, rotated_action, rotated_game)

    def handle_step_outcome(self, observation_before_action: Any, action: Any, game: Any) -> None:
        rotated_obs, rotated_action, rotated_game = self._rotate_if_needed(
            observation_before_action, action, game, self.player_id
        )
        return self._agent().handle_opponent_step_outcome(rotated_obs, rotated_action, rotated_game)

    def get_action(self, game: Any) -> Any:
        if self.player_id == "player_1":
            game = RotateWrapper(game)
            action = self._agent().get_action(game)
            return rotation.convert_rotated_action_index_to_original(self.board_size, action)
        else:
            return self._agent().get_action(game)
