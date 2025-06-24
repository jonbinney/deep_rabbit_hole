from agents.adapters.base import BaseTrainableAgentAdapter
from agents.core import rotation
from utils.misc import get_opponent_player_id


class RotateAdapter(BaseTrainableAgentAdapter):
    def _needs_rotation(self, observation, player_id):
        if observation is None:
            return False
        player_turn = observation["observation"]["my_turn"]
        return (player_id == "player_1") ^ (not player_turn)

    def _rotate_observation(self, observation):
        observation = observation.copy()
        observation["action_mask"] = rotation.rotate_action_vector(self.board_size, observation["action_mask"])
        observation["observation"] = observation["observation"].copy()
        observation["observation"]["board"] = rotation.rotate_board(observation["observation"]["board"])
        observation["observation"]["walls"] = rotation.rotate_walls(observation["observation"]["walls"])
        return observation

    def _rotate_observation_if_needed(self, observation, player_id):
        if self._needs_rotation(observation, player_id):
            return self._rotate_observation(observation)
        else:
            return observation

    def _rotate_action_if_needed(self, observation, action, player_id):
        if action is not None and self._needs_rotation(observation, player_id):
            return rotation.convert_original_action_index_to_rotated(self.board_size, action)
        else:
            return action

    def handle_opponent_step_outcome(
        self,
        opponent_observation_before_action,
        my_observation_after_opponent_action,
        opponent_observation_after_action,
        opponent_reward,
        opponent_action,
        done,
    ):
        return self._agent().handle_opponent_step_outcome(
            self._rotate_observation_if_needed(
                opponent_observation_before_action, get_opponent_player_id(self.player_id)
            ),
            self._rotate_observation_if_needed(my_observation_after_opponent_action, self.player_id),
            self._rotate_observation_if_needed(
                opponent_observation_after_action, get_opponent_player_id(self.player_id)
            ),
            opponent_reward,
            self._rotate_action_if_needed(
                opponent_observation_before_action, opponent_action, get_opponent_player_id(self.player_id)
            ),
            done,
        )

    def handle_step_outcome(
        self,
        observation_before_action,
        opponent_observation_after_action,
        observation_after_action,
        reward,
        action,
        done=False,
    ):
        self._agent().handle_step_outcome(
            self._rotate_observation_if_needed(observation_before_action, self.player_id),
            self._rotate_observation_if_needed(
                opponent_observation_after_action, get_opponent_player_id(self.player_id)
            ),
            self._rotate_observation_if_needed(observation_after_action, self.player_id),
            reward,
            self._rotate_action_if_needed(observation_before_action, action, self.player_id),
            done,
        )

    def get_action(self, observation):
        needs_rotation = self._needs_rotation(observation, self.player_id)
        if needs_rotation:
            observation = self._rotate_observation(observation)
            return rotation.convert_rotated_action_index_to_original(
                self.board_size, self._agent().get_action(observation)
            )
        else:
            return self._agent().get_action(observation)
