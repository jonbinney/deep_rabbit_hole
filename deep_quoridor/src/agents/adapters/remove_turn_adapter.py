from typing import Any

from agents.adapters.base import BaseTrainableAgentAdapter
from gymnasium import spaces


class RemoveTurnAdapter(BaseTrainableAgentAdapter):
    """
    An adapter that splits the board into one-hot representations with one channel per player
    when getting observations from the game.
    """

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
            self._transform_observation(observation_before_action),
            self._transform_observation(opponent_observation_after_action),
            self._transform_observation(observation_after_action),
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
            self._transform_observation(opponent_observation_before_action),
            self._transform_observation(my_observation_after_opponent_action),
            self._transform_observation(opponent_observation_after_action),
            opponent_reward,
            opponent_action,
            done,
        )

    def _transform_observation(self, observation: Any) -> Any:
        observation = observation.copy()
        observation["observation"] = observation["observation"].copy()
        observation["observation"].pop("my_turn")
        return observation

    def get_action(self, observation):
        transformed_observation = self._transform_observation(observation)
        return self._agent().get_action(transformed_observation)

    @classmethod
    def get_observation_space(cls, original_space):
        space = {}
        for key, value in original_space.items():
            if key != "observation":
                space[key] = value
            else:
                obs_spc = {}
                for key, value in value.items():
                    if key != "my_turn":
                        obs_spc[key] = value
                space["observation"] = spaces.Dict(obs_spc)

        return spaces.Dict(space)
