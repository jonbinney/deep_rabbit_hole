from typing import Any

import numpy as np
from agents.adapters.base import BaseTrainableAgentAdapter
from gymnasium import spaces


class DictSplitBoardAdapter(BaseTrainableAgentAdapter):
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
        """Transform the game observation by splitting the board into separate channels"""
        if observation is None:
            return None
        observation = observation.copy()
        obs = {}
        for key, value in observation["observation"].items():
            if key != "board":
                obs[key] = value
                continue
            board = value
            # Create one-hot representations for player and opponent
            player_board = (board == 1).astype(np.float32)
            opponent_board = (board == 2).astype(np.float32)

            # Update the observation dictionary
            obs["my_board"] = player_board
            obs["opponent_board"] = opponent_board

        observation["observation"] = obs
        return observation

    def get_action(self, observation):
        transformed_observation = self._transform_observation(observation)
        return self._agent().get_action(transformed_observation)

    @classmethod
    def get_observation_space(cls, original_space):
        board_shape = original_space["observation"]["board"].shape
        space = {}
        for key, value in original_space.items():
            if key != "observation":
                space[key] = value
            else:
                obs_spc = {}
                for key, value in original_space["observation"].items():
                    if key == "board":
                        obs_spc["my_board"] = original_space["observation"]["board"].__class__(
                            0, 1, shape=board_shape, dtype=np.float32
                        )
                        obs_spc["opponent_board"] = original_space["observation"]["board"].__class__(
                            0, 1, shape=board_shape, dtype=np.float32
                        )
                    else:
                        obs_spc[key] = value
                space["observation"] = spaces.Dict(obs_spc)
        return spaces.Dict(space)
