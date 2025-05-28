from typing import Any

import numpy as np
from agents.adapters.base import BaseTrainableAgentAdapter
from gymnasium import spaces


class Board3CAdapter(BaseTrainableAgentAdapter):
    """A board adapter that transforms the observation space for the uni board into three binary channels.

    The transformed board representation has 3 channels:
    - Channel 1: Binary mask for current player's position (1s and 0s)
    - Channel 2: Binary mask for opponent's position (1s and 0s)
    - Channel 3: Binary mask for walls (1s and 0s)

    This representation makes it easier for neural networks to process the game state
    by separating different elements into distinct channels.
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
        """Transform the game observation merging the boards and walls into a single board, and adding walls on the sides"""
        if observation is None:
            return None
        observation = observation.copy()
        obs = {}
        for key, value in observation["observation"].items():
            if key != "board":
                obs[key] = value
                continue
            board = value
            board_walls = (board == -1).astype(np.float32)
            board_my = (board == 1).astype(np.float32)
            board_opponent = (board == 2).astype(np.float32)
            obs["board"] = np.stack([board_my, board_opponent, board_walls], axis=0)
        observation["observation"] = obs
        return observation

    def get_action(self, observation):
        transformed_observation = self._transform_observation(observation)
        return self._agent().get_action(transformed_observation)

    @classmethod
    def get_observation_space(cls, original_space):
        board_shape = original_space["observation"]["board"].shape
        new_board_shape = (3, board_shape[0], board_shape[1])
        space = {}
        for key, value in original_space.items():
            if key != "observation":
                space[key] = value
            else:
                obs_spc = {}
                for key, value in original_space["observation"].items():
                    if key == "board":
                        obs_spc["board"] = original_space["observation"]["board"].__class__(
                            0, 1, shape=new_board_shape, dtype=np.float32
                        )
                    elif key == "walls":
                        continue
                    else:
                        obs_spc[key] = value
                space["observation"] = spaces.Dict(obs_spc)
        return spaces.Dict(space)
