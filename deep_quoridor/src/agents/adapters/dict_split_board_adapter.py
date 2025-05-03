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
        board = observation.pop("board")

        # Create one-hot representations for player and opponent
        player_board = (board == 1).astype(np.float32)
        opponent_board = (board == 2).astype(np.float32)

        # Update the observation dictionary
        observation["my_board"] = player_board
        observation["opponent_board"] = opponent_board

        return observation

    def get_action(self, observation, action_mask):
        transformed_observation = self._transform_observation(observation)
        return self._agent().get_action(transformed_observation, action_mask)

    @classmethod
    def get_observation_space(original_space):
        board_shape = original_space["observation"]["board"].shape
        return {
            "observation": spaces.Dict(
                {
                    "my_turn": original_space["observation"]["my_turn"],
                    "my_board": original_space["observation"]["board"].__class__(
                        0, 1, shape=board_shape, dtype=np.float32
                    ),
                    "opponent_board": original_space["observation"]["board"].__class__(
                        0, 1, shape=board_shape, dtype=np.float32
                    ),
                    "walls": original_space["observation"]["walls"],
                    "my_walls_remaining": original_space["observation"]["my_walls_remaining"],
                    "opponent_walls_remaining": original_space["observation"]["opponent_walls_remaining"],
                }
            ),
            "action_mask": original_space["action_mask"],
        }
