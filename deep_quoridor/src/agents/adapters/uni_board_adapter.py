from typing import Any

import numpy as np
from agents.adapters.base import BaseTrainableAgentAdapter
from gymnasium import spaces


class UnifiedBoardAdapter(BaseTrainableAgentAdapter):
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
        walls = observation["observation"]["walls"]
        for key, value in observation["observation"].items():
            if key != "board":
                obs[key] = value
                continue
            board = value
            # Creates a new board and place the walls as -1 in the odds rows and cols
            new_board = np.full(
                (board.shape[0] + board.shape[0] - 1, board.shape[1] + board.shape[1] - 1), 0, dtype=np.float32
            )
            new_board[::2, ::2] = board
            h_walls = walls[:, :, 0]
            for i in range(h_walls.shape[0]):
                for j in range(h_walls.shape[1]):
                    if h_walls[i][j] == 1:
                        new_board[2 * i + 1][2 * j] = -1
                        new_board[2 * i + 1][2 * j + 1] = -1
            v_walls = walls[:, :, 1]
            for i in range(v_walls.shape[0]):
                for j in range(v_walls.shape[1]):
                    if v_walls[i][j] == 1:
                        new_board[2 * i][2 * j + 1] = -1
                        new_board[2 * i + 1][2 * j + 1] = -1

            # Pads the board with -1 (walls)
            padded_board = np.full((new_board.shape[0] + 2, new_board.shape[1] + 2), -1, dtype=np.float32)
            padded_board[1:-1, 1:-1] = new_board
            obs["board"] = padded_board

        observation["observation"] = obs
        return observation

    def get_action(self, observation):
        transformed_observation = self._transform_observation(observation)
        return self._agent().get_action(transformed_observation)

    @classmethod
    def get_observation_space(cls, original_space):
        board_shape = original_space["observation"]["board"].shape
        new_board_shape = (board_shape[0] + board_shape[0] + 1, board_shape[1] + board_shape[1] + 1)
        space = {}
        for key, value in original_space.items():
            if key != "observation":
                space[key] = value
            else:
                obs_spc = {}
                for key, value in original_space["observation"].items():
                    if key == "board":
                        obs_spc["board"] = original_space["observation"]["board"].__class__(
                            -1, 2, shape=new_board_shape, dtype=np.float32
                        )
                    elif key == "walls":
                        continue
                    else:
                        obs_spc[key] = value
                space["observation"] = spaces.Dict(obs_spc)
        return spaces.Dict(space)
