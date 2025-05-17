from typing import Any

import numpy as np
from agents.adapters.base import BaseTrainableAgentAdapter
from gymnasium import spaces


class UnifiedBoardAdapter(BaseTrainableAgentAdapter):
    """A board adapter that unifies the game board and walls into a single representation.
    This adapter transforms the game's observation space by merging the board and walls into a
    single unified board representation. The transformed board includes:
    - Original board positions (player positions) at even indices
    - Wall representations as -1 at odd indices
    - A padding of -1 around the entire board to represent boundaries
    The unified board format:
    - Empty spaces: 0
    - Player 1: 1
    - Player 2: 2
    - Walls/Boundaries: -1
    The transformation expands the original NxN board to a (2N+1)x(2N+1) board to accommodate:
    1. Original board positions at even indices (i,j where i,j are even)
    2. Horizontal walls at odd rows, even columns
    3. Vertical walls at even rows, odd columns
    4. Wall intersections at odd rows, odd columns
    5. A border of walls (-1) padding the entire board
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
        walls = observation["observation"]["walls"]
        for key, value in observation["observation"].items():
            if key != "board":
                obs[key] = value
                continue
            board = value
            # Expands the board to include spaces for walls, representing walls with -1 in the odd rows and columns
            new_board = np.full(
                (board.shape[0] + board.shape[0] - 1, board.shape[1] + board.shape[1] - 1), 0, dtype=np.int32
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
            obs["board"] = np.pad(new_board, pad_width=1, mode="constant", constant_values=-1)

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
