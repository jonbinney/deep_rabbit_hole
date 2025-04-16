from typing import Any

import numpy as np
from agents.adapters.base import BaseTrainableAgentAdapter
from agents.core.trainable_agent import TrainableAgent
from gymnasium import spaces
from pettingzoo.utils.env import AgentID, ObsType
from pettingzoo.utils.wrappers import BaseWrapper


class DictSplitBoardAdapter(BaseTrainableAgentAdapter):
    """
    An adapter that splits the board into one-hot representations with one channel per player
    when getting observations from the game.
    """

    def __init__(self, agent: TrainableAgent):
        super().__init__(agent)
        self.board_size = None

    def start_game(self, game: Any, player_id: int) -> None:
        """Initialize board size when starting a new game"""
        self.board_size = game.unwrapped.board_size
        super().start_game(game, player_id)

    def get_action(self, game: Any) -> Any:
        """Transform the observation before getting action from the wrapped agent"""
        transformed_game = self._transform_observation(game)
        return super().get_action(transformed_game)

    def handle_opponent_step_outcome(self, observation_before_action: Any, action: Any, game: Any) -> None:
        """Transform the observation before handling opponent step outcome"""
        transformed_obs = self._transform_observation(observation_before_action)
        transformed_game = self._transform_observation(game)
        super().handle_opponent_step_outcome(transformed_obs, action, transformed_game)

    def handle_step_outcome(self, observation_before_action: Any, action: Any, game: Any) -> None:
        """Transform the observation before handling step outcome"""
        transformed_obs = self._transform_observation(observation_before_action)
        transformed_game = self._transform_observation(game)
        super().handle_step_outcome(transformed_obs, action, transformed_game)

    def _transform_observation(self, game: Any) -> Any:
        """Transform the game observation by splitting the board into separate channels"""
        game = game.copy()
        board = game.observation.pop("board")

        # Create one-hot representations for player and opponent
        player_board = (board == 1).astype(np.float32)
        opponent_board = (board == 2).astype(np.float32)

        # Update the observation dictionary
        game.observation["my_board"] = player_board
        game.observation["opponent_board"] = opponent_board

        return game


class DictSplitBoardWrapper(BaseWrapper):
    """
    A wrapper that splits the board into one-hot representations with one channel per player.
    """

    def __init__(self, env):
        super().__init__(env)
        self.board_size = env.unwrapped.board_size

    def observe(self, agent: AgentID) -> ObsType:
        """Transform the observation for the given agent."""
        obs = self.env.observe(agent)
        observation = obs["observation"].copy()
        board = observation.pop("board")

        # Create one-hot representations for player and opponent
        player_board = (board == 1).astype(np.float32)
        opponent_board = (board == 2).astype(np.float32)

        # Separate the one-hot representations
        observation["my_board"] = player_board
        observation["opponent_board"] = opponent_board

        return {"observation": observation, "action_mask": obs["action_mask"]}

    def observation_space(self, agent):
        """Define the observation space for the transformed observations."""
        original_space = self.env.observation_space(agent)
        board_shape = (self.board_size, self.board_size)  # Shape for each board (player and opponent)
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
