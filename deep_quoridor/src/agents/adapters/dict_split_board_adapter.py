from typing import Any

import numpy as np
from agents.adapters.base import BaseTrainableAgentAdapter
from environment.dict_split_board_wrapper import DictSplitBoardWrapper


class DictSplitBoardAdapter(BaseTrainableAgentAdapter):
    """
    An adapter that splits the board into one-hot representations with one channel per player
    when getting observations from the game.
    """

    def handle_step_outcome(self, observation_before_action: Any, action: Any, game: Any) -> None:
        """Transform the observation before handling step outcome"""
        transformed_obs = self._transform_observation(observation_before_action)
        transformed_game = self._transform_game(game)
        self._agent().handle_step_outcome(transformed_obs, action, transformed_game)

    def _transform_game(self, game: Any) -> Any:
        return DictSplitBoardWrapper(game)

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
