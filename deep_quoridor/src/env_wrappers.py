from typing import Any, Dict

import numpy as np
from agents.core import rotation
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from pettingzoo.utils.wrappers import BaseWrapper


class RotateWrapper(BaseWrapper):
    """
    A wrapper that rotates the board 180 degrees for player_1 to normalize the view for both players.
    Each player will see themselves at the bottom of the board.
    """

    def __init__(self, env):
        super().__init__(env)
        self.board_size = env.unwrapped.board_size

    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the observation for the current agent."""
        if self.agent_selection == "player_0":
            return obs

        observation = obs["observation"].copy()
        action_mask = obs["action_mask"].copy()

        # Rotate board and walls for player_1
        observation["board"] = rotation.rotate_board(observation["board"])
        observation["walls"] = rotation.rotate_walls(observation["walls"])

        # Rotate action mask for player_1
        rotated_mask = rotation.rotate_action_mask(self.board_size, action_mask)

        return {"observation": observation, "action_mask": rotated_mask}

    def action(self, action: int) -> int:
        """Transform the action from rotated space back to original space for player_1."""
        if self.agent_selection == "player_0":
            return action

        return rotation.convert_rotated_action_index_to_original(self.board_size, action)

    def observe(self, agent: AgentID) -> ObsType:
        """Transform the observation for the given agent."""
        obs = self.env.observe(agent)
        if agent == "player_0":
            return obs

        observation = obs["observation"].copy()
        action_mask = obs["action_mask"].copy()

        # Rotate board and walls for player_1
        observation["board"] = rotation.rotate_board(observation["board"])
        observation["walls"] = rotation.rotate_walls(observation["walls"])

        # Rotate action mask for player_1
        rotated_mask = rotation.rotate_action_mask(self.board_size, action_mask)

        return {"observation": observation, "action_mask": rotated_mask}

    def step(self, action: ActionType) -> None:
        """Transform the action if needed and perform the step."""
        if self.agent_selection == "player_1":
            action = rotation.convert_rotated_action_index_to_original(self.board_size, action)
        self.env.step(action)


class SplitBoardWrapper(BaseWrapper):
    """
    A wrapper that splits the board into one-hot representations with one channel per player.
    """

    def __init__(self, env):
        super().__init__(env)
        self.board_size = env.unwrapped.board_size

    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the observation to split the board into one-hot representations."""
        observation = obs["observation"].copy()
        board = observation["board"]

        # Create one-hot representations for player and opponent
        player_board = (board == 1).astype(np.float32)
        opponent_board = (board == 2).astype(np.float32)

        # Separate the one-hot representations
        observation["player_board"] = player_board
        observation["opponent_board"] = opponent_board

        return {"observation": observation, "action_mask": obs["action_mask"]}

    def observe(self, agent: AgentID) -> ObsType:
        """Transform the observation for the given agent."""
        obs = self.env.observe(agent)
        observation = obs["observation"].copy()
        board = observation["board"]
        del observation["board"]

        # Create one-hot representations for player and opponent
        player_board = (board == 1).astype(np.float32)
        opponent_board = (board == 2).astype(np.float32)

        # Separate the one-hot representations
        observation["player_board"] = player_board
        observation["opponent_board"] = opponent_board

        return {"observation": observation, "action_mask": obs["action_mask"]}

    def observation_space(self):
        """Define the observation space for the transformed observations."""
        original_space = self.env.observation_space
        board_shape = (self.board_size, self.board_size)  # Shape for each board (player and opponent)
        return {
            "observation": {
                "my_turn": original_space["observation"]["my_turn"],
                "player_board": original_space["observation"]["board"].__class__(shape=board_shape, dtype=np.float32),
                "opponent_board": original_space["observation"]["board"].__class__(shape=board_shape, dtype=np.float32),
                "walls": original_space["observation"]["walls"],
                "my_walls_remaining": original_space["observation"]["my_walls_remaining"],
                "opponent_walls_remaining": original_space["observation"]["opponent_walls_remaining"],
            },
            "action_mask": original_space["action_mask"],
        }
