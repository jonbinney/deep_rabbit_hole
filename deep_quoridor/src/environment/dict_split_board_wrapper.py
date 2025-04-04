import numpy as np
from pettingzoo.utils.env import AgentID, ObsType
from pettingzoo.utils.wrappers import BaseWrapper


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
        board = observation["board"]
        del observation["board"]

        # Create one-hot representations for player and opponent
        player_board = (board == 1).astype(np.float32)
        opponent_board = (board == 2).astype(np.float32)

        # Separate the one-hot representations
        observation["my_board"] = player_board
        observation["opponent_board"] = opponent_board

        return {"observation": observation, "action_mask": obs["action_mask"]}

    def observation_space(self):
        """Define the observation space for the transformed observations."""
        original_space = self.env.observation_space
        board_shape = (self.board_size, self.board_size)  # Shape for each board (player and opponent)
        return {
            "observation": {
                "my_turn": original_space["observation"]["my_turn"],
                "my_board": original_space["observation"]["board"].__class__(shape=board_shape, dtype=np.float32),
                "opponent_board": original_space["observation"]["board"].__class__(shape=board_shape, dtype=np.float32),
                "walls": original_space["observation"]["walls"],
                "my_walls_remaining": original_space["observation"]["my_walls_remaining"],
                "opponent_walls_remaining": original_space["observation"]["opponent_walls_remaining"],
            },
            "action_mask": original_space["action_mask"],
        }
