import numpy as np
from pettingzoo.utils.env import AgentID, ObsType
from pettingzoo.utils.wrappers import BaseWrapper


class ThreeDSplitBoardWrapper(BaseWrapper):
    """
    This wrapper modifies the observation space of the environment to include a 3D representation
     of the board, where each layer represents the positions of one player.

    """

    def __init__(self, env):
        super().__init__(env)
        self.board_size = env.unwrapped.board_size

    def observe(self, agent: AgentID) -> ObsType:
        """Transform the observation for the given agent."""
        obs = self.env.observe(agent)
        observation = obs["observation"].copy()

        # Combine my_board and opponent_board into a single 3D board
        board = observation.pop("board")
        # Create one-hot representations for player and opponent
        my_board = (board == 1).astype(np.float32)
        opponent_board = (board == 2).astype(np.float32)

        combined_board = np.stack([my_board, opponent_board], axis=-1)

        # Add the combined 3D board to the observation
        observation["board3d"] = combined_board

        return {"observation": observation, "action_mask": obs["action_mask"]}

    def observation_space(self):
        """Define the observation space for the transformed observations."""
        original_space = self.env.observation_space
        board_shape = (self.board_size, self.board_size, 2)  # 3D shape for the combined board
        return {
            "observation": {
                "my_turn": original_space["observation"]["my_turn"],
                "board3d": original_space["observation"]["board"].__class__(shape=board_shape, dtype=np.float32),
                "walls": original_space["observation"]["walls"],
                "my_walls_remaining": original_space["observation"]["my_walls_remaining"],
                "opponent_walls_remaining": original_space["observation"]["opponent_walls_remaining"],
            },
            "action_mask": original_space["action_mask"],
        }
