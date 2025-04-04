import numpy as np
from pettingzoo.utils.env import AgentID, ObsType
from pettingzoo.utils.wrappers import BaseWrapper


class SplitBoardTo3DWrapper(BaseWrapper):
    """
    A wrapper that combines my_board and opponent_board into a single 3D one-hot encoded board.
    """

    def __init__(self, env):
        super().__init__(env)
        self.board_size = env.unwrapped.board_size

    def observe(self, agent: AgentID) -> ObsType:
        """Transform the observation for the given agent."""
        obs = self.env.observe(agent)
        observation = obs["observation"].copy()

        # Combine my_board and opponent_board into a single 3D board
        my_board = observation.pop("my_board")
        opponent_board = observation.pop("opponent_board")
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
