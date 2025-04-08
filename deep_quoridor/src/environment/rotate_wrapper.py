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
        return self.env.step(action)
