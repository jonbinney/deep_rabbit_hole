"""
Quoridor Gym/PettingZoo Environment implementation.

The board size is parameterizable with:
 - Board size (int): size of the square board
 - Max number of walls (int): maximum number of walls that each player can place
Boards are always square.
For all explanations below, the default value of 9x9 is assumed.

The state is implemented in a two-hot encoding, as follows:
 - A 9x9 2D array, each representing a space in the board. All values are 0 except
   1 where player 1 is placed, and 2 where player 2 is placed.
   NOTE: This is represented internally in a compact way but converted in the specified
   observation space format as needed.
 - An 8x8x2 array, representing the vertical and horizontal walls placed in the board, using
   one-hot encoding.

The action is represented as follows, taking a 9x9 board as an example:
- First 9x9 = 81 values represent player positions
- The next 8x8 = 64 values reprsent vertical walls and the last 8x8 = 64 values represent horizontal walls
- The cell position of a wall represnts the first space occupied by a wall, knowing that each cell takes two spaces.
- For vertical walls, we place the wall starting at the upper right corner of the (row, col) cell
- For horizontal walls, we place the wall starting at the lower left corner of the (row, col) cell

Every time we represent a coordinate as a tuple, it is in the form (row, col)
"""

import functools
from typing import Tuple
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from gymnasium import spaces
import numpy as np


class QuoridorEnv(AECEnv):
    metadata = {"render_modes": ["ansi"], "name": "quoridor_v0"}

    def __init__(self, board_size: int = 9, max_walls: int = 10):
        super(AECEnv, self).__init__()

        self.render_mode = "human"

        self.board_size = board_size  # assumed square grid
        self.wall_size = self.board_size - 1  # grid for walls
        self.max_walls = max_walls  # Each player gets 10 walls

        self.possible_agents = ["player_0", "player_1"]

        self.reset()

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        self.agent_order = self.agents.copy()

        self.walls = np.zeros((self.wall_size, self.wall_size, 2), dtype=np.int8)
        self.walls_remaining = {agent: self.max_walls for agent in self.agents}
        # Positions are (row, col)
        self.positions = {
            "player_0": (0, self.board_size // 2),
            "player_1": (self.board_size - 1, self.board_size // 2),
        }

        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.agent_selection = self.agent_order[0]

        # I created these so that env.last() wouldn't complain, but I don't
        # really understand how should they work or why they are needed
        self._cumulative_rewards = self.rewards.copy()
        self.infos = {agent: {} for agent in self.agents}

        return None

    def step(self, action):
        """
        Players move by selecting an index from 0-80 (9×9 board).
        Wall placement is mapped to 81-208 (8×8×2).
        """
        agent = self.agent_selection
        if self.terminations[agent]:
            self._next_player()
            return

        (row, col, action_type) = self.action_index_to_params(action)
        if action_type == 0:
            self._move(agent, (row, col))
        else:
            self._place_wall(agent, (row, col), action_type - 1)

        if self._check_win(agent):
            self.terminations = {a: True for a in self.agents}
            self.rewards[agent] = 1
            self.rewards[self._opponent(agent)] = -1

        # TODO: Confirm if this is needed and if it's doing anything
        self._accumulate_rewards()

        self._next_player()

    def observe(self, agent_id):
        """
        Returns the observation and action mask in a dict, like so:
        {
            "observation": observation,
            "action_mask": action_mask
        }
        """
        return {
            "observation": self._get_observation(agent_id),
            "action_mask": self._get_action_mask(agent_id),
        }

    def _get_observation(self, agent_id):
        # TODO: Do we need to make copies of the state or can we return references directly?

        # Calculate board from self.positions
        # NOTE: The board uses 1 to indicate where the agent is and 2 to indicate where the opponent is
        # Obviously, this will be different for each player
        board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        board[self.positions[agent_id]] = 1
        board[self.positions[self._opponent(agent_id)]] = 2

        # Make a copy of walls
        walls = self.walls.copy()

        return {
            "board": board,
            "walls": walls,
            "my_walls_remaining": self.walls_remaining[agent_id],
            "opponent_walls_remaining": self.walls_remaining[self._opponent(agent_id)],
        }

    def _get_action_mask(self, agent_id):
        # Start with an empty mask (nothing possible)
        mask = np.zeros((self.board_size**2 + (self.wall_size**2) * 2,), dtype=np.int8)

        # Calculate valid "moves" (as opposed to wall placements)
        # Start with the four basic directions
        for row, col in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            old_row, old_col = self.positions[agent_id]
            new_row, new_col = old_row + row, old_col + col

            # Check if the new position is still inside the board
            if not (0 <= new_row < self.board_size and 0 <= new_col < self.board_size):
                continue

            # Is there a wall in the way we're going?
            blocked_by_wall = False
            if row == 0:  # Moving horizontally - check vertical walls
                wall_col = min(old_col, new_col)
                # NOTE: The min and max tricks there are just a lazy way to avoid overindexing
                # Since the wall grid is one smaller than the board (because boards take two spaces)
                if (
                    self.walls[min(new_row, self.wall_size - 1), wall_col, 0] == 1
                    or self.walls[max(new_row - 1, 0), wall_col, 0] == 1
                ):
                    blocked_by_wall = True
            else:  # Moving vertically - check horizontal walls
                wall_row = min(old_row, new_row)
                # NOTE: The min and max tricks there are just a lazy way to avoid overindexing
                # Since the wall grid is one smaller than the board (because boards take two spaces)
                if (
                    self.walls[wall_row, min(new_col, self.wall_size - 1), 1] == 1
                    or self.walls[wall_row, max(new_col - 1, 0), 1] == 1
                ):
                    blocked_by_wall = True

            # Is the opponent in the target position?
            opponent_in_target = self.positions[self._opponent(agent_id)] == (new_row, new_col)

            # TODO: Jumping the opponent, considering walls if there are any

            # Check if the aren't any walls in the way
            if not opponent_in_target and not blocked_by_wall:
                mask[self.action_params_to_index(new_row, new_col, 0)] = 1

        # TODO: Valid wall placements

        return mask

    def _get_info(self):
        # This is for now unused, returning empty dict
        return {}

    def render(self):
        board_str = ""

        for row in range(self.board_size):
            # Render board row with player positions and vertical walls
            for col in range(self.board_size):
                if self.positions["player_0"] == (row, col):
                    board_str += " P0 "
                elif self.positions["player_1"] == (row, col):
                    board_str += " P1 "
                else:
                    board_str += " .  "

                # Render vertical walls (|)
                if col < self.wall_size:
                    if row < self.wall_size and self.walls[row, col, 0]:
                        board_str += "|"
                    elif row > -1 and self.walls[row - 1, col, 0]:  # Continuation of a vertical wall
                        board_str += "|"
                    else:
                        board_str += " "
                else:
                    board_str += " "

            board_str += "\n"

            # Render horizontal walls (───)
            if row < self.wall_size:
                for col in range(self.board_size):
                    if col < self.wall_size and self.walls[row, col, 1]:
                        board_str += "──── "  # Wall segment
                    elif col > -1 and self.walls[row, col - 1, 1]:
                        board_str += "──── "  # Continuation of a wall from the left
                    else:
                        board_str += "     "
                board_str += "\n"

        print(board_str)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict(
            {
                "observation": spaces.Dict(
                    {
                        # For now: 0 = empty, 1 = player 1, 2 = player 2 in a board_size x board_size grid
                        "board": spaces.Box(0, 2, (self.board_size, self.board_size), dtype=np.int8),
                        # For now: 0 = no wall, 1 = wall on a grid of wall_size x wall_size x orientation (0 = vertical, 1 = horizontal)
                        "walls": spaces.Box(0, 1, (self.wall_size, self.wall_size, 2), dtype=np.int8),
                        "my_walls_remaining": spaces.Discrete(self.max_walls + 1),
                        "opponent_walls_remaining": spaces.Discrete(self.max_walls + 1),
                    }
                ),
                "action_mask": spaces.Box(0, 1, (self.board_size**2 + (self.wall_size**2) * 2,), dtype=np.int8),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(self.board_size**2 + (self.wall_size**2) * 2)

    def action_params_to_index(self, row: int, col: int, action_type: int = 0) -> int:
        """
        Takes action parameters (row, col, movement_type) to an action index
        movement_type = 0 for moving, 1 for horizontal wall placement, 2 for vertical wall placement
        """
        if action_type == 0:  # Pawn movement
            return row * self.board_size + col
        elif action_type == 1:  # Placing a vertical wall
            return self.board_size**2 + row * self.wall_size + col
        elif action_type == 2:  # Placing a horizontal wall
            return self.board_size**2 + self.wall_size**2 + row * self.wall_size + col
        else:
            raise ValueError(f"Invalid action type: {action_type}")

    def action_index_to_params(self, idx) -> Tuple[int, int, int]:
        """
        Takes an action index to action parameters (row, col, movement_type)
        movement_type = 0 for moving, 1 for horizontal wall placement, 2 for vertical wall placement
        """
        if idx < self.board_size**2:  # Pawn movement
            action_type = 0
            row, col = divmod(idx, self.board_size)
        elif idx >= self.board_size**2 and idx < self.board_size**2 + self.wall_size**2:
            # Vertical wall placement
            action_type = 1
            row, col = divmod(idx - self.board_size**2, self.wall_size)
        elif idx >= self.board_size**2 + self.wall_size**2 and idx < self.board_size**2 + (self.wall_size**2) * 2:
            # Horizontal wall placement
            action_type = 2
            row, col = divmod(idx - self.board_size**2 - self.wall_size**2, self.wall_size)
        else:
            raise ValueError(f"Invalid action index: {idx}")
        return (row, col, action_type)

    def _move(self, agent, position):
        """
        TODO: Only allow valid moves
        """
        row, col = position
        if (row, col) in self.positions.values():
            print("WTF: Invalid action: Occupied")
            return  # Invalid move (occupied)
        self.positions[agent] = (row, col)

    def _place_wall(self, agent, position, orientation):
        """
        Place a wall at the specified position and orientation.
        Since each wall is of length 2, the wall will coneinue through two spaces.

        TODO: Prevent placing walls conflicting with existing ones
        TODO: Prevent completely blocking the path from one end to the other
        """
        if self.walls_remaining[agent] == 0:
            print("WTF: Invalid action: No walls left")
            return  # No walls left
        (row, col) = position
        if self.walls[row, col, orientation] == 0:
            self.walls[row, col, orientation] = 1  # Mark as wall
            self.walls_remaining[agent] -= 1

    def _check_win(self, agent):
        row, _ = self.positions[agent]
        if agent == "player_0" and row == self.board_size - 1:
            return True
        if agent == "player_1" and row == 0:
            return True
        return False

    def _next_player(self):
        idx = self.agent_order.index(self.agent_selection)
        self.agent_selection = self.agent_order[(idx + 1) % len(self.agent_order)]

    def _opponent(self, agent):
        return "player_1" if agent == "player_0" else "player_1"


# Wrapping the environment for PettingZoo compatibility
def env():
    return wrappers.CaptureStdoutWrapper(QuoridorEnv())
