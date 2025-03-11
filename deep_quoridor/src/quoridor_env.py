"""
Quoridor Gym/PettingZoo Environment implementation.

The action is represented as follows, taking a 9x9 board as an example:
- First 9x9 = 81 values represent player positions
- The next 8x8 = 64 values reprsent vertical walls and the last 8x8 = 64 values represent horizontal walls
- The cell position of a wall represnts the first space occupied by a wall, knowing that each cell takes two spaces.
- For vertical walls, we place the wall starting at the upper right corner of the (row, col) cell
- For horizontal walls, we place the wall starting at the lower left corner of the (row, col) cell
"""

from typing import Tuple
from pettingzoo import AECEnv
from gymnasium import spaces
import numpy as np


class QuoridorEnv(AECEnv):
    metadata = {"render_modes": ["ansi"], "name": "quoridor_v0"}

    def __init__(self, board_size: int = 9, max_walls: int = 10):
        super().__init__()

        self.render_mode = "ansi"

        self.board_size = board_size  # assumed square grid
        self.wall_size = self.board_size - 1  # grid for walls
        self.max_walls = max_walls  # Each player gets 10 walls

        self.possible_agents = ["player_1", "player_2"]
        self.agent_order = self.possible_agents.copy()
        self.agent_selection = self.agent_order[0]

        # NOTE: Observation and action spaces are only meant as metadata, they don't hold actual state
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    # For now: 0 = empty, 1 = player 1, 2 = player 2 in a board_size x board_size grid
                    "board": spaces.Box(0, 2, (self.board_size, self.board_size), dtype=np.int8),
                    # For now: 0 = no wall, 1 = wall on a grid of wall_size x wall_size x orientation (0 = vertical, 1 = horizontal)
                    "walls": spaces.Box(0, 1, (self.wall_size, self.wall_size, 2), dtype=np.int8),
                    "walls_remaining": spaces.Discrete(self.max_walls + 1),
                    "action_mask": spaces.Box(0, 1, (self.board_size**2 + (self.wall_size**2) * 2,), dtype=np.int8),
                }
            )
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Discrete(self.board_size**2 + (self.wall_size**2) * 2) for agent in self.possible_agents
        }

        self.reset()

    def reset(self, seed=None, options=None):
        self.walls = np.zeros((self.wall_size, self.wall_size, 2), dtype=np.int8)
        self.walls_remaining = {agent: self.max_walls for agent in self.possible_agents}
        # Positions are (row, col)
        self.positions = {"player_1": (4, 0), "player_2": (4, 8)}

        self.rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.agent_selection = self.agent_order[0]

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
            self.terminations = {a: True for a in self.possible_agents}
            self.rewards[agent] = 1
            self.rewards[self._opponent(agent)] = -1

        self._next_player()

    def action_params_to_index(self, row: int, col: int, action_type: int = 0) -> int:
        """
        Takes action parameters (row, col, movement_type) to an action index
        movement_type = 0 for moving, 1 for horizontal wall placement, 2 for vertical wall placement
        """
        if action_type == 0:
            return row * self.board_size + col
        elif action_type == 1:
            return self.board_size**2 + row * self.wall_size + col
        elif action_type == 2:
            return self.board_size**2 + self.wall_size**2 + row * self.wall_size + col
        else:
            raise ValueError(f"Invalid action type: {action_type}")

    def action_index_to_params(self, idx) -> Tuple[int, int, int]:
        """
        Takes an action index to action parameters (row, col, movement_type)
        movement_type = 0 for moving, 1 for horizontal wall placement, 2 for vertical wall placement
        """
        if idx < self.board_size**2:
            action_type = 0
            row, col = divmod(idx, self.board_size)
        elif idx >= self.board_size**2 and idx < self.board_size**2 + self.wall_size**2:
            action_type = 1
            row, col = divmod(idx - self.board_size**2, self.wall_size)
        elif idx >= self.board_size**2 + self.wall_size**2 and idx < self.board_size**2 + (self.wall_size**2) * 2:
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
        if agent == "player_1" and row == 8:
            return True
        if agent == "player_2" and row == 0:
            return True
        return False

    def _next_player(self):
        idx = self.agent_order.index(self.agent_selection)
        self.agent_selection = self.agent_order[(idx + 1) % len(self.agent_order)]

    def _opponent(self, agent):
        return "player_2" if agent == "player_1" else "player_1"

    def render(self):
        board_str = ""

        for row in range(self.board_size):
            # Render board row with player positions and vertical walls
            for col in range(self.board_size):
                if self.positions["player_1"] == (row, col):
                    board_str += " P1 "
                elif self.positions["player_2"] == (row, col):
                    board_str += " P2 "
                else:
                    board_str += " .  "

                # Render vertical walls (|)
                if col < self.wall_size:
                    if row < self.wall_size and self.walls[row, col, 0]:
                        board_str += "|"
                    elif row > 0 and self.walls[row - 1, col, 0]:  # Continuation of a vertical wall
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
                    elif col > 0 and self.walls[row, col - 1, 1]:
                        board_str += "──── "  # Continuation of a wall from the left
                    else:
                        board_str += "     "
                board_str += "\n"

        print(board_str)

    def _can_place_wall_without_blocking(self, row, col, orientation):
        """
        Returns whether a wall can be placed in the specified coordinates an orientations such that
        both pawns can still reach the target row
        """

        def can_reach(row, col, target_row):
            def dfs(row, col, target_row, visited):
                if row == target_row:
                    return True

                if visited[row, col]:
                    return False

                visited[row, col] = True

                # Find out the forward direction to try it first and maybe get to the target faster
                fwd = 1 if target_row > row else -1

                moves = [(row + fwd, col), (row, col - 1), (row, col + 1), (row - fwd, col)]
                for new_row, new_col in moves:
                    if (
                        self._is_in_board(new_row, new_col)
                        and not self._is_wall_between(row, col, new_row, new_col)
                        and dfs(new_row, new_col, target_row, visited)
                    ):
                        return True

                return False

            visited = np.zeros((self.board_size, self.board_size), dtype="bool")
            return dfs(row, col, target_row, visited)

        # Temporarily place the wall so that we can easily check for walls
        previous = self.walls[row, col, orientation]
        self.walls[row, col, orientation] = 1
        result = can_reach(*self.positions["player_0"], self.board_size - 1) and can_reach(
            *self.positions["player_1"], 0
        )
        self.walls[row, col, orientation] = previous

        return result


# Wrapping the environment for PettingZoo compatibility
def env():
    return QuoridorEnv()
