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

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from quoridor import Quoridor


class QuoridorEnv(AECEnv):
    metadata = {"render_modes": ["ansi"], "name": "quoridor_v0"}

    def __init__(self, board_size: int, max_walls: int):
        super(AECEnv, self).__init__()
        self.render_mode = "human"
        self.possible_agents = ["player_0", "player_1"]
        self._game = Quoridor(board_size, max_walls, self.possible_agents)

    def reset(self, seed=None, options=None, game=None):
        """
        Reset the environment to the initial state.

        If a game is provided, it will be used as the game state instead.
        """
        if game is None:
            self._game = Quoridor(self._game.board_size, self._game.max_walls, self._game.players)
        else:
            self._game = game

        self.agents = self.possible_agents.copy()
        self.agent_order = self.agents.copy()

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
            self.place_wall(agent, (row, col), action_type - 1)

        if self._check_win(agent):
            self.terminations = {a: True for a in self.agents}
            self.rewards[agent] = 1
            self.rewards[self.get_opponent(agent)] = -1

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
        board[self.positions[self.get_opponent(agent_id)]] = 2

        # Make a copy of walls
        walls = self.walls.copy()

        return {
            "game": self._game,
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
            if not self._is_in_board(new_row, new_col):
                continue

            # Check if that direction is blocked by a wall
            if self.is_wall_between(old_row, old_col, new_row, new_col):
                continue

            # Check if the opponent is in the target position
            if self.positions[self.get_opponent(agent_id)] == (new_row, new_col):
                # Can we jump straight over the opponent?
                straight_row = new_row + row
                straight_col = new_col + col

                if (
                    # That new position falls within the board
                    self._is_in_board(straight_row, straight_col)
                    # That new position is not blocked by a wall behind them
                    and not self.is_wall_between(new_row, new_col, straight_row, straight_col)
                ):
                    mask[self.action_params_to_index(straight_row, straight_col, 0)] = 1
                    continue
                else:
                    # Try the diagonals, to the sides of the oponent
                    for side in [1, -1]:
                        if row == 0:
                            diag_row = new_row + side
                            diag_col = new_col
                        else:
                            diag_row = new_row
                            diag_col = new_col + side

                        if (
                            # That new position falls within the board
                            self._is_in_board(diag_row, diag_col)
                            # That new position is not blocked by a wall
                            and not self.is_wall_between(new_row, new_col, diag_row, diag_col)
                        ):
                            mask[self.action_params_to_index(diag_row, diag_col, 0)] = 1
                    continue

            # If we get to this point it's because the target is within the board,
            # not blocked by a wall and the opponent is not on the way
            mask[self.action_params_to_index(new_row, new_col, 0)] = 1

        # Calculate wall placements
        # Iterate over all possible wall positions, only of there are any left to place
        if self.walls_remaining[agent_id] > 0:
            for row, col, orientation in np.ndindex(self.walls.shape):
                if self._is_wall_overlap(row, col, orientation):
                    continue

                if not self._is_wall_potential_block(row, col, orientation) or self._can_place_wall_without_blocking(
                    row, col, orientation
                ):
                    mask[self.action_params_to_index(row, col, orientation + 1)] = 1

        return mask

    def _is_wall_potential_block(self, row, col, orientation):
        # Area to find other walls that may touch this wall
        top = max(0, row - 1)
        bottom = min(self.wall_size - 1, row + 1)
        left = max(0, col - 1)
        right = min(self.wall_size - 1, col + 1)

        # Whether the wall was touched in the left/top, middle, right/bottom
        touches = [False, False, False]

        if orientation == 1:  # Horizontal
            # On the left border or touching another horizontal wall on the left side
            if (col == 0) or (col >= 2 and self.walls[row, col - 2, 1] == 1):
                touches[0] = True

            # On the right border or touching another horizontal wall on the right sinde
            if (col == self.wall_size) - 1 or (col < self.wall_size - 2 and self.walls[row, col + 2, 1] == 1):
                touches[2] = True

            # Check for vertical walls touching it
            for r in range(top, bottom + 1):
                for c in range(left, right + 1):
                    if self.walls[r, c, 0]:
                        touches[c - left] = True
        else:  # Vertical
            # On the top border or touching another verticall wall on top
            if (row == 0) or (row >= 2 and self.walls[row - 2, col, 0] == 1):
                touches[0] = True

            # On the bottom border or touching another verticall wall on the bottom
            if (row == self.wall_size - 1) or (row < self.wall_size - 2 and self.walls[row + 2, col, 0] == 1):
                touches[2] = True

            # Check for horizontal walls touching it
            for r in range(top, bottom + 1):
                for c in range(left, right + 1):
                    if self.walls[r, c, 1]:
                        touches[r - top] = True

        return sum(touches) > 1

    def _is_wall_overlap(self, row, col, orientation):
        """
        Checks whether there is any wall overlaping where this wall would be placed
        """
        # Check if it doesn't collide with a crossing wall
        if self.walls[row, col, (orientation + 1) % 2] == 1:
            return True

        # Check that in the 2 wall segments there's no other segment already
        if orientation == 0:
            if self.is_wall_between(row, col, row, col + 1) or self.is_wall_between(row + 1, col, row + 1, col + 1):
                return True
        else:
            if self.is_wall_between(row, col, row + 1, col) or self.is_wall_between(row, col + 1, row + 1, col + 1):
                return True

        return False

    def _get_info(self):
        # This is for now unused, returning empty dict
        return {}

    def render(self):
        print(self._game.render_to_string())

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
        movement_type = 0 for moving, 1 for vertical wall placement, 2 for horizontal wall placement
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

    def get_opponent(self, agent):
        return "player_1" if agent == "player_0" else "player_0"


# Wrapping the environment for PettingZoo compatibility
def env(board_size: int = 9, max_walls: int = 10):
    return wrappers.CaptureStdoutWrapper(QuoridorEnv(board_size, max_walls))
