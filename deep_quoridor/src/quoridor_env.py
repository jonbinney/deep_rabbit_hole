"""
Quoridor Gym/PettingZoo Environment implementation.

The board size and number of walls are parameterizable:
 - board_size (int): size of the square board
 - max_walls (int): maximum number of walls that each player can place

The state is a dictionary containing:
 - "board": A board_size x board_size 2D array, representing the positions of the players.
            0 indicates an empty space, 1 indicates the player's position, and 2 indicates the opponent's position.
 - "walls": A (board_size - 1) x (board_size - 1) x 2 array, representing the vertical and horizontal walls placed on the board.
            The first channel (index 0) represents vertical walls, and the second channel (index 1) represents horizontal walls.
            A value of 1 indicates the presence of a wall.
 - "my_walls_remaining": An integer representing the number of walls remaining for the player
 - "opponent_walls_remaining": An integer representing the number of walls remaining for the opponent.
 - "my_turn": A boolean indicating whether it is the player's turn

The action space is a Discrete space with size board_size**2 + 2*(board_size-1)**2
Actions are represented as follows:
- The first board_size**2 actions represent moving to a specific cell on the board.
- The next (board_size - 1)**2 actions represent placing a vertical wall.
- The final (board_size - 1)**2 actions represent placing a horizontal wall.

The environment uses 0-based indexing for rows and columns, with (0, 0) representing the top-left corner of the board.
"""

import copy
import functools
import random
from typing import Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers


class QuoridorEnv(AECEnv):
    metadata = {"render_modes": ["ansi"], "name": "quoridor_v0"}

    def __init__(self, board_size: int, max_walls: int, step_rewards: bool):
        super(AECEnv, self).__init__()

        self.render_mode = "human"
        self.step_rewards = step_rewards

        self.board_size = board_size  # assumed square grid
        self.wall_size = self.board_size - 1  # grid for walls
        self.max_walls = max_walls  # Each player gets 10 walls

        self.possible_agents = ["player_0", "player_1"]

        self.reset()

    def copy(self):
        """
        A deep copy is not very efficient; eventually we should break out the game logic and state
        into a separate class that can be used by agents during planning
        """
        return copy.deepcopy(self)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        self.agent_order = self.agents.copy()
        self.last_action_mask = {a: [] for a in self.agents}

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

        if self.last_action_mask[agent][action] != 1:
            raise RuntimeError(f"Action not allowed by mask {action}")

        (row, col, action_type) = self.action_index_to_params(action)
        if action_type == 0:
            self._move(agent, (row, col))
        else:
            self.place_wall(agent, (row, col), action_type - 1)

        if self._check_win(agent):
            self.terminations = {a: True for a in self.agents}
            self.rewards[agent] = 1
            self.rewards[self.get_opponent(agent)] = -1
        elif self.step_rewards:
            # Assign rewards as the difference in distance to the goal divided by
            # three times the board size.
            (row, col) = self.positions[agent]
            agent_distance = self.distance_to_target(row, col, self.get_goal_row(agent), False)
            (row, col) = self.positions[self.get_opponent(agent)]
            self.rewards[agent] = self.board_size - agent_distance / self.board_size
            self.rewards[self.get_opponent(agent)] = 0

        # TODO: Confirm if this is needed and if it's doing anything
        self._accumulate_rewards()

        self._next_player()

    def is_done(self):
        """
        Returns True if the game is done
        """
        return any(self.terminations.values())

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
            "my_turn": self.agent_selection == agent_id,
            "board": board,
            "walls": walls,
            "my_walls_remaining": self.walls_remaining[agent_id],
            "opponent_walls_remaining": self.walls_remaining[self.get_opponent(agent_id)],
        }

    def is_wall_between(self, row_a, col_a, row_b, col_b):
        """
        Returns True if there is a wall between pos_a and pos_b
        NOTE: The min and max tricks there are just a lazy way to avoid overindexing
        Since the wall grid is one smaller than the board (because boards take two spaces)
        """
        if row_a == row_b:  # Horizontal movement - check vertical walls
            wall_col = min(col_a, col_b)
            return (
                self.walls[min(row_a, self.wall_size - 1), wall_col, 0] == 1
                or self.walls[max(row_a - 1, 0), wall_col, 0] == 1
            )
        elif col_a == col_b:  # Vertical movement - check horizontal walls
            wall_row = min(row_a, row_b)
            return (
                self.walls[wall_row, min(col_a, self.wall_size - 1), 1] == 1
                or self.walls[wall_row, max(col_a - 1, 0), 1] == 1
            )
        else:
            raise ValueError(f"Invalid movement from {row_a, col_a} to {row_b, col_b}")

    def is_in_board(self, row, col):
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def _get_action_mask(self, agent_id):
        # Start with an empty mask (nothing possible)
        mask = np.zeros((self.board_size**2 + (self.wall_size**2) * 2,), dtype=np.int8)

        # Calculate valid "moves" (as opposed to wall placements)
        # Start with the four basic directions
        for row, col in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            old_row, old_col = self.positions[agent_id]
            new_row, new_col = old_row + row, old_col + col

            # Check if the new position is still inside the board
            if not self.is_in_board(new_row, new_col):
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
                    self.is_in_board(straight_row, straight_col)
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
                            self.is_in_board(diag_row, diag_col)
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

        self.last_action_mask[agent_id] = mask
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
            if (col == self.wall_size - 1) or (col < self.wall_size - 2 and self.walls[row, col + 2, 1] == 1):
                touches[2] = True

            # Check for vertical walls touching it
            for r in range(top, bottom + 1):
                for c in range(left, right + 1):
                    if self.walls[r, c, 0]:
                        touches[c - col + 1] = True
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
                        touches[r - row + 1] = True

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
                        board_str += "────+"  # Wall segment
                    elif col > 0 and self.walls[row, col - 1, 1]:
                        board_str += "──── "  # Continuation of a wall from the left
                    elif col < self.wall_size and self.walls[row, col, 0]:
                        board_str += "    +"  # Vertical wall going through this intersection
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
                        "my_turn": spaces.Discrete(2),
                        # For now: 0 = empty, 1 = player 1, 2 = player 2 in a board_size x board_size grid
                        "board": spaces.Box(0, 2, (self.board_size, self.board_size), dtype=np.int8),
                        # For now: 0 = no wall, 1 = wall on a grid of wall_size x wall_size x orientation (0 = vertical, 1 = horizontal)
                        "walls": spaces.Box(0, 1, (self.wall_size, self.wall_size, 2), dtype=np.int8),
                        "my_walls_remaining": spaces.Discrete(self.max_walls + 1),
                        "opponent_walls_remaining": spaces.Discrete(self.max_walls + 1),
                    },
                    seed=random.randint(0, 2**32 - 1),
                ),
                "action_mask": spaces.Box(0, 1, (self.board_size**2 + (self.wall_size**2) * 2,), dtype=np.int8),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(self.board_size**2 + (self.wall_size**2) * 2, seed=random.randint(0, 2**32 - 1))

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

    def place_wall(self, agent, position, orientation):
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
        return row == self.get_goal_row(agent)

    def winner(self) -> Optional[int]:
        """
        Return the index of the winner (0 for player 1, 1 for player 2) or None if there's no winner
        """
        for idx, agent in enumerate(self.agent_order):
            if self._check_win(agent):
                return idx

        return None

    def _next_player(self):
        idx = self.agent_order.index(self.agent_selection)
        self.agent_selection = self.agent_order[(idx + 1) % len(self.agent_order)]

    def get_opponent(self, agent):
        return "player_1" if agent == "player_0" else "player_0"

    def get_goal_row(self, agent):
        return 0 if agent == "player_1" else self.board_size - 1

    def _dfs(self, row, col, target_row, visited, any_path=True):
        """
        Performs a depth-first search to find whether the pawn can reach the target row.

        Args:
            row (int): The current row of the pawn
            col (int): The current column of the pawn
            target_row (int): The target row to reach
            visited (numpy.array): A 2D boolean array with the same shape as the board,
                indicating which positions have been visited
            If any_path is set to true, the first path to the target row will be returned (faster).
            Otherwise, the shortest path will be returned (potentially slower)

        Returns:
            int: Number of steps to reach the target or -1 if it's unreachable
        """
        if row == target_row:
            return 0

        visited[row, col] = True

        # Find out the forward direction to try it first and maybe get to the target faster
        fwd = 1 if target_row > row else -1

        moves = [(row + fwd, col), (row, col - 1), (row, col + 1), (row - fwd, col)]
        best = -1
        for new_row, new_col in moves:
            if (
                self.is_in_board(new_row, new_col)
                and not self.is_wall_between(row, col, new_row, new_col)
                and not visited[new_row, new_col]
            ):
                dfs = self._dfs(new_row, new_col, target_row, visited)
                if dfs != -1:
                    if any_path:
                        return dfs + 1
                    if best == -1 or dfs + 1 < best:
                        best = dfs + 1

        return best

    def can_reach(self, row, col, target_row):
        return self.distance_to_target(row, col, target_row, True) != -1

    def distance_to_target(self, row, col, target_row, any_path=False):
        """
        Returns the approximate number of moves it takes to reach the target row, or -1 if it's not reachable.
        If any_path is set to true, the first path to the target row will be returned (faster).
        Otherwise, the shortest path will be returned (potentially slower)
        """
        visited = np.zeros((self.board_size, self.board_size), dtype="bool")
        return self._dfs(row, col, target_row, visited, any_path)

    def _can_place_wall_without_blocking(self, row, col, orientation):
        """
        Returns whether a wall can be placed in the specified coordinates an orientations such that
        both pawns can still reach the target row
        """

        # Temporarily place the wall so that we can easily check for walls
        previous = self.walls[row, col, orientation]
        self.walls[row, col, orientation] = 1
        result = all(self.can_reach(*self.positions[agent], self.get_goal_row(agent)) for agent in self.agents)
        self.walls[row, col, orientation] = previous

        return result


# Wrapping the environment for PettingZoo compatibility
def env(board_size: int = 9, max_walls: int = 10, step_rewards: bool = False):
    return wrappers.CaptureStdoutWrapper(QuoridorEnv(board_size, max_walls, step_rewards))
