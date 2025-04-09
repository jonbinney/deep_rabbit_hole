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
from typing import Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from deep_quoridor.src.quoridor import Action, Board, MoveAction, Player, Quoridor, WallAction, WallOrientation


class QuoridorEnv(AECEnv):
    metadata = {"render_modes": ["ansi"], "name": "quoridor_v0"}

    def __init__(
        self,
        board_size: int = 9,
        max_walls: int = 10,
        step_rewards: bool = False,
        render_mode: str = "human",
        game_start_state: Optional[Quoridor] = None,
        **kwargs,
    ):
        """
        Constructs a Quoridor environment.

        Args:
            board_size (int): Size of the square board (e.g., 9 for a 9x9 board).
            max_walls (int): Maximum number of walls each player can place (e.g., 10).
            step_rewards (bool): Whether to provide (heuristic) incremental rewards.
            game_start_state (Optional[Quoridor]): An optional starting state of the game to initialize the environment. (mainly for testing)
        """
        super(AECEnv, self).__init__()

        self.render_mode = render_mode
        self.step_rewards = step_rewards

        self.board_size = board_size  # assumed square grid
        self.wall_size = self.board_size - 1  # grid for walls
        self.max_walls = max_walls  # Each player gets 10 walls

        self._game_start_state = game_start_state
        if self._game_start_state is not None:
            assert self._game_start_state.board.board_size == self.board_size
            assert self._game_start_state.board.max_walls == self.max_walls

        self.possible_agents = ["player_0", "player_1"]

        # The Quoridor class uses an enum for the players, whereas petting zoo uses strings.
        self.agent_to_player = {"player_0": Player.ONE, "player_1": Player.TWO}
        self.player_to_agent = {Player.ONE: "player_0", Player.TWO: "player_1"}

        self.reset()

    def copy(self):
        return copy.deepcopy(self)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        self.agent_order = self.agents.copy()
        self.last_action_mask = {a: [] for a in self.agents}

        if self._game_start_state is None:
            self.game = Quoridor(Board(self.board_size, self.max_walls))
        else:
            self.game = copy.deepcopy(self._game_start_state)

        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.agent_selection = self.agent_order[0]

        # I created these so that env.last() wouldn't complain, but I don't
        # really understand how should they work or why they are needed
        self._cumulative_rewards = self.rewards.copy()
        self.infos = {agent: {} for agent in self.agents}

        return None

    def step(self, action_index):
        """
        Players move by selecting an index from 0-80 (9x9 board).
        Wall placement is mapped to 81-208 (8x8x2).
        """
        agent = self.agent_selection
        player = self.agent_to_player[agent]
        if self.terminations[agent]:
            self.game.go_to_next_player()
            return

        if self.last_action_mask[agent][action_index] != 1:
            raise RuntimeError(f"Action not allowed by mask {action_index}")

        action = self.action_index_to_params(action_index)
        self.game.step(action)

        if self._check_win(agent):
            self.terminations = {a: True for a in self.agents}
            self.rewards[agent] = 1
            for a in self.agents:
                if a != agent:
                    self.rewards[a] = -1
        elif self.step_rewards:
            # Assign rewards as the difference in distance to the goal divided by
            # three times the board size.
            position = self.game.board.get_player_position(player)
            agent_distance = self.distance_to_target(position, self.game.get_goal_row(player), False)
            opponent_position = self.positions[self.get_opponent(agent)]
            oponent_distance = self.distance_to_target(
                opponent_position, self.game.get_goal_row(self.get_opponent(agent)), False
            )
            self.rewards[agent] = (oponent_distance - agent_distance) / (self.board_size**2)
            self.rewards[self.get_opponent(agent)] = (agent_distance - oponent_distance) / (self.board_size**2)

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
        player = self.agent_to_player[agent_id]
        opponent_player = self.agent_to_player[self.get_opponent(agent_id)]
        # TODO: Do we need to make copies of the state or can we return references directly?

        # Calculate board from self.positions
        # NOTE: The board uses 1 to indicate where the agent is and 2 to indicate where the opponent is
        # Obviously, this will be different for each player
        board = np.zeros((self.game.board.board_size, self.game.board.board_size), dtype=np.int8)
        board[self.game.board.get_player_position(player)] = 1
        board[self.game.board.get_player_position(opponent_player)] = 2

        # Make a copy of walls
        walls = self.game.board.get_old_style_walls()

        return {
            "my_turn": self.agent_selection == agent_id,
            "board": board,
            "walls": walls,
            "my_walls_remaining": self.game.board.get_walls_remaining(player),
            "opponent_walls_remaining": self.game.board.get_walls_remaining(opponent_player),
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
        player = self.agent_to_player[agent_id]
        mask = np.zeros((self.board_size**2 + (self.wall_size**2) * 2,), dtype=np.int8)

        # Calculate valid moves
        current_position = self.game.board.get_player_position(player)
        for delta_row in range(-2, 3):
            for delta_col in range(-2, 3):
                destination = current_position + np.array((delta_row, delta_col))
                if self.game.board.is_position_on_board(destination):
                    move_action = MoveAction(destination)
                    if self.game.is_action_valid(move_action):
                        mask[self.action_params_to_index(move_action)] = 1

        # Calculate valid wall placements
        for row, col in np.ndindex((self.game.board.board_size - 1, self.game.board.board_size - 1)):
            for orientation in [WallOrientation.VERTICAL, WallOrientation.HORIZONTAL]:
                wall_position = np.array((row, col))
                wall_action = WallAction(wall_position, orientation)
                if self.game.is_action_valid(wall_action):
                    mask[self.action_params_to_index(wall_action)] = 1

        self.last_action_mask[agent_id] = mask
        return mask

    def _get_info(self):
        # This is for now unused, returning empty dict
        return {}

    def render(self):
        print(str(self.game))

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

    def action_params_to_index(self, action) -> int:
        """
        Converts an action object to an action index
        """
        if isinstance(action, MoveAction):
            return action.destination[0] * self.board_size + action.destination[1]
        elif isinstance(action, WallAction) and action.orientation == WallOrientation.VERTICAL:
            return self.board_size**2 + action.position[0] * self.wall_size + action.position[1]
        elif isinstance(action, WallAction) and action.orientation == WallOrientation.HORIZONTAL:
            return self.board_size**2 + self.wall_size**2 + action.position[0] * self.wall_size + action.position[1]
        else:
            raise ValueError(f"Invalid action type: {action}")

    def action_index_to_params(self, idx) -> Action:
        """
        Converts an action index to an action object.
        """
        action = None
        if idx < self.board_size**2:  # Pawn movement
            action = MoveAction(np.array(divmod(idx, self.board_size)))
        elif idx >= self.board_size**2 and idx < self.board_size**2 + self.wall_size**2:
            action = WallAction(np.array(divmod(idx - self.board_size**2, self.wall_size)), WallOrientation.VERTICAL)
        elif idx >= self.board_size**2 + self.wall_size**2 and idx < self.board_size**2 + (self.wall_size**2) * 2:
            action = WallAction(
                np.array(divmod(idx - self.board_size**2 - self.wall_size**2, self.wall_size)),
                WallOrientation.HORIZONTAL,
            )
        else:
            raise ValueError(f"Invalid action index: {idx}")

        return action

    def _check_win(self, agent):
        player = self.agent_to_player[agent]
        row, _ = self.game.board.get_player_position(player)
        return row == self.game.get_goal_row(player)

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
                self.game.is_position_on_board(new_row, new_col)
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


# Wrapping the environment for PettingZoo compatibility
def env(**kwargs):
    # Extract render_mode from kwargs with a default of "human"
    render_mode = kwargs.get("render_mode", "human")

    if render_mode == "human":
        return wrappers.CaptureStdoutWrapper(QuoridorEnv(**kwargs))
    else:
        return QuoridorEnv(**kwargs)
