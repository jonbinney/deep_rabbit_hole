from pettingzoo import AECEnv
from gymnasium import spaces
import numpy as np


class QuoridorEnv(AECEnv):
    metadata = {"render_modes": ["ansi"], "name": "quoridor_v0"}

    def __init__(self):
        super().__init__()

        self.render_mode = "ansi"

        self.board_size = 9  # 9x9 grid
        self.wall_size = self.board_size - 1  # 8x8 grid for walls
        self.max_walls = 10  # Each player gets 10 walls

        self.possible_agents = ["player_1", "player_2"]
        self.agent_order = self.possible_agents.copy()
        self.agent_selection = self.agent_order[0]

        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "board": spaces.Box(0, 2, (self.board_size, self.board_size), dtype=np.int8),
                    "walls": spaces.Box(0, 1, (self.wall_size, self.wall_size), dtype=np.int8),
                    "walls_remaining": spaces.Discrete(self.max_walls + 1),
                }
            )
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Discrete(self.board_size**2 + (self.wall_size**2) * 2) for agent in self.possible_agents
        }

        self.reset()

    def reset(self, seed=None, options=None):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.walls = np.zeros((self.wall_size, self.wall_size, 2), dtype=np.int8)
        self.walls_remaining = {agent: self.max_walls for agent in self.possible_agents}
        # Positions are (row, col)
        self.positions = {"player_1": (4, 0), "player_2": (4, 8)}
        self.board[self.positions["player_1"]] = 1
        self.board[self.positions["player_2"]] = 2

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

        if action < self.board_size**2:
            self._move(agent, action)
        else:
            self._place_wall(agent, action)

        if self._check_win(agent):
            self.terminations = {a: True for a in self.possible_agents}
            self.rewards[agent] = 1
            self.rewards[self._opponent(agent)] = -1

        self._next_player()

    def rowcol_to_idx(self, row, col):
        return row * self.board_size + col

    def idx_to_rowcol(self, idx):
        return divmod(idx, self.board_size)

    def _move(self, agent, action):
        """
        TODO: Only allow valid moves
        """
        row, col = self.idx_to_rowcol(action)
        if (row, col) in self.positions.values():
            return  # Invalid move (occupied)
        self.board[self.positions[agent]] = 0
        self.positions[agent] = (row, col)
        self.board[row, col] = 1 if agent == "player_1" else 2

    def _place_wall(self, agent, action):
        """
        Place a wall.
        In a 9x9 board:
        - First we subtract 81 to get the wall index
        - The first 8x8 = 64 values reprsent vertical walls, the last 8x8 = 64 values represent horizontal walls
        - For vertical walls, we place the wall starting atht upper right corner of the (row, col) cell
        - For horizontal walls, we place the wall starting at the lower left corner of the (row, col) cell

        TODO: Prevent placing walls conflicting with existing ones
        TODO: Prevent completely blocking the path from one end to the other
        """
        if self.walls_remaining[agent] == 0:
            return  # No walls left
        wall_idx = action - self.board_size**2
        orientation, wall_idx = divmod(wall_idx, self.wall_size**2)
        row, col = divmod(wall_idx, self.wall_size)
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
                if row < self.wall_size and col < self.wall_size:
                    if self.walls[row, col, 0]:
                        board_str += "|"
                    elif row > 0 and self.walls[row - 1, col, 0]:  # Continuation of a vertical wall
                        board_str += "|"
                    else:
                        board_str += " "

            board_str += "\n"

            # Render horizontal walls (───)
            if row < self.wall_size:
                for col in range(self.wall_size):
                    if col < self.wall_size and self.walls[row, col, 1]:
                        board_str += "───"  # Wall segment
                    elif col > 0 and self.walls[row, col - 1, 1]:
                        board_str += "───"  # Continuation of a wall from the left
                    else:
                        board_str += "   "
                board_str += "\n"

        print(board_str)


# Wrapping the environment for PettingZoo compatibility
def env():
    return QuoridorEnv()
