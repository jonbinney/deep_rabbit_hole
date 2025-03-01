from pettingzoo import AECEnv
from gymnasium import spaces
import numpy as np


class QuoridorEnv(AECEnv):
    metadata = {"render_modes": ["ansi"], "name": "quoridor_v0"}

    def __init__(self):
        super().__init__()

        self.render_mode = "ansi"

        self.board_size = 9  # 9x9 grid
        self.wall_size = 8  # 8x8 grid for walls
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
        self.walls = np.zeros((self.wall_size, self.wall_size), dtype=np.int8)
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
        row, col = self.idx_to_rowcol(action)
        if (row, col) in self.positions.values():
            return  # Invalid move (occupied)
        self.board[self.positions[agent]] = 0
        self.positions[agent] = (row, col)
        self.board[row, col] = 1 if agent == "player_1" else 2

    def _place_wall(self, agent, action):
        if self.walls_remaining[agent] == 0:
            return  # No walls left
        wall_idx = action - self.board_size**2
        row, col = divmod(wall_idx // 2, self.wall_size)
        if self.walls[row, col] == 0:
            self.walls[row, col] = 1  # Mark as wall
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
        # TODO: Render walls
        board_str = "\n".join(
            [" ".join(str(self.board[row, col]) for col in range(self.board_size)) for row in range(self.board_size)]
        )
        return board_str


# Wrapping the environment for PettingZoo compatibility
def env():
    return QuoridorEnv()
