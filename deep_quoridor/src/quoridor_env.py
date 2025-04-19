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
from quoridor import ActionEncoder, Board, MoveAction, Player, Quoridor, WallAction, WallOrientation


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

        self._action_encoder = ActionEncoder(board_size)

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
        opponent_player = self.agent_to_player[self.get_opponent(agent)]
        if self.terminations[agent]:
            self._next_player()
            self.game.go_to_next_player()
            return

        if self.last_action_mask[agent][action_index] != 1:
            raise RuntimeError(f"Action not allowed by mask {action_index}")

        # If the environment and game get out of step, weird things happen.
        assert player == self.game.get_current_player()

        action = self._action_encoder.index_to_action(action_index)
        self.game.step(action)

        if self.game.check_win(player):
            self.terminations = {a: True for a in self.agents}
            self.rewards[agent] = 1
            for a in self.agents:
                if a != agent:
                    self.rewards[a] = -1
        elif self.step_rewards:
            # Assign rewards as the difference in distance to the goal divided by
            # three times the board size.
            position = self.game.board.get_player_position(player)
            agent_distance = self.game.distance_to_target(position, self.game.get_goal_row(player), False)
            opponent_position = self.game.board.get_player_position(opponent_player)
            oponent_distance = self.game.distance_to_target(
                opponent_position, self.game.get_goal_row(opponent_player), False
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
        opponent = self.agent_to_player[self.get_opponent(agent_id)]
        # TODO: Do we need to make copies of the state or can we return references directly?

        # Calculate board from self.positions
        # NOTE: The board uses 1 to indicate where the agent is and 2 to indicate where the opponent is
        # Obviously, this will be different for each player
        board = np.zeros((self.game.board.board_size, self.game.board.board_size), dtype=np.int8)
        player_row, player_col = self.game.board.get_player_position(player)
        opponent_row, opponent_col = self.game.board.get_player_position(opponent)
        board[player_row, player_col] = 1
        board[opponent_row, opponent_col] = 2

        # Make a copy of walls
        walls = self.game.board.get_old_style_walls()

        return {
            "my_turn": self.agent_selection == agent_id,
            "board": board,
            "walls": walls,
            "my_walls_remaining": self.game.board.get_walls_remaining(player),
            "opponent_walls_remaining": self.game.board.get_walls_remaining(opponent),
        }

    def _get_action_mask(self, agent_id):
        # Start with an empty mask (nothing possible)
        player = self.agent_to_player[agent_id]
        action_mask = np.zeros((self.board_size**2 + (self.wall_size**2) * 2,), dtype=np.int8)

        # Mark valid moves
        for action in self.game.get_valid_actions(player):
            action_i = self._action_encoder.action_to_index(action)
            action_mask[action_i] = 1

        self.last_action_mask[agent_id] = action_mask
        return action_mask

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

    def winner(self) -> Optional[int]:
        """
        Return the index of the winner (0 for player 1, 1 for player 2) or None if there's no winner
        """
        for idx, agent in enumerate(self.agent_order):
            player = self.agent_to_player[agent]
            if self.game.check_win(player):
                return idx

        return None

    def _next_player(self):
        idx = self.agent_order.index(self.agent_selection)
        self.agent_selection = self.agent_order[(idx + 1) % len(self.agent_order)]

    def get_opponent(self, agent):
        return "player_1" if agent == "player_0" else "player_0"


# Wrapping the environment for PettingZoo compatibility
def env(**kwargs):
    # Extract render_mode from kwargs with a default of "human"
    render_mode = kwargs.get("render_mode", "human")

    if render_mode == "human":
        return wrappers.CaptureStdoutWrapper(QuoridorEnv(**kwargs))
    else:
        return QuoridorEnv(**kwargs)
