from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import quoridor_rs

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

from quoridor import (
    ActionEncoder,
    construct_game_from_observation,
    MoveAction,
    Player,
    WallAction,
    WallOrientation,
)
from utils import SubargsBase

from agents.core import Agent


@dataclass
class PolicyDBParams(SubargsBase):
    db_path: str = "policy_db.sqlite"
    nick: Optional[str] = None


class PolicyDBAgent(Agent):
    """Agent that looks up precomputed minimax values from a SQLite policy database."""

    def __init__(self, params=PolicyDBParams(), **kwargs):
        super().__init__()
        if not RUST_AVAILABLE:
            raise RuntimeError(
                "Rust extension (quoridor_rs) not available. Install with: cd rust && maturin develop --release"
            )
        self.params = params
        self.board_size = kwargs["board_size"]
        self.max_walls = kwargs["max_walls"]
        self.max_steps = kwargs["max_steps"]
        self.action_encoder = ActionEncoder(self.board_size)

    @classmethod
    def params_class(cls):
        return PolicyDBParams

    def name(self):
        if self.params.nick:
            return self.params.nick
        return "policydb"

    def get_action(self, observation) -> int:
        action_mask = observation["action_mask"]
        obs = observation["observation"]

        game, _, _ = construct_game_from_observation(obs)

        grid = game.board._grid
        player_positions = np.zeros((2, 2), dtype=np.int32)
        player_positions[0] = game.board.get_player_position(Player.ONE)
        player_positions[1] = game.board.get_player_position(Player.TWO)
        walls_remaining = np.zeros(2, dtype=np.int32)
        walls_remaining[0] = game.board.get_walls_remaining(Player.ONE)
        walls_remaining[1] = game.board.get_walls_remaining(Player.TWO)
        current_player = int(game.get_current_player())
        completed_steps = game.completed_steps

        result = quoridor_rs.policy_db_lookup(
            grid,
            player_positions,
            walls_remaining,
            current_player,
            completed_steps,
            self.board_size,
            self.max_walls,
            self.max_steps,
            self.params.db_path,
        )

        if result is None:
            raise RuntimeError("Unable to find action for state")
        else:
            actions, values = result

            # Choose randomly from among the best actions
            best_value = np.max(values)
            best_indices = np.where(values == best_value)[0]
            chosen_idx = int(np.random.choice(best_indices))
            r, c, action_type = actions[chosen_idx]

            # Convert to action index
            if action_type == 2:  # move
                action_idx = self.action_encoder.action_to_index(MoveAction((r, c)))
            elif action_type == 0:  # vertical wall
                action_idx = self.action_encoder.action_to_index(WallAction((r, c), WallOrientation.VERTICAL))
            elif action_type == 1:  # horizontal wall
                action_idx = self.action_encoder.action_to_index(WallAction((r, c), WallOrientation.HORIZONTAL))

            if action_mask[action_idx]:
                return action_idx
