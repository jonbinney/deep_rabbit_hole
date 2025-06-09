import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import model as az_model

from agents.core import TrainableAgent, TrainableAgentParams, rotation


@dataclass
class AlphaZeroOSParams(TrainableAgentParams):
    """Parameters for AlphaZero OpenSpiel Agent."""

    # Just used to display a user friendly name
    nick: Optional[str] = None

    # Path to the checkpoint
    checkpoint_path: str = "models/osaz/checkpoint--1"

    # UCT exploration constant
    uct_c: float = 2.0

    # Maximum number of simulations for MCTS
    max_simulations: int = 100

    # Random seed
    seed: int = 42


class AlphaZeroOSAgent(TrainableAgent):
    """Agent that uses a trained AlphaZero model from OpenSpiel to play Quoridor."""

    def __init__(self, params=AlphaZeroOSParams(), **kwargs):
        super().__init__(params=params, **kwargs)
        self.params = params
        self.board_size = kwargs["board_size"]
        self.max_walls = kwargs["max_walls"]
        self.action_space = kwargs["action_space"]

        # Player ID (set in start_game)
        self.player_id = None

        # Create the OpenSpiel game
        self.game_string = f"quoridor(board_size={self.board_size},players=2,wall_count={self.max_walls})"
        self.game = pyspiel.load_game(
            "quoridor", {"board_size": self.board_size, "players": 2, "wall_count": self.max_walls}
        )

        # Initialize the model and MCTS bot
        self.model = None
        self.bot = None

        # Internal game state tracking
        self.state = None

        # Load the model if the checkpoint path exists
        if self.params.checkpoint_path is not None:
            if not os.path.exists(f"{self.params.checkpoint_path}.index"):
                raise FileNotFoundError(f"Checkpoint file {self.params.checkpoint_path} not found")
            self.load_model()

    @classmethod
    def params_class(cls):
        return AlphaZeroOSParams

    def name(self):
        if self.params.nick:
            return self.params.nick
        return "alphazero_os"

    def load_model(self):
        """Load the AlphaZero model from checkpoint."""
        try:
            # Load the model from checkpoint
            self.model = az_model.Model.from_checkpoint(self.params.checkpoint_path)

            # Create the AlphaZero evaluator with the loaded model
            evaluator = az_evaluator.AlphaZeroEvaluator(self.game, self.model)

            # Create the MCTS bot with the AlphaZero evaluator
            self.bot = mcts.MCTSBot(
                self.game,
                self.params.uct_c,
                self.params.max_simulations,
                evaluator,
                random_state=np.random.RandomState(self.params.seed),
                solve=False,
                verbose=False,
            )
            print(f"Successfully loaded AlphaZero model from {self.params.checkpoint_path}")
        except Exception as e:
            raise ValueError(f"Error loading AlphaZero model: {e}")

    def start_game(self, game, player_id):
        """Reset the internal state when a new game starts."""
        self.player_id = player_id
        # Initialize a fresh game state
        self.state = self.game.new_initial_state()

    def end_game(self, game):
        """Clean up when the game ends."""
        self.state = None

    def _convert_gym_action_to_openspiel(self, action_idx, observation=None):
        """Convert a gym action index to an OpenSpiel action ID.

        This needs to map from the gym environment's action space to
        OpenSpiel's action IDs for the current state.

        NOTE: The action and observations are always from the current player's perspective
        """
        # The grid size in OpenSpiel is board_size * 2 - 1. Starts with cell and interleaves walls
        os_board_size = self.board_size * 2 - 1
        # The vertical / row stride in this case is then two rows of size board_size - 1
        os_row_stride = os_board_size * 2
        # On the column side it only needs to skip wall positions
        os_col_stride = 2

        if action_idx < self.board_size**2:
            # Move action
            # Calculate difference of -1, 0 or +1 with respect to previous position
            new_row = action_idx // self.board_size
            new_col = action_idx % self.board_size
            (cur_row, cur_col) = np.where(observation["board"] == 1)
            row_dif = new_row - cur_row
            col_dif = new_col - cur_col

            # If there was a straight jump, open spiel represents it as a single move
            if abs(row_dif) == 2:
                row_dif = row_dif // 2
            if abs(col_dif) == 2:
                col_dif = col_dif // 2

            # The movement is always represented with the top left of the board, to make it relative
            os_action = (1 + row_dif) * os_row_stride + (1 + col_dif) * os_col_stride

        else:
            # It's a wall placement

            # Shift by the piece move action space
            wall_idx = action_idx - self.board_size**2
            # Then identify if vertical or horizontal and shift by the vertical wall action space if needed
            if wall_idx < (self.board_size - 1) ** 2:
                row = wall_idx // (self.board_size - 1)
                col = wall_idx % (self.board_size - 1)
                os_action = 1 + (row * os_row_stride) + (col * os_col_stride)
            else:
                wall_idx -= (self.board_size - 1) ** 2
                row = wall_idx // (self.board_size - 1)
                col = wall_idx % (self.board_size - 1)
                os_action = (self.board_size * 2 - 1) + (row * os_row_stride) + (col * os_col_stride)

        return os_action

    def _convert_openspiel_action_to_gym(self, openspiel_action, observation=None):
        """Convert an OpenSpiel action ID to a gym action index."""
        # The grid size in OpenSpiel is board_size * 2 - 1. Starts with cell and interleaves walls
        os_board_size = self.board_size * 2 - 1
        # The vertical / row stride in this case is then two rows of size board_size - 1
        os_row_stride = os_board_size * 2
        # On the column side it only needs to skip wall positions
        os_col_stride = 2

        if openspiel_action % 2 == 0:
            # It's a move action: it happens on even indices, which are always cell positions
            # Remember it's relative, so all we can do here is calculate -1 / 0 / +1 with respect to the current position
            row_diff = (openspiel_action // os_row_stride) - 1
            col_diff = (openspiel_action % os_row_stride) // os_col_stride - 1
            (cur_row, cur_col) = np.where(observation["board"] == 1)
            (opp_row, opp_col) = np.where(observation["board"] == 2)

            # On straight jumps, the row and column differences should be doubled
            if cur_row + row_diff == opp_row and cur_col + col_diff == opp_col:
                row_diff *= 2
                col_diff *= 2

            gym_action = (cur_row + row_diff) * self.board_size + (cur_col + col_diff)

        else:
            # It's a wall placement
            cell_row = openspiel_action // os_row_stride
            cell_col = openspiel_action % os_board_size // os_col_stride
            is_vertical = (openspiel_action % os_board_size) % 2 == 1

            gym_action = self.board_size**2 + cell_row * (self.board_size - 1) + cell_col
            if not is_vertical:
                gym_action += (self.board_size - 1) ** 2

        return gym_action

    def handle_opponent_step_outcome(
        self,
        opponent_observation_before_action,
        my_observation_after_opponent_action,
        opponent_observation_after_action,
        opponent_reward,
        opponent_action,
        done=False,
    ):
        """Handle the outcome of the opponent's action.

        This is called after the opponent makes a move, allowing us to
        update our internal state to reflect the opponent's action.
        """
        if self.state is None:
            raise ValueError("Internal state is None")

        opponent_action = rotation.convert_rotated_action_index_to_original(self.board_size, opponent_action)
        opponent_observation_before_action = self._rotate_observation(opponent_observation_before_action)
        obs = opponent_observation_before_action["observation"]

        # Convert the opponent's gym action to an OpenSpiel action
        # Using my observation after opponent's action to help with the conversion
        openspiel_action = self._convert_gym_action_to_openspiel(opponent_action, obs)

        # Update our internal state with the opponent's action
        self.state.apply_action(openspiel_action)

    def get_action(self, observation):
        """Get an action from the agent for the current observation."""
        observation = self._rotate_observation(observation)

        action_mask = observation["action_mask"]

        # If model failed to load, fallback to random action
        if self.bot is None:
            raise ValueError("Model not loaded")

        # Ensure we have a valid state to work with
        if self.state is None:
            self.state = self.game.new_initial_state()
            raise ValueError("Internal state was None in get_action")

        # Get an action from the MCTS bot using our internal state
        openspiel_action = self.bot.step(self.state)

        # Convert the OpenSpiel action to a gym action index
        gym_action = self._convert_openspiel_action_to_gym(openspiel_action, observation["observation"])

        # Make sure it's a valid action according to the mask
        if gym_action < len(action_mask) and action_mask[gym_action]:
            # Update our internal state with our own action
            self.state.apply_action(openspiel_action)
        else:
            raise ValueError(f"AlphaZero selected invalid action {gym_action}")

        # Rotate the action back to the original player's perspective
        gym_action = rotation.convert_rotated_action_index_to_original(self.board_size, gym_action)

        return gym_action

    def _rotate_observation(self, observation):
        observation = observation.copy()
        observation["action_mask"] = rotation.rotate_action_mask(self.board_size, observation["action_mask"])
        observation["observation"] = observation["observation"].copy()
        observation["observation"]["board"] = rotation.rotate_board(observation["observation"]["board"])
        observation["observation"]["walls"] = rotation.rotate_walls(observation["observation"]["walls"])
        return observation
