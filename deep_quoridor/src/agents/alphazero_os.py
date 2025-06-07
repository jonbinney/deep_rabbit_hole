import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import model as az_model
from quoridor import ActionEncoder
from utils.subargs import SubargsBase

from agents.core import Agent


@dataclass
class AlphaZeroOSParams(SubargsBase):
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


class AlphaZeroOSAgent(Agent):
    """Agent that uses a trained AlphaZero model from OpenSpiel to play Quoridor."""

    def __init__(self, params=AlphaZeroOSParams(), **kwargs):
        super().__init__()
        self.params = params
        self.board_size = kwargs["board_size"]
        self.max_walls = kwargs["max_walls"]
        self.action_space = kwargs["action_space"]
        self.action_encoder = ActionEncoder(self.board_size)

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
        if os.path.exists(self.params.checkpoint_path):
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
            print(f"Error loading AlphaZero model: {e}")
            self.bot = None

    def start_game(self, game, player_id):
        """Reset the internal state when a new game starts."""
        self.player_id = player_id
        # Initialize a fresh game state
        self.state = self.game.new_initial_state()
        print(f"AlphaZeroOSAgent: Starting new game as player {player_id}")

    def end_game(self, game):
        """Clean up when the game ends."""
        self.state = None
        print("AlphaZeroOSAgent: Game ended")

    def _convert_gym_action_to_openspiel(self, action_idx, observation=None):
        """Convert a gym action index to an OpenSpiel action ID.

        This needs to map from the gym environment's action space to
        OpenSpiel's action IDs for the current state.
        """
        # TODO: Implement the conversion from gym action index to OpenSpiel action ID
        pass

    def _convert_openspiel_action_to_gym(self, openspiel_action, observation=None):
        """Convert an OpenSpiel action ID to a gym action index."""
        # TODO: Implement the conversion from OpenSpiel action ID to gym action index
        pass

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

        # Convert the opponent's gym action to an OpenSpiel action
        # Using my observation after opponent's action to help with the conversion
        openspiel_action = self._convert_gym_action_to_openspiel(opponent_action, opponent_observation_before_action)

        # Update our internal state with the opponent's action
        self.state.apply_action(openspiel_action)

    def get_action(self, observation):
        """Get an action from the agent for the current observation."""
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
        gym_action = self._convert_openspiel_action_to_gym(openspiel_action, observation)

        # Make sure it's a valid action according to the mask
        if gym_action < len(action_mask) and action_mask[gym_action]:
            # Update our internal state with our own action
            self.state.apply_action(openspiel_action)
        else:
            raise ValueError(f"AlphaZero selected invalid action {gym_action}")

        return gym_action
