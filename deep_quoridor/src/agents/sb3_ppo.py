import os
from glob import glob
from typing import override

from sb3_contrib import MaskablePPO

from agents.core.agent import Agent, AgentRegistry
from agents.core.trainable_agent import AbstractTrainableAgent, TrainableAgentParams
from deep_quoridor.src.train_sb3 import SB3ActionMaskWrapper


class SB3PPOAgent(AbstractTrainableAgent):
    """
    Agent that uses a trained SB3 PPO model to select actions.
    It loads the most recent trained PPO model and uses it to predict actions.
    """

    def __init__(self, board_size, max_walls, deterministic=True, params=TrainableAgentParams(), **kwargs):
        """
        Initialize the SB3 PPO agent.

        Args:
            params: Optional parameters for the agent
            **kwargs: Additional arguments to pass to the parent class
        """
        self.board_size = board_size
        self.max_walls = max_walls
        self.params = params
        self.deterministic = deterministic
        self.model = None
        self.wrapper = None
        self.steps = 0
        self.training_mode = False
        self.episodes_rewards = []

        self.fetch_model_from_wand_and_update_params()
        self.reset_episode_related_info()

    def end_game(self, game):
        pass

    def model_name(self):
        return "sb3ppo"

    def version(self):
        """Bump this version when compatibility with saved models is broken"""
        return 0

    @staticmethod
    def params_class():
        return TrainableAgentParams

    @staticmethod
    def get_model_extension():
        return "zip"

    def _calculate_action_size(self):
        """Calculate the size of the action space."""
        return self.board_size**2 + (self.board_size - 1) ** 2 * 2

    def start_game(self, game, player_id):
        """
        Load the trained model when a new game starts.

        Args:
            game: The game environment
            player_id: The ID of the player this agent controls
        """
        if self.model is None:
            # Wrap the game to get access to action_mask method
            self.wrapper = SB3ActionMaskWrapper(game)

            try:
                # Find the most recent model file
                self.model = MaskablePPO.load(self.params.model_filename)
                print(f"Loaded model from {self.params.model_filename}")
            except (ValueError, FileNotFoundError):
                print("No policy found. The agent will not work correctly.")
                return

    def get_action(self, game):
        """
        Get the action to take based on the current game state using the PPO model.

        Args:
            game: The game environment

        Returns:
            The action to take
        """
        if self.model is None:
            print("Model not loaded. Please ensure start_game was called first.")
            return None

        # Get the observation and action mask
        action_mask = self.wrapper.action_mask()

        # Check if the game is over
        observation, _, termination, truncation, _ = self.wrapper.last()
        if termination or truncation:
            return None

        # Use the model to predict the action
        action = int(self.model.predict(observation, action_masks=action_mask, deterministic=self.deterministic)[0])

        return action


# Register the agent with the registry
AgentRegistry.register(Agent._friendly_name(SB3PPOAgent.__name__), SB3PPOAgent)
