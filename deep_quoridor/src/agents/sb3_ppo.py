import os
from glob import glob

from deep_quoridor.src.train_sb3 import SB3ActionMaskWrapper
from sb3_contrib import MaskablePPO

from agents.core.agent import Agent, AgentRegistry


class SB3PPOAgent(Agent):
    """
    Agent that uses a trained SB3 PPO model to select actions.
    It loads the most recent trained PPO model and uses it to predict actions.
    """

    def __init__(self, model_prefix=None, deterministic=True, **kwargs):
        """
        Initialize the SB3 PPO agent.

        Args:
            model_prefix: Optional prefix for the model file (default: None, will use env name)
            deterministic: Whether to use deterministic action selection (default: True)
            **kwargs: Additional arguments to pass to the parent class
        """
        super().__init__()
        self.model = None
        self.model_prefix = model_prefix
        self.deterministic = deterministic
        self.wrapper = None

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

            # Determine the model prefix
            prefix = self.model_prefix if self.model_prefix is not None else self.wrapper.metadata["name"]

            try:
                # Find the most recent model file
                latest_policy = max(glob(f"{prefix}*.zip"), key=os.path.getctime)
                self.model = MaskablePPO.load(latest_policy)
                print(f"Loaded model from {latest_policy}")
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
