import torch
from environment.dict_split_board_wrapper import DictSplitBoardWrapper
from environment.rotate_wrapper import RotateWrapper
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import FlattenExtractor

from agents.core.agent import ActionLog, Agent, AgentRegistry
from agents.core.rotation import convert_rotated_action_index_to_original
from agents.core.trainable_agent import AbstractTrainableAgent, TrainableAgentParams


class SB3ActionMaskWrapper(BaseWrapper):
    """
    Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking.
    Taken from https://github.com/dm-ackerman/PettingZoo/blob/master/tutorials/SB3/connect_four/sb3_connect_four_action_mask.py

    In particular it adapts PettingZoo, since the Action Masking part of it is already implemented
    in SB3_contrib as the MaskablePPO.
    The required changes are minor:
    - Present observation_space and action_space as props instead of methods
    - return last() on step()
    - return observation on reset()
    - return only the observation (not the action mask) in observe()
    - provide a method to get to the action mask
    """

    def __init__(self, env, rewards_multiplier: float = 1000, opponent: Agent = None, **kwargs):
        super().__init__(env)
        self.rewards_multiplier = rewards_multiplier
        self.opponent = opponent

        if self.opponent is not None:
            self.opponent.start_game(self.env, "player_0")

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.

        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset(seed, options)

        # SB3 needs observation and action spaces as props instead of methods (?)
        full_observation = super().observation_space(self.possible_agents[0])
        self.observation_space = full_observation["observation"]
        self.action_space = super().action_space(self.possible_agents[0])

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info.

        The next_state observation is for the next agent (used to determine the next action), while the remaining
        items are for the agent that just acted (used to understand what just happened).

        If an opponent has been selected, at each step we also play the opponent's action as part of this step.
        If the opponent wins, we update the reward and termination flag.
        """
        current_agent = self.agent_selection

        super().step(action)

        opponent_agent = self.agent_selection

        next_state = self.observe(self.agent_selection)
        reward = self.rewards[current_agent] * self.rewards_multiplier
        termination = self.terminations[current_agent]
        truncation = self.truncations[current_agent]
        info = self.infos[current_agent]

        if not truncation and not termination and self.opponent is not None:
            opponent_action = self.opponent.get_action(self, self.action_mask())

            super().step(opponent_action)

            next_state = self.observe(self.agent_selection)
            opponent_reward = self.rewards[opponent_agent] * self.rewards_multiplier
            truncation = self.truncations[opponent_agent]
            termination = self.terminations[opponent_agent]
            reward = reward - opponent_reward

        return (
            next_state,
            reward,
            termination,
            truncation,
            info,
        )

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        obs = super().observe(agent)["observation"]
        return obs

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().observe(self.agent_selection)["action_mask"]


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
        self.action_log = ActionLog()
        self.board_size = board_size
        self.max_walls = max_walls
        self.params = params
        self.deterministic = deterministic
        self.model = None
        self.wrapper = None
        self.steps = 0
        self.training_mode = False
        self.episodes_rewards = []

        self._reset_episode_related_info()

    def end_game(self, game):
        pass

    def model_name(self):
        return "sb3ppo"

    @staticmethod
    def version():
        """Bump this version when compatibility with saved models is broken"""
        return 3

    @staticmethod
    def params_class():
        return TrainableAgentParams

    @staticmethod
    def get_model_extension():
        return "zip"

    def load_model(self, path):
        """Override the parent class method to load models using MaskablePPO.load() instead"""
        print(f"Loading pre-trained model from {path}")
        self.model = MaskablePPO.load(path)

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
        self.player_id = player_id

        if self.model is None:
            # Check if wandb_alias is provided - prioritize it if present
            if self.params.wandb_alias:
                print(f"Using wandb_alias: {self.params.wandb_alias}")
                # Try to fetch the model from wandb
                self._fetch_model_from_wand_and_update_params()

                # If we got a filename from wandb, try to load it
                if self.params.model_filename:
                    try:
                        self.load_model(self.params.model_filename)
                    except (ValueError, FileNotFoundError, TypeError):
                        print(f"Failed to load model using wandb_alias '{self.params.wandb_alias}'.")

            # If no wandb_alias or loading from wandb failed, try to load from local files
            if self.model is None:
                try:
                    print("Looking for model in local files...")
                    self.resolve_and_load_model()
                except FileNotFoundError:
                    print("No local model file found.")

            # Wrap the game to get access to action_mask method
            self.wrapper = wrap_env(game)

            if self.model is None:
                print("No policy found. The agent will not work correctly.")
                return

    def get_action(self, _observation):
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

        if self.player_id == "player_1":
            action = convert_rotated_action_index_to_original(self.board_size, action)

        return action


def wrap_env(env, **kwargs):
    env = RotateWrapper(env)
    env = DictSplitBoardWrapper(env, include_turn=False)
    env = SB3ActionMaskWrapper(env, **kwargs)
    return env


def make_env_fn(env_constructor, **mykwargs):
    def env_fn(**kwargs):
        env = env_constructor(**kwargs)
        env = wrap_env(env, **mykwargs)
        return env

    return env_fn


class DictFlattenExtractor(FlattenExtractor):
    """
    This class is necessary because the default FlattenExtractor does not work with dict spaces.
    It just tries to call "flatten" directly which of course doesn't exist on a dict.
    NOTE: It is important to be careful to not flatten the batch dimension (first one).
    """

    def __init__(self, observation_space: spaces.Box):
        super().__init__(observation_space)

    def forward(self, obs: dict) -> torch.Tensor:
        thobs = torch.tensor([], device=list(obs.values())[0].device)
        for v in obs.values():
            thobs = torch.cat((thobs, torch.tensor(v).flatten(start_dim=1)), dim=1)
        return thobs


# Register the agent with the registry
AgentRegistry.register(Agent._friendly_name(SB3PPOAgent.__name__), SB3PPOAgent)
