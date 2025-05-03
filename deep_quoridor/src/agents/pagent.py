from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from utils import my_device

from agents.adapters.base import BaseTrainableAgentAdapter
from agents.adapters.dict_split_board_adapter import DictSplitBoardAdapter
from agents.adapters.rotate_adapter import RotateAdapter
from agents.core import AbstractTrainableAgent
from agents.core.trainable_agent import TrainableAgentParams


class PAgentNetwork(nn.Module):
    """
    Neural network model for Deep Q-learning.
    Takes observation from the Quoridor game and outputs Q-values for each action.
    """

    def __init__(self, observation_size, action_size):
        super(PAgentNetwork, self).__init__()

        # Define network architecture
        self.model = nn.Sequential(
            nn.Linear(observation_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, action_size),
        )
        self.model.to(my_device())

    def forward(self, x):
        return self.model(x)


@dataclass
class PAgentParams(TrainableAgentParams):
    # Whether oppoments actions are used for training
    # This is used for training only, not for playing
    # This should not be combined with assign_negative_reward
    use_opponents_actions: bool = False

    @classmethod
    def training_only_params(cls) -> set[str]:
        """
        Returns a set of parameters that are used only during training.
        These parameters should not be used during playing.
        """
        return super().training_only_params() | {
            "use_opponents_actions",
        }


class PAgentAgent(AbstractTrainableAgent):
    """Diego experimental Agent using DRL."""

    def __init__(
        self,
        params=PAgentParams(),
        **kwargs,
    ):
        super().__init__(params=params, **kwargs)

    def name(self):
        if self.params.nick:
            return self.params.nick
        return "PAgent"

    def model_name(self):
        return "pagent"

    @classmethod
    def params_class(cls):
        return PAgentParams

    def version(self):
        """Bump this version when compatibility with saved models is broken"""
        return 1

    def _calculate_action_size(self):
        """Calculate the size of the action space."""
        return self.action_space.flatten().shape[0]

    def _calculate_observation_size(self):
        """Calculate the size of the action space."""
        return self.observation_space.flatten().shape[0]

    def _create_network(self):
        """Create the neural network model."""
        return PAgentNetwork(self._calculate_observation_size(), self._calculate_action_size())

    def handle_opponent_step_outcome(
        self,
        opponent_observation_before_action,
        my_observation_after_opponent_action,
        opponent_observation_after_action,
        opponent_reward,
        opponent_action,
        done=False,
    ):
        if not self.training_mode or not self.params.use_opponents_actions:
            return

        self._handle_step_outcome_all(
            opponent_observation_before_action,
            my_observation_after_opponent_action,
            opponent_observation_after_action,
            opponent_reward,
            opponent_action,
            self._get_opponent_player_id(self.player_id),
            done,
        )

    def _observation_to_tensor(self, observation, obs_player_id):
        """Convert the observation dict to a flat tensor."""
        flat_obs = []
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                flat_obs.extend(value.flatten())
            else:
                flat_obs.append(value)
        return torch.FloatTensor(flat_obs).to(self.device)

    def _convert_action_mask_to_tensor_for_player(self, mask, player_id):
        return torch.tensor(mask, dtype=torch.float32, device=self.device)


class NewDexpAgent(BaseTrainableAgentAdapter):
    def __init__(
        self,
        board_size,
        max_walls,
        observation_space,
        action_space,
        params: TrainableAgentParams = TrainableAgentParams(),
        **kwargs,
    ):
        new_action_space = RotateAdapter.get_action_space(DictSplitBoardAdapter.get_action_space(action_space))
        new_observation_space = RotateAdapter.get_observation_space(
            DictSplitBoardAdapter.get_observation_space(action_space)
        )
        agent = PAgentAgent(board_size, max_walls, new_observation_space, new_action_space, params=params, **kwargs)
        rotate_agent = RotateAdapter(agent)
        dict_agent = DictSplitBoardAdapter(rotate_agent)
        super().__init__(dict_agent)
