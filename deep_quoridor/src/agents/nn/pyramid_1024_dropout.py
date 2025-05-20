import numpy as np
import torch
import torch.nn as nn
from utils.misc import my_device

from agents.nn.core.nn import BaseNN


class P1024DropoutNetwork(BaseNN):
    """Neural network architecture for policy/value evaluation with dropout layers.

    This network consists of fully connected layers with ReLU activations and dropout regularization:
    - Input layer → 512 units with ReLU and 10% dropout
    - 512 units → 256 units with ReLU and 10% dropout
    - 256 units → 256 units with ReLU
    - 256 units → output layer

    Args:
        observation_size (int): Size of the input observation vector
        action_size (int): Size of the output action vector

    Attributes:
        model (nn.Sequential): Sequential container of the network layers
    """

    def __init__(self, obs_spc, action_spc):
        super(P1024DropoutNetwork, self).__init__()
        action_size = self._calculate_action_size(action_spc)
        observation_size = self._calculate_observation_size(obs_spc)

        # Define network architecture
        self.model = nn.Sequential(
            nn.Linear(observation_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
        )

        self.model.to(my_device())

    def forward(self, x):
        # When training, we receive a tuple of all the tensors, so we need to stack all of them
        if isinstance(x, tuple):
            x = torch.stack([i for i in x]).to(my_device())
        return self.model(x)

    @classmethod
    def id(cls):
        return "p1024do"

    def observation_to_tensor(self, observation):
        """Convert the observation dict to a tensor required by the Network."""
        flat_obs = []
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                flat_obs.extend(value.flatten())
            else:
                flat_obs.append(value)
        return torch.FloatTensor(flat_obs).to(my_device())
