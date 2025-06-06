import numpy as np
import torch
import torch.nn as nn
from utils.misc import my_device

from agents.nn.core.nn import BaseNN


class Flat1024Network(BaseNN):
    """Neural Network architecture for Quoridor game with flat layers.
    This network consists of fully connected layers with ReLU activations:
    - Input layer -> 1024 neurons
    - Hidden layer 1: 1024 -> 2048 neurons
    - Hidden layer 2: 2048 -> 1024 neurons
    - Output layer: 1024 -> action_size neurons
    Args:
        observation_size (int): Size of the input observation space
        action_size (int): Size of the output action space
    Returns:
        torch.Tensor: Output tensor representing action probabilities/values
    Example:
        >>> network = Flat1024Network(observation_size=81, action_size=20)
        >>> observation = torch.randn(1, 81)
        >>> output = network(observation)  # Shape: (1, 20)
    """

    def __init__(self, obs_spc, action_spc):
        super(Flat1024Network, self).__init__()
        action_size = self._calculate_action_size(action_spc)
        observation_size = self._calculate_observation_size(obs_spc)

        # Define network architecture
        self.model = nn.Sequential(
            nn.Linear(observation_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_size),
        )
        self.model.to(my_device())

    def forward(self, x):
        # When training, we receive a tuple of all the tensors, so we need to stack all of them
        if isinstance(x, tuple):
            x = torch.stack([i for i in x]).to(my_device())
        return self.model(x)

    @classmethod
    def id(cls):
        return "flat1024"

    def observation_to_tensor(self, observation):
        """Convert the observation dict to a tensor required by the network."""
        flat_obs = []
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                flat_obs.extend(value.flatten())
            else:
                flat_obs.append(value)
        return torch.FloatTensor(flat_obs).to(my_device())
