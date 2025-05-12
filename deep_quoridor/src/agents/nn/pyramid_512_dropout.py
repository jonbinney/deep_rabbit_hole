import numpy as np
import torch
import torch.nn as nn
from utils.misc import my_device


class P512DropoutNetwork(nn.Module):
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

    def __init__(self, observation_size, action_size):
        super(P512DropoutNetwork, self).__init__()

        # Define network architecture
        self.model = nn.Sequential(
            nn.Linear(observation_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

        self.model.to(my_device())

    def forward(self, x):
        return self.model(x)

    @classmethod
    def id(cls):
        return "p512do"

    def observation_to_tensor(self, observation):
        """Convert the observation dict to a tensor required by the Network."""
        flat_obs = []
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                flat_obs.extend(value.flatten())
            else:
                flat_obs.append(value)
        return torch.FloatTensor(flat_obs).to(my_device())
