import torch.nn as nn
from utils.misc import my_device


class Flat1024Network(nn.Module):
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

    def __init__(self, observation_size, action_size):
        super(Flat1024Network, self).__init__()

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
        return self.model(x)

    @classmethod
    def id(cls):
        return "flat1024"
