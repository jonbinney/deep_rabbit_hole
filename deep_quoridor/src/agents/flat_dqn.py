import numpy as np
import torch
import torch.nn as nn

from agents.core import AbstractTrainableAgent


class DQNNetwork(nn.Module):
    """
    Neural network model for Deep Q-learning.
    Takes observation from the Quoridor game and outputs Q-values for each action.
    """

    def __init__(self, board_size, action_size):
        super(DQNNetwork, self).__init__()

        # Calculate input dimensions based on observation space
        # Board is board_size x board_size with 2 channels (player position and opponent position)
        # Walls are (board_size-1) x (board_size-1) with 2 channels (vertical and horizontal walls)
        board_input_size = board_size * board_size
        walls_input_size = (board_size - 1) * (board_size - 1) * 2

        # Additional features: walls remaining for both players
        flat_input_size = board_input_size + walls_input_size + 2

        # Define network architecture
        self.model = nn.Sequential(
            nn.Linear(flat_input_size, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.model(x)


class FlatDQNAgent(AbstractTrainableAgent):
    """Agent that uses Deep Q-Network with flat state representation."""

    @classmethod
    def params_class(cls):
        """If we want to receive parameters from the command line, return a class that uses @dataclass
        containing the fields.   They will be parsed using subargs.
        """
        return None

    def name(self):
        return "flatdqn"

    def version(self):
        """Bump this version when compatibility with saved models is broken"""
        return 1

    def _calculate_action_size(self):
        """Calculate the size of the action space."""
        return self.board_size**2 + (self.board_size - 1) ** 2 * 2

    def _create_network(self):
        """Create the neural network model."""
        return DQNNetwork(self.board_size, self.action_size)

    def observation_to_tensor(self, observation):
        """Convert the observation dict to a flat tensor."""
        obs = observation["observation"]
        board = obs["board"].flatten()
        walls = obs["walls"].flatten()
        my_walls = np.array([obs["my_walls_remaining"]])
        opponent_walls = np.array([obs["opponent_walls_remaining"]])

        flat_obs = np.concatenate([board, walls, my_walls, opponent_walls])
        return torch.FloatTensor(flat_obs).to(self.device)


class Pretrained01FlatDQNAgent(FlatDQNAgent):
    """
    A FlatDQNAgent that is initialized with the pre-trained model from main.py.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_pretrained_file()
