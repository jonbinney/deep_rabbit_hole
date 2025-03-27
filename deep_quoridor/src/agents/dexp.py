import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from agents.core import AbstractTrainableAgent


class DExpNetwork(nn.Module):
    """
    Neural network model for Deep Q-learning.
    Takes observation from the Quoridor game and outputs Q-values for each action.
    """

    def __init__(self, board_size, action_size):
        super(DExpNetwork, self).__init__()

        # Calculate input dimensions based on observation space
        # Board is board_size x board_size with 2 channels (player position and opponent position)
        # Walls are (board_size-1) x (board_size-1) with 2 channels (vertical and horizontal walls)
        board_input_size = board_size * board_size
        walls_input_size = (board_size - 1) * (board_size - 1) * 2

        # Additional features: walls remaining for both players
        flat_input_size = board_input_size + walls_input_size + 2

        # Define network architecture
        self.model = nn.Sequential(
            nn.Linear(flat_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, x):
        return self.model(x)


class DExpAgent(AbstractTrainableAgent):
    """Diego experimental Agent using DRL."""

    def _calculate_action_size(self):
        """Calculate the size of the action space."""
        return self.board_size**2 + (self.board_size - 1) ** 2 * 2

    def _create_network(self):
        """Create the neural network model."""
        return DExpNetwork(self.board_size, self.action_size)

    def observation_to_tensor(self, observation):
        """Convert the observation dict to a flat tensor."""
        obs = observation["observation"]

        board = obs["board"].flatten()
        walls = obs["walls"].flatten()
        my_walls = np.array([obs["my_walls_remaining"]])
        opponent_walls = np.array([obs["opponent_walls_remaining"]])

        flat_obs = np.concatenate([board, walls, my_walls, opponent_walls])
        return torch.FloatTensor(flat_obs).to(self.device)


class DExpPretrainedAgent(DExpAgent):
    """
    A DExpAgent that is initialized with the pre-trained model from main.py.
    """

    def __init__(self, board_size, **kwargs):
        super().__init__(board_size, epsilon=0.0)
        model_path = Path(__file__).resolve().parents[3] / "models" / "dexp_final.pt"
        if os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}")
            self.load_model(model_path)
        else:
            print(
                f"Warning: Model file {model_path} not found, using untrained agent. Ask Julian for the weights file."
            )
            raise FileNotFoundError(f"Model file {model_path} not found. Please provide the weights file.")
