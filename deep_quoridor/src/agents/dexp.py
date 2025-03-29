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
            nn.Linear(flat_input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
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

        board = obs["board"] if (self.player_id == "player_0") else self.rotate_board(obs["board"])
        walls = obs["walls"] if (self.player_id == "player_0") else self.rotate_walls(obs["walls"])
        board = board.flatten()
        walls = walls.flatten()
        my_walls = np.array([obs["my_walls_remaining"]])
        opponent_walls = np.array([obs["opponent_walls_remaining"]])

        flat_obs = np.concatenate([board, walls, my_walls, opponent_walls])
        return torch.FloatTensor(flat_obs).to(self.device)

    def convert_action_mask_to_tensor(self, mask):
        """Convert action mask to tensor, rotating it for player_1."""
        if self.player_id == "player_0":
            return super().convert_action_mask_to_tensor(mask)
        total_actions = self.board_size * self.board_size  # Movement actions
        wall_actions = (self.board_size - 1) ** 2  # Actions for each wall type

        # Split the mask into board moves and wall placements
        indices = np.array([total_actions, total_actions + wall_actions])
        board_mask, walls_v, walls_h = np.split(mask, indices)

        # Rotate board moves (first part of mask)
        board_mask = board_mask.reshape(self.board_size, self.board_size)
        board_mask = np.rot90(board_mask, k=2).flatten()

        # Rotate wall placements
        walls_v = walls_v.reshape(self.board_size - 1, self.board_size - 1)
        walls_h = walls_h.reshape(self.board_size - 1, self.board_size - 1)
        walls_v = np.rot90(walls_v, k=2).flatten()
        walls_h = np.rot90(walls_h, k=2).flatten()

        # Combine rotated masks back together
        rotated_mask = np.concatenate([board_mask, walls_v, walls_h])

        return torch.tensor(rotated_mask, dtype=torch.float32, device=self.device)

    def _rotate_index_to_original(self, index, grid_size, offset=0):
        """
        Helper method to convert rotated indices back to original space.

        Args:
            index: The index within the current section (board/walls)
            grid_size: Size of the grid for this section (board_size or board_size-1)
            offset: Offset to add to final result (0 for board moves, total_actions for walls)
        """
        row, col = divmod(index, grid_size)
        rotated_row = grid_size - 1 - row
        rotated_col = grid_size - 1 - col
        return offset + rotated_row * grid_size + rotated_col

    def convert_to_action_from_tensor_index(self, action_index_in_tensor):
        """Convert action index from rotated tensor back to original action space."""
        if self.player_id == "player_0":
            return super().convert_to_action_from_tensor_index(action_index_in_tensor)

        total_actions = self.board_size * self.board_size
        wall_actions = (self.board_size - 1) ** 2

        # Determine which section of the action space we're in
        if action_index_in_tensor < total_actions:
            # Board movement action
            return self._rotate_index_to_original(action_index_in_tensor, self.board_size)

        elif action_index_in_tensor < total_actions + wall_actions:
            # Vertical wall action
            wall_index = action_index_in_tensor - total_actions
            return self._rotate_index_to_original(wall_index, self.board_size - 1, total_actions)

        else:
            # Horizontal wall action
            wall_index = action_index_in_tensor - (total_actions + wall_actions)
            return self._rotate_index_to_original(wall_index, self.board_size - 1, total_actions + wall_actions)

    def rotate_board(self, board):
        """Rotate the board 180 degrees."""
        return np.rot90(board, k=2)

    def rotate_walls(self, walls):
        """Rotate the walls array 180 degrees for each layer."""
        rotated = np.zeros_like(walls)
        for i in range(walls.shape[2]):
            rotated[:, :, i] = np.rot90(walls[:, :, i], k=2)
        return rotated


class DExpPretrainedAgent(DExpAgent):
    """
    A DExpAgent that is initialized with the pre-trained model from main.py.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_pretrained_file()
