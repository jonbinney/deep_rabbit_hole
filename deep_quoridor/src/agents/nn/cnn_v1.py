import numpy as np
import torch
import torch.nn as nn
from utils.misc import my_device

from agents.nn.core.nn import BaseNN


class CnnV1Network(BaseNN):
    """Initial CNN hardcoded for 5x5 board, doing transformation as part of forward
    Input tensors x are a flattened tensor with items in this order:
    - player board: 5x5 board with 1s at player location (25 items)
    - opponent board: 5x5 board with 1s at opponent location (25 items)
    - horizontal walls: 4x4 board with 1s at wall locations (16 items)
    - vertical walls: 4x4 board with 1s at wall locations (16 items)
    - remaining walls: [player walls, opponent walls] (2 items)
    Total size: 84 items

    Kept for historical reference
    """

    def __init__(self, observation_spc, action_spc):
        super(CnnV1Network, self).__init__()

        # CNN layers for board feature extraction
        self.board_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # CNN layers for walls feature extraction
        self.walls_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Flatten CNN outputs and combine with extra info
        board_flat_size = 32 * 5 * 5  # Output from board_cnn (32 channels, 5x5)
        walls_flat_size = 32 * 4 * 4  # Output from walls_cnn (32 channels, 4x4)
        action_size = self._calculate_action_size(action_spc)

        # Update Linear layer input size
        self.modelx = nn.Sequential(
            nn.Linear(board_flat_size + walls_flat_size + 2, 512),  # +2 for remaining walls,  # *2 for both CNNs
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

        self.modelx.to(my_device())

    def forward(self, x):
        # When training, we receive a tuple of all the tensors, so we need to stack all of them
        if isinstance(x, tuple):
            x = torch.stack([i for i in x]).to(my_device())

        # Split input into board position, walls, and extra info
        # Get batch size and reshape tensors for batch processing
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        n = 5
        board_size = n * n
        walls_size = (n - 1) * (n - 1)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension if not present

        player_board = x[:, :board_size].view(batch_size, 1, n, n)
        opponent_board = x[:, board_size : 2 * board_size].view(batch_size, 1, n, n)
        h_walls = x[:, 2 * board_size : 2 * board_size + walls_size].view(batch_size, 1, n - 1, n - 1)
        v_walls = x[:, 2 * board_size + walls_size : 2 * board_size + walls_size * 2].view(batch_size, 1, n - 1, n - 1)
        remaining_walls = x[:, 2 * board_size + walls_size * 2 :]  # my walls, opponent walls

        # Combine boards for board position features
        board_position = torch.cat([player_board, opponent_board], dim=1)  # shape: (batch_size, 2, 5, 5)

        # Combine walls for wall features
        walls = torch.cat([h_walls, v_walls], dim=1)  # shape: (batch_size, 2, 4, 4)

        # Process through CNNs
        board_features = self.board_cnn(board_position)
        walls_features = self.walls_cnn(walls)

        # Flatten CNN outputs
        board_flat = board_features.view(batch_size, -1)
        walls_flat = walls_features.view(batch_size, -1)

        # Concatenate all features
        combined = torch.cat([board_flat, walls_flat, remaining_walls], dim=1)

        return self.modelx(combined)

    @classmethod
    def id(cls):
        return "cnn1"

    def observation_to_tensor(self, observation):
        """Convert the observation dict to a tensor required by the Network."""
        flat_obs = []
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                flat_obs.extend(value.flatten())
            else:
                flat_obs.append(value)
        return torch.FloatTensor(flat_obs).to(my_device())
