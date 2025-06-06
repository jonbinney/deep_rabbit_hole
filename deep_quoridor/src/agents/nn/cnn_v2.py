import numpy as np
import torch
import torch.nn as nn
from utils.misc import cnn_output_size_per_channel, my_device

from agents.nn.core.nn import BaseNN


class CnnV2Network(BaseNN):
    def __init__(self, obs_spc, action_spc):
        super(CnnV2Network, self).__init__()

        # CNN layers for board feature extraction
        self.board_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(),
        )

        board_size = obs_spc["observation"]["board"].shape[0]
        m1 = cnn_output_size_per_channel(board_size, 0, 3, 1)
        m2 = cnn_output_size_per_channel(m1, 0, 3, 1)
        board_layer_output_size = 32 * m2**2
        action_size = self._calculate_action_size(action_spc)

        # Update Linear layer input size
        self.modelx = nn.Sequential(
            nn.Linear(board_layer_output_size + 2, 512),  # +2 for remaining walls
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

        self.modelx.to(my_device())

    def forward(self, x):
        # Expects batch or non batch, to be ( (tensor(board2D),tensor(my_remaining_walls, opponent_remaining_walls)) )
        if isinstance(x, tuple) and not isinstance(x[0], tuple):
            # If we get non bach version, me make it a batch
            x = [x]

        board = torch.stack([t[0] for t in x])
        board = board.unsqueeze(1).to(my_device())  # Add channel dimension for CNN
        remaining_walls = torch.stack([t[1] for t in x]).to(my_device())

        # Combine boards for board position features
        # board_position = torch.cat([player_board, opponent_board], dim=1)  # shape: (batch_size, 2, 5, 5)

        # Process through CNNs
        board_features = self.board_cnn(board)
        board_flat = board_features.view(board_features.size(0), -1)

        # Concatenate all features
        combined = torch.cat([board_flat, remaining_walls], dim=1)

        return self.modelx(combined)

    def observation_to_tensor(self, observation):
        board = np.ascontiguousarray(observation["board"])
        player_walls = observation["my_walls_remaining"]
        opponent_walls = observation["opponent_walls_remaining"]
        # Convert numpy arrays to tensors and move to the appropriate device
        board_tensor = torch.FloatTensor(board).to(my_device())
        walls_tensor = torch.FloatTensor([player_walls, opponent_walls]).to(my_device())

        # Return tensors as expected by forward method
        return (board_tensor, walls_tensor)

    @classmethod
    def id(cls):
        return "cnn2"
