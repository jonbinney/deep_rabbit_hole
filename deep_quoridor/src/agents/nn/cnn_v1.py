import torch
import torch.nn as nn
from utils.misc import my_device


class CnnV1Network(nn.Module):
    """A convolutional neural network implementation for the Quoridor game board state evaluation.
    This network uses two parallel CNN branches to process the board position and wall placements
    separately, combining their features with additional game state information to predict action values.
    Architecture:
        - Two parallel CNN branches with identical structure:
            - Conv2d(2->16) + ReLU
            - Conv2d(16->32) + ReLU
        - Flattened CNN outputs are concatenated with extra game state info
        - Fully connected layers:
            - 2592->512 with ReLU and 0.1 dropout
            - 512->256 with ReLU and 0.1 dropout
            - 256->256 with ReLU
            - 256->action_size
    Args:
        observation_size (int): The size of the input observation space
        action_size (int): The size of the action space (number of possible actions)
    Input shape:
        x: Tensor of shape (batch_size, 4, 10, 9) containing:
            - First 2 channels (9x9): Board position features
            - Next 2 channels (9x9): Wall placement features
            - Extra 4 values: Additional game state information
    Returns:
        Tensor of shape (batch_size, action_size): Predicted action values
    """

    def __init__(self, observation_size, action_size):
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
