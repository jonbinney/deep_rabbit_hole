from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from quoridor import ActionEncoder, Player, Quoridor


@dataclass
class ResnetConfig:
    num_blocks: Optional[int] = None
    num_channels: int = 32
    # Maximum number of moves in the game. Used for the "moves remaining"
    # input channel. None means "unknown" — that channel is filled with 0.
    max_steps: Optional[int] = None


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and skip connection."""

    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = torch.relu(out)
        return out


class ResnetNetwork(nn.Module):
    def __init__(self, action_encoder: ActionEncoder, device, config: ResnetConfig):
        """
        ResNet-based network following AlphaZero architecture.

        Args:
            action_encoder: ActionEncoder for the game
            device: torch device to use
            num_blocks: Number of residual blocks.  Alphazero used 20 for chess and 40 for go, which
             is a bit more than twice the board dimension. We default to the twice the dimension of
             the combined grid, plus two.
            num_channels: Number of channels in convolutional layers (Alphazero used 256)
        """
        super(ResnetNetwork, self).__init__()

        self.action_encoder = action_encoder
        self.device = device
        self.max_steps = config.max_steps
        num_channels = config.num_channels
        num_blocks = config.num_blocks

        # Calculate input dimensions: MxMx6 where M = board_size * 2 + 3
        self.input_size = action_encoder.board_size * 2 + 3

        if num_blocks is None:
            # Alphazero used 20 blocks for chess and 40 for Go, which seems to be a bit more
            # than double the board dimension.
            num_blocks = self.input_size * 2 + 2

        # Initial convolutional block (6 input channels — see game_to_input_array)
        self.conv_input = nn.Conv2d(6, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.residual_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.input_size * self.input_size, action_encoder.num_actions)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.input_size * self.input_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

        self.to(self.device)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 5, M, M) where M = board_size * 2 + 3
               or tuple of such tensors

        Returns:
            policy: Policy logits of shape (batch_size, num_actions)
            value: Value estimate of shape (batch_size, 1) in range [-1, 1]
        """
        if isinstance(x, tuple):
            x = torch.stack([i for i in x]).to(self.device)

        # Initial convolution
        out = torch.relu(self.bn_input(self.conv_input(x)))

        # Residual tower
        for block in self.residual_blocks:
            out = block(out)

        # Policy head
        policy = torch.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value head
        value = torch.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = torch.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    def game_to_input_array(self, game: Quoridor) -> np.ndarray:
        """
        Convert Quoridor game state to an input array for the neural network.

        The returned array is 6xMxM, where M is the dimension of the combined
        grid representation, which is twice the board dimension plus three.

        The 6 planes in the array are:

            1. Walls (1 where there is a wall, zero otherwise)
            2. Current player's position, as a 1-hot encoding
            3. Opponent's position, as a 1-hot encoding
            4. Current player walls remaining (same value for entire plane)
            5. Opponent walls remaining (same value for the entire plane)
            6. Moves remaining (max_steps - completed_steps; same value for
               entire plane). Filled with 0 if max_steps is unknown.
        """
        input_array = np.zeros((6, self.input_size, self.input_size), dtype=np.float32)

        # First channel is a 1 where there are walls
        input_array[0, :, :] = game.board._grid == game.board.WALL

        # Second channel is a 1-hot encoding of current player position.
        player = game.get_current_player()
        input_array[1, :, :] = game.board._grid == player

        # Third channel is a 1-hot encoding of opponent position.
        opponent = Player(1 - player)
        input_array[2, :, :] = game.board._grid == opponent

        # Fourth channel is current player walls remaining (all values the same)
        input_array[3, :, :] = game.board._walls_remaining[player]

        # Fifth channel is opponent walls remaining (all values the same)
        input_array[4, :, :] = game.board._walls_remaining[opponent]

        # Sixth channel is the fraction of moves remaining (broadcast). We
        # divide by max_steps so this stays in [0, 1] alongside the other
        # channels — keeping channel 5 within roughly the same magnitude
        # as channels 0–4 prevents BatchNorm running stats from lagging
        # the training-mode batch stats.
        if self.max_steps is not None and self.max_steps > 0:
            input_array[5, :, :] = max(0.0, (self.max_steps - game.completed_steps) / self.max_steps)

        return input_array
