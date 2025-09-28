import numpy as np
import torch
import torch.nn as nn
from quoridor import ActionEncoder, Board, Player, Quoridor


class MLPNetwork(nn.Module):
    def __init__(self, action_encoder: ActionEncoder, device):
        super(MLPNetwork, self).__init__()

        self.action_encoder = action_encoder
        self.device = device

        # Create a temporary game just to see how  big the input tensors are.
        temp_game = Quoridor(Board(self.action_encoder.board_size))
        temp_input = self.game_to_input_array(temp_game)
        self.input_size = len(temp_input)

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Policy head - outputs action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_encoder.num_actions),
        )

        # value head - outputs position evaluation (-1 to 1)
        self.value_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

        self.to(self.device)

    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.stack([i for i in x]).to(self.device)

        shared_features = self.shared(x)
        policy = self.policy_head(shared_features)
        value = self.value_head(shared_features)

        return policy, value

    def game_to_input_array(self, game: Quoridor) -> np.ndarray:
        """Convert Quoridor game state to tensor format for neural network."""
        player = game.get_current_player()
        opponent = Player(1 - player)

        player_position = game.board.get_player_position(player)
        opponent_position = game.board.get_player_position(opponent)

        player_board = np.zeros((game.board.board_size, game.board.board_size), dtype=np.float32)
        player_board[player_position] = 1
        opponent_board = np.zeros((game.board.board_size, game.board.board_size), dtype=np.float32)
        opponent_board[opponent_position] = 1

        # Make a copy of walls
        walls = game.board.get_old_style_walls()

        my_walls = game.board.get_walls_remaining(player)
        opponent_walls = game.board.get_walls_remaining(opponent)

        # Combine all features into single tensor
        input_array = np.concatenate(
            [player_board.flatten(), opponent_board.flatten(), walls.flatten(), [my_walls, opponent_walls]],
            dtype=np.float32,
        )

        return input_array
