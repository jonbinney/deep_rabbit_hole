import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from quoridor import ActionEncoder, Board, Quoridor


class NNEvaluator:
    def __init__(
        self,
        board_size: int,
        learning_rate: float,
        batch_size: int,
        optimization_iterations: int,
        action_encoder: ActionEncoder,
        device,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimization_iterations = optimization_iterations
        self.action_encoder = action_encoder
        self.device = device

        # Create a temporary game just to see how  big the input tensors are.
        temp_game = Quoridor(Board(board_size))
        temp_input, _ = self.game_to_input_tensor(temp_game)
        self.input_size = len(temp_input)

        self.nn = AlphaZeroNetwork(self.input_size, self.action_encoder.num_actions, self.device)

        self.rotated_action_mapping = np.zeros(action_encoder.num_actions, dtype=int)
        for action_i in range(len(self.rotated_action_mapping)):
            action = action_encoder.index_to_action(action_i)
            rotated_action = temp_game.rotate_action(action)
            rotated_action_i = action_encoder.action_to_index(rotated_action)
            self.rotated_action_mapping[rotated_action_i] = action_i

    def evaluate(self, game: Quoridor):
        with torch.no_grad():
            input_array, is_board_rotated = self.game_to_input_tensor(game)
            raw_policy, value = self.nn(torch.from_numpy(input_array).float())
            value = value.item()

        if is_board_rotated:
            raw_policy = raw_policy[self.rotated_action_mapping]

        # Mask the policy to ignore invalid actions
        valid_actions = game.get_valid_actions()
        valid_action_indices = [self.action_encoder.action_to_index(action) for action in valid_actions]
        policy_masked = np.zeros_like(raw_policy)
        policy_masked[valid_action_indices] = raw_policy[valid_action_indices]

        # Re-normalize probabilities after masking.
        policy_probs = policy_masked
        policy_probs = policy_probs / policy_probs.sum()

        return value, policy_probs

    def game_to_input_tensor(self, game: Quoridor) -> tuple[torch.FloatTensor, bool]:
        """Convert Quoridor game state to tensor format for neural network."""
        player = game.get_current_player()
        opponent = int(1 - player)

        rotate = True if (player == "player_2") else False
        if rotate:
            game = copy.deepcopy(game)
            game.rotate()

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
        features = np.concatenate(
            [player_board.flatten(), opponent_board.flatten(), walls.flatten(), [my_walls, opponent_walls]]
        )

        return features, rotate

    def train_network(self, replay_buffer):
        """Train the neural network on collected self-play data."""
        if len(replay_buffer) < self.batch_size:
            return

        optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.learning_rate)

        for _ in range(self.optimizer_iterations):
            # Sample random batch from replay buffer
            batch_data = random.sample(list(replay_buffer), self.batch_size)

            # Prepare batch tensors
            states = []
            target_policies = []
            target_values = []

            for data in batch_data:
                game = data["game"]
                if game.get_goal_row(game.get_current_player()) != game.board.board_size - 1:
                    raise ValueError("Training expects board to be oriented with player moving in positive direction")

                state_tensor = self.game_to_input_tensor(game, self.device)
                states.append(state_tensor)
                target_policies.append(torch.FloatTensor(data["mcts_policy"]).to(self.device))
                target_values.append(torch.FloatTensor([data["value"]]).to(self.device))

            states = torch.stack(states)
            target_policies = torch.stack(target_policies)
            target_values = torch.stack(target_values)

            # Forward pass
            pred_policies, pred_values = self.nn(states)

            # Compute losses
            policy_loss = F.cross_entropy(pred_policies, target_policies, reduction="mean")
            value_loss = F.mse_loss(pred_values.squeeze(), target_values.squeeze(), reduction="mean")
            total_loss = policy_loss + value_loss
            # print(f"{total_loss.item():3.3f} {policy_loss.item():3.3f} {value_loss.item():3.3f}")

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Store loss for metrics
        self.recent_losses.append(total_loss.item())
        if len(self.recent_losses) > 1000:  # Keep only recent losses
            self.recent_losses = self.recent_losses[-1000:]

        return {"total_loss": total_loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item()}


class AlphaZeroNetwork(nn.Module):
    def __init__(self, input_size, action_size, device):
        super(AlphaZeroNetwork, self).__init__()

        self.device = device

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Policy head - outputs action probabilities
        self.policy_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_size), nn.Softmax(dim=0))

        # Value head - outputs position evaluation (-1 to 1)
        self.value_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

        self.to(self.device)

    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.stack([i for i in x]).to(self.device)

        shared_features = self.shared(x)
        policy = self.policy_head(shared_features)
        value = self.value_head(shared_features)

        return policy, value
