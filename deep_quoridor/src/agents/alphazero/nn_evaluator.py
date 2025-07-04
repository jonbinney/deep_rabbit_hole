import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from quoridor import ActionEncoder, Board, Player, Quoridor

from agents.alphazero.mlp_network import MLPNetwork


class NNEvaluator:
    def __init__(
        self,
        action_encoder: ActionEncoder,
        device,
    ):
        self.action_encoder = action_encoder
        self.device = device

        # Create a temporary game just to see how  big the input tensors are.
        temp_game = Quoridor(Board(self.action_encoder.board_size))
        temp_input = self.game_to_input_array(temp_game)
        self.input_size = len(temp_input)

        self.network = MLPNetwork(self.input_size, self.action_encoder.num_actions, self.device)

        self.rotated_action_mapping = np.zeros(action_encoder.num_actions, dtype=int)
        for action_i in range(len(self.rotated_action_mapping)):
            action = action_encoder.index_to_action(action_i)
            rotated_action = temp_game.rotate_action(action)
            rotated_action_i = action_encoder.action_to_index(rotated_action)
            self.rotated_action_mapping[rotated_action_i] = action_i

    def evaluate(self, game: Quoridor):
        # Rotate the board if player 2 is playing so that we always work with player 1's perspective.
        game, is_board_rotated = self.rotate_game_for_player_two(game)

        with torch.no_grad():
            input_array = self.game_to_input_array(game)
            unmasked_policy, value = self.network(torch.from_numpy(input_array).float().to(self.device))
            unmasked_policy = unmasked_policy.cpu().numpy()
            value = value.item()

        # Mask the policy to ignore invalid actions. NOTE: Game is already rotated so the valid actions will be rotated too
        valid_actions = game.get_valid_actions()
        valid_action_indices = [self.action_encoder.action_to_index(action) for action in valid_actions]
        policy_masked = np.zeros_like(unmasked_policy)
        policy_masked[valid_action_indices] = unmasked_policy[valid_action_indices]

        if np.all(policy_masked == 0):
            # If the policy ends up as all zeros after masking, turn it into a uniform distribution among
            # the valid actions.
            policy_masked[valid_action_indices] = 1 / len(valid_action_indices)
            print("Policy is all zeros after masking, turning it into a uniform distribution")
        else:
            # Otherwise, just renormalize after masking
            policy_masked = policy_masked / policy_masked.sum()

        # Sanity checks
        assert np.all(policy_masked >= 0), "Policy contains negative probabilities"
        assert np.all(policy_masked <= 1), "Policy contains probabilities greater than 1"
        assert np.any(policy_masked > 0), "Policy is all zeros"
        assert np.isfinite(policy_masked).all() and np.isfinite(value), "Policy or value is non-finite"

        # If the game was originally rotated, rotate the resulting back to player 2's perspective
        if is_board_rotated:
            policy_masked = policy_masked[self.rotated_action_mapping]

        return value, policy_masked

    def rotate_game_for_player_two(self, game: Quoridor):
        player = game.get_current_player()

        is_rotated = True if (player == Player.TWO) else False
        if is_rotated:
            game = copy.deepcopy(game)
            game.rotate_board()

        return game, is_rotated

    def game_to_input_array(self, game: Quoridor) -> tuple[torch.FloatTensor, bool]:
        """Convert Quoridor game state to tensor format for neural network."""
        player = game.get_current_player()
        opponent = int(1 - player)

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

    def train_network(self, replay_buffer, learning_rate, batch_size, optimizer_iterations):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        for _ in range(optimizer_iterations):
            # Sample random batch from replay buffer
            batch_data = random.sample(list(replay_buffer), batch_size)

            # Prepare batch tensors
            inputs = []

            # The network predicts the natural log of the probability of each next action. When we use
            # it to make predictions, we take the softamx of its output which un-logifies it and also
            # normalizes it to a valid probability distribution.
            target_policies = []

            target_values = []

            for data in batch_data:
                game, is_rotated = self.rotate_game_for_player_two(data["game"])
                inputs.append(torch.from_numpy(self.game_to_input_array(game)))
                if is_rotated:
                    mcts_policy = data["mcts_policy"][self.rotated_action_mapping]
                else:
                    mcts_policy = data["mcts_policy"]
                target_policies.append(torch.FloatTensor(mcts_policy))
                target_values.append(torch.FloatTensor([data["value"]]))

            inputs = torch.stack(inputs).to(self.device)
            target_policies = torch.stack(target_policies).to(self.device)
            target_values = torch.stack(target_values).to(self.device)

            if inputs.isnan().any() or target_policies.isnan().any() or target_values.isnan().any():
                raise ValueError("NaN in training data")

            # Forward pass
            pred_policies, pred_values = self.network(inputs)

            # Compute losses
            policy_loss = F.cross_entropy(pred_policies, target_policies, reduction="mean")
            value_loss = F.mse_loss(pred_values.squeeze(), target_values.squeeze(), reduction="mean")
            total_loss = policy_loss + value_loss
            # print(f"{total_loss.item():3.3f} {policy_loss.item():3.3f} {value_loss.item():3.3f}")

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        return {"total_loss": total_loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item()}
