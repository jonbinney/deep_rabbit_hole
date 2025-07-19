import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from quoridor import ActionEncoder, Board, Player, Quoridor

from agents.alphazero.mlp_network import MLPNetwork
from agents.core.rotation import create_rotation_mapping

INVALID_ACTION_VALUE = -1e32


def function_stats(f):
    def wrapper(*args, **kwargs):
        print(wrapper.calls)
        t_start = time.time()
        return_value = f(*args, **kwargs)
        t_end = time.time()
        wrapper.calls.append(t_end - t_start)
        return return_value

    wrapper.calls = []
    return wrapper


class NNEvaluator:
    def __init__(
        self,
        action_encoder: ActionEncoder,
        device,
    ):
        self.action_encoder = action_encoder
        self.device = device
        self.evaluate_call_stats = []
        self.train_iteration_call_stats = []

        # Create a temporary game just to see how  big the input tensors are.
        temp_game = Quoridor(Board(self.action_encoder.board_size))
        temp_input = self.game_to_input_array(temp_game)
        self.input_size = len(temp_input)

        self.network = MLPNetwork(self.input_size, self.action_encoder.num_actions, self.device)

        [self.action_mapping_original_to_rotated, self.action_mapping_rotated_to_original] = create_rotation_mapping(
            self.action_encoder.board_size
        )

    def evaluate_batch(self, games: list[Quoridor]):
        self.network.eval()  # Disables dropout

        inputs = []
        rotation_flags = []
        action_masks = []
        for game in games:
            # Rotate the board if player 2 is playing so that we always work with player 1's perspective.
            game, is_board_rotated = self.rotate_if_needed_to_point_downwards(game)
            inputs.append(self.game_to_input_array(game))
            rotation_flags.append(is_board_rotated)

            valid_actions = game.get_valid_actions()
            valid_action_indices = [self.action_encoder.action_to_index(action) for action in valid_actions]
            valid_indices_tensor = torch.tensor(valid_action_indices, device=self.device)
            invalid_mask = torch.ones(self.action_encoder.num_actions, dtype=torch.bool, device=self.device)
            invalid_mask[valid_indices_tensor] = False
            action_masks.append(invalid_mask)
        inputs_tensor = torch.Tensor(np.stack(inputs)).to(self.device).float()
        action_masks_tensor = torch.Tensor(np.stack[action_masks]).to(torch.float32).to(self.device)

        # Run the network on the entire batch
        with torch.no_grad():
            policy_logits_tensor, values_tensor = self.network(inputs_tensor)

        # Leave the policy tensors on the device while we mask and softmax
        assert torch.isfinite(policy_logits_tensor).all(), "Policy logits contains non-finite values"
        policy_logits_tensor[action_masks_tensor] = INVALID_ACTION_VALUE
        normalized_policy_tensor = F.softmax(policy_logits_tensor, dim=-1)

        # Transfer policies and values back to CPU and turn them into arrays
        policy_logits_array = policy_logits_tensor.cpu().numpy()
        values_array = values_tensor.cpu().numpy()

        # Prepare a tensor with True only on INVALID action positions so that we can set those values to
        # a large negative number below, to discourage choosing them.
        for game_i in range(len(games)):
            policy_logit = policy_

            is_board_rotated = rotation_flags[game_i]
            if is_board_rotated

    def evaluate(self, game: Quoridor):
        game, is_board_rotated = self.rotate_if_needed_to_point_downwards(game)
        input_array = self.game_to_input_array(game)
        action_mask = torch.from_numpy(game.get_action_mask()).to(torch.float32).to(self.device)

        if self.network.training:
            self.network.eval()  # Disables dropout

        with torch.no_grad():
            policy_logits, value = self.network(torch.from_numpy(input_array).float().to(self.device))
            assert torch.isfinite(policy_logits).all(), "Policy logits contains non-finite values"
            policy_logits = policy_logits * action_mask + INVALID_ACTION_VALUE * (1 - action_mask)
            policy_masked = F.softmax(policy_logits, dim=-1).cpu().numpy()
            value = value.item()

        # Sanity checks
        assert np.all(policy_masked >= 0), "Policy contains negative probabilities"
        assert np.all(policy_masked <= 1), "Policy contains probabilities greater than 1"
        assert np.any(policy_masked > 0), "Policy is all zeros"
        assert np.isfinite(policy_masked).all() and np.isfinite(value), "Policy or value is non-finite"

        # If the game was originally rotated, rotate the resulting back to player 2's perspective
        if is_board_rotated:
            policy_masked = policy_masked[self.action_mapping_rotated_to_original]

        return value, policy_masked

    def rotate_policy_from_original(self, policy: np.ndarray):
        """
        Rotate the policy vector to match the current player's perspective.
        """
        return policy[self.action_mapping_original_to_rotated]

    def rotate_if_needed_to_point_downwards(self, game: Quoridor):
        """
        Rotates the game so that the current player's goal is the row with the largest index.

        This makes it easier for the neural network to learn, since it doesn't need to
        understand that one player wants to move up the board and the other wants to move down it.
        If the current player is Player.ONE, this is a no-op since the board
        is already rotated correctly for it, and the same game instance is returned. If
        the current player is Player.TWO, the returned game is a copy of the original game
        which has beeen rotated appropriately.
        """
        player = game.get_current_player()

        is_rotated = True if (player == Player.TWO) else False
        if is_rotated:
            game = game.create_new()
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

    def train_prepare(self, learning_rate, batch_size, batches_per_iteration, weight_decay: float = 0):
        assert not hasattr(self, "optimizer") or self.optimizer is None, "train_prepare should be called only once"

        self.batch_size = batch_size
        self.batches_per_iteration = batches_per_iteration
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train_iteration(self, replay_buffer):
        assert self.optimizer is not None, "Call train_prepare before training"
        if not self.network.training:
            self.network.train()  # Make sure we aren't in eval mode, which disables dropout

        for _ in range(self.batches_per_iteration):
            # Sample random batch from replay buffer
            batch_data = random.sample(list(replay_buffer), self.batch_size)

            # Prepare batch tensors
            inputs = []

            # The network predicts the natural log of the probability of each next action. When we use
            # it to make predictions, we take the softamx of its output which un-logifies it and also
            # normalizes it to a valid probability distribution.
            target_policies = []

            target_values = []

            for data in batch_data:
                inputs.append(torch.from_numpy(data["input_array"]))
                target_policies.append(torch.FloatTensor(data["mcts_policy"]))
                target_values.append(torch.FloatTensor([data["value"]]))

            inputs = torch.stack(inputs).to(self.device)
            target_policies = torch.stack(target_policies).to(self.device)
            target_values = torch.stack(target_values).to(self.device)

            assert not (inputs.isnan().any() or target_policies.isnan().any() or target_values.isnan().any()), (
                "NaN in training data"
            )

            # Forward pass
            pred_logits, pred_values = self.network(inputs)
            # TODO: Should we apply masking before calculating cross-entropy here?

            # Compute losses
            policy_loss = F.cross_entropy(pred_logits, target_policies, reduction="mean")
            value_loss = F.mse_loss(pred_values.squeeze(), target_values.squeeze(), reduction="mean")
            total_loss = policy_loss + value_loss
            # print(f"{total_loss.item():3.3f} {policy_loss.item():3.3f} {value_loss.item():3.3f}")

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return {"total_loss": total_loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item()}
