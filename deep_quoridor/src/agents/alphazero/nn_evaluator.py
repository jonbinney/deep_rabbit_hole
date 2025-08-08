import random

import numpy as np
import torch
import torch.nn.functional as F
from quoridor import ActionEncoder, Board, Player, Quoridor

from agents.alphazero.mlp_network import MLPNetwork
from agents.core.rotation import create_rotation_mapping

INVALID_ACTION_VALUE = -1e32


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

        [self.action_mapping_original_to_rotated, self.action_mapping_rotated_to_original] = create_rotation_mapping(
            self.action_encoder.board_size
        )
        # fast hash -> (value, policy)
        self.cache = {}

    def evaluate(self, game: Quoridor, extra_games_to_evaluate: list[Quoridor] = []):
        if game.get_fast_hash() in self.cache:
            return self.cache[game.get_fast_hash()]

        all_games = [game] + extra_games_to_evaluate
        all_hashes = [g.get_fast_hash() for g in all_games]
        all_games = [self.rotate_if_needed_to_point_downwards(g)[0] for g in all_games]
        all_games_input_arrays = [torch.from_numpy(self.game_to_input_array(g)) for g in all_games]
        all_games_tensors = torch.stack(all_games_input_arrays).to(device=self.device)

        if self.network.training:
            self.network.eval()  # Disables dropout

        with torch.no_grad():
            action_masks = torch.stack([torch.from_numpy(g.get_action_mask()) for g in all_games]).to(
                device=self.device
            )
            policy_logits, values = self.network(all_games_tensors)
            policy_logits = policy_logits * action_masks + INVALID_ACTION_VALUE * (1 - action_masks)
            policy_masked = F.softmax(policy_logits, dim=-1).cpu().numpy()
            values = values.cpu().numpy()

        for i, g in enumerate(all_games):
            pm = policy_masked[i]
            # Sanity checks
            assert np.all(pm >= 0), "Policy contains negative probabilities"
            assert np.abs(np.sum(pm) - 1) < 1e-6, "Policy does not sum to 1"
            assert np.isfinite(values[i]), "Policy or value is non-finite"

            # If the game was originally rotated, rotate the resulting back to player 2's perspective
            if g.get_current_player() == Player.TWO:
                policy_masked[i] = policy_masked[i][self.action_mapping_rotated_to_original]
            self.cache[all_hashes[i]] = (values[i][0], policy_masked[i])

        return values[0][0], policy_masked[0]

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

    def train_prepare(self, learning_rate, batch_size, batches_per_iteration, weight_decay: float = 0):
        assert not hasattr(self, "optimizer") or self.optimizer is None, "train_prepare should be called only once"

        self.batch_size = batch_size
        self.batches_per_iteration = batches_per_iteration
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train_iteration(self, replay_buffer):
        assert self.optimizer is not None, "Call train_prepare before training"
        self.cache = {}
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
