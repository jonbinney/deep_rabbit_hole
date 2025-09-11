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
        temp_input = NNEvaluator.game_to_input_array(temp_game)
        self.input_size = len(temp_input)

        self.network = MLPNetwork(self.input_size, self.action_encoder.num_actions, self.device)

        [self.action_mapping_original_to_rotated, self.action_mapping_rotated_to_original] = create_rotation_mapping(
            self.action_encoder.board_size
        )
        # fast hash -> (value, policy)
        self.cache = {}

    def evaluate_tensors(
        self, inputs_tensor: torch.Tensor, action_masks_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Caller is responsible for converting from games to a stack of input tensors, as well as rotating results
        """
        if self.network.training:
            self.network.eval()  # Disables dropout

        # Run the network on the entire batch
        with torch.no_grad():
            policy_logits_tensor, values_tensor = self.network(inputs_tensor)

            assert torch.isfinite(policy_logits_tensor).all(), "Policy logits contains non-finite values"

            # Leave the policy tensors on the device while we mask and softmax
            masked_policy_logits_tensor = policy_logits_tensor * action_masks_tensor + INVALID_ACTION_VALUE * (
                1 - action_masks_tensor
            )
            policies_tensor = F.softmax(masked_policy_logits_tensor, dim=-1)

            assert torch.all(policies_tensor >= 0), "Policy contains negative probabilities"
            assert torch.all(torch.sum(policies_tensor, dim=-1) - 1 < 1e-6), "Policy does not sum to 1"
            assert torch.all(torch.isfinite(values_tensor)), "Value is non-finite"

        return values_tensor, policies_tensor

    def evaluate(self, game: Quoridor, extra_games_to_evaluate: list[Quoridor] = []):
        if game.get_fast_hash() in self.cache:
            return self.cache[game.get_fast_hash()]

        all_games = [game] + extra_games_to_evaluate
        all_hashes = [g.get_fast_hash() for g in all_games]
        all_games = [NNEvaluator.rotate_if_needed_to_point_downwards(g)[0] for g in all_games]
        all_games_input_arrays = [torch.from_numpy(NNEvaluator.game_to_input_array(g)) for g in all_games]
        all_games_tensors = torch.stack(all_games_input_arrays).to(device=self.device)

        with torch.no_grad():
            action_masks = torch.stack([torch.from_numpy(g.get_action_mask()) for g in all_games]).to(
                device=self.device
            )
            values, policy_masked = self.evaluate_tensors(all_games_tensors, action_masks)
            values = values.cpu().numpy()
            policy_masked = policy_masked.cpu().numpy()

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

    @staticmethod
    def rotate_if_needed_to_point_downwards(game: Quoridor):
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

    @staticmethod
    def game_to_input_array(game: Quoridor) -> np.ndarray:
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

    def train_prepare(self, learning_rate, batch_size, batches_per_iteration, weight_decay: float):
        assert not hasattr(self, "optimizer") or self.optimizer is None, "train_prepare should be called only once"

        self.batch_size = batch_size
        self.batches_per_iteration = batches_per_iteration
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def split_data(self, replay_buffer, validation_ratio: float) -> tuple[list, list]:
        """
        Splits the replay buffer into a training set and a validation set.
        There are usually multiple entries with the same input (e.g. the initial state is always repeated in each game),
        so this function makes sure that all entries with the same input are either in the training set or in the validation set.
        """
        if validation_ratio == 0.0:
            return list(replay_buffer), []

        by_hash = {}

        for entry in replay_buffer:
            input_hash = hash(entry["input_array"].tobytes())
            if input_hash in by_hash:
                by_hash[input_hash].append(entry)
            else:
                by_hash[input_hash] = [entry]

        validation_size = int(len(replay_buffer) * validation_ratio)
        validation_set = []

        while len(validation_set) < validation_size:
            key = random.choice(list(by_hash.keys()))
            validation_set.extend(by_hash[key])
            del by_hash[key]

        training_set = []
        for entries in by_hash.values():
            training_set.extend(entries)

        return training_set, validation_set

    def compute_losses(self, batch_data):
        # Prepare batch tensors
        inputs = []

        # The network predicts the natural log of the probability of each next action. When we use
        # it to make predictions, we take the softmax of its output which un-logifies it and also
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

        return policy_loss, value_loss, total_loss

    def train_iteration(self, replay_buffer, validation_ratio: float = 0.0):
        assert self.optimizer is not None, "Call train_prepare before training"
        self.cache = {}
        if not self.network.training:
            self.network.train()  # Make sure we aren't in eval mode, which disables dropout

        training_set, validation_set = self.split_data(replay_buffer, validation_ratio)

        if validation_ratio > 0.0:
            print("==== Training & Validation Loss ====")
            print("  Epoch       Total   Val Total   Policy  Val Policy  Value   Val Value")
        else:
            print("==== Training Loss ====")
            print("  Epoch   Total   Policy  Value")

        # Show a fixed number of losses
        show_loss_every = max(1, self.batches_per_iteration // 25)

        for i in range(self.batches_per_iteration):
            # Sample random batch from replay buffer
            batch_data = random.sample(training_set, self.batch_size)

            policy_loss, value_loss, total_loss = self.compute_losses(batch_data)

            if validation_ratio > 0.0:
                with torch.no_grad():
                    val_policy_loss, val_value_loss, val_total_loss = self.compute_losses(validation_set)

                if i % show_loss_every == 0:
                    t, p, v = total_loss.item(), policy_loss.item(), value_loss.item()
                    vt, vp, vv = val_total_loss.item(), val_policy_loss.item(), val_value_loss.item()
                    print(f"{i:>7}     {t:>7.3f} {vt:>7.3f}     {p:>7.3f} {vp:>7.3f}     {v:>7.3f} {vv:>7.3f}")
            else:
                if i % show_loss_every == 0:
                    t, p, v = total_loss.item(), policy_loss.item(), value_loss.item()
                    print(f"{i:>7} {t:>7.3f} {p:>7.3f} {v:>7.3f}")

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return {"total_loss": total_loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item()}
