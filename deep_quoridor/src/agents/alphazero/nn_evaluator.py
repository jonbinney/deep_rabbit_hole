import random
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from quoridor import ActionEncoder, Player, Quoridor

from agents.alphazero.mlp_network import MLPNetwork
from agents.core.rotation import create_rotation_mapping

INVALID_ACTION_VALUE = -1e32


@dataclass
class EvaluatorStatistics:
    """
    Statistics about all evaluations since this evaluator was created.
    """

    duty_cycle: float
    average_evaluation_time: float
    num_evaluations: int
    num_cache_hits: int


class NNEvaluator:
    def __init__(
        self,
        action_encoder: ActionEncoder,
        device,
        network_type: str = "mlp",
    ):
        self.action_encoder = action_encoder
        self.device = device

        if network_type == "mlp":
            self.network = MLPNetwork(self.action_encoder, self.device)
        else:
            raise ValueError(f"Unknown network type: {network_type}")

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

    @torch.no_grad
    def evaluate_batch(self, games: list[Quoridor]):
        hashes = [g.get_fast_hash() for g in games]

        if all([h in self.cache for h in hashes]):
            # everything was cached!
            return [self.cache[h][0] for h in hashes], [self.cache[h][1] for h in hashes]

        # We may receive the game more than once, so this deduplicates it
        games_by_hash = {h: g for g, h in zip(games, hashes) if h not in self.cache}
        games_to_evaluate = games_by_hash.values()

        games_to_evaluate = [self.rotate_if_needed_to_point_downwards(g)[0] for g in games_to_evaluate]
        input_arrays = [torch.from_numpy(self.game_to_input_array(g)) for g in games_to_evaluate]
        tensors = torch.stack(input_arrays).to(device=self.device)
        action_masks = torch.stack([torch.from_numpy(g.get_action_mask()) for g in games_to_evaluate]).to(
            device=self.device
        )

        values, policy_masked = self.evaluate_tensors(tensors, action_masks)
        values = values.cpu().numpy()
        policy_masked = policy_masked.cpu().numpy()

        for i, g, h in zip(range(len(games_to_evaluate)), games_to_evaluate, games_by_hash.keys()):
            # If the game was originally rotated, rotate the resulting back to player 2's perspective
            if g.get_current_player() == Player.TWO:
                policy_masked[i] = policy_masked[i][self.action_mapping_rotated_to_original]

            self.cache[h] = (values[i][0], policy_masked[i])

        # list of values, list of policies
        return [self.cache[h][0] for h in hashes], [self.cache[h][1] for h in hashes]

    def evaluate(self, game: Quoridor):
        value, policy_masked = self.evaluate_batch([game])
        return value[0], policy_masked[0]

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
        return self.network.game_to_input_array(game)

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
        if len(batch_data) == 0:
            return None, None, None

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

    def train_iteration(
        self,
        replay_buffer,
        validation_ratio: float = 0.0,
        max_entries=25,
        on_new_entry: Optional[Callable[[dict], None]] = None,
    ):
        def make_entry(i, t, p, v, vt=None, vp=None, vv=None):
            if vt is None:
                return {
                    "step": i,
                    "completion": float(i) / self.batches_per_iteration,
                    "loss_total": t.item(),
                    "loss_policy": p.item(),
                    "loss_value": v.item(),
                }
            return {
                "step": i,
                "completion": float(i) / self.batches_per_iteration,
                "loss_total": t.item(),
                "loss_policy": p.item(),
                "loss_value": v.item(),
                "loss_total_val": vt.item(),
                "loss_policy_val": vp.item(),
                "loss_value_val": vv.item(),
            }

        assert self.optimizer is not None, "Call train_prepare before training"
        self.cache = {}
        if not self.network.training:
            self.network.train()  # Make sure we aren't in eval mode, which disables dropout

        training_set, validation_set = self.split_data(replay_buffer, validation_ratio)

        # Show a fixed number of losses
        show_loss_every = max(1, self.batches_per_iteration // (max_entries - 1))

        for i in range(self.batches_per_iteration):
            # Sample random batch from replay buffer
            batch_data = random.sample(training_set, self.batch_size)

            policy_loss, value_loss, total_loss = self.compute_losses(batch_data)

            if i % show_loss_every == 0 and on_new_entry is not None:
                with torch.no_grad():
                    v_policy_loss, v_value_loss, v_total_loss = self.compute_losses(validation_set)

                entry = make_entry(i, total_loss, policy_loss, value_loss, v_total_loss, v_policy_loss, v_value_loss)
                on_new_entry(entry)

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # Get a last entry for the losses after the last backprop
        with torch.no_grad():
            batch_data = random.sample(training_set, self.batch_size)
            policy_loss, value_loss, total_loss = self.compute_losses(batch_data)
            v_policy_loss, v_value_loss, v_total_loss = self.compute_losses(validation_set)
            entry = make_entry(i + 1, total_loss, policy_loss, value_loss, v_total_loss, v_policy_loss, v_value_loss)

            if on_new_entry is not None:
                on_new_entry(entry)

        return entry
