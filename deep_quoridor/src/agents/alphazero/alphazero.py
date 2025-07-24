import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import wandb
from quoridor import ActionEncoder, construct_game_from_observation
from utils import my_device
from utils.subargs import SubargsBase

from agents.alphazero.mcts import MCTS
from agents.alphazero.nn_evaluator import NNEvaluator
from agents.core import TrainableAgent


@dataclass
class AlphaZeroParams(SubargsBase):
    training_mode: bool = False

    # After how many self play games we train the network
    train_every: int = 10

    # Learning rate to use for the optimizer
    learning_rate: float = 0.001

    # Exploration vs exploitation.  0 is pure exploitation, infinite is random exploration.
    temperature: float = 1.0

    # How many moves to remember. The training batches are sampled from this replay buffer.
    replay_buffer_size: int = 1000

    # Batch size for training
    batch_size: int = 100

    # How many iterations of optimizer each time we update the model
    optimizer_iterations: int = 100

    # Number of MCTS selections
    mcts_n: int = 100

    # A higher number favors exploration over exploitation
    mcts_ucb_c: float = 1.4

    # If wandb_alias is provided, the model will be fetched from wandb using the model_id and the alias
    wandb_alias: Optional[str] = None

    # When loading from wandb, the project name to be used
    wandb_project: str = "deep_quoridor"

    # If a filename is provided, the model will be loaded from disc
    model_filename: Optional[str] = None

    # Directory where wandb models are stored
    wandb_dir: str = "wandbmodels"

    # Directory where local models are stored
    model_dir = "models"

    @classmethod
    def training_only_params(cls) -> set[str]:
        """
        Returns a set of parameters that are used only during training.
        These parameters should not be used during playing.
        """
        return {
            "training_mode",
            "train_every",
            "learning_rate",
            "optimizer_iterations",
            "batch_size",
            "replay_buffer_size",
        }


class AlphaZeroAgent(TrainableAgent):
    def __init__(
        self,
        board_size,
        max_walls,
        observation_space,
        action_space,
        params=AlphaZeroParams(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.params = params
        self.board_size = board_size
        self.max_walls = max_walls
        self.action_space = action_space
        self.device = my_device()

        self.action_encoder = ActionEncoder(board_size)
        self.evaluator = NNEvaluator(self.action_encoder, self.device)
        self.mcts = MCTS(params.mcts_n, params.mcts_ucb_c, self.evaluator)
        if params.training_mode:
            self.evaluator.train_prepare(params.learning_rate, params.batch_size, params.optimizer_iterations)

        self.episode_count = 0

        # When playing use 0.0 for temperature so we always chose the best available action.
        self.temperature = params.temperature if params.training_mode else 0.0

        self.replay_buffer = deque(maxlen=params.replay_buffer_size)

        # Metrics tracking
        self.recent_losses = []

        # Just to make the training status plugin happy
        self.epsilon = 0.0

        # Avoid circular imports
        from metrics import Metrics

        self.metrics = Metrics(board_size, max_walls)

    def version(self):
        return "1.0"

    def model_name(self):
        return "alphazero"

    def is_training(self):
        return self.params.training_mode

    @classmethod
    def params_class(cls):
        return AlphaZeroParams

    def model_id(self):
        return f"{self.model_name()}_B{self.board_size}W{self.max_walls}_mv{self.version()}"

    def resolve_filename(self, suffix):
        return f"{self.model_id()}{suffix}.pt"

    def wandb_local_filename(self, artifact: wandb.Artifact) -> str:
        return f"{self.model_id()}_{artifact.digest[:5]}.pt"

    def save_model(self, path):
        # Create directory for saving models if it doesn't exist
        os.makedirs(Path(path).absolute().parents[0], exist_ok=True)

        # Save the neural network state dict
        nn = self.evaluator.network
        model_state = {
            "network_state_dict": nn.state_dict(),
            "episode_count": self.episode_count,
            "board_size": self.board_size,
            "max_walls": self.max_walls,
            "params": self.params.__dict__,
        }
        torch.save(model_state, path)
        print(f"AlphaZero model saved to {path}")

    def load_model(self, path):
        self.evaluator.network.load_state_dict(torch.load(path, map_location=my_device()))

    def end_game(self, env):
        if not self.params.training_mode:
            return

        # Assign the final game outcome to all positions in this episode
        # For Quoridor: reward = 1 for win, -1 for loss, 0 for draw
        episode_positions = []
        while self.replay_buffer and self.replay_buffer[-1]["value"] is None:
            position = self.replay_buffer.pop()
            agent = env.player_to_agent[position["player"]]
            position["value"] = env.rewards[agent]
            episode_positions.append(position)

        # Add back the positions with assigned values
        self.replay_buffer.extend(reversed(episode_positions))

        self.episode_count += 1

        # Train network if we have enough episodes
        if self.episode_count % self.params.train_every == 0:
            self.train_iteration()

    def compute_loss_and_reward(self, length: int) -> Tuple[float, float]:
        # Return some basic metrics if available
        if hasattr(self, "recent_losses") and self.recent_losses:
            avg_loss = np.mean(self.recent_losses[-length:])
        else:
            avg_loss = 0.0

        return float(avg_loss), 0.0

    def train_iteration(self):
        """Train the neural network on collected data."""
        if len(self.replay_buffer) < self.params.batch_size:
            return
        t0 = time.time()
        print(
            f"Training the network (buffer size: {len(self.replay_buffer)}, batch size: {self.params.batch_size})...",
            end="",
        )

        metrics = self.evaluator.train_iteration(self.replay_buffer)

        # Store loss for metrics
        self.recent_losses.append(metrics["total_loss"])
        if len(self.recent_losses) > 100:  # Keep only recent losses
            self.recent_losses = self.recent_losses[-100:]

        print(f"done in {time.time() - t0:.2f}s")

    def _log_action(
        self,
        visit_probs,
        root_children,
    ):
        if not self.action_log.is_enabled():
            return

        self.action_log.clear()

        _, top_indices = torch.topk(torch.tensor(visit_probs), min(5, len(visit_probs)))

        scores = {root_children[i].action_taken: visit_probs[i] for i in top_indices}
        self.action_log.action_score_ranking(scores)

    def get_action(self, observation) -> int:
        game, player, _ = construct_game_from_observation(observation["observation"])

        # Run MCTS to get action visit counts
        root_children = self.mcts.search(game)
        visit_counts = np.array([child.visit_count for child in root_children])
        visit_counts_sum = np.sum(visit_counts)
        if visit_counts_sum == 0:
            raise RuntimeError("No nodes visited during MCTS")

        visit_probs = visit_counts / visit_counts_sum
        self._log_action(visit_probs, root_children)

        if self.temperature == 0.0:
            max_value = np.max(visit_probs)
            visit_probs = np.array([1.0 if v == max_value else 0.0 for v in visit_probs])
            visit_probs /= np.sum(visit_probs)
        else:
            visit_probs = visit_probs ** (1.0 / self.temperature)
            visit_probs = visit_probs / np.sum(visit_probs)

        # Sample from probability distribution
        best_child = np.random.choice(root_children, p=visit_probs)
        action = best_child.action_taken

        # Store training data if in training mode
        if self.params.training_mode:
            # Convert visit counts to policy target (normalized)
            policy_target = np.zeros(self.action_encoder.num_actions, dtype=np.float32)
            for child in root_children:
                action_index = self.action_encoder.action_to_index(child.action_taken)
                policy_target[action_index] = child.visit_count / visit_counts_sum
            self.store_training_data(game, policy_target, player)

        return self.action_encoder.action_to_index(action)

    def store_training_data(self, game, mcts_policy, player):
        """Store training data for later use in training."""
        game, is_rotated = self.evaluator.rotate_if_needed_to_point_downwards(game)
        input_array = self.evaluator.game_to_input_array(game)
        if is_rotated:
            mcts_policy = self.evaluator.rotate_policy_from_original(mcts_policy)
        self.replay_buffer.append(
            {
                "input_array": input_array,
                "mcts_policy": mcts_policy,
                "value": None,  # Will be filled in at end of episode
                "player": player,
            }
        )
