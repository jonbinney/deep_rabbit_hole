import copy
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from quoridor import Action, ActionEncoder, Quoridor, construct_game_from_observation
from utils import my_device
from utils.subargs import SubargsBase

from agents.core import Agent


def game_to_tensor(game: Quoridor, device):
    """Convert Quoridor observation to tensor format for neural network."""
    player = game.get_current_player()
    opponent = int(1 - player)

    player_position = game.board.get_player_position(player)
    opponent_position = game.board.get_player_position(opponent)

    board = np.zeros((game.board.board_size, game.board.board_size), dtype=np.int8)
    board[player_position] = 1
    board[opponent_position] = 2

    # Make a copy of walls
    walls = game.board.get_old_style_walls()

    my_walls = game.board.get_walls_remaining(player)
    opponent_walls = game.board.get_walls_remaining(opponent)
    my_turn = 1.0  # Assume it is our turn

    # Flatten board and walls
    board_flat = board.flatten()
    walls_flat = walls.flatten()

    # Combine all features into single tensor
    features = np.concatenate([board_flat, walls_flat, [my_walls, opponent_walls, my_turn]])

    return torch.FloatTensor(features).to(device)


@dataclass
class AlphaZeroParams(SubargsBase):
    training_mode: bool = False

    # After how many self play games we train the network
    train_every: int = 10

    # Learning rate to use for the optimizer
    learning_rate: float = 0.001

    # Exploration vs exploitation.  0 is pure exploitation, infinite is random exploration.
    temperature: float = 1.0

    # Training parameters specific to AlphaZero
    replay_buffer_size: int = 10000

    # Number of MCTS selections
    n: int = 1000
    # A higher number favors exploration over exploitation
    c: float = 1.4


class AzNode:
    def __init__(
        self,
        game: Quoridor,
        parent: Optional["AzNode"] = None,
        action_taken: Optional[Action] = None,
        ucb_c: float = 1.0,
        prior: float = 0.0,
    ):
        self.game = game
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0

        self.ucb_c = ucb_c
        self.prior = prior

        self.action_encoder = ActionEncoder(game.board.board_size)

    def should_expand(self):
        return len(self.children) == 0

    def expand(self, policy_probs: np.ndarray):
        """
        Create a child of the current node.
        """
        for action_index, prob in enumerate(policy_probs):
            if prob == 0.0:
                continue

            action = self.action_encoder.index_to_action(action_index)
            game = copy.deepcopy(self.game)
            game.step(action)

            child = AzNode(game, parent=self, action_taken=action, ucb_c=self.ucb_c, prior=prob)
            self.children.append(child)

    def select(self) -> "AzNode":
        """
        Return the child of the current node with the highest ucb
        """
        return max(self.children, key=self.get_ucb)

    def get_ucb(self, child):
        if child.visit_count == 0:
            return self.ucb_c * self.prior * math.sqrt(self.visit_count)

        # value_sum is in between -1 and 1, so doing (avg + 1) / 2 would make it in the range [0, 1]
        q_value = ((child.value_sum / child.visit_count) + 1) / 2

        return q_value + self.ucb_c * self.prior * math.sqrt(self.visit_count) / (child.visit_count + 1)

    def backpropagate(self, value: float):
        """
        Update the nodes from the current node up to the tree by increasing the visit count and adding the value
        """
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(-value)


class AzMCTS:
    def __init__(self, nn: nn.Module, params: AlphaZeroParams):
        self.nn = nn
        self.params = params

    def select(self, node: AzNode) -> AzNode:
        """
        Select a node to expand, starting from the given node.
        As long as the passed node has leaves to expand, that one will be selected.
        Otherwise, if the node is fully expanded, then its best child will be selected.
        """
        while not node.should_expand():
            node = node.select()
        return node

    def search(self, initial_game: Quoridor):
        root = AzNode(initial_game, ucb_c=self.params.c)
        good_player = initial_game.current_player

        for _ in range(self.params.n):
            # Traverse down the tree guided by maximum UCB until we find a node to expand
            node = self.select(root)

            if node.game.check_win(good_player):
                value = 1
            elif node.game.check_win(1 - good_player):
                value = -1
            else:
                with torch.no_grad():
                    input_tensor = game_to_tensor(node.game, my_device())
                    policy, value = self.nn(input_tensor)

                # Mask the policy to ignore invalid actions
                valid_actions = node.game.get_valid_actions()
                valid_action_indices = [node.game.action_encoder.action_to_index(action) for action in valid_actions]
                policy_masked = torch.zeros_like(policy)
                policy_masked[valid_action_indices] = policy[valid_action_indices]

                # Re-normalize probbailities after masking.
                policy_probs = policy_masked.cpu().numpy()
                policy_probs = policy_probs / policy_probs.sum()
                node.expand(policy_probs)

            node.backpropagate(value)

        return root.children


class AlphaZeroNetwork(nn.Module):
    def __init__(self, board_size, action_size):
        super(AlphaZeroNetwork, self).__init__()

        # Calculate input dimensions
        # Board: board_size x board_size (combined player positions)
        # Walls: (board_size-1) x (board_size-1) x 2 channels (horizontal/vertical)
        # Additional features: my_walls_remaining, opponent_walls_remaining, my_turn
        board_input_size = board_size * board_size
        walls_input_size = (board_size - 1) * (board_size - 1) * 2
        additional_features = 3  # walls remaining for both players + turn indicator

        input_size = board_input_size + walls_input_size + additional_features

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
        self.policy_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_size), nn.Softmax())

        # Value head - outputs position evaluation (-1 to 1)
        self.value_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

        self.to(my_device())

    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.stack([i for i in x]).to(my_device())

        shared_features = self.shared(x)
        policy = self.policy_head(shared_features)
        value = self.value_head(shared_features)

        return policy, value


class AlphaZeroAgent(Agent):
    def __init__(
        self,
        board_size,
        max_walls,
        observation_space,
        action_space,
        training_instance=None,
        params=AlphaZeroParams(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.params = params
        self.board_size = board_size
        self.max_walls = max_walls
        self.action_space = action_space
        self.action_size = self.board_size**2 + (self.board_size - 1) ** 2 * 2

        # The parent class may cause us to load a model. If not, and we're in training mode, create it ourselves.
        if params.training_mode:
            self.nn = AlphaZeroNetwork(self.board_size, self.action_size)
        else:
            assert False, "Only training mode supported"

        self.episode_count = 0

        self.mcts = AzMCTS(self.nn, params)
        self.action_encoder = ActionEncoder(board_size)

        # When playing use 0.0 for temperature so we always chose the best available action.
        self.temperature = params.temperature if params.training_mode else 0.0

        # Training data storage
        self.replay_buffer = deque(maxlen=params.replay_buffer_size)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=params.learning_rate)

        # Metrics tracking
        self.recent_losses = []
        self.recent_rewards = []

    def version(self):
        return "1.0"

    def model_name(self):
        return "alphazero"

    @classmethod
    def params_class(cls):
        return AlphaZeroParams

    def model_id(self):
        return f"{self.model_name()}_B{self.board_size}W{self.max_walls}_mv{self.version()}"

    def resolve_filename(self, suffix):
        return f"{self.model_id()}{suffix}.pt"

    def save_model(self, path):
        # Create directory for saving models if it doesn't exist
        os.makedirs(Path(path).absolute().parents[0], exist_ok=True)

        # Save the neural network state dict
        model_state = {
            "network_state_dict": self.nn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode_count": self.episode_count,
            "board_size": self.board_size,
            "max_walls": self.max_walls,
            "params": self.params.__dict__,
        }
        torch.save(model_state, path)
        print(f"AlphaZero model saved to {path}")

    def load_model(self, path):
        """Load the model from disk."""
        print(f"Loading pre-trained model from {path}")

        try:
            model_state = torch.load(path, map_location=my_device())

            # Load network state
            self.nn = AlphaZeroNetwork(self.board_size, self.action_size)
            self.nn.load_state_dict(model_state["network_state_dict"])

            # Load optimizer state if available
            if "optimizer_state_dict" in model_state and hasattr(self, "optimizer"):
                self.optimizer.load_state_dict(model_state["optimizer_state_dict"])

            # Load episode count if available
            if "episode_count" in model_state:
                self.episode_count = model_state["episode_count"]

            print(f"Successfully loaded AlphaZero model from {path}")
            print(f"Model was trained for {self.episode_count} episodes")

        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            raise

    def start_game(self, game, player_id):
        self.player_id = player_id

    def end_game(self, game):
        if not self.params.training_mode:
            return

        # Track current episode reward for metrics
        reward = game.rewards[self.agent_id]

        # Store reward for metrics
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 1000:  # Keep only recent rewards
            self.recent_rewards = self.recent_rewards[-1000:]

        # Assign the final game outcome to all positions in this episode
        # For Quoridor: reward = 1 for win, -1 for loss, 0 for draw
        episode_positions = []
        while self.replay_buffer and self.replay_buffer[-1]["value"] is None:
            position = self.replay_buffer.pop()
            position["value"] = reward
            episode_positions.append(position)

        # Add back the positions with assigned values
        self.replay_buffer.extend(reversed(episode_positions))

        self.episode_count += 1

        # Train network if we have enough episodes
        if self.episode_count % self.params.train_every == 0:
            self.train_network()

    def compute_loss_and_reward(self, length: int) -> Tuple[float, float]:
        # Return some basic metrics if available
        if hasattr(self, "recent_losses") and self.recent_losses:
            avg_loss = sum(self.recent_losses[-length:]) / min(length, len(self.recent_losses))
        else:
            avg_loss = 0.0

        # For AlphaZero, reward is based on win/loss, not cumulative
        avg_reward = 0.0
        if hasattr(self, "recent_rewards") and self.recent_rewards:
            avg_reward = sum(self.recent_rewards[-length:]) / min(length, len(self.recent_rewards))

        return avg_loss, avg_reward

    def train_network(self):
        """Train the neural network on collected self-play data."""
        if len(self.replay_buffer) < self.params.batch_size:
            return

        # Sample random batch from replay buffer
        batch_data = random.sample(list(self.replay_buffer), self.params.batch_size)

        # Prepare batch tensors
        states = []
        target_policies = []
        target_values = []

        for data in batch_data:
            state_tensor = game_to_tensor(data["game"], self.board_size, self.device)
            states.append(state_tensor)
            target_policies.append(torch.FloatTensor(data["mcts_policy"]).to(self.device))
            target_values.append(torch.FloatTensor([data["value"]]).to(self.device))

        states = torch.stack(states)
        target_policies = torch.stack(target_policies)
        target_values = torch.stack(target_values)

        # Forward pass
        pred_policies, pred_values = self.nn(states)

        # Compute losses
        policy_loss = F.cross_entropy(pred_policies, target_policies)
        value_loss = F.mse_loss(pred_values.squeeze(), target_values.squeeze())
        total_loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Store loss for metrics
        self.recent_losses.append(total_loss.item())
        if len(self.recent_losses) > 1000:  # Keep only recent losses
            self.recent_losses = self.recent_losses[-1000:]

        return {"total_loss": total_loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item()}

    def get_action(self, observation) -> int:
        game, _, _ = construct_game_from_observation(observation["observation"], self.player_id)

        # Run MCTS to get action visit counts
        root_children = self.mcts.search(game)
        visit_counts = np.zeros(self.action_size, dtype=np.float32)
        for child in root_children:
            action_index = self.action_encoder.action_to_index(child.action_taken)
            visit_counts[action_index] = child.visit_count

        if self.temperature == 0.0:
            # Greedy selection - choose action with highest visit count
            action = np.argmax(visit_counts)
        else:
            # Temperature-based selection
            # Convert visit counts to probabilities
            if np.sum(visit_counts) == 0:
                # If no visits, uniform over valid actions
                valid_actions = np.where(observation["action_mask"])[0]
                action = np.random.choice(valid_actions)
            else:
                # Apply temperature
                visit_probs = visit_counts / np.sum(visit_counts)
                if self.temperature != 1.0:
                    visit_probs = visit_probs ** (1.0 / self.temperature)
                    visit_probs = visit_probs / np.sum(visit_probs)

                # Sample from probability distribution
                action = np.random.choice(len(visit_counts), p=visit_probs)

        # Store training data if in training mode
        if self.params.training_mode:
            # Convert visit counts to policy target (normalized)
            policy_target = visit_counts / np.sum(visit_counts) if np.sum(visit_counts) > 0 else visit_counts
            self.store_training_data(game, policy_target)

        return int(action)

    def store_training_data(self, game, mcts_policy):
        """Store training data for later use in training."""
        self.replay_buffer.append(
            {
                "game": copy.deepcopy(game),
                "mcts_policy": mcts_policy.copy(),
                "value": None,  # Will be filled in at end of episode
            }
        )
