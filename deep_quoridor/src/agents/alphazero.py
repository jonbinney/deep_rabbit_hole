import math
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import my_device

from agents.core.trainable_agent import AbstractTrainableAgent, TrainableAgentParams


def observation_to_tensor(observation, board_size, device):
    """Convert Quoridor observation to tensor format for neural network."""
    # Extract components from observation - handle nested structure
    if "observation" in observation:
        obs_data = observation["observation"]
    else:
        obs_data = observation
        
    board = obs_data["board"]  # board_size x board_size with player positions
    walls = obs_data["walls"]  # (board_size-1) x (board_size-1) x 2 for h/v walls
    my_walls = obs_data["my_walls_remaining"]
    opponent_walls = obs_data["opponent_walls_remaining"]
    my_turn = 1.0 if obs_data["my_turn"] else 0.0

    # Flatten board and walls
    board_flat = board.flatten()
    walls_flat = walls.flatten()

    # Combine all features into single tensor
    features = np.concatenate([board_flat, walls_flat, [my_walls, opponent_walls, my_turn]])

    return torch.FloatTensor(features).to(device)


@dataclass
class AlphaZeroParams(TrainableAgentParams):
    # After how many self play games we train the network
    train_every: int = 100

    # Exploration vs exploitation.  0 is pure exploitation, infinite is random exploration.
    temperature: float = 1.0

    # MCTS parameters
    mcts_simulations: int = 800
    c_puct: float = 1.0  # UCB exploration constant

    # Training parameters specific to AlphaZero
    replay_buffer_size: int = 10000


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
        self.policy_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_size))

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


class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior_prob=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior_prob = prior_prob

        self.children = {}  # action -> child node
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def get_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def get_ucb_score(self, c_puct):
        if self.visit_count == 0:
            return float("inf")

        exploration = c_puct * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.get_value() + exploration

    def select_child(self, c_puct):
        return max(self.children.values(), key=lambda child: child.get_ucb_score(c_puct))

    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = MCTSNode(None, parent=self, action=action, prior_prob=prob)
        self.is_expanded = True

    def backup(self, value):
        self.visit_count += 1
        self.value_sum += value
        if not self.is_root():
            self.parent.backup(-value)  # Flip value for opponent


class MCTS:
    def __init__(self, network, c_puct=1.0):
        self.network = network
        self.c_puct = c_puct

    def search(self, root_state, action_mask, num_simulations):
        root = MCTSNode(root_state)

        for _ in range(num_simulations):
            node = root

            # Selection - traverse down to leaf
            while not node.is_leaf() and node.is_expanded:
                node = node.select_child(self.c_puct)

            # Expansion and Evaluation
            if not node.is_expanded:
                # Get network predictions for this state
                with torch.no_grad():
                    state_tensor = self._state_to_tensor(node.state if node.state else root_state)
                    policy_logits, value = self.network(state_tensor.unsqueeze(0))

                    # Apply action mask and get valid action probabilities
                    valid_actions = np.where(action_mask)[0]
                    policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]

                    # Normalize probabilities for valid actions only
                    valid_probs = [(action, policy_probs[action]) for action in valid_actions]
                    total_prob = sum(prob for _, prob in valid_probs)
                    if total_prob > 0:
                        valid_probs = [(action, prob / total_prob) for action, prob in valid_probs]
                    else:
                        # Uniform distribution if all probabilities are 0
                        uniform_prob = 1.0 / len(valid_actions)
                        valid_probs = [(action, uniform_prob) for action in valid_actions]

                    node.expand(valid_probs)

                    # Backup the value
                    node.backup(value.item())
            else:
                # If already expanded, just backup a default value
                node.backup(0.0)

        # Return visit counts for all actions
        action_visits = np.zeros(len(action_mask))
        for child in root.children.values():
            action_visits[child.action] = child.visit_count

        return action_visits

    def _state_to_tensor(self, observation):
        return observation_to_tensor(observation, self.board_size, my_device())

    def set_board_size(self, board_size):
        self.board_size = board_size


class AlphaZeroAgent(AbstractTrainableAgent):
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
        super().__init__(board_size, max_walls, observation_space, action_space, params=params, **kwargs)
        self.episode_count = 0

        # If a training instance is passed, this instance is playing to train the other, sharing the NN and temperature
        if training_instance:
            self.nn = training_instance.nn
            self.mcts = training_instance.mcts
            self.temperature = training_instance.temperature
        else:
            self.nn = self._create_network()
            self.mcts = MCTS(self.nn, params.c_puct)
            self.mcts.set_board_size(board_size)

            # When playing use 0.0 for temperature so we always chose the best available action.
            self.temperature = params.temperature if self.training_mode else 0.0

        # Training data storage
        self.replay_buffer = deque(maxlen=params.replay_buffer_size)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=params.learning_rate)

        # Metrics tracking
        self.recent_losses = []
        self.recent_rewards = []

        # TODO remove, this is because TrainingStatusRenderer assumes we have epsilon, we need a workaround
        self.epsilon = 0

    def _calculate_action_size(self):
        """Calculate the size of the action space."""
        return self.board_size**2 + (self.board_size - 1) ** 2 * 2

    def _create_network(self):
        """Create the neural network model."""
        return AlphaZeroNetwork(self.board_size, self.action_size)

    def _observation_to_tensor(self, observation, obs_player_id):
        """Convert observation to tensor format."""
        return observation_to_tensor(observation, self.board_size, self.device)

    def version(self):
        return "1.0"

    def model_name(self):
        return "alphazero"

    @classmethod
    def params_class(cls):
        return AlphaZeroParams

    def is_training(self):
        return self.training_mode

    def model_id(self):
        return f"{self.model_name()}_B{self.board_size}W{self.max_walls}_mv{self.version()}"

    def resolve_filename(self, suffix):
        return f"{self.model_id()}{suffix}.pt"

    def save_model(self, path):
        # Create directory for saving models if it doesn't exist
        os.makedirs(Path(path).absolute().parents[0], exist_ok=True)
        
        # Save the neural network state dict
        model_state = {
            'network_state_dict': self.nn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'board_size': self.board_size,
            'max_walls': self.max_walls,
            'params': self.params.__dict__,
        }
        torch.save(model_state, path)
        print(f"AlphaZero model saved to {path}")

    def load_model(self, path):
        """Load the model from disk."""
        print(f"Loading pre-trained model from {path}")
        
        try:
            model_state = torch.load(path, map_location=my_device())
            
            # Load network state
            self.nn.load_state_dict(model_state['network_state_dict'])
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in model_state and hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(model_state['optimizer_state_dict'])
            
            # Load episode count if available
            if 'episode_count' in model_state:
                self.episode_count = model_state['episode_count']
            
            print(f"Successfully loaded AlphaZero model from {path}")
            print(f"Model was trained for {self.episode_count} episodes")
            
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            raise

    def handle_step_outcome(
        self,
        observation_before_action,
        opponent_observation_after_action,
        observation_after_action,
        reward,
        action,
        done=False,
    ):
        """Override to handle episode end properly for AlphaZero."""
        # For AlphaZero, we don't use the standard replay buffer from the parent class
        # Instead, we store training data in our own format and handle episode end differently
        
        # Only increment steps counter from parent class
        self.steps += 1
        
        if not self.training_mode:
            return
            
        # Track current episode reward for metrics
        self.current_episode_reward += reward

        # Handle episode end for AlphaZero training
        if done:
            self.handle_episode_end(reward, done)

    def end_game(self, game):
        """Override to handle episode end for AlphaZero."""
        super().end_game(game)
        # Additional AlphaZero-specific cleanup if needed

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

    def handle_episode_end(self, reward: float, done: bool):
        """Handle end of episode - assign final rewards to training data."""
        if not self.training_mode or not done:
            return

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
            state_tensor = observation_to_tensor(data["observation"], self.board_size, self.device)
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
        action_mask = observation["action_mask"]

        # Run MCTS to get action visit counts
        visit_counts = self.mcts.search(observation, action_mask, self.params.mcts_simulations)

        if self.temperature == 0.0:
            # Greedy selection - choose action with highest visit count
            action = np.argmax(visit_counts)
        else:
            # Temperature-based selection
            # Convert visit counts to probabilities
            if np.sum(visit_counts) == 0:
                # If no visits, uniform over valid actions
                valid_actions = np.where(action_mask)[0]
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
        if self.training_mode:
            # Convert visit counts to policy target (normalized)
            policy_target = visit_counts / np.sum(visit_counts) if np.sum(visit_counts) > 0 else visit_counts
            self.store_training_data(observation, policy_target)

        return int(action)

    def store_training_data(self, observation, mcts_policy):
        """Store training data for later use in training."""
        self.replay_buffer.append(
            {
                "observation": observation.copy(),
                "mcts_policy": mcts_policy.copy(),
                "value": None,  # Will be filled in at end of episode
            }
        )
