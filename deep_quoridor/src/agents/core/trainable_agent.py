import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.core.agent import Agent
from agents.core.replay_buffer import ReplayBuffer


class AbstractTrainableAgent(Agent):
    """Base class for trainable agents using neural networks."""

    def __init__(
        self,
        board_size,
        max_walls,
        epsilon=0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        gamma=0.99,
        batch_size=64,
        update_target_every=100,
        assing_negative_reward=False,
        **kwargs,
    ):
        super().__init__()
        self.board_size = board_size
        self.max_walls = max_walls
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.assign_negative_reward = assing_negative_reward
        self.final_reward_multiplier = 1
        self.action_size = self._calculate_action_size()

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.online_network = self._create_network()
        self.target_network = self._create_network()
        self.online_network.to(self.device)
        self.target_network.to(self.device)

        self.update_target_network()

        # Setup training components
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_criterion()
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.training_mode = False
        self.episodes_rewards = []
        self.train_call_losses = []
        self.reset_episode_related_info()

    def reset_episode_related_info(self):
        self.current_episode_reward = 0
        self.player_id = None
        self.games_count = 0

    def is_trainable(self):
        return True

    def start_game(self, game, player_id):
        self.reset_episode_related_info()
        self.player_id = player_id

    def end_game(self, game):
        """Store episode results and reset tracking"""
        self.episodes_rewards.append(self.current_episode_reward)
        self.games_count += 1
        self._update_epsilon()
        if (self.games_count % self.update_target_every) == 0:
            self.update_target_network()

    def handle_opponent_step_outcome(self, observation_before_action, action, game):
        pass

    def handle_step_outcome(self, observation_before_action, action, game):
        if not self.training_mode:
            return
        reward = game.rewards[self.player_id]

        if game.is_done():
            reward *= self.final_reward_multiplier

        # Handle end of episode
        if action is None:
            if self.assign_negative_reward and len(self.replay_buffer) > 0:
                last = self.replay_buffer.get_last()
                last[2] = reward  # update final reward
                last[4] = 1.0  # mark as done
                self.current_episode_reward += reward
            return

        state_before_action = self.observation_to_tensor(observation_before_action)
        state_after_action = self.observation_to_tensor(game.observe(self.player_id))
        done = game.is_done()
        self.current_episode_reward += reward

        self.replay_buffer.add(
            state_before_action.cpu().numpy(),
            action,
            reward,
            state_after_action.cpu().numpy()
            if state_after_action is not None
            else np.zeros_like(state_before_action.cpu().numpy()),
            float(done),
        )

        if len(self.replay_buffer) > self.batch_size:
            loss = self.train(self.batch_size)
            if loss is not None:
                self.train_call_losses.append(loss)

    def _calculate_action_size(self):
        """Calculate the size of the action space."""
        raise NotImplementedError("Subclasses must implement _calculate_action_size")

    def _create_network(self):
        """Create the neural network model."""
        raise NotImplementedError("Subclasses must implement _create_network")

    def _create_optimizer(self):
        """Create the optimizer for training."""
        return optim.Adam(self.online_network.parameters(), lr=0.001)

    def _create_criterion(self):
        """Create the loss criterion."""
        return nn.MSELoss().to(self.device)

    def observation_to_tensor(self, observation):
        """Convert observation to tensor format."""
        raise NotImplementedError("Subclasses must implement observation_to_tensor")

    def update_target_network(self):
        """Copy parameters from online network to target network."""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def get_action(self, game):
        """Select an action using epsilon-greedy policy."""
        observation, _, termination, truncation, _ = game.last()
        if termination or truncation:
            return None

        mask = observation["action_mask"]

        if random.random() < self.epsilon:
            valid_actions = self._get_valid_actions(mask)
            return self._select_random_action(valid_actions)

        return self._get_best_action(observation, mask)

    def _get_valid_actions(self, mask):
        """Get valid actions from the action mask."""
        return np.where(mask == 1)[0]

    def _select_random_action(self, valid_actions):
        """Select a random action from valid actions."""
        return np.random.choice(valid_actions)

    def convert_action_mask_to_tensor(self, mask):
        return torch.tensor(mask, dtype=torch.float32, device=self.device)

    def convert_to_action_from_tensor_index(self, action_index_in_tensor):
        return action_index_in_tensor

    def _log_action(self, q_values):
        if not self.action_log.is_enabled():
            return

        self.action_log.clear()

        # Log the 5 best actions, as long as the value is > -100 (arbitrary value)
        top_values, top_indices = torch.topk(q_values, min(5, len(q_values)))
        scores = {
            int(self.convert_to_action_from_tensor_index(i.item())): v.item()
            for v, i in zip(top_values, top_indices)
            if v.item() >= -100
        }
        self.action_log.action_score_ranking(scores)

    def _get_best_action(self, observation, mask):
        """Get the best action based on Q-values."""
        state = self.observation_to_tensor(observation)
        with torch.no_grad():
            q_values = self.online_network(state)

        mask_tensor = self.convert_action_mask_to_tensor(mask)
        q_values = q_values * mask_tensor - 1e9 * (1 - mask_tensor)
        self._log_action(q_values)

        selected_action = torch.argmax(q_values).item()
        idx = self.convert_to_action_from_tensor_index(selected_action)
        assert mask[idx] == 1
        return idx

    def train(self, batch_size):
        """Train the network on a batch of samples."""
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        loss = self._train_on_batch(batch)

        return loss

    def _train_on_batch(self, batch):
        """Train the network on a single batch."""
        states, actions, rewards, next_states, dones = self._prepare_batch(batch)

        current_q_values = self._compute_current_q_values(states, actions)
        target_q_values = self._compute_target_q_values(rewards, next_states, dones)

        loss = self._update_network(current_q_values, target_q_values)
        return loss

    def _prepare_batch(self, batch):
        """Prepare batch data for training."""
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([torch.FloatTensor(s).to(self.device) for s in states])
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack([torch.FloatTensor(s).to(self.device) for s in next_states])
        dones = torch.FloatTensor(dones).to(self.device)

        return states, actions, rewards, next_states, dones

    def _compute_current_q_values(self, states, actions):
        """Compute current Q-values."""
        return self.online_network(states).gather(1, actions).squeeze()

    def _compute_target_q_values(self, rewards, next_states, dones):
        """Compute target Q-values."""
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            return rewards + (1 - dones) * self.gamma * next_q_values

    def _update_network(self, current_q_values, target_q_values):
        """Update the network using computed Q-values."""
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _update_epsilon(self):
        """Update epsilon value for exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def resolve_filename(self, suffix):
        filename = (
            f"{Agent._friendly_name(self.__class__.__base__.__name__)}_B{self.board_size}W{self.max_walls}_{suffix}.pt"
        )
        return filename

    def save_model(self, path):
        """Save the model to disk."""
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        """Load the model from disk."""
        print(f"Loading pre-trained model from {path}")
        self.online_network.load_state_dict(torch.load(path))
        self.update_target_network()

    def load_pretrained_file(self):
        filename = self.resolve_filename("final")
        model_path = Path(__file__).resolve().parents[4] / "models" / filename
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file {model_path} not found. Please provide the weights file.")
