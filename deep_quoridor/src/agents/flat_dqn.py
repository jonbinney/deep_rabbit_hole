import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from agents import SelfRegisteringAgent


class DQNNetwork(nn.Module):
    """
    Neural network model for Deep Q-learning.
    Takes observation from the Quoridor game and outputs Q-values for each action.
    """

    def __init__(self, board_size, action_size):
        super(DQNNetwork, self).__init__()

        # Calculate input dimensions based on observation space
        # Board is board_size x board_size with 2 channels (player position and opponent position)
        # Walls are (board_size-1) x (board_size-1) with 2 channels (vertical and horizontal walls)
        board_input_size = board_size * board_size
        walls_input_size = (board_size - 1) * (board_size - 1) * 2

        # Additional features: walls remaining for both players
        flat_input_size = board_input_size + walls_input_size + 2

        # Define network architecture
        self.model = nn.Sequential(
            nn.Linear(flat_input_size, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.model(x)


class ReplayBuffer:
    """
    Experience replay buffer to store and sample transitions.
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class FlatDQNAgent(SelfRegisteringAgent):
    """
    Agent that uses Deep Q-Network for action selection.
    """

    def __init__(self, board_size, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, gamma=0.99):
        super(FlatDQNAgent, self).__init__()
        self.board_size = board_size
        # Assumes action representation is a flat array of size board_size**2 + (board_size - 1)**2 * 2
        # See quoridor_env.py for details
        self.action_size = board_size**2 + (board_size - 1) ** 2 * 2
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma  # Discount factor

        # Initialize Q-networks (online and target)
        self.online_network = DQNNetwork(board_size, self.action_size)
        self.target_network = DQNNetwork(board_size, self.action_size)
        self.update_target_network()

        # Set up optimizer
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online_network.to(self.device)
        self.target_network.to(self.device)

    def update_target_network(self):
        """Copy parameters from online network to target network."""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def preprocess_observation(self, observation):
        """
        Convert the observation dict to a flat tensor.
        """
        obs = observation["observation"]
        board = obs["board"].flatten()
        walls = obs["walls"].flatten()
        my_walls = np.array([obs["my_walls_remaining"]])
        opponent_walls = np.array([obs["opponent_walls_remaining"]])

        # Concatenate all components
        flat_obs = np.concatenate([board, walls, my_walls, opponent_walls])
        return torch.FloatTensor(flat_obs).to(self.device)

    def get_action(self, game):
        """
        Select an action using epsilon-greedy policy.
        """
        observation, _, termination, truncation, _ = game.last()
        if termination or truncation:
            return None

        mask = observation["action_mask"]
        valid_actions = np.where(mask == 1)[0]

        # With probability epsilon, select a random action (exploration)
        if random.random() < self.epsilon:
            return np.random.choice(valid_actions)

        # Otherwise, select the action with the highest Q-value (exploitation)
        state = self.preprocess_observation(observation)
        with torch.no_grad():
            q_values = self.online_network(state)

        # Apply action mask to q_values
        mask_tensor = torch.FloatTensor(mask).to(self.device)
        q_values = q_values * mask_tensor - 1e9 * (1 - mask_tensor)

        return torch.argmax(q_values).item()

    def train(self, batch_size):
        """
        Train the network on a batch of samples from the replay buffer.
        """
        if len(self.replay_buffer) < batch_size:
            return

        # Sample a batch of transitions
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.stack([torch.FloatTensor(s) for s in states]).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack([torch.FloatTensor(s) for s in next_states]).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q values
        current_q_values = self.online_network(states).gather(1, actions).squeeze()

        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and update online network
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save_model(self, path):
        """Save the model to disk."""
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        """Load the model from disk."""
        self.online_network.load_state_dict(torch.load(path))
        self.update_target_network()


class Pretrained01FlatDQNAgent(FlatDQNAgent):
    """
    A FlatDQNAgent that is initialized with the pre-trained model from main.py.
    """

    def __init__(self, board_size, **kwargs):
        super(Pretrained01FlatDQNAgent, self).__init__(board_size)
        model_path = "/home/julian/aaae/deep-rabbit-hole/code/deep_rabbit_hole/models/dqn_agent_final.pt"
        if os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}")
            self.load_model(model_path)
        else:
            print(
                f"Warning: Model file {model_path} not found, using untrained agent. Ask Julian for the weights file."
            )
