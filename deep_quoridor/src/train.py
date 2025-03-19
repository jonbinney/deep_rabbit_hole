import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from agents import Agent
from quoridor_env import env


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


class DQNAgent(Agent):
    """
    Agent that uses Deep Q-Network for action selection.
    """

    def __init__(self, board_size, action_size, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, gamma=0.99):
        super(DQNAgent, self).__init__()
        self.board_size = board_size
        self.action_size = action_size
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma  # Discount factor

        # Initialize Q-networks (online and target)
        self.online_network = DQNNetwork(board_size, action_size)
        self.target_network = DQNNetwork(board_size, action_size)
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


def train_dqn(
    episodes,
    batch_size,
    update_target_every,
    board_size,
    max_walls,
    save_path="models",
    model_name="dqn_agent",
    save_frequency=100,
    step_rewards=True,
):
    """
    Train a DQN agent to play Quoridor.

    Args:
        episodes: Number of episodes to train for
        batch_size: Size of batches to sample from replay buffer
        update_target_every: Number of episodes between target network updates
        board_size: Size of the Quoridor board
        max_walls: Maximum number of walls per player
        save_path: Directory to save trained models
        model_name: Base name for saved models
        save_frequency: How often to save the model (in episodes)
        step_rewards: Whether to use step rewards
    """
    game = env(board_size=board_size, max_walls=max_walls, step_rewards=step_rewards)

    # Calculate action space size
    action_size = board_size**2 + ((board_size - 1) ** 2) * 2

    # Create the DQN agent
    dqn_agent = DQNAgent(board_size, action_size)

    # Create a random opponent
    from agents import RandomAgent

    random_agent = RandomAgent()

    # Create directory for saving models if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    total_rewards = []
    losses = []

    for episode in range(episodes):
        game.reset()

        # Reset episode-specific variables
        episode_reward = 0
        episode_losses = []

        # Agent iteration loop
        for agent_name in game.agent_iter():
            observation, reward, termination, truncation, _ = game.last()

            # If the game is over, break the loop
            if termination or truncation:
                break

            # If it's the DQN agent's turn
            if agent_name == "player_0":
                # Get current state
                state = dqn_agent.preprocess_observation(observation)

                # Select action using epsilon-greedy
                action = dqn_agent.get_action(game)

                # Execute action
                game.step(action)

                # Get new state, reward, etc.
                next_observation, reward, termination, truncation, _ = game.last()

                # Add to episode reward
                episode_reward += reward

                # Store transition in replay buffer
                next_state = (
                    dqn_agent.preprocess_observation(next_observation) if not (termination or truncation) else None
                )
                done = 1.0 if (termination or truncation) else 0.0
                dqn_agent.replay_buffer.add(
                    state.cpu().numpy(),
                    action,
                    reward,
                    next_state.cpu().numpy() if next_state is not None else np.zeros_like(state.cpu().numpy()),
                    done,
                )

                # Train the agent
                if len(dqn_agent.replay_buffer) > batch_size:
                    loss = dqn_agent.train(batch_size)
                    if loss is not None:
                        episode_losses.append(loss)

            # If it's the random opponent's turn
            else:
                # Get action from random agent
                action = random_agent.get_action(game)

                # Execute action
                game.step(action)

        # Aggregate episode statistics
        total_rewards.append(episode_reward)
        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
            losses.append(avg_loss)

        # Update target network periodically
        if episode % update_target_every == 0:
            dqn_agent.update_target_network()
            print(
                f"Episode {episode}/{episodes}, Avg Reward: {sum(total_rewards[-100:]) / min(100, len(total_rewards)):.2f}, "
                f"Avg Loss: {sum(losses[-100:]) / min(100, len(losses)):.4f}, Epsilon: {dqn_agent.epsilon:.4f}"
            )

        # Save model periodically
        if episode % save_frequency == 0 and episode > 0:
            save_file = os.path.join(save_path, f"{model_name}_episode_{episode}.pt")
            dqn_agent.save_model(save_file)
            print(f"Model saved to {save_file}")

    # Save final model
    final_save_file = os.path.join(save_path, f"{model_name}_final.pt")
    dqn_agent.save_model(final_save_file)
    print(f"Final model saved to {final_save_file}")

    return dqn_agent, total_rewards, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent for Quoridor")
    parser.add_argument("-N", "--board_size", type=int, default=9, help="Board Size")
    parser.add_argument("-W", "--max_walls", type=int, default=10, help="Max walls per player")
    parser.add_argument("-e", "--episodes", type=int, default=10000, help="Number of episodes to train for")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("-u", "--update_target", type=int, default=100, help="Episodes between target network updates")
    parser.add_argument("--step_rewards", action="store_true", default=True, help="Enable step rewards")
    parser.add_argument("--save_path", type=str, default="models", help="Directory to save models")
    parser.add_argument("--model_name", type=str, default="dqn_agent", help="Base name for saved models")
    parser.add_argument("--save_frequency", type=int, default=500, help="How often to save the model (in episodes)")

    args = parser.parse_args()

    print("Starting DQN training...")
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Max walls: {args.max_walls}")
    print(f"Training for {args.episodes} episodes")
    print(f"Using step rewards: {args.step_rewards}")

    agent, rewards, losses = train_dqn(
        episodes=args.episodes,
        batch_size=args.batch_size,
        update_target_every=args.update_target,
        board_size=args.board_size,
        max_walls=args.max_walls,
        save_path=args.save_path,
        model_name=args.model_name,
        save_frequency=args.save_frequency,
        step_rewards=args.step_rewards,
    )

    print("Training completed!")
