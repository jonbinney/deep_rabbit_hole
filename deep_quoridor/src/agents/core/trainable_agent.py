import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from quoridor import ActionEncoder
from utils import SubargsBase, my_device, resolve_path

import wandb
from agents.core.agent import Agent
from agents.core.replay_buffer import ReplayBuffer


@dataclass
class TrainableAgentParams(SubargsBase):
    # Just used to display a user friendly name
    nick: Optional[str] = None
    # If wandb_alias is provided, the model will be fetched from wandb using the model_id and the alias
    wandb_alias: Optional[str] = None
    # If a filename is provided, the model will be loaded from disc
    model_filename: Optional[str] = None
    # Directory where wandb models are stored
    wandb_dir: str = "wandbmodels"
    # Directory where local models are stored
    model_dir: str = "models"
    # Epsilon value for exploration
    epsilon: float = 0.0
    # Minimum epsilon value that epsilon can decay to
    epsilon_min: float = 0.0
    # Decay rate for epsilon
    epsilon_decay: float = 0.995
    # Discount factor for future rewards
    gamma: float = 0.99
    # Batch size for training
    batch_size: int = 64
    # Number of episodes between target network updates
    # This is the number of games played, not the number of training steps
    update_target_every: int = 100
    # If True, the agent will receive a negative reward when game is lost
    assign_negative_reward: bool = False
    # If True, the agent is running in training mode
    training_mode: bool = False
    # If True, final reward will be multiplied by this value (positive or negative)
    final_reward_multiplier: float = 1.0
    # If True, the target q-value function will substract maxq_a'(s', a') instead of adding it
    use_negative_qvalue_function: bool = False
    # If True, the target q-value calculation will be masked with the action mask for the next state
    # This is used to prevent qvalues from invalid actions to be used as part of the qvalue function
    mask_targetq: bool = False
    # If True, the agent will use softmax exploration even when explotation is happening
    softmax_exploration: bool = False
    # Inspect the opponent's possible actions
    # This is used to log the opponent's possible actions based on agent's qvalues
    inspect_opponent_possible_actions: bool = False

    @classmethod
    def training_only_params(cls) -> set[str]:
        """Returns a set of parameter names that are only used during training."""
        # TODO: we should ideally have two set of params: one for training and one for inference
        return {
            "assign_negative_reward",
            "batch_size",
            "epsilon",
            "epsilon_decay",
            "epsilon_min",
            "final_reward_multiplier",
            "gamma",
            "inspect_opponent_possible_actions",
            "mask_targetq",
            "model_dir",
            "model_filename",
            "nick",
            "softmax_exploration",
            "training_mode",
            "update_target_every",
            "use_negative_qvalue_function",
            "wandb_alias",
            "wandb_dir",
        }


class AbstractTrainableAgent(Agent):
    """Base class for trainable agents using neural networks."""

    def __init__(
        self,
        board_size,
        max_walls,
        params: TrainableAgentParams = TrainableAgentParams(),
        **kwargs,
    ):
        super().__init__()
        self.board_size = board_size
        self.max_walls = max_walls
        self.initial_epsilon = params.epsilon
        self.epsilon = params.epsilon
        self.epsilon_min = params.epsilon_min
        self.epsilon_decay = params.epsilon_decay
        self.gamma = params.gamma
        self.batch_size = params.batch_size
        self.update_target_every = params.update_target_every
        self.assign_negative_reward = params.assign_negative_reward
        self.final_reward_multiplier = params.final_reward_multiplier
        self.training_mode = params.training_mode
        self.params = params
        self.action_size = self._calculate_action_size()
        self.action_encoder = ActionEncoder(self.board_size)

        # Setup device
        self.device = my_device()
        self.fetch_model_from_wand_and_update_params()

        # Initialize networks
        self.online_network = self._create_network()
        self.target_network = self._create_network()
        self.online_network.to(self.device)
        self.target_network.to(self.device)

        self.update_target_network()

        # Setup training components
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_criterion()
        self.replay_buffer = ReplayBuffer(capacity=(400000 if self.training_mode else 1))
        self.games_count = 0
        self.episodes_rewards = []
        self.train_call_losses = []
        self.reset_episode_related_info()
        self.resolve_and_load_model()
        self.games_count = 0

    def reset_episode_related_info(self):
        self.current_episode_reward = 0
        self.player_id = None
        self.steps = 0

    def is_trainable(self):
        return True

    def start_game(self, game, player_id):
        self.reset_episode_related_info()
        self.player_id = player_id

    def end_game(self, game):
        if not self.training_mode:
            return
        """Store episode results and reset tracking"""
        self.episodes_rewards.append(self.current_episode_reward)
        self.games_count += 1
        self._update_epsilon()
        if (self.games_count % self.update_target_every) == 0:
            self.update_target_network()

    def get_opponent_player_id(self, player_id):
        """Get the opponent player ID."""
        return "player_1" if player_id == "player_0" else "player_0"

    def handle_opponent_step_outcome(
        opponent_observation_before_action,
        my_observation_after_opponent_action,
        opponent_observation_after_action,
        opponent_reward,
        opponent_action,
        done=False,
    ):
        pass

    def adjust_reward(self, r, done):
        if done:
            r *= self.final_reward_multiplier
        return r

    def handle_step_outcome(
        self,
        observation_before_action,
        opponent_observation_after_action,
        observation_after_action,
        reward,
        action,
        done=False,
    ):
        self.steps += 1
        if not self.training_mode:
            return
        reward = self.handle_step_outcome_all(
            observation_before_action,
            opponent_observation_after_action,
            observation_after_action,
            reward,
            action,
            self.player_id,
            done,
        )
        self.current_episode_reward += reward

    def handle_step_outcome_all(
        self,
        observation_before_action,
        opponent_observation_after_action,
        observation_after_action,
        reward,
        action,
        agent_id,
        done,
    ):
        # reward = self.adjust_reward(game.rewards[player_id], game)
        reward = self.adjust_reward(reward, done)

        # Handle end of episode
        if action is None:
            ## TODO: Revisit this since it won't work for the case in which
            ## opponents actions are used
            if self.assign_negative_reward:
                if len(self.replay_buffer) > 0:
                    last = self.replay_buffer.get_last()
                    last[2] = reward  # update final reward
                    last[4] = 1.0  # mark as done
                    return reward
            return 0

        state_before_action = self.observation_to_tensor(observation_before_action["observation"], agent_id)
        state_after_action = self.observation_to_tensor(observation_after_action["observation"], agent_id)
        next_state_mask = None
        if self.params.mask_targetq:
            # next action mask is stored with the same rotation of the next state
            # if we want to mask actions on next state
            next_state_mask = self.convert_action_mask_to_tensor_for_player(
                opponent_observation_after_action["action_mask"], self.get_opponent_player_id(agent_id)
            )

        # If next_state_mask is None, we just add a zero tensor. It is not really used anyway
        # Ideally for off policy training we could collect all moves and all the information
        # store it and use it for training without running "matches" every time.
        self.replay_buffer.add(
            state_before_action.cpu().numpy(),
            self.convert_to_tensor_index_from_action(action, agent_id),
            reward,
            state_after_action.cpu().numpy()
            if state_after_action is not None
            else np.zeros_like(state_before_action.cpu().numpy()),
            float(done),
            np.zeros_like(1) if next_state_mask is None else next_state_mask.cpu().numpy(),
        )

        if len(self.replay_buffer) > self.batch_size:
            loss = self.train(self.batch_size)
            if loss is not None:
                self.train_call_losses.append(loss)
        return reward

    def compute_loss_and_reward(self, length: int) -> Tuple[float, float]:
        avg_reward = (
            sum(self.episodes_rewards[-length:]) / min(length, len(self.episodes_rewards))
            if self.episodes_rewards
            else 0.0
        )
        if self.train_call_losses:
            losses = torch.stack(self.train_call_losses[-length:])
            avg_loss = losses.mean().item()
        else:
            avg_loss = 0.0

        return avg_loss, avg_reward

    def model_hyperparameters(self):
        return {}

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
        # TODO Try HuberLoss
        return nn.MSELoss().to(self.device)

    def observation_to_tensor(self, observation, obs_player_id):
        """Convert observation to tensor format."""
        raise NotImplementedError("Subclasses must implement observation_to_tensor")

    def update_target_network(self):
        """Copy parameters from online network to target network."""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def get_action(self, observation, action_mask):
        """Select an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            valid_actions = self._get_valid_actions(action_mask)
            return self._select_random_action(valid_actions)

        return self._get_best_action(observation, action_mask)

    def _get_valid_actions(self, mask):
        """Get valid actions from the action mask."""
        return np.where(mask == 1)[0]

    def _select_random_action(self, valid_actions):
        """Select a random action from valid actions."""
        return np.random.choice(valid_actions)

    def convert_action_mask_to_tensor(self, mask):
        return self.convert_action_mask_to_tensor_for_player(mask, self.player_id)

    def convert_action_mask_to_tensor_for_player(self, mask, player_id):
        return torch.tensor(mask, dtype=torch.float32, device=self.device)

    def convert_to_action_from_tensor_index(self, action_index_in_tensor):
        return self.convert_to_action_from_tensor_index_for_player(action_index_in_tensor, self.player_id)

    def convert_to_action_from_tensor_index_for_player(self, action_index_in_tensor, player_id):
        return action_index_in_tensor

    def convert_to_tensor_index_from_action(self, action, player_id):
        return action

    def _log_action(self, q_values):
        if not self.action_log.is_enabled():
            return

        self.action_log.clear()

        # Log the 5 best actions, as long as the value is > -100 (arbitrary value)
        top_values, top_indices = torch.topk(q_values, min(5, len(q_values)))
        scores = {
            self.action_encoder.index_to_action(int(self.convert_to_action_from_tensor_index(i.item()))): v.item()
            for v, i in zip(top_values, top_indices)
            if v.item() >= -100
        }
        self.action_log.action_score_ranking(scores)

    def _get_best_action(self, observation, mask):
        """Get the best action based on Q-values."""
        state = self.observation_to_tensor(observation, self.player_id)
        with torch.no_grad():
            q_values = self.online_network(state)

        mask_tensor = self.convert_action_mask_to_tensor(mask)
        q_values = q_values * mask_tensor - 1e9 * (1 - mask_tensor)
        self._log_action(q_values)

        if self.training_mode and self.params.softmax_exploration:
            # Apply softmax to the Q-values to get action probabilities
            q_values = q_values.detach().cpu().numpy()
            exp_q_values = np.exp(q_values)
            probabilities = exp_q_values / np.sum(exp_q_values)
            # Select an action based on the probabilities
            selected_action = np.random.choice(len(probabilities), p=probabilities)
        else:
            selected_action = torch.argmax(q_values).item()

        idx = self.convert_to_action_from_tensor_index(selected_action)
        assert mask[idx] == 1
        return idx

    def inspect_opponent_possible_actions(self, game, observation, action_log):
        if not self.params.inspect_opponent_possible_actions:
            return
        """Get the best action based on Q-values."""
        opponent_player_id = self.get_opponent_player_id(self.player_id)
        state = self.observation_to_tensor(observation, opponent_player_id)
        with torch.no_grad():
            q_values = self.online_network(state)

        mask = observation["action_mask"]
        mask_tensor = self.convert_action_mask_to_tensor_for_player(mask, opponent_player_id)
        q_values = q_values * mask_tensor - 1e9 * (1 - mask_tensor)

        # Log the 5 best actions, as long as the value is > -100 (arbitrary value)
        top_values, top_indices = torch.topk(q_values, min(5, len(q_values)))
        scores = {
            game.action_index_to_params(
                int(self.convert_to_action_from_tensor_index_for_player(i.item(), opponent_player_id))
            ): v.item()
            for v, i in zip(top_values, top_indices)
            if v.item() >= -100
        }
        action_log.action_score_ranking(scores)

    def train(self, batch_size):
        """Train the network on a batch of samples."""
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        loss = self._train_on_batch(batch)

        return loss

    def _train_on_batch(self, batch):
        """Train the network on a single batch."""
        states, actions, rewards, next_states, dones, next_state_masks = self._prepare_batch(batch)

        current_q_values = self._compute_current_q_values(states, actions)
        target_q_values = self._compute_target_q_values(rewards, next_states, dones, next_state_masks)

        loss = self._update_network(current_q_values, target_q_values)
        return loss

    def _prepare_batch(self, batch):
        """Prepare batch data for training."""
        states, actions, rewards, next_states, dones, next_state_masks = zip(*batch)

        states = torch.stack([torch.FloatTensor(s) for s in states]).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack([torch.FloatTensor(s) for s in next_states]).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        next_state_masks = torch.stack([torch.FloatTensor(s) for s in next_state_masks]).to(self.device)

        return states, actions, rewards, next_states, dones, next_state_masks

    def _compute_current_q_values(self, states, actions):
        """Compute current Q-values."""
        return self.online_network(states).gather(1, actions).squeeze()

    def _compute_target_q_values(self, rewards, next_states, dones, next_state_masks):
        """Compute target Q-values foe next states."""
        with torch.no_grad():
            next_q_values = self.target_network(next_states)  # Shape: [batch_size, num_actions]

        if self.params.mask_targetq:
            # Apply mask: set invalid actions to large negative value, so those qvalues are ignored
            next_q_values = next_q_values * next_state_masks - 1e9 * (1 - next_state_masks)

        # Get the maximum Q-value for each next state
        max_next_q_values = next_q_values.max(1)[0]
        negate = -1 if self.params.use_negative_qvalue_function else 1
        return rewards + negate * (1 - dones) * self.gamma * max_next_q_values

    def _update_network(self, current_q_values, target_q_values):
        """Update the network using computed Q-values."""
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def _update_epsilon(self):
        """Update epsilon value for exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def version(self):
        raise NotImplementedError("Trainable agents should return a version")

    def model_id(self):
        return f"{self.model_name()}_B{self.board_size}W{self.max_walls}_mv{self.version()}"

    def model_name(self):
        raise NotImplementedError("Trainable agents should return a model name")

    def wandb_local_filename(self, artifact: wandb.Artifact) -> str:
        return f"{self.model_id()}_{artifact.digest[:5]}.{self.get_model_extension()}"

    def resolve_filename(self, suffix):
        return f"{self.model_id()}_{suffix}.{self.get_model_extension()}"

    def save_model(self, path):
        """Save the model to disk."""
        # Create directory for saving models if it doesn't exist
        os.makedirs(Path(path).absolute().parents[0], exist_ok=True)
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        """Load the model from disk."""
        print(f"Loading pre-trained model from {path}")
        self.online_network.load_state_dict(torch.load(path, map_location=my_device()))
        self.update_target_network()

    def resolve_and_load_model(self):
        """Figure out what model needs to be loaded based on the settings and loads it."""
        if self.params.model_filename:
            filename = self.params.model_filename
        else:
            # If no filename is passed in training mode, assume we are not loading a model
            if self.training_mode:
                return

            # If it's not training mode, we definitely need to load a pretrained model, so try the
            # default path for local files
            filename = resolve_path(self.params.model_dir, self.resolve_filename("final"))

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found.")

        self.load_model(filename)

    @classmethod
    def params_class(cls):
        raise NotImplementedError("Trainable agents must implement method params_class")

    @classmethod
    def get_model_extension(cls):
        return "pt"

    def fetch_model_from_wand_and_update_params(self):
        """
        This function doesn't do anything if wandb_alias is not set in self.params.
        Otherwise, it will download the file if there's not a local copy.
        The params are updated to the artifact metadata.

        """
        alias = self.params.wandb_alias
        if not alias:
            return

        api = wandb.Api()
        print(f"Fetching model from wandb: the-lazy-learning-lair/deep_quoridor/{self.model_id()}:{alias}")
        artifact = api.artifact(f"the-lazy-learning-lair/deep_quoridor/{self.model_id()}:{alias}", type="model")
        local_filename = resolve_path(self.params.wandb_dir, self.wandb_local_filename(artifact))

        all_params = self.params_class()(**artifact.metadata)

        # Override params, but only the ones that are not training only
        for key, value in artifact.metadata.items():
            if key not in all_params.training_only_params():
                setattr(self.params, key, value)

        self.params.model_filename = str(local_filename)

        if os.path.exists(local_filename):
            return local_filename

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = artifact.download(root=tmpdir)

            # NOTE: This picks the first .pt file it finds in the artifact
            tmp_filename = next(Path(artifact_dir).glob(f"**/*.{self.get_model_extension()}"), None)
            if tmp_filename is None:
                raise FileNotFoundError(f"No model file found in artifact {artifact.name}")

            os.rename(tmp_filename, local_filename)

            print(f"Model downloaded from wandb to {local_filename}")
