from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import yaml

from agents.core import AbstractTrainableAgent, rotation
from agents.core.trainable_agent import TrainableAgentParams


class DExpNetwork(nn.Module):
    """
    Neural network model for Deep Q-learning.
    Takes observation from the Quoridor game and outputs Q-values for each action.
    """

    def __init__(self, board_size, action_size, split_board, include_turn):
        super(DExpNetwork, self).__init__()

        # Calculate input dimensions based on observation space
        # Board is board_size x board_size with 2 channels (player position and opponent position)
        # Walls are (board_size-1) x (board_size-1) with 2 channels (vertical and horizontal walls)
        board_input_size = board_size * board_size * (2 if split_board else 1)
        walls_input_size = (board_size - 1) * (board_size - 1) * 2

        # Additional features: walls remaining for both players
        # turn, board player, board opponent, wall positions, my remaining walls, opponent's remaining walls
        flat_input_size = board_input_size + walls_input_size + (3 if include_turn else 2)

        # Define network architecture
        self.model = nn.Sequential(
            nn.Linear(flat_input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, x):
        return self.model(x)


@dataclass
class DExpPlayParams(TrainableAgentParams):
    use_rotate_board: bool = False
    include_turn: bool = False
    split_board: bool = False

    @classmethod
    def from_str(cls, agent_params_str="000"):
        p = DExpPlayParams(
            use_rotate_board=bool(int(agent_params_str[0])),
            include_turn=bool(int(agent_params_str[1])),
            split_board=bool(int(agent_params_str[2])),
        )
        return p

    def __str__(self):
        return f"{int(self.use_rotate_board)}{int(self.include_turn)}{int(self.split_board)}"


class DExpAgent(AbstractTrainableAgent):
    """Diego experimental Agent using DRL."""

    def __init__(
        self,
        use_opponents_actions=False,
        params=DExpPlayParams(),
        **kwargs,
    ):
        self.use_opponents_actions = use_opponents_actions
        self.params = params
        super().__init__(params=params, **kwargs)

    def name(self):
        if self.params.nick:
            return self.params.nick
        return f"dexp ({self.params})"

    def model_name(self):
        return "dexp"

    @classmethod
    def params_class(cls):
        return DExpPlayParams

    def version(self):
        """Bump this version when compatibility with saved models is broken"""
        return 1

    def resolve_filename(self, suffix):
        return f"{self.model_id()}_C{self.params}_{suffix}.pt"

    def _calculate_action_size(self):
        """Calculate the size of the action space."""
        return self.board_size**2 + (self.board_size - 1) ** 2 * 2

    def _create_network(self):
        """Create the neural network model."""
        return DExpNetwork(self.board_size, self.action_size, self.params.split_board, self.params.include_turn)

    def adjust_reward(self, r, game):
        r = super().adjust_reward(r, game)
        if game.is_done():
            r = r - self.steps / self.board_size**2
        return r

    def handle_opponent_step_outcome(self, observation_before_action, action, game):
        if not self.training_mode or not self.use_opponents_actions:
            return

        opponent_player = "player_1" if self.player_id == "player_0" else "player_0"

        reward = game.rewards[opponent_player]

        # Handle end of episode
        if action is None:
            # TODO: Check whether it is worth it
            # if self.assign_negative_reward and len(self.replay_buffer) > 0:
            #    last = self.replay_buffer.get_last()
            #    last[2] = reward  # update final reward
            #    last[4] = 1.0  # mark as done
            #    self.current_episode_reward += reward
            return

        state_before_action = self.observation_to_tensor_by_player(observation_before_action, True, opponent_player)
        state_after_action = self.observation_to_tensor_by_player(game.observe(opponent_player), False, opponent_player)
        done = game.is_done()

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

    def observation_to_tensor(self, observation):
        """Convert the observation dict to a flat tensor."""
        obs = observation["observation"]
        my_turn = 1 if obs["my_turn"] else 0
        return self.observation_to_tensor_by_player(observation, my_turn, self.player_id)

    def observation_to_tensor_by_player(self, observation, my_turn, player_id):
        obs = observation["observation"]

        # Rotate board and walls if needed for player_1
        should_rotate = player_id == "player_1" and self.params.use_rotate_board
        board = rotation.rotate_board(obs["board"]) if should_rotate else obs["board"]
        walls = rotation.rotate_walls(obs["walls"]) if should_rotate else obs["walls"]

        # Create position matrices for player and opponent
        is_player_turn = my_turn
        player_board = (board == 1 if is_player_turn else board == 2).astype(np.float32)
        opponent_board = (board == 2 if is_player_turn else board == 1).astype(np.float32)

        # Get wall counts
        my_walls = np.array([obs["my_walls_remaining"]])
        opponent_walls = np.array([obs["opponent_walls_remaining"]])

        # Swap positions and walls if not player's turn and include_turn is True
        if not is_player_turn and self.params.include_turn:
            my_walls, opponent_walls = opponent_walls, my_walls
            player_board, opponent_board = opponent_board, player_board

        # Prepare board representation
        board = np.stack([player_board, opponent_board]) if self.params.split_board else board

        # Flatten all components
        board_flat = board.flatten()
        walls_flat = walls.flatten()
        turn_info = [my_turn] if self.params.include_turn else []

        # Combine all features into single tensor
        flat_obs = np.concatenate([turn_info, board_flat, walls_flat, my_walls, opponent_walls])
        # print(f"Obs {flat_obs}")
        return torch.FloatTensor(flat_obs).to(self.device)

    def convert_action_mask_to_tensor(self, mask):
        """Convert action mask to tensor, rotating it for player_1."""
        if self.player_id == "player_0" or not self.params.use_rotate_board:
            # print(f"Mask {mask}")
            return torch.tensor(mask, dtype=torch.float32, device=self.device)
        rotated_mask = rotation.rotate_action_mask(self.board_size, mask)
        # print(f"Mask(r) {rotated_mask}")
        return torch.tensor(rotated_mask, dtype=torch.float32, device=self.device)

    def convert_to_action_from_tensor_index(self, action_index_in_tensor):
        """Convert action index from rotated tensor back to original action space."""
        # print(f"action {action_index_in_tensor}")
        if self.player_id == "player_0" or not self.params.use_rotate_board:
            return super().convert_to_action_from_tensor_index(action_index_in_tensor)

        return rotation.convert_rotated_action_index_to_original(self.board_size, action_index_in_tensor)

    def yaml_config(self) -> str:
        config = {
            "board_size": self.board_size,
            "max_walls": self.max_walls,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "update_target_every": self.update_target_every,
            "training_mode": self.training_mode,
            "final_reward_multiplier": self.final_reward_multiplier,
            "params": {
                "use_rotate_board": self.params.use_rotate_board,
                "split_board": self.params.split_board,
                "include_turn": self.params.include_turn,
            },
        }
        return yaml.dump(config, indent=2)
