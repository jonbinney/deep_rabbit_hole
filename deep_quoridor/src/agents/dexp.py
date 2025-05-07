import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from utils import my_device
from utils.misc import get_opponent_player_id

from agents.core import AbstractTrainableAgent, rotation
from agents.core.trainable_agent import TrainableAgentParams


class DExpNetwork(nn.Module):
    """
    Neural network model for Deep Q-learning.
    Takes observation from the Quoridor game and outputs Q-values for each action.
    """

    def __init__(self, board_size, action_size, split_board, include_turn, nn_version):
        super(DExpNetwork, self).__init__()

        # Calculate input dimensions based on observation space
        # Board is board_size x board_size with 2 channels (player position and opponent position)
        # Walls are (board_size-1) x (board_size-1) with 2 channels (vertical and horizontal walls)
        board_input_size = board_size * board_size * (2 if split_board else 1)
        walls_input_size = (board_size - 1) * (board_size - 1) * 2

        # Additional features: walls remaining for both players
        # turn, board player, board opponent, wall positions, my remaining walls, opponent's remaining walls
        flat_input_size = board_input_size + walls_input_size + (3 if include_turn else 2)

        if nn_version == "3":
            self.model = nn.Sequential(
                nn.Linear(flat_input_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, action_size),
            )
        elif nn_version == "4":
            # Define network architecture
            self.model = nn.Sequential(
                nn.Linear(flat_input_size, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_size),
            )
        else:
            raise RuntimeError(f"Model version does not exist: {nn_version}")

        self.model.to(my_device())

    def forward(self, x):
        return self.model(x)


@dataclass
class DExpAgentParams(TrainableAgentParams):
    # Indicates whether to rotate the board for player_1
    rotate: bool = False
    # Indicates whether to include the turn information in tensor
    turn: bool = False
    # Indicates whether to split the board into two channels (one for each player)
    split: bool = False

    # Whether oppoments actions are used for training
    # This is used for training only, not for playing
    # This should not be combined with assign_negative_reward
    use_opponents_actions: bool = False

    # Whether the target state generated for a player should match the source state of the opponent
    # in the next step. This should not be combined with assign_negative_reward
    target_as_source_for_opponent: bool = False

    register_for_reuse: bool = False

    # Parameters used for training which are required to be used with the same set of values during training
    #  and playing are used to generate a 'key' to identify the model.
    def __str__(self):
        return f"{int(self.rotate)}{int(self.turn)}{int(self.split)}{'1' if self.target_as_source_for_opponent else ''}"

    @classmethod
    def training_only_params(cls) -> set[str]:
        """
        Returns a set of parameters that are used only during training.
        These parameters should not be used during playing.
        """
        return super().training_only_params() | {
            "use_opponents_actions",
        }


class DExpAgent(AbstractTrainableAgent):
    """Diego experimental Agent using DRL."""

    def check_congiguration(self):
        """
        Check the configuration of the agent.
        This is used to check if the agent is configured correctly.
        """
        if self.params.use_opponents_actions and self.params.assign_negative_reward:
            raise ValueError("use_opponents_actions and assign_negative_reward cannot be used together. ")
        if self.params.target_as_source_for_opponent and self.params.assign_negative_reward:
            raise ValueError("target_as_source_for_opponent and assign_negative_reward cannot be used together. ")

    def __init__(
        self,
        params=DExpAgentParams(),
        **kwargs,
    ):
        super().__init__(params=params, **kwargs)
        self.check_congiguration()
        DExpAgent._instance_being_trained = None
        if params.register_for_reuse and self.is_training():
            DExpAgent._instance_being_trained = self

    def name(self):
        if self.params.nick:
            return self.params.nick
        return f"dexp ({self.params})"

    def model_name(self):
        return "dexp"

    @classmethod
    def params_class(cls):
        return DExpAgentParams

    def version(self):
        """Bump this version when compatibility with saved models is broken"""
        return 4

    def resolve_filename(self, suffix):
        return f"{self.model_id()}_C{self.params}_{suffix}.pt"

    def _calculate_action_size(self):
        """Calculate the size of the action space."""
        return self.board_size**2 + (self.board_size - 1) ** 2 * 2

    def _create_network(self):
        """Create the neural network model."""
        return DExpNetwork(
            self.board_size,
            self.action_size,
            self.params.split,
            self.params.turn,
            self.params.nn_version if self.params.nn_version is not None else "4",
        )

    def handle_opponent_step_outcome(
        self,
        opponent_observation_before_action,
        my_observation_after_opponent_action,
        opponent_observation_after_action,
        opponent_reward,
        opponent_action,
        done=False,
    ):
        if not self.training_mode or not self.params.use_opponents_actions:
            return

        self._handle_step_outcome_all(
            opponent_observation_before_action,
            my_observation_after_opponent_action,
            opponent_observation_after_action,
            opponent_reward,
            opponent_action,
            get_opponent_player_id(self.player_id),
            done,
        )

    def _observation_to_tensor(self, observation, obs_player_id):
        """Convert the observation dict to a flat tensor."""
        obs_player_turn = 1 if observation["my_turn"] else 0

        should_rotate = False
        if self.params.rotate and not self.params.target_as_source_for_opponent:
            should_rotate = obs_player_id == "player_1"
        elif self.params.rotate and self.params.target_as_source_for_opponent:
            # Rotate board and walls if needed for player_1 xor (not players turn)
            # This ensures board always faces to the player that will act on it
            should_rotate = (obs_player_id == "player_1") ^ (not obs_player_turn)

        board = rotation.rotate_board(observation["board"]) if should_rotate else observation["board"]
        walls = rotation.rotate_walls(observation["walls"]) if should_rotate else observation["walls"]

        # Create position matrices for player and opponent
        player_board = (board == 1).astype(np.float32)
        opponent_board = (board == 2).astype(np.float32)

        # Get wall counts
        player_walls = np.array([observation["my_walls_remaining"]])
        opponent_walls = np.array([observation["opponent_walls_remaining"]])

        # Swap boards and walls if not player's turn. It means this is a target state
        # Target states are played by the opponents, so board and walls should be in the opponents POV
        if not obs_player_turn and self.params.target_as_source_for_opponent:
            player_walls, opponent_walls = opponent_walls, player_walls
            player_board, opponent_board = opponent_board, player_board

        # Prepare board representation
        board = np.stack([player_board, opponent_board]) if self.params.split else board

        # Flatten all components
        board_flat = board.flatten()
        walls_flat = walls.flatten()
        turn_info = [obs_player_turn] if self.params.turn else []

        # Combine all features into single tensor
        flat_obs = np.concatenate([turn_info, board_flat, walls_flat, player_walls, opponent_walls])
        # print(f"Obs {flat_obs}")
        return torch.FloatTensor(flat_obs).to(self.device)

    def _convert_action_mask_to_tensor_for_player(self, mask, player_id):
        """
        Convert action mask to tensor, rotating it for player_1.
        This method should be call only when it is agent's turn.
        """
        if player_id == "player_0" or not self.params.rotate:
            return torch.tensor(mask, dtype=torch.float32, device=self.device)
        rotated_mask = rotation.rotate_action_mask(self.board_size, mask)
        return torch.tensor(rotated_mask, dtype=torch.float32, device=self.device)

    def _convert_to_action_from_tensor_index_for_player(self, action_index_in_tensor, player_id):
        if player_id == "player_0" or not self.params.rotate:
            return super()._convert_to_action_from_tensor_index_for_player(action_index_in_tensor, player_id)

        return rotation.convert_rotated_action_index_to_original(self.board_size, action_index_in_tensor)

    def _convert_to_tensor_index_from_action(self, action, action_player_id):
        if action_player_id == "player_0" or not self.params.rotate:
            return super()._convert_to_tensor_index_from_action(action, action_player_id)
        return rotation.convert_original_action_index_to_rotated(self.board_size, action)

    @classmethod
    def create_from_trained_instance(_cls, **kwargs):
        """Create a new mimic model for the agent."""
        if DExpAgent._instance_being_trained is None:
            raise RuntimeError("Dexp version being trained must have param register_for_reuse set to True")
        s = DExpAgent._instance_being_trained
        p = copy.deepcopy(s.params)
        p.nick = p.nick + "_mimic"
        # No exploration for this player
        p.epsilon = 0
        # Do not register
        p.register_for_reuse = False
        agent = DExpAgent(params=p, **kwargs)
        # We set the online network to be the same instance as the trained agent is using
        agent.online_network = s.online_network
        return agent
