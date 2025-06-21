import copy
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from quoridor import Action, ActionEncoder, Player, Quoridor, construct_game_from_observation
from quoridor_env import make_observation
from utils import my_device
from utils.misc import get_opponent_player_id

from agents.core import AbstractTrainableAgent, rotation
from agents.core.trainable_agent import TrainableAgentParams

player_to_agent = {Player.ONE: "player_0", Player.TWO: "player_1"}


def daz_observation_to_tensor(observation, obs_player_id, device):
    """Convert the observation dict to a flat tensor."""
    obs_player_turn = 1 if observation["my_turn"] else 0

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
    if not obs_player_turn:
        player_walls, opponent_walls = opponent_walls, player_walls
        player_board, opponent_board = opponent_board, player_board

    # Prepare board representation
    board = np.stack([player_board, opponent_board])

    # Flatten all components
    board_flat = board.flatten()
    walls_flat = walls.flatten()

    # Combine all features into single tensor
    flat_obs = np.concatenate([board_flat, walls_flat, player_walls, opponent_walls])
    return torch.FloatTensor(flat_obs).to(device)


class DAZNetwork(nn.Module):
    """
    Neural network model for Deep Q-learning.
    Takes observation from the Quoridor game and outputs Q-values for each action.
    """

    def __init__(self, board_size, action_size, nn_version):
        super(DAZNetwork, self).__init__()

        # Calculate input dimensions based on observation space
        # Board is board_size x board_size with 2 channels (player position and opponent position)
        # Walls are (board_size-1) x (board_size-1) with 2 channels (vertical and horizontal walls)
        board_input_size = board_size * board_size * 2
        walls_input_size = (board_size - 1) * (board_size - 1) * 2

        # Additional features: walls remaining for both players
        # turn, board player, board opponent, wall positions, my remaining walls, opponent's remaining walls
        flat_input_size = board_input_size + walls_input_size + 2

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
        if isinstance(x, tuple):
            x = torch.stack([i for i in x]).to(my_device())
        return self.model(x)


@dataclass
class DAZAgentParams(TrainableAgentParams):
    register_for_reuse: bool = False

    # After how many self play games we train the network
    train_every: int = 1

    # Exploration vs exploitation.  0 is pure exploitation, infinite is random exploration.
    temperature: float = 1.0

    # Batch size for training
    batch_size: int = 2

    # Number of MCTS selections
    n: int = 1000

    # A higher number favors exploration over exploitation
    c: float = 1.4

    @classmethod
    def training_only_params(cls) -> set[str]:
        """
        Returns a set of parameters that are used only during training.
        These parameters should not be used during playing.
        """
        return super().training_only_params() | {
            "use_opponents_actions",
        }


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
        Create all the children of the current node.
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


class DAZAgent(AbstractTrainableAgent):
    """Diego experimental Agent using DRL."""

    def check_congiguration(self):
        """
        Check the configuration of the agent.
        This is used to check if the agent is configured correctly.
        """
        if self.params.assign_negative_reward:
            raise ValueError("use_opponents_actions and assign_negative_reward cannot be used together. ")

    def __init__(
        self,
        params=DAZAgentParams(),
        **kwargs,
    ):
        super().__init__(params=params, **kwargs)
        self.check_congiguration()
        DAZAgent._instance_being_trained = None
        if params.register_for_reuse and self.is_training():
            DAZAgent._instance_being_trained = self

    def name(self):
        if self.params.nick:
            return self.params.nick
        return f"daz ({self.params})"

    def model_name(self):
        return "daz"

    @classmethod
    def params_class(cls):
        return DAZAgentParams

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
        return DAZNetwork(
            self.board_size,
            self.action_size,
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
        if not self.training_mode:
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
        return daz_observation_to_tensor(observation, obs_player_id, self.device)

    def _convert_action_mask_to_tensor_for_player(self, mask, player_id):
        """
        Convert action mask to tensor, rotating it for player_1.
        This method should be call only when it is agent's turn.
        """
        if player_id == "player_0":
            return torch.tensor(mask, dtype=torch.float32, device=self.device)
        rotated_mask = rotation.rotate_action_mask(self.board_size, mask)
        return torch.tensor(rotated_mask, dtype=torch.float32, device=self.device)

    def _convert_to_action_from_tensor_index_for_player(self, action_index_in_tensor, player_id):
        if player_id == "player_0":
            return super()._convert_to_action_from_tensor_index_for_player(action_index_in_tensor, player_id)

        return rotation.convert_rotated_action_index_to_original(self.board_size, action_index_in_tensor)

    def _convert_to_tensor_index_from_action(self, action, action_player_id):
        if action_player_id == "player_0":
            return super()._convert_to_tensor_index_from_action(action, action_player_id)
        return rotation.convert_original_action_index_to_rotated(self.board_size, action)

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

    def _compute_qvalues_softmax(self, observation, mask, player_id):
        """Compute Q-values using softmax exploration."""
        state = self._observation_to_tensor(observation, player_id)
        with torch.no_grad():
            q_values = self.online_network(state)

        mask_tensor = self._convert_action_mask_to_tensor(mask)
        q_values = q_values * mask_tensor - 1e9 * (1 - mask_tensor)

        # Apply softmax to the Q-values to get action probabilities
        exp_q_values = torch.exp(q_values)
        probabilities = exp_q_values / torch.sum(exp_q_values)

        return probabilities.squeeze().detach().cpu().numpy()

    def _get_best_action(self, observation, mask):
        """Get the best action based on Q-values."""
        state = self._observation_to_tensor(observation, self.player_id)
        with torch.no_grad():
            q_values = self.online_network(state)

        mask_tensor = self._convert_action_mask_to_tensor(mask)
        q_values = q_values * mask_tensor - 1e9 * (1 - mask_tensor)
        self._log_action(q_values)

        if self.training_mode and self.params.softmax_exploration:
            # Apply softmax to the Q-values to get action probabilities
            q_values = q_values.squeeze().detach().cpu().numpy()
            exp_q_values = np.exp(q_values)
            probabilities = exp_q_values / np.sum(exp_q_values)
            # Select an action based on the probabilities
            selected_action = np.random.choice(len(probabilities), p=probabilities)
        else:
            selected_action = torch.argmax(q_values).item()

        idx = self._convert_to_action_from_tensor_index(selected_action)
        assert mask[idx] == 1
        return idx

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
                agent_id = player_to_agent[node.game.current_player]
                observation = make_observation(
                    node.game,
                    agent_id,
                    get_opponent_player_id(agent_id),
                    True,
                )
                policy = self._compute_qvalues_softmax(observation["observation"], observation["action_mask"], agent_id)
                policy_probs = np.zeros_like(policy)
                for i, prob in enumerate(policy):
                    idx = self._convert_to_action_from_tensor_index_for_player(i, agent_id)
                    policy_probs[idx] = prob
                node.expand(policy_probs)

            node.backpropagate(value)

        return root.children

    @classmethod
    def create_from_trained_instance(_cls, **kwargs):
        """Create a new mimic model for the agent."""
        if DAZAgent._instance_being_trained is None:
            raise RuntimeError("Dexp version being trained must have param register_for_reuse set to True")
        s = DAZAgent._instance_being_trained
        p = copy.deepcopy(s.params)
        p.nick = s.name() + "_m"
        # No exploration for this player
        p.epsilon = 0
        # No training
        p.training_mode = False
        # Do not register
        p.register_for_reuse = False
        agent = DAZAgent(params=p, load_model_if_needed=False, **kwargs)
        # We set the online network to be the same instance as the trained agent is using
        agent.online_network = s.online_network
        return agent
