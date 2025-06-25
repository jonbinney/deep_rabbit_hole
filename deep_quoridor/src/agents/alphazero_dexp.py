import copy
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from quoridor import Action, ActionEncoder, Player, Quoridor, construct_game_from_observation
from quoridor_env import make_action_mask, make_observation
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
    temperature: float = 0

    # Batch size for training
    batch_size: int = 2

    # Number of MCTS selections
    n: int = 1000

    # A higher number favors exploration over exploitation
    c: float = 1.8

    # not used but kept for compatibility with dexp models
    split: bool = True
    turn: bool = False
    rotate: bool = True
    use_opponents_actions: bool = True
    target_as_source_for_opponent: bool = True

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
        self.action_taken = action_taken
        self.parent = parent
        self.leaf_state = game.is_game_over()

        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.total_wins = 0
        self.total_losses = 0

        self.ucb_c = ucb_c
        self.prior = prior
        self.node_key = game.get_state_hash()

        self.action_encoder = ActionEncoder(game.board.board_size)

    def is_expanded(self):
        return len(self.children) > 0

    def is_loop(self, child_node: str) -> bool:
        ancestors = set()
        current = self
        while current is not None:
            ancestors.add(current)
            current = current.parent

        if child_node in ancestors:
            return True

        # DFS to check if any successor of child_node is in ancestors
        stack = [child_node] + self.children
        visited = set()
        while stack:
            node = stack.pop()
            if node in ancestors:
                # print(
                #     f"Adding child_node {child_node.node_key} as child of {self.node_key} would create a loop via successor {node.node_key}"
                # )
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend([c for c in node.children if c is not None])

    def expand(self, policy_probs: np.ndarray, nodes: dict[str, "AzNode"]):
        """
        Create all the children of the current node.
        """
        self.expanded = True
        for action_index, prob in enumerate(policy_probs):
            if prob == 0.0:
                continue

            action = self.action_encoder.index_to_action(action_index)
            game = copy.deepcopy(self.game)
            game.step(action)

            child = AzNode(game, parent=self, action_taken=action, ucb_c=self.ucb_c, prior=prob)
            self.children.append(child)

    def select_child_by_ucb(self) -> "AzNode":
        """
        Return the child of the current node with the highest ucb
        """
        if not self.children:
            # self.parent.children.remove(self)
            # print("No children, returning parent")
            # return self.parent
            print(
                f"Unexpected for node {self.game.board}, previous node: {self.parent.game.board if self.parent else None}"
            )
        return max(self.children, key=self.get_ucb)

    def get_ucb(self, child):
        if child.visit_count == 0:
            return self.ucb_c * child.prior * math.sqrt(self.visit_count)

        # value_sum is in between -1 and 1, so doing (avg + 1) / 2 would make it in the range [0, 1]
        q_value = ((child.value_sum / child.visit_count) + 1) / 2

        return q_value + self.ucb_c * child.prior * math.sqrt(self.visit_count) / (child.visit_count + 1)

    def backpropagate(self, value: float):
        """
        Update the nodes from the current node up to the tree by increasing the visit count and adding the value
        """
        self.value_sum += value
        self.visit_count += 1
        self.total_wins += 1 if value > 0 else 0
        self.total_losses += 1 if value < 0 else 0
        if self.parent is not None:
            self.parent.backpropagate(-value)


class DAZAgent(AbstractTrainableAgent):
    nodes: dict[str, AzNode] = {}

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
        return "dexp"

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
        rotated_mask = rotation.rotate_action_vector(self.board_size, mask)
        return torch.tensor(rotated_mask, dtype=torch.float32, device=self.device)

    def _convert_to_action_from_tensor_index_for_player(self, action_index_in_tensor, player_id):
        if player_id == "player_0":
            return super()._convert_to_action_from_tensor_index_for_player(action_index_in_tensor, player_id)

        return rotation.convert_rotated_action_index_to_original(self.board_size, action_index_in_tensor)

    def _convert_to_tensor_index_from_action(self, action, action_player_id):
        if action_player_id == "player_0":
            return super()._convert_to_tensor_index_from_action(action, action_player_id)
        return rotation.convert_original_action_index_to_rotated(self.board_size, action)

    def score(self, child, total_visits=None, total_wins=None, total_losses=None):
        if True:
            if child.visit_count == 0:
                return -np.inf if child.prior == 0.0 else child.prior
            # Fraction of visits that are wins or losses (i.e., decisive outcomes)
            decisive_frac = (child.total_wins + child.total_losses) / child.visit_count
            # When decisive_frac is low, rely more on prior; when high, rely more on value_sum
            # Increase value_weight using a tunable constant (e.g., value_weight_scale)
            value_weight_scale = 5.0  # You can tune this constant as needed
            value_weight = min(1.0, decisive_frac * value_weight_scale)
            prior_weight = 1.0 - value_weight
            # avg_value = child.value_sum / child.visit_count
            avg_value = child.total_wins / (child.total_losses + 1)
            return value_weight * avg_value + prior_weight * child.prior
        return (child.value_sum + child.prior) / (child.visit_count + 1) if child.visit_count > 0 else -np.inf

    def scores(self, child_nodes):
        """
        Compute scores for all child nodes.
        Returns a dictionary: action_taken -> score
        """
        total_visits = 0
        total_wins = 0
        total_losses = 0
        for child in child_nodes:
            total_visits += child.visit_count
            total_wins += child.total_wins
            total_losses += child.total_losses
        return {child: self.score(child, total_visits, total_wins, total_losses) for child in child_nodes}

    def _log_action(self, children_scores):
        if not self.action_log.is_enabled():
            return

        self.action_log.clear()

        # Build a dictionary: action_taken -> average value (for children with visit_count > 0)
        action_scores = {child.action_taken: score for child, score in list(children_scores.items())[:5]}
        self.action_log.action_score_ranking(action_scores)

    def get_action(self, observation) -> int:
        game, _, _ = construct_game_from_observation(observation["observation"], self.player_id)

        # Run MCTS to get action visit counts
        root_children = self.search(game)
        visit_counts = np.zeros(self.action_size, dtype=np.float32)
        for child in root_children:
            action_index = self.action_encoder.action_to_index(child.action_taken)
            visit_counts[action_index] = child.visit_count

        # Build a dictionary: action_taken -> average value (for children with visit_count > 0)
        action_scores = self.scores(root_children)
        action_scores = dict(sorted(action_scores.items(), key=lambda item: item[1], reverse=True))
        self._log_action(action_scores)
        print(f"Action visit counts: {visit_counts}")
        for child, score in action_scores.items():
            print(
                f"Action: {str(child.action_taken):<25} "
                f"Score: {score:>10.8f} "
                f"Visit count: {child.visit_count:>5} "
                f"Value sum: {child.value_sum:>7} "
                f"Prior: {child.prior:>10.8f} "
                f"Avg: {child.value_sum / child.visit_count if child.visit_count > 0 else 0:>10.8f} "
                f"Wins: {child.total_wins:>3} "
                f"Losses: {child.total_losses:>3} "
            )
        best_child = max(action_scores, key=action_scores.get)
        action = self.action_encoder.action_to_index(best_child.action_taken)

        # Store training data if in training mode
        # if self.params.training_mode:
        # Convert visit counts to policy target (normalized)
        # policy_target = visit_counts / np.sum(visit_counts) if np.sum(visit_counts) > 0 else visit_counts
        # self.store_training_data(game, policy_target)

        return int(action)

    def _compute_qvalues_softmax(self, observation, mask, player_id):
        """Compute Q-values using softmax exploration."""
        state = self._observation_to_tensor(observation, player_id)
        with torch.no_grad():
            q_values = self.online_network(state)

        mask_tensor = self._convert_action_mask_to_tensor_for_player(mask, player_id)
        q_values = q_values * mask_tensor - 1e9 * (1 - mask_tensor)

        # Apply softmax to the Q-values to get action probabilities
        exp_q_values = torch.exp(q_values)
        probabilities = exp_q_values / torch.sum(exp_q_values)

        # This is an action vector with probabilities for each action
        # but it might be rotated for player_1
        p = probabilities.squeeze().detach().cpu().numpy()
        if player_id == "player_1":
            # Rotate probabilities for player_1
            p = rotation.rotate_action_vector(self.board_size, p)
        return p

    def select_for_expansion(self, node: AzNode) -> AzNode:
        """
        Select a node to expand, starting from the given node.
        As long as the passed node has leaves to expand, that one will be selected.
        Otherwise, if the node is fully expanded, then its best child will be selected.
        """
        nodes = set()
        while node.is_expanded():
            if node in nodes:
                print(f"Loop detected in node {node}, returning None")
            nodes.add(node)
            node = node.select_child_by_ucb()
        return node

    def search(self, initial_game: Quoridor):
        root = AzNode(initial_game, ucb_c=self.params.c)
        self.nodes[initial_game.get_state_hash()] = root
        for _ in range(self.params.n):
            # Traverse down the tree guided by maximum UCB until we find a node to expand
            node = self.select_for_expansion(root)

            if node.leaf_state:
                # If the node is a leaf, we can assign a value based on the game outcome
                # Since the end state belongs to the winner, we give a value of 1
                value = 1.0
            else:
                agent_id = player_to_agent[node.game.current_player]
                observation = make_observation(
                    node.game,
                    node.game.current_player,
                    True,
                )
                action_mask = make_action_mask(
                    self.action_size, node.game.get_valid_actions(node.game.current_player), self.action_encoder
                )
                policy = self._compute_qvalues_softmax(observation, action_mask, agent_id)
                node.expand(policy, self.nodes)
                non_zero_policy = policy[policy > 0]
                # value = float(np.mean(non_zero_policy)) if non_zero_policy.size > 0 else 0.0
                value = 0

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
