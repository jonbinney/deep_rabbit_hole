import copy
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from quoridor import Action, Player, Quoridor
from utils.subargs import SubargsBase

from agents.core.trainable_agent import TrainableAgent


@dataclass
class AlphaZeroParams(SubargsBase):
    # Just used to display a user friendly name
    nick: Optional[str] = None
    # If wandb_alias is provided, the model will be fetched from wandb using the model_id and the alias
    wandb_alias: Optional[str] = None
    # When loading from wandb, the project name to be used
    wandb_project: str = "deep_quoridor"
    # If a filename is provided, the model will be loaded from disc
    model_filename: Optional[str] = None
    # Directory where wandb models are stored
    wandb_dir: str = "wandbmodels"
    # Directory where local models are stored
    model_dir: str = "models"

    # After how many self play games we train the network
    train_every: int = 100

    # Exploration vs exploitation.  0 is pure exploitation, infinite is random exploration.
    temperature: float = 1.0

    # Number of searches
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

    def should_expand(self):
        return len(self.children) == 0

    def expand(self, policy_probs: np.ndarray):
        """
        Create all the children of the current node.
        """
        for action, prob in policy_probs:
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
                    policy, value = self.nn(node.game)

                policy = torch.softmax(policy, dim=1).squeeze(0)

                # Mask the policy to make
                valid_actions = node.game.get_valid_actions()
                valid_action_indices = [node.game.action_encoder.action_to_index(action) for action in valid_actions]
                policy_masked = torch.zeros_like(policy)
                policy_masked[valid_action_indices] = policy[valid_action_indices]

                policy_probs = policy_masked.cpu().numpy()
                policy_probs = policy_probs / policy_probs.sum()
                node.expand(policy_probs)

            node.backpropagate(value)

        return root.children


class AlphaZeroAgent(TrainableAgent):
    def __init__(
        self, board_size, max_walls, training_mode=False, training_instance=None, params=AlphaZeroParams(), **kwargs
    ):
        super().__init__(**kwargs)
        self.board_size = board_size
        self.max_walls = max_walls
        self.params = params
        self.episode_count = 0
        self.training_mode = training_mode

        # If a training instance is passed, this instance is playing to train the other, sharing the NN and temperature
        if training_instance:
            self.nn = training_instance.nn
            self.temperature = training_instance.temperature
        else:
            # TO DO, design and implement the NN
            self.nn = nn.Sequential(nn.Linear(2, 2))

            # When playing use 0.0 for temperature so we always chose the best available action.
            self.temperature = params.temperature if training_mode else 0.0

        # TODO remove, used just to return a random action
        self.action_space = kwargs["action_space"]

        # TODO remove, this is because TrainingStatusRenderer assumes we have epsilon, we need a workaround
        self.epsilon = 0

    @classmethod
    def params_class(cls):
        return AlphaZeroParams

    def is_training(self):
        return self.training_mode

    def version(self):
        """Bump this version when compatibility with saved models is broken"""
        return 1

    def model_name(self):
        return "alphazero"

    def model_id(self):
        return f"{self.model_name()}_B{self.board_size}W{self.max_walls}_mv{self.version()}"

    def resolve_filename(self, suffix):
        return f"{self.model_id()}{suffix}.pt"

    def save_model(self, path):
        # Create directory for saving models if it doesn't exist
        os.makedirs(Path(path).absolute().parents[0], exist_ok=True)
        # TO DO
        # torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        """Load the model from disk."""
        print(f"Loading pre-trained model from {path}")
        # TO DO
        # self.online_network.load_state_dict(torch.load(path, map_location=my_device()))

    def compute_loss_and_reward(self, length: int) -> Tuple[float, float]:
        # TO DO, btw, we don't have a reward here
        return 0.0, 0.0

    def get_action(self, observation) -> int:
        # TO DO: Use MCTS with the NN to get the probabilities for the actions for the current state.
        # When temperature is 0, just return the argmax.
        # Otherwise, compute p = p ** (1 / temperature) and use np.random.choice to chose

        action_mask = observation["action_mask"]
        return self.action_space.sample(action_mask)
