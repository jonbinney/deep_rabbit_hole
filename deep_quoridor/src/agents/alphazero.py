import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from numpy import copy
from quoridor import Action, Quoridor
from quoridor_env import QuoridorEnv
from utils import my_device
from utils.subargs import SubargsBase

from agents.core.trainable_agent import TrainableAgent
from agents.nn.flat_1024 import Flat1024Network


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


class Node:
    def __init__(
        self, game: Quoridor, parent: Optional["Node"] = None, action_taken: Optional[Action] = None, depth=0, prior=0
    ):
        self.game = game
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

        # Just for debugging, remove
        self.depth = depth
        self.winner = ""
        # TODO maybe better to do it lazily
        # self.valid_actions = list(game.get_valid_actions())
        # random.shuffle(self.valid_actions)

        wall_actions = game.get_valid_wall_actions()
        random.shuffle(wall_actions)
        self.valid_actions = game.get_valid_move_actions() + wall_actions

        # print("Created a node, valid actions:", self.valid_actions)
        # TOOD arg
        self.ucb_c = 1.4

    def should_expand(self):
        return len(self.valid_actions) > 0 or len(self.children) == 0

    def expand(self, probs):
        for action, prob in enumerate(probs):
            game = copy.deepcopy(self.game)
            game.step(action)
            # do we need to change the perspective?
            child = Node(game, self, action, self.depth + 1, prob)

            self.children.append(child)

    def select(self) -> "Node":
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        assert best_child
        return best_child

    def get_ucb(self, child):
        # value_sum is in -1 or 1, so doing (avg + 1) / 2 would make it in the range [0, 1]
        # Then we invert it because we're switching players
        q_value = ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.ucb_c * math.sqrt(math.log(self.visit_count) / child.visit_count)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(-value)

    # for debug
    def print(self):
        if self.action_taken:
            q_value = ((self.value_sum / self.visit_count) + 1) / 2

            print(
                f"{' ' * self.depth * 4}{self.game.action_encoder.action_to_index(self.action_taken)}: ({self.parent.get_ucb(self):.2f}) Q:{q_value} {self.value_sum}/{self.visit_count} {self.winner}"
            )
        else:
            print(f"{self.value_sum}/{self.visit_count}")
        for ch in self.children:
            ch.print()


class MCTS:
    def __init__(self, num_searches: int, nn):
        self.num_searches = num_searches
        self.nn = nn

    def select(self, node):
        """
        Select a node to expand, starting from the given node.
        As long as the passed node has leaves to expand, that one will be selected.
        Otherwise, if the node is fully expanded, then its best child will be selected.
        """
        while not node.should_expand():
            node = node.select()
        return node

    def search(self, initial_game: Quoridor):
        root = Node(initial_game)
        good_player = initial_game.current_player
        # print("====== SEARCH ========")
        # print(initial_game.board)
        # print(f"Player: {good_player}")

        for _ in range(self.num_searches):
            # Select a node to expand
            node = self.select(root)

            # TODO, ok? also, should this check be in the node?
            if node.game.check_win(good_player):
                terminal = True
                value = 1
                node.winner = "Me"
            elif node.game.check_win(1 - good_player):
                terminal = True
                value = -1
                node.winner = "Opp"
            else:
                terminal = False

            if not terminal:
                state = self.nn.game_to_tensor(node.game)
                policy, value = self.nn(state)
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                # TO DO compute valid moves
                # valid_moves = self.game.get_valid_moves(node.ga,e)
                # policy *= valid_moves
                # policy /= np.sum(policy)

                value = value.item()
                node.expand(policy)

            node.backpropagate(value)

        # root.print()

        return {ch.action_taken: ch.visit_count for ch in root.children}


class BaselineNetwork(nn.Module):
    """
    Simple baseline network to implement AlphaZero.
    It should later be improved by using CNN
    """

    def __init__(self, observation_size, action_size):
        super().__init__()
        action_size = action_size
        observation_size = observation_size
        backbone_size = 1024

        # Define network architecture
        self.backbone = nn.Sequential(
            nn.Linear(observation_size, 2048), nn.ReLU(), nn.Linear(2048, backbone_size), nn.ReLU()
        )
        self.backbone.to(my_device())

        self.policy_head = nn.Linear(backbone_size, action_size)

        self.value_head = nn.Sequential(nn.Linear(backbone_size, 1), nn.Tanh())

    def forward(self, x):
        # When training, we receive a tuple of all the tensors, so we need to stack all of them
        if isinstance(x, tuple):
            x = torch.stack([i for i in x]).to(my_device())

        x = self.backbone(x)
        action_probabilities = self.policy_head(x)
        value = self.value_head(x)
        return action_probabilities, value

    @classmethod
    def id(cls):
        return "flat1024"

    def game_to_tensor(self, game: Quoridor):
        """Convert the observation dict to a tensor required by the network."""
        # TODO: Implement conversion from Quoridor game to a flat tensor
        return None


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
            self.memory = training_instance.memory
        else:
            # When playing use 0.0 for temperature so we always chose the best available action.
            self.nn = BaselineNetwork(
                (board_size + board_size - 1) ** 2 + max_walls * 2, board_size**2 + (board_size - 1) ** 2 * 2
            )
            self.temperature = 0.0
            self.memory = []

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

    def end_game(self, game):
        """This method is called when a game ends.
        It allows the agent to clean up its state if needed.
        """
        pass
