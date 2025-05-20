import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from quoridor import Quoridor
from utils.subargs import SubargsBase

from agents.core.trainable_agent import SelfPlayTrainableAgent


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


class AlphaZeroAgent(SelfPlayTrainableAgent):
    def __init__(self, board_size, max_walls, params=AlphaZeroParams(), **kwargs):
        super().__init__(**kwargs)
        self.board_size = board_size
        self.max_walls = max_walls
        self.params = params
        self.episode_count = 0

    @classmethod
    def params_class(cls):
        return AlphaZeroParams

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

    # def end_game(self, game):

    def compute_loss_and_reward(self, length: int) -> Tuple[float, float]:
        # TO DO
        return 0.0, 0.0

    def self_play(self, game: Quoridor) -> tuple[str, int]:
        self.episode_count += 1
        if self.episode_count % self.params.train_every == 0 or self.episode_count == self.total_episodes:
            print(f"To do: train the net! {self.episode_count}")
            # After the training we could check if the new network plays better and otherwise reject it and go back
            # to the previous parameters

        # Self play a game using MCTS and store each move in a list, something like (state, probs, winner)
        return "player_0", 42

    def get_action(self, observation) -> int:
        # This should be called only during play, not training
        # TO DO: Use MCTS with the NN to get the probabilities for the actions for the current state, and just chose the argmax

        raise NotImplementedError("You must implement the get_action method")
