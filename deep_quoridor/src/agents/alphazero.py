import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch.nn as nn
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
