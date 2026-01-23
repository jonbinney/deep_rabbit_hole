import re
import sys
import time
from abc import abstractmethod
from pathlib import Path
from typing import Optional

from pydantic_yaml import parse_yaml_file_as, to_yaml_file

sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: F821

from agents.alphazero.alphazero import AlphaZeroAgent, AlphaZeroParams
from config import AlphaZeroPlayConfig, AlphaZeroSelfPlayConfig, Config, load_config_and_setup_run
from pydantic import BaseModel


class LatestModel(BaseModel):
    filename: str
    version: int

    @classmethod
    def load(cls, config: Config):
        return parse_yaml_file_as(cls, config.paths.latest_model_yaml)

    @classmethod
    def write(cls, config: Config, filename: str, version: int):
        latest = LatestModel(filename=filename, version=version)
        to_yaml_file(config.paths.latest_model_yaml, latest)

    @classmethod
    def wait_for_creation(cls, config: Config, timeout: int = 60):
        start_time = time.time()
        while not config.paths.latest_model_yaml.exists():
            if time.time() - start_time > timeout:
                raise RuntimeError(f"Timeout: {config.paths.latest_model_yaml} not found after {timeout} seconds.")
        time.sleep(1)


class JobTrigger:
    @classmethod
    def from_string(cls, config: Config, s: str):
        match = re.match(r"^\s*(\d+)\s*(second|seconds|minute|minutes|hour|hours|model|models)\s*$", s.lower())
        if not match:
            raise ValueError(f"Invalid frequency string: {s!r}")

        value = int(match.group(1))
        if value <= 0:
            raise ValueError(f"Frequency must be a positive integer, got {value}")

        unit = match.group(2)
        if unit in ("model", "models"):
            return ModelJobTrigger(config, value)

        seconds_per_unit = {
            "second": 1,
            "seconds": 1,
            "minute": 60,
            "minutes": 60,
            "hour": 3600,
            "hours": 3600,
        }
        return TimeJobTrigger(value * seconds_per_unit[unit])

    @abstractmethod
    def wait(self):
        pass


class TimeJobTrigger:
    def __init__(self, every_s: int):
        self.every_s = every_s
        self.next_time = None

    def wait(self):
        if self.next_time is not None and time.time() < self.next_time:
            time.sleep(self.next_time - time.time())

        self.next_time = time.time() + self.every_s


class ModelJobTrigger:
    def __init__(self, config: Config, every_model: int):
        self.every_model = every_model
        self.next_model = None
        self.config = config

    def wait(self):
        current_model = LatestModel.load(self.config).version

        while self.next_model is not None and current_model < self.next_model:
            models_left = self.next_model - current_model
            # We assume each model will take at least 1s to be created to avoid
            # re-opening the file too often
            time.sleep(1.0 * models_left)
            current_model = LatestModel.load(self.config).version

        self.next_model = current_model + self.every_model


def create_alphazero(
    config: Config,
    sub_config: Optional[AlphaZeroPlayConfig | AlphaZeroSelfPlayConfig],
    training_mode: bool,
) -> AlphaZeroAgent:
    mcts_n = config.alphazero.mcts_n
    mcts_ucb_c = config.alphazero.mcts_c_puct
    # TODO temperature

    if config.alphazero.network.type == "mlp":
        nn_type = "mlp"
        nn_mask_training_predictions = config.alphazero.network.mask_training_predictions
        # Those 2 are not needed for ml
        nn_resnet_num_blocks = None
        nn_resnet_num_channels = 32
    elif config.alphazero.network.type == "resnet":
        nn_type = "resnet"
        nn_mask_training_predictions = config.alphazero.network.mask_training_predictions
        nn_resnet_num_blocks = config.alphazero.network.num_blocks
        nn_resnet_num_channels = config.alphazero.network.num_channels

    else:
        raise ValueError(f"Unknown nn_type {config.alphazero.network.type}")

    if isinstance(sub_config, AlphaZeroPlayConfig):
        if sub_config.mcts_n is not None:
            mcts_n = sub_config.mcts_n
        if sub_config.mcts_c_puct is not None:
            mcts_ucb_c = sub_config.mcts_c_puct

    if isinstance(sub_config, AlphaZeroSelfPlayConfig):
        mcts_noise_epsilon = sub_config.mcts_noise_epsilon
        mcts_noise_alpha = sub_config.mcts_noise_alpha
    else:
        mcts_noise_epsilon = 0.25
        mcts_noise_alpha = None

    params = AlphaZeroParams(
        mcts_n=mcts_n,
        mcts_ucb_c=mcts_ucb_c,
        training_mode=training_mode,
        nn_type=nn_type,
        nn_mask_training_predictions=nn_mask_training_predictions,
        nn_resnet_num_blocks=nn_resnet_num_blocks,
        nn_resnet_num_channels=nn_resnet_num_channels,
        mcts_noise_epsilon=mcts_noise_epsilon,
        mcts_noise_alpha=mcts_noise_alpha,
    )
    return AlphaZeroAgent(
        config.quoridor.board_size,
        config.quoridor.max_walls,
        config.quoridor.max_steps,
        params=params,
    )


config = load_config_and_setup_run("deep_quoridor/experiments/B5W3/demo.yaml", "/Users/amarcu/code/deep_rabbit_hole")

az = create_alphazero(config, config.benchmarks[0].jobs[0].alphazero, True)
print(az.__dict__)

print(LatestModel.load(config))
