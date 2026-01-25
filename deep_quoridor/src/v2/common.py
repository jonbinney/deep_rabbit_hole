import re
import sys
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

from pydantic_yaml import parse_yaml_file_as, to_yaml_file

sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: F821

from agents.alphazero.alphazero import AlphaZeroAgent, AlphaZeroParams
from config import AlphaZeroPlayConfig, AlphaZeroSelfPlayConfig, Config
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


def _validate_overrides(overrides: dict[str, Any]) -> None:
    """Validate that override keys are valid AlphaZeroParams fields."""
    if not overrides:
        return

    from dataclasses import fields

    valid_fields = {f.name for f in fields(AlphaZeroParams)}
    invalid_keys = set(overrides.keys()) - valid_fields

    if invalid_keys:
        raise ValueError(f"Invalid override keys: {invalid_keys}. Valid AlphaZeroParams fields: {sorted(valid_fields)}")


def create_alphazero(
    config: Config,
    sub_config: Optional[AlphaZeroPlayConfig | AlphaZeroSelfPlayConfig] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> AlphaZeroAgent:
    """
    Create an AlphaZero agent with configurable parameters.

    Parameters are merged with precedence: config < sub_config < overrides

    Args:
        config: Base configuration
        sub_config: Optional play or self-play specific config
        overrides: Optional dict to override any AlphaZeroParams field

    Returns:
        Configured AlphaZeroAgent
    """
    # Normalize overrides to avoid None checks
    overrides = overrides or {}

    # Validate overrides early
    _validate_overrides(overrides)

    # Build params dict from config
    params_dict = {
        "mcts_n": config.alphazero.mcts_n,
        "mcts_ucb_c": config.alphazero.mcts_c_puct,
    }

    # Add network config
    if config.alphazero.network.type == "mlp":
        params_dict.update(
            {
                "nn_type": "mlp",
                "nn_mask_training_predictions": config.alphazero.network.mask_training_predictions,
            }
        )
    elif config.alphazero.network.type == "resnet":
        params_dict.update(
            {
                "nn_type": "resnet",
                "nn_mask_training_predictions": config.alphazero.network.mask_training_predictions,
                "nn_resnet_num_blocks": config.alphazero.network.num_blocks,
                "nn_resnet_num_channels": config.alphazero.network.num_channels,
            }
        )
    else:
        raise ValueError(f"Unknown nn_type {config.alphazero.network.type}")

    # Apply sub_config overrides
    if isinstance(sub_config, AlphaZeroPlayConfig):
        if sub_config.mcts_n is not None:
            params_dict["mcts_n"] = sub_config.mcts_n
        if sub_config.mcts_c_puct is not None:
            params_dict["mcts_ucb_c"] = sub_config.mcts_c_puct
        params_dict["temperature"] = sub_config.temperature
        params_dict["drop_t_on_step"] = sub_config.drop_t_on_step

    if isinstance(sub_config, AlphaZeroSelfPlayConfig):
        params_dict["mcts_noise_epsilon"] = sub_config.mcts_noise_epsilon
        params_dict["mcts_noise_alpha"] = sub_config.mcts_noise_alpha
        params_dict["temperature"] = sub_config.temperature
        params_dict["drop_t_on_step"] = sub_config.drop_t_on_step

    # Apply overrides (highest priority)
    params_dict.update(overrides)

    params = AlphaZeroParams(**params_dict)
    return AlphaZeroAgent(
        config.quoridor.board_size,
        config.quoridor.max_walls,
        config.quoridor.max_steps,
        params=params,
    )


class MockWandb:
    def log(
        self,
        data: dict[str, Any],
        step: int | None = None,
        commit: bool | None = None,
    ) -> None:
        data_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in data.items())
        print(f"[MockWandb] step={step} | {data_str}")
