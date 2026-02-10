import re
import time
from abc import abstractmethod
from typing import Any, Callable, Optional

import wandb
from agents.alphazero import AlphaZeroAgent, AlphaZeroParams
from v2.config import AlphaZeroPlayConfig, AlphaZeroSelfPlayConfig, Config
from v2.yaml_models import LatestModel


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
    def wait(self, should_exit: Callable[[], bool] = lambda: False) -> bool:
        pass

    @abstractmethod
    def is_ready(self):
        pass


class TimeJobTrigger:
    def __init__(self, every_s: int):
        self.every_s = every_s
        self.next_time = time.time() + self.every_s

    def wait(self, should_exit: Callable[[], bool] = lambda: False) -> bool:
        while time.time() < self.next_time:
            sleep_time = min(1.0, self.next_time - time.time())
            if sleep_time > 0:
                time.sleep(sleep_time)
            if should_exit():
                return False

        self.next_time = time.time() + self.every_s
        return True

    def is_ready(self):
        return self.next_time < time.time()


class ModelJobTrigger:
    def __init__(self, config: Config, every_model: int):
        self.every_model = every_model
        self.config = config
        LatestModel.wait_for_creation(config)
        current_model = LatestModel.load(config).version
        self.next_model = current_model + every_model

    def wait(self, should_exit: Callable[[], bool] = lambda: False) -> bool:
        current_model = LatestModel.load(self.config).version

        while current_model < self.next_model:
            time.sleep(5.0)
            current_model = LatestModel.load(self.config).version
            if should_exit():
                return False

        self.next_model = current_model + self.every_model
        return True

    def is_ready(self):
        current_model = LatestModel.load(self.config).version
        return self.next_model <= current_model


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


class ShutdownSignal:
    @staticmethod
    def file_path(config: Config):
        return config.paths.run_dir / ".shutdown"

    @staticmethod
    def signal(config: Config):
        ShutdownSignal.file_path(config).touch()

    @staticmethod
    def is_set(config: Config):
        return ShutdownSignal.file_path(config).exists()

    @staticmethod
    def clear(config: Config):
        ShutdownSignal.file_path(config).unlink(missing_ok=True)


def upload_model(wandb_run, config: Config, model: LatestModel, model_id, aliases: list[str]):
    print(f"Uploading {model.filename} with aliases {aliases}")
    try:
        metadata = {"model_version": model.version}
        artifact = wandb.Artifact(model_id, type="model", metadata=metadata)
        artifact.add_file(local_path=model.filename)
        artifact.add_file(local_path=str(config.paths.config_file))
        wandb_run.log_artifact(artifact, aliases=aliases)
    except Exception as e:
        print(f"!!! Exception during wandb upload: {e}")
