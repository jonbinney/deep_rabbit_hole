from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QuoridorConfig(StrictBaseModel):
    board_size: int
    max_walls: int
    max_steps: int


class MLPConfig(StrictBaseModel):
    type: Literal["mlp"] = "mlp"
    mask_training_predictions: bool = False


class ResnetConfig(StrictBaseModel):
    type: Literal["resnet"] = "resnet"
    num_blocks: Optional[int] = None
    num_channels: int
    mask_training_predictions: bool = False


NetworkConfig = Union[MLPConfig, ResnetConfig]


class AlphaZeroBaseConfig(StrictBaseModel):
    network: NetworkConfig = Field(default_factory=MLPConfig, discriminator="type")
    mcts_n: int
    mcts_c_puct: float


class UploadModel(StrictBaseModel):
    every: Optional[str] = ""
    when_max: list[str] = []
    when_min: list[str] = []


class WandbConfig(StrictBaseModel):
    project: str
    upload_model: Optional[UploadModel] = None


class AlphaZeroPlayConfig(StrictBaseModel):
    # temperature here or where
    mcts_n: Optional[int] = None
    mcts_c_puct: Optional[float] = None
    drop_t_on_step: Optional[int] = None
    temperature: Optional[float] = None


class AlphaZeroSelfPlayConfig(StrictBaseModel):
    mcts_noise_epsilon: float
    mcts_noise_alpha: Optional[float] = None
    drop_t_on_step: Optional[int] = None
    temperature: Optional[float] = None


class SelfPlayConfig(StrictBaseModel):
    num_workers: int
    parallel_games: int
    alphazero: Optional[AlphaZeroSelfPlayConfig] = None


class TrainingConfig(StrictBaseModel):
    games_per_training_step: float
    learning_rate: float
    batch_size: int
    weight_decay: float
    replay_buffer_size: int
    model_save_timing: bool = False
    save_pytorch: bool = True
    save_onnx: bool = False
    finish_after: Optional[str] = None


class TournamentBenchmarkConfig(StrictBaseModel):
    type: Literal["tournament"] = "tournament"
    alphazero: Optional[AlphaZeroPlayConfig] = None
    prefix: str
    times: int
    opponents: list[str]


class DumbScoreBenchmarkConfig(StrictBaseModel):
    type: Literal["dumb_score"] = "dumb_score"
    alphazero: Optional[AlphaZeroPlayConfig] = None
    prefix: str


class AgentEvolutionBenchmarkConfig(StrictBaseModel):
    type: Literal["agent_evolution"] = "agent_evolution"
    alphazero: Optional[AlphaZeroPlayConfig] = None
    prefix: str
    times: int
    top_n: int


BenchmarkJobConfig = Annotated[
    Union[
        TournamentBenchmarkConfig,
        DumbScoreBenchmarkConfig,
        AgentEvolutionBenchmarkConfig,
    ],
    Field(discriminator="type"),
]


class BenchmarkScheduleConfig(StrictBaseModel):
    every: str
    jobs: list[BenchmarkJobConfig]


class UserConfig(StrictBaseModel):
    """A normal pydantic model that can be used as an inner class."""

    run_id: str
    quoridor: QuoridorConfig
    alphazero: AlphaZeroBaseConfig
    wandb: Optional[WandbConfig] = None
    self_play: SelfPlayConfig
    training: TrainingConfig
    benchmarks: list[BenchmarkScheduleConfig] = []

    @field_validator("run_id")
    @classmethod
    def replace_datetime_placeholder(cls, v: str) -> str:
        """Replace $DATETIME with current datetime in format YYYYMMDD-HHMM."""
        if "$DATETIME" in v:
            current_datetime = datetime.now().strftime("%Y%m%d-%H%M")
            return v.replace("$DATETIME", current_datetime)
        return v


class PathsConfig(StrictBaseModel):
    run_dir: Path
    models: Path
    latest_model_yaml: Path
    checkpoints: Path
    replay_buffers: Path
    replay_buffers_ready: Path
    replay_buffers_tmp: Path
    config_file: Path

    @classmethod
    def create(cls, base_dir: str, run_id: str, create_dirs: bool = True) -> "PathsConfig":
        run_root = Path(base_dir) / "runs"
        run_dir = run_root / run_id
        config_file = run_dir / "config.yaml"
        models = run_dir / "models"
        latest_model_yaml = models / "latest.yaml"
        checkpoints = models / "checkpoints"
        replay_buffers = run_dir / "replay_buffers"
        replay_buffers_ready = replay_buffers / "ready"
        replay_buffers_tmp = replay_buffers_ready / "tmp"

        if create_dirs:
            for path in (models, checkpoints, replay_buffers, replay_buffers_ready, replay_buffers_tmp):
                path.mkdir(parents=True, exist_ok=True)

        return cls(
            run_dir=run_dir,
            models=models,
            latest_model_yaml=latest_model_yaml,
            checkpoints=checkpoints,
            replay_buffers=replay_buffers,
            replay_buffers_ready=replay_buffers_ready,
            replay_buffers_tmp=replay_buffers_tmp,
            config_file=config_file,
        )


class Config(UserConfig):
    paths: PathsConfig

    @classmethod
    def from_user(cls, user: UserConfig, base_dir: str, create_dirs: bool = True) -> "Config":
        paths = PathsConfig.create(base_dir, user.run_id, create_dirs=create_dirs)
        return cls(**user.model_dump(), paths=paths)


def to_yaml_str_ordered(model: BaseModel) -> str:
    return yaml.safe_dump(model.model_dump(by_alias=True, exclude_none=True, exclude_unset=True), sort_keys=False)


def _merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_data(file: str) -> dict:
    contents = Path(file).read_text()
    data = yaml.safe_load(contents) or {}
    extend = data.pop("extend", None)
    if extend:
        extend_path = Path(extend)
        if not extend_path.is_absolute():
            extend_path = Path(file).parent / extend_path
        base = _load_config_data(str(extend_path))
        data = _merge_dicts(base, data)
    return data


def _parse_override_value(value: str):
    """Parse a string value into an appropriate Python type."""
    if value.lower() == "none":
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_override_value(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _as_index(part: str) -> int:
    try:
        return int(part)
    except ValueError:
        raise ValueError(f"Expected a numeric index for list, got '{part}'")


def _ensure_and_navigate(target, part: str):
    """Navigate into an intermediate path part, creating a dict if the key is missing."""
    if isinstance(target, list):
        return target[_as_index(part)]
    if part not in target:
        target[part] = {}
    return target[part]


def _set_value(target, part: str, value):
    """Set a value on a dict key or list index."""
    if isinstance(target, list):
        target[_as_index(part)] = value
    else:
        target[part] = value


def _apply_overrides(data: dict, overrides: list[str]) -> dict:
    """Apply dotted-key overrides (e.g. 'alphazero.mcts_n=250', 'wandb=None') to a config dict.

    Supports numeric indices for lists: 'benchmarks.0.every=5m'
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format '{override}', expected 'key=value'")
        key, value = override.split("=", 1)
        parts = key.split(".")
        parsed_value = _parse_override_value(value)

        target = data
        for part in parts[:-1]:
            target = _ensure_and_navigate(target, part)
        _set_value(target, parts[-1], parsed_value)

    return data


def load_user_config(file: str, overrides: list[str] | None = None) -> UserConfig:
    data = _load_config_data(file)
    if overrides:
        _apply_overrides(data, overrides)
    return UserConfig.model_validate(data)


def load_config_and_setup_run(
    file: str, base_dir: str, overrides: list[str] | None = None, create_dirs: bool = True
) -> Config:
    user_config = load_user_config(file, overrides=overrides)
    config = Config.from_user(user_config, base_dir, create_dirs=create_dirs)

    config_filename = config.paths.config_file
    with config_filename.open(mode="w") as f:
        f.write(to_yaml_str_ordered(user_config))

    return config
