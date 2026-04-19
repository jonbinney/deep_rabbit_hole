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
    program: Literal["python", "rust"] = "python"
    rust_selfplay_binary: Optional[str] = None


class InitialModel(StrictBaseModel):
    file: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_alias: Optional[str] = None

    @field_validator("wandb_alias")
    @classmethod
    def file_and_wandb_mutually_exclusive(cls, v, info):
        if v is not None and info.data.get("file") is not None:
            raise ValueError("Cannot specify both 'file' and 'wandb_alias' in initial_model")
        return v


class TrainingConfig(StrictBaseModel):
    games_per_training_step: float
    learning_rate: float
    batch_size: int
    weight_decay: float
    replay_buffer_size: int
    max_cached_games: int = 100000
    model_save_timing: bool = False
    save_onnx: bool = False
    finish_after: Optional[str] = None
    initial_model: Optional[InitialModel] = None


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


class AIReportConfig(StrictBaseModel):
    """Configuration for periodic AI-generated training reports.

    When this section is present in the yaml, train_v2 spawns a sibling process
    that periodically invokes an AI tool (currently only Claude) to generate a
    markdown report summarizing training progress. When absent, no reports are
    generated.
    """

    every: str
    """How often to generate a report. Parsed by JobTrigger.from_string
    (e.g. '100 models', '30 minutes')."""

    ai: str = "claude"
    """Which AI backend to use. Currently only 'claude' is supported."""

    model: Optional[str] = None
    """Model identifier passed to the AI backend. For Claude, values like
    'sonnet', 'opus', 'haiku', or a full model ID work. If None, the backend's
    default model is used."""


class UserConfig(StrictBaseModel):
    """A normal pydantic model that can be used as an inner class."""

    run_id: str
    quoridor: QuoridorConfig
    alphazero: AlphaZeroBaseConfig
    wandb: Optional[WandbConfig] = None
    self_play: SelfPlayConfig
    training: TrainingConfig
    benchmarks: list[BenchmarkScheduleConfig] = []
    ai_report: Optional[AIReportConfig] = None

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
    reports: Path

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
        reports = run_dir / "reports"

        if create_dirs:
            for path in (
                models,
                checkpoints,
                replay_buffers,
                replay_buffers_ready,
                replay_buffers_tmp,
                reports,
            ):
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
            reports=reports,
        )


class Config(UserConfig):
    paths: PathsConfig

    @classmethod
    def from_user(cls, user: UserConfig, base_dir: str, create_dirs: bool = True) -> "Config":
        paths = PathsConfig.create(base_dir, user.run_id, create_dirs=create_dirs)
        return cls(**user.model_dump(), paths=paths)


def to_yaml_str_ordered(model: BaseModel) -> str:
    return yaml.safe_dump(
        model.model_dump(by_alias=True, exclude_none=True, exclude_unset=True),
        sort_keys=False,
    )


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
    file: str,
    base_dir: str,
    overrides: list[str] | None = None,
    create_dirs: bool = True,
) -> Config:
    user_config = load_user_config(file, overrides=overrides)
    config = Config.from_user(user_config, base_dir, create_dirs=create_dirs)

    config_filename = config.paths.config_file
    with config_filename.open(mode="w") as f:
        f.write(to_yaml_str_ordered(user_config))

    use_rust = config.self_play.program == "rust"
    if use_rust:
        # Apply default Rust binary path if not specified in config
        if config.self_play.rust_selfplay_binary is None:
            config.self_play.rust_selfplay_binary = str(
                Path(__file__).parent.parent.parent / "rust" / "target" / "release" / "selfplay"
            )
        rust_binary = config.self_play.rust_selfplay_binary
        if not Path(rust_binary).exists():
            print(f"ERROR: Rust self-play binary not found at {rust_binary}")
            print("Build it with: cd deep_quoridor/rust && cargo build --release --features binary --bin selfplay")
            exit(1)
        # Rust self-play requires ONNX model exports
        config.training.save_onnx = True

    return config
