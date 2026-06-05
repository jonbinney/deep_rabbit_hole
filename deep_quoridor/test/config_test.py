import pytest
import yaml
from v2.config import load_user_config

EXAMPLE_CONFIG = {
    "run_id": "test-run",
    "quoridor": {"board_size": 5, "max_walls": 3, "max_steps": 50},
    "alphazero": {"network": {"type": "mlp"}, "mcts_n": 300, "mcts_c_puct": 1.2},
    "wandb": {"project": "example", "upload_model": {"every": "20 models", "when_max": ["raw_win_perc", "elo_score"]}},
    "self_play": {"num_processes": 2, "games_per_process": 16, "alphazero": {"mcts_noise_epsilon": 0.25}},
    "training": {
        "games_per_training_step": 25.0,
        "learning_rate": 0.001,
        "batch_size": 256,
        "weight_decay": 0.0001,
        "replay_buffer_size": 1000000,
    },
    "benchmarks": [
        {
            "every": "10 models",
            "jobs": [
                {"type": "tournament", "prefix": "raw", "times": 10, "opponents": ["random", "greedy"]},
                {"type": "dumb_score", "prefix": "raw"},
            ],
        },
    ],
}


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(EXAMPLE_CONFIG, sort_keys=False))
    return str(path)


def test_no_overrides(config_file):
    config = load_user_config(config_file)
    assert config.wandb is not None
    assert config.wandb.project == "example"
    assert config.training.learning_rate == 0.001


def test_override_none(config_file):
    config = load_user_config(config_file, overrides=["wandb=None"])
    assert config.wandb is None


def test_override_boolean_true(config_file):
    config = load_user_config(config_file, overrides=["training.model_save_timing=True"])
    assert config.training.model_save_timing is True


def test_override_int(config_file):
    config = load_user_config(config_file, overrides=["alphazero.mcts_n=500"])
    assert config.alphazero.mcts_n == 500


def test_override_float(config_file):
    config = load_user_config(config_file, overrides=["training.learning_rate=0.01"])
    assert config.training.learning_rate == 0.01


def test_override_string(config_file):
    config = load_user_config(config_file, overrides=["run_id=my-custom-run"])
    assert config.run_id == "my-custom-run"


def test_override_list(config_file):
    config = load_user_config(config_file, overrides=["wandb.upload_model.when_max=[dumb_score,tournament]"])
    assert config.wandb.upload_model.when_max == ["dumb_score", "tournament"]


def test_override_empty_list(config_file):
    config = load_user_config(config_file, overrides=["wandb.upload_model.when_max=[]"])
    assert config.wandb.upload_model.when_max == []


def test_override_list_index(config_file):
    config = load_user_config(config_file, overrides=["benchmarks.0.every=5 models"])
    assert config.benchmarks[0].every == "5 models"


def test_override_nested_list_index(config_file):
    config = load_user_config(config_file, overrides=["benchmarks.0.jobs.0.times=20"])
    assert config.benchmarks[0].jobs[0].times == 20


def test_multiple_overrides(config_file):
    config = load_user_config(
        config_file, overrides=["alphazero.mcts_n=100", "training.learning_rate=0.05", "wandb=None"]
    )
    assert config.alphazero.mcts_n == 100
    assert config.training.learning_rate == 0.05
    assert config.wandb is None


def test_invalid_override_format(config_file):
    with pytest.raises(ValueError, match="Invalid override format"):
        load_user_config(config_file, overrides=["no_equals_sign"])


def test_invalid_key_rejected_by_pydantic(config_file):
    with pytest.raises(Exception):
        load_user_config(config_file, overrides=["nonexistent_key=value"])


def test_initial_model_run_accepted(config_file):
    config = load_user_config(config_file, overrides=["training.initial_model.run=/some/old/run"])
    assert config.training.initial_model is not None
    assert config.training.initial_model.run == "/some/old/run"
    assert config.training.initial_model.file is None
    assert config.training.initial_model.wandb_alias is None


def test_initial_model_rejects_file_plus_run(config_file):
    with pytest.raises(Exception, match="initial_model"):
        load_user_config(
            config_file,
            overrides=[
                "training.initial_model.file=/a.pt",
                "training.initial_model.run=/some/old/run",
            ],
        )


def test_initial_model_rejects_wandb_alias_plus_run(config_file):
    with pytest.raises(Exception, match="initial_model"):
        load_user_config(
            config_file,
            overrides=[
                "training.initial_model.wandb_alias=m1",
                "training.initial_model.run=/some/old/run",
            ],
        )


def test_initial_model_rejects_file_plus_wandb_alias(config_file):
    # Existing behavior; restated under the new model_validator.
    with pytest.raises(Exception, match="initial_model"):
        load_user_config(
            config_file,
            overrides=[
                "training.initial_model.file=/a.pt",
                "training.initial_model.wandb_alias=m1",
            ],
        )


def test_initial_replay_buffer_accepted(config_file):
    config = load_user_config(config_file, overrides=["training.initial_replay_buffer.run=/some/old/run"])
    assert config.training.initial_replay_buffer is not None
    assert config.training.initial_replay_buffer.run == "/some/old/run"


def test_initial_replay_buffer_defaults_to_none(config_file):
    config = load_user_config(config_file)
    assert config.training.initial_replay_buffer is None


def test_source_run_field_no_longer_exists(config_file):
    # Removed in favor of training.initial_replay_buffer.
    with pytest.raises(Exception, match="source_run|extra"):
        load_user_config(config_file, overrides=["training.source_run=/some/old/run"])


def test_self_play_enabled_defaults_true(config_file):
    config = load_user_config(config_file)
    assert config.self_play.enabled is True


def test_self_play_enabled_can_be_false(config_file):
    config = load_user_config(
        config_file,
        overrides=[
            "self_play.enabled=False",
            "training.initial_replay_buffer.run=/some/old/run",
        ],
    )
    assert config.self_play.enabled is False


def test_selfplay_off_without_replay_buffer_is_rejected(config_file):
    with pytest.raises(Exception, match="initial_replay_buffer"):
        load_user_config(config_file, overrides=["self_play.enabled=False"])


def test_initial_model_run_resolves_to_latest_filename(tmp_path):
    """alphazero_params_dict_from_config translates initial_model.run into the
    .pt filename recorded in <run>/models/latest.yaml."""
    from pydantic_yaml import to_yaml_file
    from v2.common import alphazero_params_dict_from_config
    from v2.config import Config, load_user_config
    from v2.yaml_models import LatestModel

    # Build a fake "old run" with a latest.yaml pointing at a model file.
    old_run = tmp_path / "old_run"
    models_dir = old_run / "models"
    models_dir.mkdir(parents=True)
    to_yaml_file(
        models_dir / "latest.yaml",
        LatestModel(filename=str(old_run / "models" / "checkpoints" / "model_42.pt"), version=42),
    )

    # Build a config that points initial_model.run at the fake run.
    cfg_data = dict(EXAMPLE_CONFIG)
    cfg_data["training"] = {
        **EXAMPLE_CONFIG["training"],
        "initial_model": {"run": str(old_run)},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data, sort_keys=False))

    user = load_user_config(str(cfg_path))
    config = Config.from_user(user, str(tmp_path), create_dirs=False)

    params = alphazero_params_dict_from_config(config)
    assert params["model_filename"] == str(old_run / "models" / "checkpoints" / "model_42.pt")
