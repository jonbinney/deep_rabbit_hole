import pytest
import yaml
from v2.config import load_user_config

EXAMPLE_CONFIG = {
    "run_id": "test-run",
    "quoridor": {"board_size": 5, "max_walls": 3, "max_steps": 50},
    "alphazero": {"network": {"type": "mlp"}, "mcts_n": 300, "mcts_c_puct": 1.2},
    "wandb": {"project": "example", "upload_model": {"every": "20 models", "when_max": ["raw_win_perc", "elo_score"]}},
    "self_play": {"num_workers": 2, "parallel_games": 8, "alphazero": {"mcts_noise_epsilon": 0.25}},
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
