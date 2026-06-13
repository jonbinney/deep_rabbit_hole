from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from run_benchmarks_v2 import _check_run_dir, _derive_base_dir, _load_config, main


EXAMPLE_CONFIG = {
    "run_id": "test-run",
    "quoridor": {"board_size": 5, "max_walls": 3, "max_steps": 50},
    "alphazero": {"network": {"type": "mlp"}, "mcts_n": 300, "mcts_c_puct": 1.2},
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
            ],
        },
    ],
}


def _make_run_dir(tmp_path: Path, run_id: str = "test-run") -> Path:
    """Create a runs/<run_id>/ structure with a valid config.yaml inside."""
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    cfg = dict(EXAMPLE_CONFIG)
    cfg["run_id"] = run_id
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return run_dir


def test_derive_base_dir_uses_grandparent(tmp_path):
    run_dir = tmp_path / "runs" / "my-run"
    assert _derive_base_dir(run_dir) == str(tmp_path)


def test_load_config_returns_full_config(tmp_path):
    run_dir = _make_run_dir(tmp_path, run_id="my-run")
    config = _load_config(run_dir, overrides=None)
    assert config.run_id == "my-run"
    assert config.training.learning_rate == 0.001
    assert len(config.benchmarks) == 1
    # paths derived from the run dir
    assert config.paths.run_dir == run_dir


def test_load_config_applies_overrides(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    config = _load_config(run_dir, overrides=["training.learning_rate=0.05"])
    assert config.training.learning_rate == 0.05


def test_load_config_does_not_create_dirs(tmp_path):
    """Config.from_user(..., create_dirs=False) — no replay_buffers/, etc. spawned."""
    run_dir = _make_run_dir(tmp_path)
    _load_config(run_dir, overrides=None)
    assert not (run_dir / "replay_buffers").exists()
    assert not (run_dir / "models").exists()


def test_load_config_raises_when_config_yaml_missing(tmp_path):
    run_dir = tmp_path / "runs" / "my-run"
    run_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        _load_config(run_dir, overrides=None)


def test_check_run_dir_raises_when_run_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run directory not found"):
        _check_run_dir(tmp_path / "does-not-exist")


def test_check_run_dir_raises_when_config_yaml_missing(tmp_path):
    run_dir = tmp_path / "runs" / "my-run"
    run_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        _check_run_dir(run_dir)


def test_check_run_dir_raises_when_latest_yaml_missing(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    with pytest.raises(FileNotFoundError, match="models/latest.yaml"):
        _check_run_dir(run_dir)


def test_check_run_dir_passes_when_all_present(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "models").mkdir()
    (run_dir / "models" / "latest.yaml").write_text("filename: /tmp/m.pt\nversion: 0\n")
    _check_run_dir(run_dir)  # no exception


def test_main_exits_zero_when_no_benchmarks(tmp_path, capsys):
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "models").mkdir()
    (run_dir / "models" / "latest.yaml").write_text("filename: /tmp/m.pt\nversion: 0\n")

    # Strip the benchmarks section from the config.yaml.
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    cfg["benchmarks"] = []
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    args = Namespace(run_dir=str(run_dir), overrides=None)
    exit_code = main(args)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "No benchmarks configured" in captured.out
