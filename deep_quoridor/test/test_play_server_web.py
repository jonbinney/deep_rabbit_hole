from pathlib import Path

from v2.play_server_web.model_listing import default_model, list_onnx_models


def _touch(dir_: Path, *names: str) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for n in names:
        (dir_ / n).write_bytes(b"")


def test_list_onnx_models_natural_sorted(tmp_path):
    _touch(tmp_path, "model_2.onnx", "model_10.onnx", "model_1.onnx", "model_0.pt", "notes.txt")
    assert list_onnx_models(tmp_path) == ["model_1.onnx", "model_2.onnx", "model_10.onnx"]


def test_list_onnx_models_missing_dir_is_empty(tmp_path):
    assert list_onnx_models(tmp_path / "nope") == []


def test_default_model_is_highest_version_or_none():
    assert default_model(["model_1.onnx", "model_2.onnx", "model_10.onnx"]) == "model_10.onnx"
    assert default_model([]) is None


from v2.config import load_user_config
from v2.play_server_web.config_view import build_config_view

MINIMAL_CONFIG_YAML = """\
run_id: test-run
quoridor:
  board_size: 5
  max_walls: 2
  max_steps: 50
alphazero:
  mcts_n: 123
  mcts_c_puct: 1.4
self_play:
  num_processes: 1
  games_per_process: 1
training:
  games_per_training_step: 1.0
  learning_rate: 0.001
  batch_size: 64
  weight_decay: 0.0001
  replay_buffer_size: 1000
"""


def _write_config(tmp_path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(MINIMAL_CONFIG_YAML)
    return p


def test_build_config_view_board_and_defaults(tmp_path):
    cfg = load_user_config(str(_write_config(tmp_path)))
    view = build_config_view(cfg)
    assert view["board_size"] == 5
    assert view["max_walls"] == 2
    assert view["max_steps"] == 50
    d = view["defaults"]
    assert d["mcts_n"] == 123
    assert d["mcts_c_puct"] == 1.4
    assert d["leaf_parallelism"] == 16  # SelfPlayConfig default
    assert d["virtual_loss"] == 3       # SelfPlayConfig default
    # No self_play.alphazero block in the minimal config:
    assert d["temperature"] is None
    assert d["mcts_noise_epsilon"] == 0.0
    assert d["mcts_worker_threads"] is None
