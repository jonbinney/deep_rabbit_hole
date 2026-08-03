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


from fastapi.testclient import TestClient

from v2.play_server_web.app import create_app


def _make_play_dir(tmp_path):
    # A play dir: config.yaml + a flat models/ dir of .onnx files.
    (tmp_path / "config.yaml").write_text(MINIMAL_CONFIG_YAML)
    models = tmp_path / "models"
    _touch(models, "model_1.onnx", "model_2.onnx")
    return tmp_path, models


def test_api_config_endpoint(tmp_path):
    play_dir, _ = _make_play_dir(tmp_path)
    client = TestClient(create_app(play_dir))
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body["board_size"] == 5
    assert body["defaults"]["mcts_n"] == 123


def test_api_models_endpoint(tmp_path):
    play_dir, _ = _make_play_dir(tmp_path)
    client = TestClient(create_app(play_dir))
    r = client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    assert body["models"] == ["model_1.onnx", "model_2.onnx"]
    assert body["default"] == "model_2.onnx"


def test_static_serving_headers_and_wasm_mime(tmp_path):
    play_dir, _ = _make_play_dir(tmp_path)
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>hi</title>")
    (static / "app_bg.wasm").write_bytes(b"\x00asm")
    client = TestClient(create_app(play_dir, static_dir=static))

    idx = client.get("/")
    assert idx.status_code == 200
    assert idx.headers["cross-origin-opener-policy"] == "same-origin"
    assert idx.headers["cross-origin-embedder-policy"] == "require-corp"

    wasm = client.get("/app_bg.wasm")
    assert wasm.status_code == 200
    assert wasm.headers["content-type"] == "application/wasm"


def test_models_are_served_as_static(tmp_path):
    play_dir, models = _make_play_dir(tmp_path)
    (models / "model_1.onnx").write_bytes(b"ONNXDATA")
    client = TestClient(create_app(play_dir))
    r = client.get("/models/model_1.onnx")
    assert r.status_code == 200
    assert r.content == b"ONNXDATA"


def test_placeholder_when_no_static_dir(tmp_path):
    play_dir, _ = _make_play_dir(tmp_path)
    client = TestClient(create_app(play_dir))
    r = client.get("/")
    assert r.status_code == 200
    assert "SPA build not found" in r.text


import importlib


def test_main_returns_1_on_missing_config(tmp_path):
    mod = importlib.import_module("run_play_server_web")

    class Args:
        play_dir = str(tmp_path)  # no config.yaml here
        static_dir = None
        models_dir = None
        host = "127.0.0.1"
        port = 8080

    assert mod.main(Args()) == 1


def test_api_and_models_resolve_with_static_dir_mounted(tmp_path):
    # Regression: the SPA catch-all mount ("/") must not shadow /api or /models.
    play_dir, models = _make_play_dir(tmp_path)
    (models / "model_1.onnx").write_bytes(b"ONNXDATA")
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>spa</title>")
    # Decoy files under the SPA dir that would shadow the API if routing were wrong.
    (static / "api").mkdir()
    (static / "api" / "config").write_text("DECOY")
    client = TestClient(create_app(play_dir, static_dir=static))

    assert client.get("/api/config").json()["board_size"] == 5
    assert client.get("/api/models").json()["models"] == ["model_1.onnx", "model_2.onnx"]
    r = client.get("/models/model_1.onnx")
    assert r.status_code == 200 and r.content == b"ONNXDATA"


def test_coop_coep_headers_on_model_file(tmp_path):
    play_dir, models = _make_play_dir(tmp_path)
    (models / "model_1.onnx").write_bytes(b"ONNXDATA")
    client = TestClient(create_app(play_dir))
    r = client.get("/models/model_1.onnx")
    assert r.status_code == 200
    assert r.headers["cross-origin-opener-policy"] == "same-origin"
    assert r.headers["cross-origin-embedder-policy"] == "require-corp"
