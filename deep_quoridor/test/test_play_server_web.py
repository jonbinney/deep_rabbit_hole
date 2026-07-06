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
