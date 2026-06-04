import importlib

# train_v2 is a top-level module under src/; import the helper directly.
train_v2 = importlib.import_module("train_v2")


def test_source_run_overrides_when_unset():
    assert train_v2.source_run_overrides(None) == []


def test_source_run_overrides_when_set():
    result = train_v2.source_run_overrides("/path/to/old/run")
    assert result == [
        "training.source_run=/path/to/old/run",
        "self_play.program=python",
    ]
