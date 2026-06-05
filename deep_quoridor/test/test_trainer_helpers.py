from v2.trainer import _build_game_log, _should_skip_iteration
from v2.yaml_models import GameInfo


def test_should_skip_when_not_enough_moves():
    # Below batch_size: must skip regardless of mode.
    assert _should_skip_iteration(
        total_moves=10, batch_size=64, games_per_training_step=1.0,
        training_steps=0, last_game=100, selfplay_disabled=False,
    ) is True
    assert _should_skip_iteration(
        total_moves=10, batch_size=64, games_per_training_step=1.0,
        training_steps=0, last_game=100, selfplay_disabled=True,
    ) is True


def test_selfplay_on_honors_games_per_step_gate():
    # Enough moves, but games_per_training_step * (steps+1) > last_game: skip.
    assert _should_skip_iteration(
        total_moves=1000, batch_size=64, games_per_training_step=1.0,
        training_steps=99, last_game=100, selfplay_disabled=False,
    ) is False  # 1.0 * 100 == last_game; not greater, so train.
    assert _should_skip_iteration(
        total_moves=1000, batch_size=64, games_per_training_step=1.0,
        training_steps=100, last_game=100, selfplay_disabled=False,
    ) is True  # 1.0 * 101 > 100; throttle.


def test_selfplay_off_skips_games_per_step_gate():
    # Same parameters that would throttle now train.
    assert _should_skip_iteration(
        total_moves=1000, batch_size=64, games_per_training_step=1.0,
        training_steps=100, last_game=100, selfplay_disabled=True,
    ) is False
    assert _should_skip_iteration(
        total_moves=1000, batch_size=64, games_per_training_step=1.0,
        training_steps=10_000, last_game=100, selfplay_disabled=True,
    ) is False


def _gi(model_version: int, game_length: int) -> GameInfo:
    return GameInfo(model_version=model_version, game_length=game_length, creator="test")


def test_build_game_log_includes_model_lag_by_default():
    log = _build_game_log(
        game_info=_gi(model_version=5, game_length=42),
        model_version=8, last_game=123, omit_model_lag=False,
    )
    assert log == {
        "game_length": 42,
        "model_lag": 8 - 1 - 5,
        "Game num": 123,
        "Model version": 8,
    }


def test_build_game_log_omits_model_lag_when_requested():
    log = _build_game_log(
        game_info=_gi(model_version=5, game_length=42),
        model_version=8, last_game=123, omit_model_lag=True,
    )
    assert log == {
        "game_length": 42,
        "Game num": 123,
        "Model version": 8,
    }
    assert "model_lag" not in log
