from v2.trainer import _build_game_log, _should_skip_iteration
from v2.yaml_models import GameInfo


def test_should_skip_when_not_enough_moves():
    # Below batch_size: must skip regardless of mode.
    assert _should_skip_iteration(
        total_moves=10, batch_size=64, games_per_training_step=1.0,
        training_steps=0, last_game=100, offline_mode=False,
    ) is True
    assert _should_skip_iteration(
        total_moves=10, batch_size=64, games_per_training_step=1.0,
        training_steps=0, last_game=100, offline_mode=True,
    ) is True


def test_online_mode_honors_games_per_step_gate():
    # Enough moves, but games_per_training_step * (steps+1) > last_game: skip.
    assert _should_skip_iteration(
        total_moves=1000, batch_size=64, games_per_training_step=1.0,
        training_steps=99, last_game=100, offline_mode=False,
    ) is False  # 1.0 * 100 == last_game; not greater, so train.
    assert _should_skip_iteration(
        total_moves=1000, batch_size=64, games_per_training_step=1.0,
        training_steps=100, last_game=100, offline_mode=False,
    ) is True  # 1.0 * 101 > 100; throttle.


def test_offline_mode_skips_games_per_step_gate():
    # Same parameters that would throttle in online mode now train.
    assert _should_skip_iteration(
        total_moves=1000, batch_size=64, games_per_training_step=1.0,
        training_steps=100, last_game=100, offline_mode=True,
    ) is False
    # And keeps training even at very high step counts.
    assert _should_skip_iteration(
        total_moves=1000, batch_size=64, games_per_training_step=1.0,
        training_steps=10_000, last_game=100, offline_mode=True,
    ) is False


def _gi(model_version: int, game_length: int) -> GameInfo:
    return GameInfo(model_version=model_version, game_length=game_length, creator="test")


def test_build_game_log_online_includes_model_lag():
    log = _build_game_log(
        game_info=_gi(model_version=5, game_length=42),
        model_version=8, last_game=123, offline_mode=False,
    )
    assert log == {
        "game_length": 42,
        "model_lag": 8 - 1 - 5,
        "Game num": 123,
        "Model version": 8,
    }


def test_build_game_log_offline_omits_model_lag():
    log = _build_game_log(
        game_info=_gi(model_version=5, game_length=42),
        model_version=8, last_game=123, offline_mode=True,
    )
    assert log == {
        "game_length": 42,
        "Game num": 123,
        "Model version": 8,
    }
    assert "model_lag" not in log
