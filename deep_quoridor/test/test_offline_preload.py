from pathlib import Path

import pytest

from v2.offline_preload import select_games


def test_select_games_source_larger_than_buffer():
    # Source has 5 games totaling 50 moves; buffer holds 25 moves.
    # Newest games are at the end; take from the end until cumulative >= 25.
    entries = [
        ("game_0000001.npz", 10),
        ("game_0000002.npz", 10),
        ("game_0000003.npz", 10),
        ("game_0000004.npz", 10),
        ("game_0000005.npz", 10),
    ]
    result = select_games(entries, buffer_size=25)
    # Newest 3 games (4, 5 wouldn't be enough; need 3 to reach >= 25).
    # Returned in ascending (chronological) order.
    assert result == ["game_0000003.npz", "game_0000004.npz", "game_0000005.npz"]


def test_select_games_source_smaller_than_buffer():
    entries = [
        ("game_0000001.npz", 10),
        ("game_0000002.npz", 10),
    ]
    result = select_games(entries, buffer_size=100)
    assert result == ["game_0000001.npz", "game_0000002.npz"]


def test_select_games_empty_source():
    assert select_games([], buffer_size=100) == []


def test_select_games_exact_equal_cumulative():
    entries = [
        ("game_0000001.npz", 10),
        ("game_0000002.npz", 10),
    ]
    # Newest one alone has exactly 10 moves; buffer wants >= 10.
    result = select_games(entries, buffer_size=10)
    assert result == ["game_0000002.npz"]


def test_select_games_input_order_does_not_matter():
    # The function sorts by filename internally, so any input order yields the
    # same chronological result.
    entries = [
        ("game_0000005.npz", 10),
        ("game_0000001.npz", 10),
        ("game_0000003.npz", 10),
        ("game_0000002.npz", 10),
        ("game_0000004.npz", 10),
    ]
    result = select_games(entries, buffer_size=25)
    assert result == ["game_0000003.npz", "game_0000004.npz", "game_0000005.npz"]


import numpy as np
from pydantic_yaml import to_yaml_file

from v2.offline_preload import preload_symlinks
from v2.yaml_models import GameInfo


def _make_source_game(source_replay_dir: Path, name: str, game_length: int, model_version: int = 0) -> None:
    """Create a tiny .npz + .yaml sidecar pair, the same shape the real trainer writes."""
    npz_path = source_replay_dir / f"{name}.npz"
    np.savez(
        npz_path,
        input_arrays=np.zeros((game_length, 1), dtype=np.float32),
        policies=np.zeros((game_length, 1), dtype=np.float32),
        action_masks=np.zeros((game_length, 1), dtype=np.float32),
        values=np.zeros(game_length, dtype=np.float32),
        players=np.zeros(game_length, dtype=np.int32),
    )
    to_yaml_file(
        source_replay_dir / f"{name}.yaml",
        GameInfo(model_version=model_version, game_length=game_length, creator="test"),
    )


def _make_source_run(tmp_path: Path, num_games: int, moves_per_game: int) -> Path:
    """Build a fake source run directory with `replay_buffers/` populated."""
    source_run = tmp_path / "source_run"
    replay_dir = source_run / "replay_buffers"
    replay_dir.mkdir(parents=True)
    for i in range(1, num_games + 1):
        _make_source_game(replay_dir, f"game_{i:07d}", moves_per_game, model_version=i)
    return source_run


def test_preload_symlinks_creates_npz_and_yaml_symlinks(tmp_path):
    source_run = _make_source_run(tmp_path, num_games=5, moves_per_game=10)
    dest_ready = tmp_path / "new_run" / "ready"
    dest_ready.mkdir(parents=True)

    count = preload_symlinks(source_run, dest_ready, buffer_size=25)

    # Newest 3 games (totaling 30 >= 25) are selected.
    assert count == 3
    expected = {"game_0000003", "game_0000004", "game_0000005"}
    npz_links = {p.stem for p in dest_ready.glob("*.npz")}
    yaml_links = {p.stem for p in dest_ready.glob("*.yaml")}
    assert npz_links == expected
    assert yaml_links == expected
    # All entries in dest_ready are symlinks, not copies.
    for p in dest_ready.iterdir():
        assert p.is_symlink(), f"{p} is not a symlink"


def test_preload_symlinks_target_resolves_via_np_load(tmp_path):
    source_run = _make_source_run(tmp_path, num_games=2, moves_per_game=5)
    dest_ready = tmp_path / "new_run" / "ready"
    dest_ready.mkdir(parents=True)

    preload_symlinks(source_run, dest_ready, buffer_size=100)

    # np.load through the symlink should yield the same arrays as the source.
    link = dest_ready / "game_0000001.npz"
    with np.load(link) as npz:
        assert npz["values"].shape == (5,)


def test_preload_symlinks_source_smaller_than_buffer_takes_all(tmp_path):
    source_run = _make_source_run(tmp_path, num_games=2, moves_per_game=3)
    dest_ready = tmp_path / "new_run" / "ready"
    dest_ready.mkdir(parents=True)

    count = preload_symlinks(source_run, dest_ready, buffer_size=1_000_000)

    assert count == 2


def test_preload_symlinks_aborts_when_replay_buffers_missing(tmp_path):
    source_run = tmp_path / "empty_run"
    source_run.mkdir()
    dest_ready = tmp_path / "new_run" / "ready"
    dest_ready.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="replay_buffers"):
        preload_symlinks(source_run, dest_ready, buffer_size=10)


def test_preload_symlinks_aborts_when_replay_buffers_empty(tmp_path):
    source_run = tmp_path / "empty_run"
    (source_run / "replay_buffers").mkdir(parents=True)
    dest_ready = tmp_path / "new_run" / "ready"
    dest_ready.mkdir(parents=True)

    with pytest.raises(ValueError, match="no .npz files"):
        preload_symlinks(source_run, dest_ready, buffer_size=10)


def test_preload_symlinks_aborts_when_yaml_sidecar_missing(tmp_path):
    source_run = _make_source_run(tmp_path, num_games=2, moves_per_game=5)
    # Delete one yaml sidecar to simulate corruption.
    (source_run / "replay_buffers" / "game_0000002.yaml").unlink()
    dest_ready = tmp_path / "new_run" / "ready"
    dest_ready.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="game_0000002.yaml"):
        preload_symlinks(source_run, dest_ready, buffer_size=10)
