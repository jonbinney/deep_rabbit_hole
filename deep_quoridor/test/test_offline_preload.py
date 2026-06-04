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
