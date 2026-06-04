"""Preload selected games from a previous run's replay_buffers into a new run's ready/ dir.

Used by `train_v2.py` when `--source-run` (config.training.source_run) is set, to seed
the replay buffer for a fresh-architecture training run without spawning self-play.
"""

from __future__ import annotations


def select_games(entries: list[tuple[str, int]], buffer_size: int) -> list[str]:
    """Pick newest games (by filename) whose cumulative game_length covers `buffer_size`.

    `entries` is a list of (filename, game_length) pairs. The source numbers games
    monotonically (`game_NNNNNNN.npz`), so sorting filenames ascending is chronological.

    Returns the selected filenames in ascending (chronological) order. If the source has
    fewer total moves than `buffer_size`, returns every entry.
    """
    sorted_asc = sorted(entries, key=lambda e: e[0])
    # Walk newest-first (descending), collect names until cumulative >= buffer_size.
    selected: list[str] = []
    cumulative = 0
    for name, length in reversed(sorted_asc):
        selected.append(name)
        cumulative += length
        if cumulative >= buffer_size:
            break
    # Return in ascending (chronological) order to match the trainer's ready/-sort.
    selected.reverse()
    return selected
