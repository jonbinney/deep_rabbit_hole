"""Preload selected games from a previous run's replay_buffers into a new run's ready/ dir.

Used by `train_v2.py` when `--source-run` (config.training.source_run) is set, to seed
the replay buffer for a fresh-architecture training run without spawning self-play.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_yaml import parse_yaml_file_as

from v2.yaml_models import GameInfo


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


def preload_symlinks(source_run: Path, dest_ready: Path, buffer_size: int) -> int:
    """Symlink the newest source games (.npz + .yaml each) into `dest_ready`.

    Reads `<source_run>/replay_buffers/` for `.npz` files, parses each sibling `.yaml` for
    its `game_length`, picks games newest-first until cumulative >= `buffer_size`, and
    creates symlinks (preserving source basenames) for both files in `dest_ready`.

    Returns the number of games linked. Raises:
      - FileNotFoundError if `<source_run>/replay_buffers/` does not exist, or if any
        selected `.npz` lacks its `.yaml` sidecar.
      - ValueError if the source replay_buffers dir contains no `.npz` files.
    """
    source_replay = Path(source_run) / "replay_buffers"
    if not source_replay.is_dir():
        raise FileNotFoundError(f"Source replay_buffers dir not found: {source_replay}")

    npz_paths = sorted(source_replay.glob("*.npz"))
    if not npz_paths:
        raise ValueError(f"Source dir contains no .npz files: {source_replay}")

    # Build (name, game_length) entries; abort if any yaml sidecar is missing.
    entries: list[tuple[str, int]] = []
    for npz_path in npz_paths:
        yaml_path = npz_path.with_suffix(".yaml")
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Missing yaml sidecar: {yaml_path}")
        info = parse_yaml_file_as(GameInfo, yaml_path)
        entries.append((npz_path.name, info.game_length))

    selected = select_games(entries, buffer_size)

    for name in selected:
        npz_src = source_replay / name
        yaml_src = npz_src.with_suffix(".yaml")
        npz_dst = Path(dest_ready) / name
        yaml_dst = npz_dst.with_suffix(".yaml")
        npz_dst.symlink_to(npz_src.resolve())
        yaml_dst.symlink_to(yaml_src.resolve())

    return len(selected)
