"""Preload selected games from a previous run's replay_buffers into a new run's ready/ dir.

Used by `train_v2.py` when `config.training.initial_replay_buffer` is set, to seed the
replay buffer from a previous run's games (typically when training a new architecture
on the same lineage of self-play data).
"""

from __future__ import annotations

from pathlib import Path


def select_games(filenames: list[str], buffer_size: int) -> list[str]:
    """Return the newest `buffer_size` filenames in chronological (ascending) order.

    Source replay-buffer filenames are monotonically numbered (`game_NNNNNNN.npz`), so
    sorting ascending is chronological. If `filenames` has fewer than `buffer_size`
    entries, returns them all.
    """
    return sorted(filenames)[-buffer_size:]


def preload_symlinks(source_run: Path, dest_ready: Path, buffer_size: int) -> int:
    """Symlink the newest source games (.npz + .yaml each) into `dest_ready`.

    Reads `<source_run>/replay_buffers/` for `.npz` files, picks the newest
    `buffer_size` games by filename (which the trainer trims by count, not by
    move total), and creates symlinks (preserving source basenames) for both
    files in `dest_ready`.

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

    selected = select_games([p.name for p in npz_paths], buffer_size)

    for name in selected:
        npz_src = source_replay / name
        yaml_src = npz_src.with_suffix(".yaml")
        if not yaml_src.is_file():
            raise FileNotFoundError(f"Missing yaml sidecar: {yaml_src}")
        npz_dst = Path(dest_ready) / name
        yaml_dst = npz_dst.with_suffix(".yaml")
        npz_dst.symlink_to(npz_src.resolve())
        yaml_dst.symlink_to(yaml_src.resolve())

    return len(selected)
