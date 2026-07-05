# Train on Existing Self-play Games — Implementation Plan

> **AMENDMENT (2026-06-04, post-completion):** Tasks 2 and 3 of this plan implemented
> `replay_buffer_size` as a **moves** budget (cumulative `game_length` ≥ `buffer_size`).
> That was wrong — the trainer's existing trim at `trainer.py:208` (`len(moves_per_game) >
> replay_buffer_size`) treats it as a **count of games**. The follow-up commit
> "vibe: fix preload to use replay_buffer_size as games count" corrects the implementation
> and tests. See the (updated) spec at `docs/superpowers/specs/2026-06-04-train-on-existing-selfplay-design.md`
> for the authoritative semantics. The task descriptions below still show the original
> (incorrect) wording — preserved as a historical record.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

I'm using AGENTS.md

**Goal:** Add an "offline" mode to `train_v2.py` that consumes a previous run's stored self-play games as a fixed replay buffer, trains a new (fresh-weights) model on it, and runs benchmarks normally — no self-play subprocess, no new games arriving.

**Architecture:** A new CLI flag `--source-run <run_dir>` on `train_v2.py` is sugar for two config overrides — `training.source_run=<run_dir>` and `self_play.program=python`. When `config.training.source_run` is set, `train_v2.py` symlinks the newest source games into the new run's `replay_buffers/ready/` and skips spawning self-play workers; the trainer's existing pickup loop ingests them indistinguishably from live games. Two tiny conditionals in the trainer drop the `games_per_training_step` pacing gate and suppress the meaningless `model_lag` metric in offline mode.

**Tech Stack:** Python 3.12, pydantic, pytest, numpy, pydantic-yaml.

**Spec:** `docs/superpowers/specs/2026-06-04-train-on-existing-selfplay-design.md`

---

## File structure

- `deep_quoridor/src/v2/config.py` — add `source_run` field to `TrainingConfig`. (modify)
- `deep_quoridor/src/v2/offline_preload.py` — `select_games()` + `preload_symlinks()`. (create)
- `deep_quoridor/src/v2/__init__.py` — export `preload_symlinks`. (modify)
- `deep_quoridor/src/v2/trainer.py` — extract `_should_skip_iteration` and `_build_game_log` helpers; wire them in `train()`. (modify)
- `deep_quoridor/src/train_v2.py` — `--source-run` flag, override injection, preload call, skip self-play spawning. (modify)
- `deep_quoridor/test/config_test.py` — add `source_run` override test. (modify)
- `deep_quoridor/test/test_offline_preload.py` — unit tests for `select_games` + `preload_symlinks`. (create)
- `deep_quoridor/test/test_trainer_helpers.py` — unit tests for `_should_skip_iteration` + `_build_game_log`. (create)
- `deep_quoridor/test/test_train_v2_args.py` — unit tests for the override-injection helper. (create)

**Run all Python tests with:**
```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/<file> -v
```

**Commit style (AGENTS.md):** `vibe: ` imperative subject ≤ 50 chars. Keep functional changes in one commit and formatting/lint in a separate commit. Activate the venv when running Python.

---

## Task 1: Add `source_run` field to TrainingConfig

**Files:**
- Modify: `deep_quoridor/src/v2/config.py:105-117` (the `TrainingConfig` class)
- Modify: `deep_quoridor/test/config_test.py` (add one test)

- [ ] **Step 1: Write the failing test**

Append to `deep_quoridor/test/config_test.py`:

```python
def test_override_source_run(config_file):
    config = load_user_config(config_file, overrides=["training.source_run=/path/to/old/run"])
    assert config.training.source_run == "/path/to/old/run"


def test_source_run_defaults_to_none(config_file):
    config = load_user_config(config_file)
    assert config.training.source_run is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/config_test.py::test_override_source_run deep_quoridor/test/config_test.py::test_source_run_defaults_to_none -v
```

Expected: `test_override_source_run` fails with a pydantic `extra="forbid"` validation error (unknown field `source_run`); `test_source_run_defaults_to_none` fails with `AttributeError: 'TrainingConfig' object has no attribute 'source_run'`.

- [ ] **Step 3: Add the field to TrainingConfig**

In `deep_quoridor/src/v2/config.py`, modify the `TrainingConfig` class (around line 105). Add `source_run` after `initial_model`:

```python
class TrainingConfig(StrictBaseModel):
    games_per_training_step: float
    learning_rate: float
    batch_size: int
    weight_decay: float
    replay_buffer_size: int
    max_cached_games: int = 100000
    model_save_timing: bool = False
    save_onnx: bool = False
    finish_after: Optional[str] = None
    initial_model: Optional[InitialModel] = None
    source_run: Optional[str] = None
    lr_scheduler: Optional[LRSchedulerConfig] = None
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/config_test.py -v
```

Expected: all tests pass (the two new ones plus the existing 11).

- [ ] **Step 5: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/v2/config.py deep_quoridor/test/config_test.py
git commit -m "vibe: add training.source_run config field"
```

---

## Task 2: `select_games()` — pure selector function

**Files:**
- Create: `deep_quoridor/src/v2/offline_preload.py`
- Create: `deep_quoridor/test/test_offline_preload.py`

- [ ] **Step 1: Write the failing tests**

Create `deep_quoridor/test/test_offline_preload.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_offline_preload.py -v
```

Expected: `ModuleNotFoundError: No module named 'v2.offline_preload'`.

- [ ] **Step 3: Implement `select_games`**

Create `deep_quoridor/src/v2/offline_preload.py`:

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_offline_preload.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/v2/offline_preload.py deep_quoridor/test/test_offline_preload.py
git commit -m "vibe: add select_games offline-preload helper"
```

---

## Task 3: `preload_symlinks()` — I/O function

**Files:**
- Modify: `deep_quoridor/src/v2/offline_preload.py`
- Modify: `deep_quoridor/test/test_offline_preload.py`

- [ ] **Step 1: Write the failing tests**

Append to `deep_quoridor/test/test_offline_preload.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_offline_preload.py -v
```

Expected: the 5 new tests fail with `ImportError` on `preload_symlinks` (the older 5 `select_games` tests still pass).

- [ ] **Step 3: Implement `preload_symlinks`**

Append to `deep_quoridor/src/v2/offline_preload.py`:

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_offline_preload.py -v
```

Expected: all 11 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/v2/offline_preload.py deep_quoridor/test/test_offline_preload.py
git commit -m "vibe: add preload_symlinks offline-preload helper"
```

---

## Task 4: Export `preload_symlinks` from `v2/__init__.py`

**Files:**
- Modify: `deep_quoridor/src/v2/__init__.py`

- [ ] **Step 1: Add the export**

In `deep_quoridor/src/v2/__init__.py`, add `"preload_symlinks"` to `__all__` and add the import. The full file should be:

```python
__all__ = [
    "load_config_and_setup_run",
    "create_benchmark_processes",
    "create_alphazero",
    "LatestModel",
    "JobTrigger",
    "MockWandb",
    "self_play",
    "train",
    "GameInfo",
    "ShutdownSignal",
    "upload_model",
    "check_ai_available",
    "run_ai_reporter",
    "generate_on_demand_report",
    "metrics_dir_for",
    "run_selfplay_metrics",
    "preload_symlinks",
]

from v2.ai_report import check_ai_available, generate_on_demand_report, run_ai_reporter
from v2.benchmarks import create_benchmark_processes
from v2.selfplay_metrics import metrics_dir_for, run_selfplay_metrics
from v2.common import JobTrigger, MockWandb, ShutdownSignal, create_alphazero, upload_model
from v2.config import load_config_and_setup_run
from v2.offline_preload import preload_symlinks
from v2.self_play import self_play
from v2.trainer import train
from v2.yaml_models import GameInfo, LatestModel
```

- [ ] **Step 2: Sanity-check the import works**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/python -c "from v2 import preload_symlinks; print(preload_symlinks)"
```

Expected: prints `<function preload_symlinks at 0x...>`.

- [ ] **Step 3: Re-run the offline_preload tests**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_offline_preload.py -v
```

Expected: all 11 tests pass (this confirms the export change didn't break anything).

- [ ] **Step 4: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/v2/__init__.py
git commit -m "vibe: export preload_symlinks from v2"
```

---

## Task 5: Trainer changes — extract helpers, wire offline mode

**Files:**
- Modify: `deep_quoridor/src/v2/trainer.py`
- Create: `deep_quoridor/test/test_trainer_helpers.py`

This task does TDD for two small extracted helpers (`_should_skip_iteration`, `_build_game_log`), then wires them into the trainer's loop.

- [ ] **Step 1: Write failing tests for the helpers**

Create `deep_quoridor/test/test_trainer_helpers.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_trainer_helpers.py -v
```

Expected: ImportError on `_should_skip_iteration` and `_build_game_log`.

- [ ] **Step 3: Add the helpers to trainer.py**

In `deep_quoridor/src/v2/trainer.py`, just below the `Sampler` class and above `def model_uploader(...)`, add:

```python
def _should_skip_iteration(
    total_moves: int,
    batch_size: int,
    games_per_training_step: float,
    training_steps: int,
    last_game: int,
    offline_mode: bool,
) -> bool:
    """Decide whether to skip this iteration of the trainer's main loop.

    Always skip when the buffer holds fewer moves than one batch. In online mode also
    skip when the trainer is ahead of self-play (the `games_per_training_step` gate).
    In offline mode the buffer is static and there is no production cadence to wait on,
    so we train every iteration once enough moves are available.
    """
    if total_moves < batch_size:
        return True
    if offline_mode:
        return False
    games_needed_to_train = games_per_training_step * (training_steps + 1)
    return games_needed_to_train > last_game


def _build_game_log(
    game_info,
    model_version: int,
    last_game: int,
    offline_mode: bool,
) -> dict:
    """Per-game wandb log payload emitted when a game is ingested from ready/.

    `model_lag` is meaningful only when games arrive from live self-play, since it
    compares the trainer's current model version against the version that *produced*
    the game. In offline mode the source's `game_info.model_version` came from a
    different training run and the subtraction is nonsense, so the key is omitted.
    """
    log = {
        "game_length": game_info.game_length,
        "Game num": last_game,
        "Model version": model_version,
    }
    if not offline_mode:
        log["model_lag"] = model_version - 1 - game_info.model_version
    return log
```

- [ ] **Step 4: Run helper tests to confirm they pass**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_trainer_helpers.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Wire the helpers into `train()`**

In `deep_quoridor/src/v2/trainer.py`, modify `train(config)`. Near the top of the function (after `batch_size = config.training.batch_size`), add:

```python
    offline_mode = config.training.source_run is not None
```

Replace the per-game log construction. The current code (around lines 158-165) reads:

```python
            wandb_run.log(
                {
                    "game_length": game_info.game_length,
                    "model_lag": model_version - 1 - game_info.model_version,
                    "Game num": last_game,
                    "Model version": model_version,
                }
            )
```

Change it to:

```python
            wandb_run.log(_build_game_log(game_info, model_version, last_game, offline_mode))
```

Replace the gate. The current code (around lines 175-179) reads:

```python
        games_needed_to_train = config.training.games_per_training_step * (training_steps + 1)

        if total_moves < batch_size or games_needed_to_train > last_game:
            time.sleep(1)
            continue
```

Change it to:

```python
        if _should_skip_iteration(
            total_moves=total_moves,
            batch_size=batch_size,
            games_per_training_step=config.training.games_per_training_step,
            training_steps=training_steps,
            last_game=last_game,
            offline_mode=offline_mode,
        ):
            time.sleep(1)
            continue
```

- [ ] **Step 6: Re-run helper tests and the full test directory to confirm nothing regressed**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_trainer_helpers.py deep_quoridor/test/config_test.py deep_quoridor/test/test_offline_preload.py deep_quoridor/test/test_selfplay_metrics.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/v2/trainer.py deep_quoridor/test/test_trainer_helpers.py
git commit -m "vibe: wire offline mode into trainer loop"
```

---

## Task 6: `train_v2.py` — extract argument helper, add `--source-run`

**Files:**
- Modify: `deep_quoridor/src/train_v2.py`
- Create: `deep_quoridor/test/test_train_v2_args.py`

- [ ] **Step 1: Write the failing tests**

Create `deep_quoridor/test/test_train_v2_args.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_train_v2_args.py -v
```

Expected: `AttributeError: module 'train_v2' has no attribute 'source_run_overrides'`.

- [ ] **Step 3: Add the helper to `train_v2.py`**

In `deep_quoridor/src/train_v2.py`, just below the `_selfplay_subprocess_env` function (above `if __name__ == "__main__":`), add:

```python
def source_run_overrides(source_run: str | None) -> list[str]:
    """Build the config overrides implied by ``--source-run <run_dir>``.

    When ``--source-run`` is set, the run executes in offline mode: no self-play
    workers are spawned. We inject two overrides:
      - ``training.source_run=<run_dir>`` (the single source of truth for "offline mode")
      - ``self_play.program=python`` so ``load_config_and_setup_run`` doesn't reject the
        run when the source's old config has ``program=rust`` but no rust binary is
        available locally.

    Returns the empty list when ``source_run`` is None.
    """
    if source_run is None:
        return []
    return [
        f"training.source_run={source_run}",
        "self_play.program=python",
    ]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_train_v2_args.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Wire `--source-run` into the argparse and main flow**

In `deep_quoridor/src/train_v2.py`, modify the `if __name__ == "__main__":` block.

Add the new argparse flag (after the existing `--overrides` argument, around line 66):

```python
    parser.add_argument(
        "--source-run",
        type=str,
        default=None,
        help=(
            "Run in offline mode: symlink the newest games from <run_dir>/replay_buffers/ "
            "into this run's replay_buffers/ready/ and skip spawning self-play. "
            "Use when training a new network architecture on a previous run's games."
        ),
    )
```

Just after `args = parser.parse_args()` (around line 68), merge the source-run overrides:

```python
    extra_overrides = source_run_overrides(args.source_run)
    if extra_overrides:
        print(f"Offline mode: injecting overrides {extra_overrides}")
    overrides = (args.overrides or []) + extra_overrides
```

Change the next line from `config = load_config_and_setup_run(args.config_file, runs_dir, overrides=args.overrides)` to:

```python
    config = load_config_and_setup_run(args.config_file, runs_dir, overrides=overrides)
```

Update the top of file imports — add `preload_symlinks` to the `from v2 import (...)` block:

```python
from v2 import (
    benchmarks,
    check_ai_available,
    load_config_and_setup_run,
    metrics_dir_for,
    preload_symlinks,
    run_ai_reporter,
    run_selfplay_metrics,
    self_play,
    train,
)
```

After the `ShutdownSignal.clear(config)` call and **before** `train_process = mp.Process(target=train, args=[config])`, add the preload + skip-self-play decision:

```python
    offline_mode = config.training.source_run is not None
    if offline_mode:
        n_loaded = preload_symlinks(
            source_run=Path(config.training.source_run),
            dest_ready=config.paths.replay_buffers_ready,
            buffer_size=config.training.replay_buffer_size,
        )
        print(f"Offline mode: linked {n_loaded} games from {config.training.source_run}")
```

Then wrap the entire `if config.self_play.program == "rust":` / `else:` block (currently lines 102-135) so it is **skipped** when `offline_mode` is true:

```python
    if not offline_mode:
        if config.self_play.program == "rust":
            # ... existing rust branch unchanged ...
        else:
            # ... existing python branch unchanged ...
```

(`self_play_processes` and `rust_subprocesses` are initialized to `[]` immediately above this block in the existing code — those initializations stay outside the `if not offline_mode:` so the shutdown wait loop below still works.)

- [ ] **Step 6: Sanity-check the script still parses**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/python -c "import train_v2; print('ok')"
```

Expected: prints `ok`.

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/python deep_quoridor/src/train_v2.py --help 2>&1 | head -30
```

Expected: usage text including the `--source-run` flag.

- [ ] **Step 7: Re-run the affected tests**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_train_v2_args.py deep_quoridor/test/test_offline_preload.py deep_quoridor/test/test_trainer_helpers.py deep_quoridor/test/config_test.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/train_v2.py deep_quoridor/test/test_train_v2_args.py
git commit -m "vibe: add --source-run offline mode to train_v2"
```

---

## Task 7: End-to-end smoke test

This task verifies the wired-together behavior against a real (tiny) source run. It does not add a test file — it is a manual procedure with explicit success criteria, because the end-to-end path involves `multiprocessing.Process` spawning and a real neural network init that's expensive to mock cleanly. Use any existing small training config in `deep_quoridor/experiments/` as the base, or use the snippet below.

**Files:**
- No code changes. Verification only.

- [ ] **Step 1: Prepare a fake "source" run dir**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
PYTHONPATH=deep_quoridor/src .venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
from pydantic_yaml import to_yaml_file
from v2.yaml_models import GameInfo

source = Path("/tmp/smoke_source_run/replay_buffers")
source.mkdir(parents=True, exist_ok=True)

# Five small games matching board_size=5, max_walls=3 shape would normally be
# required for real training. For this smoke test we won't actually train --
# we'll let train_v2 boot, preload, then Ctrl-C. So array shapes don't matter
# beyond what the trainer's ingestion needs (game_length from yaml; npz can be
# minimal).
for i in range(1, 6):
    game_length = 8
    np.savez(
        source / f"game_{i:07d}.npz",
        input_arrays=np.zeros((game_length, 1), dtype=np.float32),
        policies=np.zeros((game_length, 1), dtype=np.float32),
        action_masks=np.zeros((game_length, 1), dtype=np.float32),
        values=np.zeros(game_length, dtype=np.float32),
        players=np.zeros(game_length, dtype=np.int32),
    )
    to_yaml_file(
        source / f"game_{i:07d}.yaml",
        GameInfo(model_version=i, game_length=game_length, creator="smoke"),
    )
print("source_run prepared at /tmp/smoke_source_run")
PY
```

- [ ] **Step 2: Create a minimal config**

Save this to `/tmp/smoke_config.yaml`:

```yaml
run_id: smoke-offline-$DATETIME
quoridor:
  board_size: 5
  max_walls: 3
  max_steps: 50
alphazero:
  network:
    type: mlp
  mcts_n: 25
  mcts_c_puct: 1.2
self_play:
  num_processes: 1
  games_per_process: 4
  alphazero:
    mcts_noise_epsilon: 0.25
training:
  games_per_training_step: 1.0
  learning_rate: 0.001
  batch_size: 32
  weight_decay: 0.0001
  replay_buffer_size: 30
  finish_after: "3 models"
```

- [ ] **Step 3: Run train_v2 in offline mode**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/python deep_quoridor/src/train_v2.py /tmp/smoke_config.yaml -r /tmp/smoke_runs --source-run /tmp/smoke_source_run 2>&1 | tee /tmp/smoke_run.log
```

(Stop after a few seconds with Ctrl-C if it doesn't reach `finish_after`; the array shapes in the fake .npz won't satisfy the real network, but startup logging will show whether offline mode wired up correctly.)

- [ ] **Step 4: Verify the smoke test output**

Check the log:
```bash
grep -E "Offline mode|Started Rust self-play|linked .* games from" /tmp/smoke_run.log
```

Expected to see:
- `Offline mode: injecting overrides ['training.source_run=/tmp/smoke_source_run', 'self_play.program=python']`
- `Offline mode: linked N games from /tmp/smoke_source_run` (N between 3 and 5 — newest until cumulative ≥ 30 moves)
- No `Started Rust self-play process` lines.

Verify the symlinks landed in the new run:
```bash
ls -la /tmp/smoke_runs/runs/smoke-offline-*/replay_buffers/ready/ 2>/dev/null || \
  ls -la /tmp/smoke_runs/runs/smoke-offline-*/replay_buffers/
```

Expected: symlinks (shown by `ls -la` as `->`) targeting `/tmp/smoke_source_run/replay_buffers/game_*.npz` and `.yaml` files. Files may have been already moved into `replay_buffers/` (from `ready/`) by the trainer's pickup loop.

- [ ] **Step 5: Cleanup**

```bash
rm -rf /tmp/smoke_source_run /tmp/smoke_runs /tmp/smoke_config.yaml /tmp/smoke_run.log
```

- [ ] **Step 6: Commit (formatting / lint pass if any)**

Run formatters across the touched files:

```bash
cd /home/jbinney/ws/deep_rabbit_hole && .venv/bin/ruff format deep_quoridor/src/v2/config.py deep_quoridor/src/v2/offline_preload.py deep_quoridor/src/v2/__init__.py deep_quoridor/src/v2/trainer.py deep_quoridor/src/train_v2.py deep_quoridor/test/test_offline_preload.py deep_quoridor/test/test_trainer_helpers.py deep_quoridor/test/test_train_v2_args.py deep_quoridor/test/config_test.py
```

If anything reformatted, commit it separately per AGENTS.md:

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add -u
git diff --cached --quiet || git commit -m "vibe: ruff format"
```

---

## Done criteria

- All unit tests in `test_offline_preload.py`, `test_trainer_helpers.py`, `test_train_v2_args.py`, and the new `config_test.py` cases pass.
- `train_v2.py --source-run <dir>` boots, logs the override injection and symlink count, does not spawn any rust self-play subprocess, and proceeds into the training loop.
- Existing tests in `config_test.py`, `test_selfplay_metrics.py` still pass.
- Files left clean by `ruff format`.
