# Train from a Previous Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

I'm using AGENTS.md

**Goal:** Replace the single `--source-run` mechanism (shipped yesterday) with three independent config knobs — `training.initial_model.run`, `training.initial_replay_buffer.run`, and `self_play.enabled` — so each concern can be toggled separately and mixed-mode (preload + self-play on) becomes a legitimate combination.

**Architecture:** Three additive schema changes (one new `InitialReplayBuffer` sub-model, a new `run` option on the existing `InitialModel`, and an `enabled` flag on `SelfPlayConfig`). Two cross-field validators (three-way mutual exclusion on `InitialModel` sources; refuse `enabled=False` with no replay buffer source). The trainer's two existing helpers (`_should_skip_iteration`, `_build_game_log`) keep their shapes but their bool params rename to reflect the split concerns. `train_v2.py` loses its `--source-run` CLI flag and the `source_run_overrides` helper; everything drives off config.

**Tech Stack:** Python 3.12, pydantic, pytest, pydantic-yaml.

**Spec:** `docs/superpowers/specs/2026-06-05-train-from-previous-run-design.md`

---

## File structure

- `deep_quoridor/src/v2/config.py` — `InitialModel.run` field + 3-way validator; new `InitialReplayBuffer` sub-model; remove `TrainingConfig.source_run`; add `TrainingConfig.initial_replay_buffer`; add `SelfPlayConfig.enabled`; UserConfig-level validator; rust-binary check gating. (modify)
- `deep_quoridor/src/v2/common.py` — `alphazero_params_dict_from_config`: resolve `initial_model.run` to the latest checkpoint via `LatestModel`. (modify)
- `deep_quoridor/src/v2/trainer.py` — rename helper bool params (`offline_mode` → `selfplay_disabled` / `omit_model_lag`); wire from new config fields. (modify)
- `deep_quoridor/src/train_v2.py` — remove `--source-run` flag and `source_run_overrides`; derive two booleans from config; wrap self-play block accordingly. (modify)
- `deep_quoridor/test/config_test.py` — replace `source_run` tests with `initial_replay_buffer` tests; add `initial_model.run` + 3-way exclusion tests; add `self_play.enabled` tests + the no-source-of-games validator test. (modify)
- `deep_quoridor/test/test_trainer_helpers.py` — rename bool params in the 5 existing test cases. (modify)
- `deep_quoridor/test/test_train_v2_args.py` — **delete** (its only purpose was the removed `source_run_overrides` helper). (delete)

**Run Python tests with:**
```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/<file> -v
```

**Commit style (AGENTS.md):** `vibe: ` imperative subject ≤ 50 chars. Functional vs formatting changes in separate commits. Activate the venv when running Python.

**Branch state:** This continues on `jdb/train-on-existing-selfplay-games`. The just-shipped `source_run` mechanism is on the same branch (commit `31ff54e` and follow-ups); we replace it in place — no main has ever seen it.

---

## Task 1: `InitialModel.run` + 3-way mutual exclusion validator

**Files:**
- Modify: `deep_quoridor/src/v2/config.py:82-92` (the `InitialModel` class)
- Modify: `deep_quoridor/test/config_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `deep_quoridor/test/config_test.py`. The override parser supports dotted keys with auto-created intermediate dicts (see `_apply_overrides` → `_ensure_and_navigate`), so we can target `training.initial_model.run=...` directly without needing a dict literal.

```python
def test_initial_model_run_accepted(config_file):
    config = load_user_config(
        config_file, overrides=["training.initial_model.run=/some/old/run"]
    )
    assert config.training.initial_model is not None
    assert config.training.initial_model.run == "/some/old/run"
    assert config.training.initial_model.file is None
    assert config.training.initial_model.wandb_alias is None


def test_initial_model_rejects_file_plus_run(config_file):
    with pytest.raises(Exception, match="initial_model"):
        load_user_config(
            config_file,
            overrides=[
                "training.initial_model.file=/a.pt",
                "training.initial_model.run=/some/old/run",
            ],
        )


def test_initial_model_rejects_wandb_alias_plus_run(config_file):
    with pytest.raises(Exception, match="initial_model"):
        load_user_config(
            config_file,
            overrides=[
                "training.initial_model.wandb_alias=m1",
                "training.initial_model.run=/some/old/run",
            ],
        )


def test_initial_model_rejects_file_plus_wandb_alias(config_file):
    # Existing behavior; restated under the new model_validator.
    with pytest.raises(Exception, match="initial_model"):
        load_user_config(
            config_file,
            overrides=[
                "training.initial_model.file=/a.pt",
                "training.initial_model.wandb_alias=m1",
            ],
        )
```

`pytest` is already imported in the file.

- [ ] **Step 2: Run the new tests to confirm they fail**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/config_test.py::test_initial_model_run_accepted deep_quoridor/test/config_test.py::test_initial_model_rejects_file_plus_run deep_quoridor/test/config_test.py::test_initial_model_rejects_wandb_alias_plus_run -v
```

Expected: `test_initial_model_run_accepted` fails with `extra_forbidden` (unknown field `run`); the two `rejects_*` tests with `run` may fail or pass depending on whether the field exists. Either way they will be correct after Step 3.

- [ ] **Step 3: Replace `InitialModel`'s validator with a 3-way `model_validator`**

In `deep_quoridor/src/v2/config.py`, ensure `model_validator` is imported from pydantic at the top:

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
```

Replace the `InitialModel` class (lines 82-92) with:

```python
class InitialModel(StrictBaseModel):
    file: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_alias: Optional[str] = None
    run: Optional[str] = None

    @model_validator(mode="after")
    def at_most_one_source(self) -> "InitialModel":
        sources = [
            ("file", self.file),
            ("wandb_alias", self.wandb_alias),
            ("run", self.run),
        ]
        set_sources = [name for name, val in sources if val is not None]
        if len(set_sources) > 1:
            raise ValueError(
                "At most one of file, wandb_alias, run may be set in initial_model; "
                f"got: {set_sources}"
            )
        return self
```

This replaces the old `file_and_wandb_mutually_exclusive` field_validator. The new validator covers the same 2-way case and the two new pair cases involving `run`.

- [ ] **Step 4: Run the full `config_test.py` to confirm everything passes**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/config_test.py -v
```

Expected: all tests pass (the 4 new ones plus the pre-existing 15).

- [ ] **Step 5: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/v2/config.py deep_quoridor/test/config_test.py
git commit -m "vibe: add initial_model.run + 3-way exclusion"
```

End the commit message with the Co-Authored-By trailer.

---

## Task 2: `InitialReplayBuffer` sub-model + replace `source_run`

**Files:**
- Modify: `deep_quoridor/src/v2/config.py` (TrainingConfig)
- Modify: `deep_quoridor/test/config_test.py`

- [ ] **Step 1: Write the failing test (and remove the old `source_run` tests)**

In `deep_quoridor/test/config_test.py`, **delete** these two tests (lines around 108-115):
```python
def test_override_source_run(config_file):
    ...
def test_source_run_defaults_to_none(config_file):
    ...
```

Append these new tests:

```python
def test_initial_replay_buffer_accepted(config_file):
    config = load_user_config(
        config_file, overrides=["training.initial_replay_buffer.run=/some/old/run"]
    )
    assert config.training.initial_replay_buffer is not None
    assert config.training.initial_replay_buffer.run == "/some/old/run"


def test_initial_replay_buffer_defaults_to_none(config_file):
    config = load_user_config(config_file)
    assert config.training.initial_replay_buffer is None


def test_source_run_field_no_longer_exists(config_file):
    # Removed in favor of training.initial_replay_buffer.
    with pytest.raises(Exception, match="source_run|extra"):
        load_user_config(config_file, overrides=["training.source_run=/some/old/run"])
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/config_test.py::test_initial_replay_buffer_accepted deep_quoridor/test/config_test.py::test_initial_replay_buffer_defaults_to_none deep_quoridor/test/config_test.py::test_source_run_field_no_longer_exists -v
```

Expected: the first two fail with `extra_forbidden` for `initial_replay_buffer`; the third passes today (source_run still exists) but will need to start failing after Step 3 — so it confirms the schema change actually took effect.

- [ ] **Step 3: Add `InitialReplayBuffer` and replace `source_run` in `TrainingConfig`**

In `deep_quoridor/src/v2/config.py`, just above the `CosineWarmRestartsSchedulerConfig` class (around line 95), add:

```python
class InitialReplayBuffer(StrictBaseModel):
    """Configures preloading the replay buffer from a previous run.

    `run` points at a run directory (parent of `replay_buffers/`), mirroring
    `InitialModel.run`. At preload time the loader reads `<run>/replay_buffers/`.
    """

    run: str
```

In `TrainingConfig` (around line 105-117), replace the line `source_run: Optional[str] = None` with:

```python
    initial_replay_buffer: Optional[InitialReplayBuffer] = None
```

Final `TrainingConfig`:

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
    initial_replay_buffer: Optional[InitialReplayBuffer] = None
    lr_scheduler: Optional[LRSchedulerConfig] = None
```

- [ ] **Step 4: Run `config_test.py` to confirm everything passes**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/config_test.py -v
```

Expected: all tests pass. The two old `source_run` tests are gone; three new tests in place.

- [ ] **Step 5: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/v2/config.py deep_quoridor/test/config_test.py
git commit -m "vibe: replace source_run with initial_replay_buffer"
```

End with the Co-Authored-By trailer.

---

## Task 3: `self_play.enabled` field + Config validators + rust-binary gate

**Files:**
- Modify: `deep_quoridor/src/v2/config.py` (SelfPlayConfig, UserConfig, `load_config_and_setup_run`)
- Modify: `deep_quoridor/test/config_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `deep_quoridor/test/config_test.py`:

```python
def test_self_play_enabled_defaults_true(config_file):
    config = load_user_config(config_file)
    assert config.self_play.enabled is True


def test_self_play_enabled_can_be_false(config_file):
    config = load_user_config(
        config_file,
        overrides=[
            "self_play.enabled=False",
            "training.initial_replay_buffer.run=/some/old/run",
        ],
    )
    assert config.self_play.enabled is False


def test_selfplay_off_without_replay_buffer_is_rejected(config_file):
    with pytest.raises(Exception, match="initial_replay_buffer"):
        load_user_config(config_file, overrides=["self_play.enabled=False"])
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/config_test.py::test_self_play_enabled_defaults_true deep_quoridor/test/config_test.py::test_self_play_enabled_can_be_false deep_quoridor/test/config_test.py::test_selfplay_off_without_replay_buffer_is_rejected -v
```

Expected: all three fail with `extra_forbidden` for the unknown `enabled` field.

- [ ] **Step 3: Add `enabled` to `SelfPlayConfig` and the Config-level validator**

In `deep_quoridor/src/v2/config.py`, modify the `SelfPlayConfig` class (around lines 66-79). Add `enabled: bool = True` near the top of the field list:

```python
class SelfPlayConfig(StrictBaseModel):
    enabled: bool = True
    num_processes: int
    games_per_process: int
    # Leaf-parallel MCTS knobs (Rust self-play only).
    leaf_parallelism: int = 16
    virtual_loss: int = 3
    enable_tree_reuse: bool = True
    mcts_worker_threads: Optional[int] = None
    eval_batch_size: int = 2048
    eval_max_wait_ms: int = 0
    eval_cache_max_size: int = 100000
    alphazero: Optional[AlphaZeroSelfPlayConfig] = None
    program: Literal["python", "rust"] = "python"
    rust_selfplay_binary: Optional[str] = None
```

In the `UserConfig` class (around lines 178-198), add a `model_validator(mode="after")` after the existing `replace_datetime_placeholder` field_validator. The full class becomes:

```python
class UserConfig(StrictBaseModel):
    """A normal pydantic model that can be used as an inner class."""

    run_id: str
    quoridor: QuoridorConfig
    alphazero: AlphaZeroBaseConfig
    wandb: Optional[WandbConfig] = None
    self_play: SelfPlayConfig
    training: TrainingConfig
    benchmarks: list[BenchmarkScheduleConfig] = []
    ai_report: Optional[AIReportConfig] = None

    @field_validator("run_id")
    @classmethod
    def replace_datetime_placeholder(cls, v: str) -> str:
        """Replace $DATETIME with current datetime in format YYYYMMDD-HHMM."""
        if "$DATETIME" in v:
            current_datetime = datetime.now().strftime("%Y%m%d-%H%M")
            return v.replace("$DATETIME", current_datetime)
        return v

    @model_validator(mode="after")
    def selfplay_off_requires_replay_buffer(self) -> "UserConfig":
        if not self.self_play.enabled and self.training.initial_replay_buffer is None:
            raise ValueError(
                "When self_play.enabled is False, training.initial_replay_buffer must be set "
                "(otherwise the trainer would hang forever waiting for games)."
            )
        return self
```

- [ ] **Step 4: Gate the rust-binary check on `enabled`**

In `deep_quoridor/src/v2/config.py`, find `load_config_and_setup_run` (around line 362-390). The current code reads:

```python
    use_rust = config.self_play.program == "rust"
    if use_rust:
        # Apply default Rust binary path if not specified in config
        if config.self_play.rust_selfplay_binary is None:
            config.self_play.rust_selfplay_binary = str(
                Path(__file__).parent.parent.parent / "rust" / "target" / "release" / "selfplay"
            )
        rust_binary = config.self_play.rust_selfplay_binary
        if not Path(rust_binary).exists():
            print(f"ERROR: Rust self-play binary not found at {rust_binary}")
            print("Build it with: cd deep_quoridor/rust && cargo build --release --features binary --bin selfplay")
            exit(1)
        # Rust self-play requires ONNX model exports
        config.training.save_onnx = True
```

Change the first line to:

```python
    use_rust = config.self_play.enabled and config.self_play.program == "rust"
```

Rest unchanged. When self-play is disabled, no rust binary is needed.

- [ ] **Step 5: Run `config_test.py` to confirm everything passes**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/config_test.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/v2/config.py deep_quoridor/test/config_test.py
git commit -m "vibe: add self_play.enabled + cross-field guards"
```

End with the Co-Authored-By trailer.

---

## Task 4: Resolve `initial_model.run` in `alphazero_params_dict_from_config`

**Files:**
- Modify: `deep_quoridor/src/v2/common.py:134-142` (the initial_model branch in `alphazero_params_dict_from_config`)
- Test: covered by Task 7 smoke verification (a unit test for this would require mocking the AlphaZero model load, which is heavyweight and low-value; the smoke run with `initial_model.run` set covers it end-to-end). Add one targeted unit test of the resolution helper alone — see Step 1.

- [ ] **Step 1: Write the failing test**

Append to `deep_quoridor/test/config_test.py` (kept in the same test file since it's a small assertion on config-derived behavior):

```python
def test_initial_model_run_resolves_to_latest_filename(tmp_path):
    """alphazero_params_dict_from_config translates initial_model.run into the
    .pt filename recorded in <run>/models/latest.yaml."""
    from pydantic_yaml import to_yaml_file
    from v2.common import alphazero_params_dict_from_config
    from v2.config import Config, load_user_config
    from v2.yaml_models import LatestModel

    # Build a fake "old run" with a latest.yaml pointing at a model file.
    old_run = tmp_path / "old_run"
    models_dir = old_run / "models"
    models_dir.mkdir(parents=True)
    to_yaml_file(
        models_dir / "latest.yaml",
        LatestModel(filename=str(old_run / "models" / "checkpoints" / "model_42.pt"), version=42),
    )

    # Build a config that points initial_model.run at the fake run.
    cfg_data = dict(EXAMPLE_CONFIG)
    cfg_data["training"] = {
        **EXAMPLE_CONFIG["training"],
        "initial_model": {"run": str(old_run)},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data, sort_keys=False))

    user = load_user_config(str(cfg_path))
    config = Config.from_user(user, str(tmp_path), create_dirs=False)

    params = alphazero_params_dict_from_config(config)
    assert params["model_filename"] == str(old_run / "models" / "checkpoints" / "model_42.pt")
```

(Imports inside the test keep them local; `EXAMPLE_CONFIG` and `yaml` are already imported at the top of `config_test.py`.)

- [ ] **Step 2: Run the new test to confirm it fails**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/config_test.py::test_initial_model_run_resolves_to_latest_filename -v
```

Expected: passes the config validation (Task 1 added `run`) but `params["model_filename"]` is not set — KeyError or missing key. The current `if im.file: ...; if im.wandb_alias: ...` branches don't handle `run`.

- [ ] **Step 3: Wire `run` in `alphazero_params_dict_from_config`**

In `deep_quoridor/src/v2/common.py`, the file already imports `LatestModel` (line 9). Add `parse_yaml_file_as` and `Path` if not present. Current imports at the top of `common.py`:

```python
import re
import time
from abc import abstractmethod
from typing import Any, Callable, Optional

import wandb
from agents.alphazero import AlphaZeroAgent, AlphaZeroParams
from v2.config import AlphaZeroPlayConfig, AlphaZeroSelfPlayConfig, Config
from v2.yaml_models import LatestModel
```

Add:

```python
from pathlib import Path

from pydantic_yaml import parse_yaml_file_as
```

Then modify the `initial_model` block inside `alphazero_params_dict_from_config` (currently lines 134-142):

```python
    if config.training.initial_model:
        im = config.training.initial_model
        if im.file:
            params_dict["model_filename"] = im.file
        elif im.wandb_alias:
            params_dict["wandb_alias"] = im.wandb_alias
            params_dict["wandb_project"] = im.wandb_project or (
                config.wandb.project if config.wandb else "deep_quoridor"
            )
        elif im.run:
            latest_yaml = Path(im.run) / "models" / "latest.yaml"
            latest = parse_yaml_file_as(LatestModel, latest_yaml)
            params_dict["model_filename"] = latest.filename
```

Note the `if/elif/elif` chain (the three sources are mutually exclusive per the Task 1 validator).

- [ ] **Step 4: Run the test to confirm it passes**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/config_test.py::test_initial_model_run_resolves_to_latest_filename -v
```

Expected: PASS.

Also re-run the full `config_test.py` to confirm no regressions:

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/config_test.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/v2/common.py deep_quoridor/test/config_test.py
git commit -m "vibe: resolve initial_model.run via latest.yaml"
```

End with the Co-Authored-By trailer.

---

## Task 5: Rename trainer helper bool params; wire from new config

**Files:**
- Modify: `deep_quoridor/src/v2/trainer.py` (both helpers + the `train()` site that calls them)
- Modify: `deep_quoridor/test/test_trainer_helpers.py`

- [ ] **Step 1: Update the 5 tests to use the new param names**

In `deep_quoridor/test/test_trainer_helpers.py`, replace all five test functions. The truth table is unchanged; only the bool parameter names change (`offline_mode` → `selfplay_disabled` for `_should_skip_iteration`, and `offline_mode` → `omit_model_lag` for `_build_game_log`). The file's full contents should be:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_trainer_helpers.py -v
```

Expected: tests fail with `TypeError: _should_skip_iteration() got an unexpected keyword argument 'selfplay_disabled'` (same for `omit_model_lag`).

- [ ] **Step 3: Rename the helper bool params**

In `deep_quoridor/src/v2/trainer.py`, find the two helpers (added around lines 56-100 in the recent commits). Replace them with:

```python
def _should_skip_iteration(
    total_moves: int,
    batch_size: int,
    games_per_training_step: float,
    training_steps: int,
    last_game: int,
    selfplay_disabled: bool,
) -> bool:
    """Decide whether to skip this iteration of the trainer's main loop.

    Always skip when the buffer holds fewer moves than one batch. When self-play is
    enabled, also skip when the trainer is ahead of self-play (the
    `games_per_training_step` gate). When self-play is disabled the buffer is static
    and there is no production cadence to wait on, so we train every iteration once
    enough moves are available.
    """
    if total_moves < batch_size:
        return True
    if selfplay_disabled:
        return False
    games_needed_to_train = games_per_training_step * (training_steps + 1)
    return games_needed_to_train > last_game


def _build_game_log(
    game_info,
    model_version: int,
    last_game: int,
    omit_model_lag: bool,
) -> dict:
    """Per-game wandb log payload emitted when a game is ingested from ready/.

    `model_lag` is meaningful only when games arrive from live self-play of the
    current run, since it compares the trainer's current model version against the
    version that *produced* the game. When games come from a preloaded buffer
    (different lineage), the subtraction is nonsense, so the caller asks us to
    omit the key.
    """
    log = {
        "game_length": game_info.game_length,
        "Game num": last_game,
        "Model version": model_version,
    }
    if not omit_model_lag:
        log["model_lag"] = model_version - 1 - game_info.model_version
    return log
```

- [ ] **Step 4: Wire `train()` to derive the new booleans from config**

In `deep_quoridor/src/v2/trainer.py`'s `train()` function, find the line `offline_mode = config.training.source_run is not None` (added in commit `583b3d9`, around line 121 after the helpers shifted positions). Replace it with:

```python
    selfplay_disabled = not config.self_play.enabled
    omit_model_lag = config.training.initial_replay_buffer is not None
```

Then find the `wandb_run.log(_build_game_log(game_info, model_version, last_game, offline_mode))` call (around line 205) and change it to:

```python
            wandb_run.log(_build_game_log(game_info, model_version, last_game, omit_model_lag))
```

Then find the `_should_skip_iteration(...)` call (around line 215-222). The keyword arg `offline_mode=offline_mode` becomes `selfplay_disabled=selfplay_disabled`:

```python
        if _should_skip_iteration(
            total_moves=total_moves,
            batch_size=batch_size,
            games_per_training_step=config.training.games_per_training_step,
            training_steps=training_steps,
            last_game=last_game,
            selfplay_disabled=selfplay_disabled,
        ):
            time.sleep(1)
            continue
```

- [ ] **Step 5: Run helper tests to confirm they pass**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_trainer_helpers.py -v
```

Expected: all 5 pass.

- [ ] **Step 6: Sanity-check trainer.py still imports**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/python -c "from v2.trainer import train; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 7: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/v2/trainer.py deep_quoridor/test/test_trainer_helpers.py
git commit -m "vibe: split trainer offline-mode into two booleans"
```

End with the Co-Authored-By trailer.

---

## Task 6: `train_v2.py` refactor — remove `--source-run`, wire from new config

**Files:**
- Modify: `deep_quoridor/src/train_v2.py`
- Delete: `deep_quoridor/test/test_train_v2_args.py`

- [ ] **Step 1: Delete the obsolete test file**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git rm deep_quoridor/test/test_train_v2_args.py
```

That file only tested `source_run_overrides`, which is being removed.

- [ ] **Step 2: Refactor `train_v2.py`**

Replace the entire `deep_quoridor/src/train_v2.py` with the following. The diff from the current file: removes the `source_run_overrides` helper, removes the `--source-run` argparse flag, removes the `extra_overrides` merge block, replaces the `offline_mode = ...` derivation with two booleans, updates the preload call to use `config.training.initial_replay_buffer.run`, and wraps the self-play branch in `if config.self_play.enabled:` instead of `if not offline_mode:`.

```python
import argparse
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

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
from v2.common import ShutdownSignal

# Prevents getting messages in the console every few lines telling you to install weave
os.environ["WANDB_DISABLE_WEAVE"] = "true"


def _selfplay_subprocess_env():
    """Environment for the Rust self-play subprocess.

    A selfplay binary built with the ``gpu`` feature loads ONNX Runtime
    dynamically, so it needs ``ORT_DYLIB_PATH`` pointing at the onnxruntime-gpu
    shared library and the CUDA/cuDNN wheel libs on ``LD_LIBRARY_PATH``. We
    discover both from the installed packages so GPU self-play works without
    manual shell setup. Returns ``None`` (inherit the current environment) when
    onnxruntime isn't installed, in which case a CPU build runs unchanged.
    """
    import importlib.util

    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or not spec.origin:
        return None
    pkg_dir = Path(spec.origin).parent
    dylibs = sorted(pkg_dir.glob("capi/libonnxruntime.so*"))
    if not dylibs:
        return None

    site_packages = pkg_dir.parent
    nvidia_libs = [str(p) for p in sorted((site_packages / "nvidia").glob("*/lib")) if p.is_dir()]

    env = dict(os.environ)
    env["ORT_DYLIB_PATH"] = str(dylibs[-1])
    ld_parts = nvidia_libs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    if ld_parts:
        env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Quoridor agent")
    parser.add_argument("config_file", type=str, help="Path to YAML configuration file")
    parser.add_argument("-r", "--runs-dir", type=str, default=None, help="Directory for runs")
    # TODO: implement this
    # parser.add_argument("-c", "--continue", dest="continue_run", action="store_true", help="Continue an existing run")
    parser.add_argument(
        "-o",
        "--overrides",
        nargs="*",
        help="Configuration overrides (e.g., run_id=my_run self_play.program=rust)",
    )

    args = parser.parse_args()

    runs_dir = args.runs_dir if args.runs_dir is not None else str(Path(__file__).parent.parent)

    config = load_config_and_setup_run(args.config_file, runs_dir, overrides=args.overrides)

    # Validate AI report prerequisites before spawning anything, so a misconfigured
    # run aborts early instead of failing silently inside a sibling process.
    if config.ai_report is not None:
        try:
            check_ai_available(config.ai_report.ai)
        except Exception as e:
            print(f"ERROR: {e}")
            exit(1)

    mp.set_start_method("spawn", force=True)

    # Make sure we don't have the shutdown signal from a previous run
    ShutdownSignal.clear(config)

    if config.training.initial_replay_buffer is not None:
        n_loaded = preload_symlinks(
            source_run=Path(config.training.initial_replay_buffer.run),
            dest_ready=config.paths.replay_buffers_ready,
            buffer_size=config.training.replay_buffer_size,
        )
        print(f"Preloaded {n_loaded} games from {config.training.initial_replay_buffer.run}")

    train_process = mp.Process(target=train, args=[config])
    train_process.start()

    benchmark_processes = benchmarks.create_benchmark_processes(config)
    [p.start() for p in benchmark_processes]

    ai_report_process = None
    if config.ai_report is not None:
        ai_report_process = mp.Process(target=run_ai_reporter, args=[config])
        ai_report_process.start()

    self_play_processes = []
    rust_subprocesses = []

    if config.self_play.enabled:
        if config.self_play.program == "rust":
            # Spawn Rust self-play processes in continuous mode
            selfplay_env = _selfplay_subprocess_env()
            if selfplay_env is not None:
                print(f"Self-play GPU env: ORT_DYLIB_PATH={selfplay_env['ORT_DYLIB_PATH']}")
            metrics_dir = metrics_dir_for(config)
            os.makedirs(metrics_dir, exist_ok=True)
            config_file_path = str(config.paths.config_file)
            for i in range(config.self_play.num_processes):
                cmd = [
                    config.self_play.rust_selfplay_binary,
                    "--config",
                    config_file_path,
                    "--output-dir",
                    str(config.paths.replay_buffers_ready),
                    "--continuous",
                    "--latest-model-yaml",
                    str(config.paths.latest_model_yaml),
                    "--shutdown-file",
                    str(ShutdownSignal.file_path(config)),
                    "--metrics-dir",
                    metrics_dir,
                ]
                proc = subprocess.Popen(cmd, env=selfplay_env)
                rust_subprocesses.append(proc)
                print(f"Started Rust self-play process {proc.pid}")
            selfplay_metrics_process = mp.Process(target=run_selfplay_metrics, args=[config])
            selfplay_metrics_process.start()
            self_play_processes.append(selfplay_metrics_process)
        else:
            for i in range(config.self_play.num_processes):
                p = mp.Process(target=self_play, args=[config])
                p.start()
                self_play_processes.append(p)

    train_process.join()
    ShutdownSignal.signal(config)
    print("Shutting down!")

    b_count_prev, sf_count_prev, ai_count_prev = -1, -1, -1
    while True:
        b_count = sum([p.is_alive() for p in benchmark_processes])
        sf_count = sum([p.is_alive() for p in self_play_processes])
        sf_count += sum([p.poll() is None for p in rust_subprocesses])
        ai_count = 1 if ai_report_process is not None and ai_report_process.is_alive() else 0
        if b_count_prev != b_count or sf_count_prev != sf_count or ai_count_prev != ai_count:
            print(
                f"Waiting for {b_count} benchmark processes, {sf_count} self_play processes"
                f" and {ai_count} ai_report processes"
            )
            b_count_prev, sf_count_prev, ai_count_prev = b_count, sf_count, ai_count

        if (b_count + sf_count + ai_count) == 0:
            break
        time.sleep(1)

    ShutdownSignal.clear(config)
```

- [ ] **Step 3: Sanity-check the script imports + `--help`**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/python -c "import train_v2; print('ok')"
```

Expected: prints `ok`.

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/python deep_quoridor/src/train_v2.py --help 2>&1 | head -20
```

Expected: usage text with `config_file`, `-r/--runs-dir`, `-o/--overrides` — and **no** `--source-run` flag.

- [ ] **Step 4: Re-run all affected tests**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/config_test.py deep_quoridor/test/test_offline_preload.py deep_quoridor/test/test_trainer_helpers.py deep_quoridor/test/test_selfplay_metrics.py -v
```

Expected: all pass. (`test_train_v2_args.py` has been deleted so it's not in the list.)

- [ ] **Step 5: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/train_v2.py deep_quoridor/test/test_train_v2_args.py
git commit -m "vibe: drop --source-run; wire offline from config"
```

End with the Co-Authored-By trailer.

---

## Task 7: End-to-end smoke verification + format pass

This task verifies the wired-together behavior against a tiny source run. Two smoke configurations: pure offline (yesterday's behavior, reproduced via the new config form) and mixed mode (preload + self-play on). It also runs a formatting pass.

**Files:** No code changes for the smoke tests. The formatting commit may touch the files modified in Tasks 1-6.

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

- [ ] **Step 2: Create the offline-mode config**

Save this to `/tmp/smoke_offline.yaml`:

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
  enabled: false
  num_processes: 1
  games_per_process: 4
  alphazero:
    mcts_noise_epsilon: 0.25
training:
  games_per_training_step: 1.0
  learning_rate: 0.001
  batch_size: 32
  weight_decay: 0.0001
  replay_buffer_size: 3
  finish_after: "3 models"
  initial_replay_buffer:
    run: /tmp/smoke_source_run
```

- [ ] **Step 3: Run pure-offline smoke**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src timeout 30 .venv/bin/python deep_quoridor/src/train_v2.py /tmp/smoke_offline.yaml -r /tmp/smoke_runs 2>&1 | tee /tmp/smoke_offline.log
```

(Like yesterday: the trainer will eventually crash on a tensor-shape mismatch because the fake arrays don't match the real network's input shape. We're verifying boot + preload + self-play-skip, not training correctness.)

- [ ] **Step 4: Verify offline-smoke output**

```bash
grep -E "Preloaded .* games from|Started Rust self-play|self_play process" /tmp/smoke_offline.log
```

Expected:
- `Preloaded 3 games from /tmp/smoke_source_run` line present.
- NO `Started Rust self-play process` lines.
- NO `self_play process` spawning messages (from the python branch).

```bash
ls -la /tmp/smoke_runs/runs/smoke-offline-*/replay_buffers/ 2>/dev/null | head -10
```

Expected: at least 3 symlinks pointing into `/tmp/smoke_source_run/replay_buffers/`.

- [ ] **Step 5: Create the mixed-mode config and run it**

Save this to `/tmp/smoke_mixed.yaml`:

```yaml
run_id: smoke-mixed-$DATETIME
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
  enabled: true
  program: python
  num_processes: 1
  games_per_process: 4
  alphazero:
    mcts_noise_epsilon: 0.25
training:
  games_per_training_step: 1.0
  learning_rate: 0.001
  batch_size: 32
  weight_decay: 0.0001
  replay_buffer_size: 3
  finish_after: "3 models"
  initial_replay_buffer:
    run: /tmp/smoke_source_run
```

Run:

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src timeout 15 .venv/bin/python deep_quoridor/src/train_v2.py /tmp/smoke_mixed.yaml -r /tmp/smoke_runs 2>&1 | tee /tmp/smoke_mixed.log || true
```

(`|| true` so a non-zero exit from `timeout` doesn't fail the script — we expect the trainer to crash on the same shape mismatch and possibly kill the run before clean shutdown.)

- [ ] **Step 6: Verify mixed-mode output**

```bash
grep -E "Preloaded .* games from" /tmp/smoke_mixed.log
```

Expected: `Preloaded 3 games from /tmp/smoke_source_run` is present.

```bash
ps -eo pid,cmd 2>/dev/null | grep -i "self_play\|train_v2" | grep -v grep | head
```

May show python self-play processes if they're still alive. Either way, the key signal is in the log — verify that self-play python processes are mentioned by checking the trainer's behavior. (Lower-confidence signal in mixed mode because the trainer crashes early; the `Preloaded` line is the primary check that `initial_replay_buffer` is honored independently of self-play.)

- [ ] **Step 7: Cleanup**

```bash
rm -rf /tmp/smoke_source_run /tmp/smoke_runs /tmp/smoke_offline.yaml /tmp/smoke_mixed.yaml /tmp/smoke_offline.log /tmp/smoke_mixed.log
```

- [ ] **Step 8: Format pass**

Run ruff format on the touched files:

```bash
cd /home/jbinney/ws/deep_rabbit_hole && .venv/bin/ruff format deep_quoridor/src/v2/config.py deep_quoridor/src/v2/common.py deep_quoridor/src/v2/trainer.py deep_quoridor/src/train_v2.py deep_quoridor/test/config_test.py deep_quoridor/test/test_trainer_helpers.py
```

If anything was reformatted, commit it separately per AGENTS.md:

```bash
cd /home/jbinney/ws/deep_rabbit_hole && git status --short
cd /home/jbinney/ws/deep_rabbit_hole && git add -u && git commit -m "vibe: ruff format"
```

(End with the Co-Authored-By trailer.) If nothing changed, skip this commit.

---

## Done criteria

- All tests in `config_test.py`, `test_offline_preload.py`, `test_trainer_helpers.py`, `test_selfplay_metrics.py` pass.
- `test_train_v2_args.py` is gone.
- `train_v2.py --help` does not show `--source-run`.
- Offline smoke run prints `Preloaded N games from <source>` and spawns NO self-play processes.
- Mixed-mode smoke run prints `Preloaded N games from <source>` and DOES proceed to spawn self-play.
- Touched files clean under `ruff format`.
