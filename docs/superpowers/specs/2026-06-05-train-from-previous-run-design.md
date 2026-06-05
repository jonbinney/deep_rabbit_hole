# Train from a previous run — design

**Date:** 2026-06-05
**Branch:** `jdb/train-on-existing-selfplay-games`
**Status:** approved (design), pending implementation plan

**Supersedes:** `docs/superpowers/specs/2026-06-04-train-on-existing-selfplay-design.md`
(the offline-mode feature shipped yesterday; this refactor generalizes it).

## Problem

Yesterday we shipped `--source-run`, a single CLI flag that did three things at once: pull
the previous run's replay buffer, infer "skip self-play", and force `self_play.program=python`
to dodge the rust-binary check. The three concerns are actually independent:

1. **Inherit weights** from a previous run's latest checkpoint (continue training, possibly
   with the same architecture).
2. **Inherit replay buffer** from a previous run's stored games (warm-start the buffer).
3. **Disable self-play** entirely (consume a static buffer; the "old offline mode").

Wanting any single one without the others is a real use case. Wanting all three is the
yesterday-shipped offline mode. The current schema can't express the partial cases cleanly,
and the CLI flag bypasses config — leaking the mode into the call site rather than the run's
recorded config.

This design replaces the single `--source-run` mechanism with three independent config knobs,
removes the CLI flag, and keeps the existing `preload_symlinks` machinery unchanged.

## Approach

Three independent additions to the existing config, all opt-in, all default to today's
backward-compatible behavior:

```yaml
training:
  initial_model:
    run: /path/to/old/run          # NEW — joins file: and wandb_alias: as a third source
  initial_replay_buffer:
    run: /path/to/old/run          # NEW — replaces the just-shipped training.source_run

self_play:
  enabled: false                   # NEW — defaults true; explicit on/off
```

`--source-run` and the `source_run_overrides` helper are removed. The user expresses each
concern explicitly, either in the yaml or via the existing `-o key=val` override mechanism.

The three concerns are honored independently inside `train_v2.py` and `trainer.py` — see
Section "Trainer wiring" below. In particular, mixed mode (preload + self-play on) is now a
legitimate combination: preloaded games seed the buffer, new self-play games stream in on
top.

## Config schema

```python
class InitialModel(StrictBaseModel):
    file: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_alias: Optional[str] = None
    run: Optional[str] = None    # NEW

    # Validator: at most one of {file, wandb_alias, run} may be set.

class InitialReplayBuffer(StrictBaseModel):
    run: str    # required when initial_replay_buffer is set.
                # Points at a run dir (parent of replay_buffers/), mirroring
                # initial_model.run. The preloader reads <run>/replay_buffers/.

class TrainingConfig(StrictBaseModel):
    # ... existing fields ...
    initial_replay_buffer: Optional[InitialReplayBuffer] = None   # replaces source_run
    # ... existing fields ...

class SelfPlayConfig(StrictBaseModel):
    enabled: bool = True       # NEW
    num_processes: int
    # ... other existing fields unchanged; all ignored when enabled=False
```

### Cross-field validators

- `InitialModel.run` joins the existing mutual-exclusion validator. At most one of
  `{file, wandb_alias, run}` may be set.
- `Config`-level (new): refuse to load when `not self_play.enabled and
  training.initial_replay_buffer is None` — the trainer would hang forever waiting for
  games that never arrive.

### Loader changes

- `load_config_and_setup_run`'s rust-binary existence check is now gated on
  `config.self_play.enabled and config.self_play.program == "rust"`. Today's check fires
  whenever `program=="rust"`; the `enabled=False` shortcut means a user can disable
  self-play without needing a rust binary present.

### Removed

- `training.source_run` (only landed yesterday; nothing depends on it).
- CLI flag `--source-run`.
- Helper `source_run_overrides` in `train_v2.py`.
- Test file `deep_quoridor/test/test_train_v2_args.py` (only existed to cover the removed
  helper).

## Resolving `initial_model.run` → weights file

`v2/common.py`'s `create_alphazero` already builds `params_dict["model_filename"]` from
`initial_model.file`. The `run` branch reuses that hand-off:

```python
if im.file:
    params_dict["model_filename"] = im.file
elif im.wandb_alias:
    params_dict["wandb_alias"] = im.wandb_alias
    params_dict["wandb_project"] = im.wandb_project or (...)
elif im.run:
    latest = parse_yaml_file_as(LatestModel, Path(im.run) / "models" / "latest.yaml")
    params_dict["model_filename"] = latest.filename
```

The existing `LatestModel` schema (`v2/yaml_models.py`) already exposes `filename` as an
absolute path to a `.pt` file — exactly what `model_filename` expects. No new on-disk format
or path-resolution rules.

Architecture-mismatch failure mode: if the prior `.pt` doesn't match the new network shape,
loading fails at `create_alphazero` time with the existing torch error. We don't try to
guard this — same behavior as setting `initial_model.file` to an incompatible checkpoint
today.

## Trainer wiring

Today's trainer has a single `offline_mode = config.training.source_run is not None` that
both (a) skips the `games_per_training_step` gate and (b) suppresses the `model_lag` wandb
metric. The two concerns split cleanly with the new config:

- `selfplay_disabled = not config.self_play.enabled` — the static-buffer property. No
  production cadence to wait on; the gate skip in `_should_skip_iteration` is tied to this.
- `omit_model_lag = config.training.initial_replay_buffer is not None` — the lineage
  property. Preloaded games' `game_info.model_version` came from a different training run,
  making the `model_version - 1 - game_info.model_version` subtraction meaningless. The
  `_build_game_log` suppression is tied to this. In mixed mode (preload + self-play on),
  later self-play games also lose `model_lag` for the rest of the run; acceptable v1
  simplification — the metric is most useful for diagnosing a self-play-only setup.

Both helpers (`_should_skip_iteration`, `_build_game_log`) keep their shapes; only the bool
parameter is renamed (`offline_mode` → `selfplay_disabled` / `omit_model_lag`).

The split matters: with preload-only mode (mixed self-play on), the gate must still be
honored — new games arrive at the normal cadence, and skipping the gate would let the
trainer race ahead of self-play. Tying the gate skip to `not self_play.enabled` gets that
right.

## `train_v2.py` wiring

Inside `if __name__ == "__main__":`:

```python
config = load_config_and_setup_run(args.config_file, runs_dir, overrides=args.overrides)

# ...existing AI-report check, mp.set_start_method, ShutdownSignal.clear...

if config.training.initial_replay_buffer is not None:
    n_loaded = preload_symlinks(
        source_run=Path(config.training.initial_replay_buffer.run),
        dest_ready=config.paths.replay_buffers_ready,
        buffer_size=config.training.replay_buffer_size,
    )
    print(f"Preloaded {n_loaded} games from {config.training.initial_replay_buffer.run}")

# ... train_process, benchmark_processes, ai_report_process as today ...

self_play_processes = []
rust_subprocesses = []

if config.self_play.enabled:
    if config.self_play.program == "rust":
        # ... existing rust branch ...
    else:
        # ... existing python branch ...
```

The shutdown wait loop at the bottom of `train_v2.py` works unchanged — `self_play_processes`
and `rust_subprocesses` are always initialized.

## Migration

`source_run` has been on the branch since commit `31ff54e` (2026-06-04) and was never
released or used in any committed yaml config. We delete it outright — no alias, no
deprecation period.

A user with an existing yaml that sets `training.source_run` will get a pydantic
`extra_forbidden` error on load, with the field name in the error message; they update to
the new schema.

## Edge cases & failure modes

- `initial_model.run` set, but `<run>/models/latest.yaml` is missing → fails at
  `create_alphazero` time with a clear `FileNotFoundError` (matches today's behavior for
  `initial_model.file` pointing at a missing file).
- `initial_model.run` set with two of `{file, wandb_alias, run}` → caught by the pydantic
  validator at load time.
- `initial_replay_buffer.run` missing dir or no `.npz` files → fails at preload time via
  the existing `preload_symlinks` errors.
- `self_play.enabled=False` AND `initial_replay_buffer is None` → caught by the new
  Config-level validator at load time. Clear message: no source of games.
- `self_play.enabled=True` AND `initial_replay_buffer` set → mixed mode. Trainer's existing
  ready/-pickup loop handles both naturally. The `omit_model_lag` flag is honored for the
  whole run.
- `self_play.enabled=False` AND `program=rust` in yaml → rust binary check skipped; the
  program field is unused. No need to override it manually.
- Source run still being written to by another process → out of scope (same as
  yesterday's spec); document "point at a dormant source run."

## Testing

**Unit tests:**

- `config_test.py` additions:
  - `initial_model.run` accepted, default None.
  - `initial_model` rejects setting two or more of `{file, wandb_alias, run}`.
  - `self_play.enabled` accepted, defaults True.
  - `initial_replay_buffer.run` parses correctly.
  - Config-level validator rejects `enabled=False` with no `initial_replay_buffer`.
- `test_trainer_helpers.py` updates:
  - Rename bool params; verify the gate-skip is tied to `selfplay_disabled` and `model_lag`
    suppression is tied to `omit_model_lag`. The existing 5 tests already cover the truth
    table — just rename the params.

**Deleted:**

- `test_train_v2_args.py` (the only thing it tested no longer exists).

**No changes:**

- `test_offline_preload.py` — preload functions are unchanged.

**Manual verification:**

- Re-run a variant of yesterday's smoke test with the new config: build a tiny source run;
  run train_v2 with `initial_replay_buffer.run=<source>` and `self_play.enabled=false` in the
  yaml; assert the same observable behavior (preload print, symlinks present, no rust
  spawn).
- Spot-check mixed mode: same yaml but with `self_play.enabled=true` — boot should preload
  AND spawn self-play.

## Out of scope (YAGNI)

- Loading replay buffer from anywhere other than a previous run dir (e.g. arbitrary file
  list). The sub-config shape leaves room for it (`initial_replay_buffer.from_files: [...]`)
  but we don't add it now.
- Loading model from a specific older checkpoint of a run (i.e. `model_5.pt` rather than
  latest). Users wanting this still use `initial_model.file: <run>/models/checkpoints/model_5.pt`.
- Cross-run partial weight transfer (architecture mismatch tolerated). Same out-of-scope
  note as yesterday's spec.
- Startup validation that prior-run quoridor params match the new config. Same deferred
  note as yesterday's spec — failure mode is loud at first batch.
