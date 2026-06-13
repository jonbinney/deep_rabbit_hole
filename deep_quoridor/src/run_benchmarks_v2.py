"""Run just the benchmark schedules from an existing run's config.yaml.

Usage:
    python deep_quoridor/src/run_benchmarks_v2.py <run_dir> [-o key=val ...]

Spawns one process per `config.benchmarks` schedule and waits until Ctrl-C.
Reuses `benchmarks.create_benchmark_processes` from the v2 package; does not
train, run self-play, or generate AI reports.
"""

from pathlib import Path

from v2.config import Config, load_user_config


def _derive_base_dir(run_dir: Path) -> str:
    """Given a run dir laid out as `base_dir/runs/<run_id>/`, return `base_dir`.

    The run-dir convention used by `train_v2.py`'s `load_config_and_setup_run`
    places each run under `<base_dir>/runs/<run_id>/`, so the parent of `runs/`
    is the base_dir the rest of the v2 machinery expects.
    """
    return str(run_dir.parent.parent)


def _load_config(run_dir: Path, overrides: list[str] | None) -> Config:
    """Load `<run_dir>/config.yaml` and build a Config without touching disk.

    Uses `Config.from_user(..., create_dirs=False)` so the existing run directory
    isn't disturbed and no `config.yaml` snapshot is rewritten. Raises
    `FileNotFoundError` if the config file is missing.
    """
    config_yaml = run_dir / "config.yaml"
    if not config_yaml.is_file():
        raise FileNotFoundError(f"No config.yaml in {run_dir}")
    user_config = load_user_config(str(config_yaml), overrides=overrides)
    return Config.from_user(user_config, _derive_base_dir(run_dir), create_dirs=False)
