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


def _check_run_dir(run_dir: Path) -> None:
    """Verify the run directory has the layout we need before spawning processes.

    Aborts early on a missing `latest.yaml` so the benchmark processes don't enter
    `LatestModel.wait_for_creation`'s blocking wait (no training is producing
    models in this script).
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not (run_dir / "config.yaml").is_file():
        raise FileNotFoundError(f"No config.yaml in {run_dir}")
    latest_yaml = run_dir / "models" / "latest.yaml"
    if not latest_yaml.is_file():
        raise FileNotFoundError(
            f"No models/latest.yaml in {run_dir}; the run has no trained model to benchmark."
        )


def main(args) -> int:
    """Entry point. Returns the exit code."""
    run_dir = Path(args.run_dir).resolve()
    _check_run_dir(run_dir)
    config = _load_config(run_dir, args.overrides)

    if not config.benchmarks:
        print(f"No benchmarks configured in {run_dir}/config.yaml; nothing to run.")
        return 0

    # Spawning is added in Task 3.
    return 0
