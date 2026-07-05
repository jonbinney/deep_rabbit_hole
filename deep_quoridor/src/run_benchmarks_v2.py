"""Run just the benchmark schedules from an existing run's config.yaml.

Usage:
    python deep_quoridor/src/run_benchmarks_v2.py <run_dir> [-o key=val ...]

Spawns one process per `config.benchmarks` schedule and waits until Ctrl-C.
Reuses `benchmarks.create_benchmark_processes` from the v2 package; does not
train, run self-play, or generate AI reports.
"""

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

from v2 import benchmarks
from v2.common import ShutdownSignal
from v2.config import Config, load_user_config

# Match train_v2.py: suppress wandb's "install weave" log spam.
os.environ["WANDB_DISABLE_WEAVE"] = "true"


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
        raise FileNotFoundError(f"No models/latest.yaml in {run_dir}; the run has no trained model to benchmark.")


def main(args) -> int:
    """Entry point. Returns the exit code."""
    run_dir = Path(args.run_dir).resolve()
    _check_run_dir(run_dir)
    config = _load_config(run_dir, args.overrides)

    if not config.benchmarks:
        print(f"No benchmarks configured in {run_dir}/config.yaml; nothing to run.")
        return 0

    mp.set_start_method("spawn", force=True)
    ShutdownSignal.clear(config)

    benchmark_processes = benchmarks.create_benchmark_processes(config)
    for p in benchmark_processes:
        p.start()
    print(f"Started {len(benchmark_processes)} benchmark processes")

    try:
        b_count_prev = -1
        while True:
            b_count = sum(p.is_alive() for p in benchmark_processes)
            if b_count != b_count_prev:
                print(f"Waiting for {b_count} benchmark processes")
                b_count_prev = b_count
            if b_count == 0:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCaught Ctrl-C; signaling shutdown...")
        ShutdownSignal.signal(config)
        for p in benchmark_processes:
            p.join()

    ShutdownSignal.clear(config)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run just the benchmark schedules from an existing run's config.yaml.",
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to an existing run directory (e.g. /path/to/runs/<run_id>/).",
    )
    parser.add_argument(
        "-o",
        "--overrides",
        nargs="*",
        help="Configuration overrides (e.g., benchmarks.0.every=2 minutes).",
    )
    args = parser.parse_args()
    raise SystemExit(main(args))
