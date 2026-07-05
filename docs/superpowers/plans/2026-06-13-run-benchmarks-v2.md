# `run_benchmarks_v2.py` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

I'm using AGENTS.md

**Goal:** A new script `deep_quoridor/src/run_benchmarks_v2.py` that takes an existing run directory and spawns just the benchmark processes described in that run's saved `config.yaml`, looping like `train_v2.py` until Ctrl-C.

**Architecture:** Reuse `benchmarks.create_benchmark_processes(config)` unchanged. The script loads the saved `config.yaml` via the existing `load_user_config(...)`, builds a `Config` with `Config.from_user(..., create_dirs=False)` so existing dirs are untouched, runs three startup checks (run_dir, config.yaml, models/latest.yaml all exist), then spawns the benchmark processes and waits with the same shutdown-loop pattern train_v2 uses. Ctrl-C signals `ShutdownSignal` and joins.

**Tech Stack:** Python 3.12, pytest, pydantic, multiprocessing.

**Spec:** `docs/superpowers/specs/2026-06-11-run-benchmarks-v2-design.md`

---

## File structure

- `deep_quoridor/src/run_benchmarks_v2.py` — new script. (create)
- `deep_quoridor/test/test_run_benchmarks_v2.py` — unit tests for path-derivation, config-loading, and startup checks. (create)

**Run Python tests with:**
```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_run_benchmarks_v2.py -v
```

**Commit style (AGENTS.md):** `vibe: ` imperative subject ≤ 50 chars. End commit messages with:
```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

**Branch state:** This continues on `jdb/train-on-existing-selfplay-games`. The spec is committed at `557f22e`.

---

## Task 1: Path-derivation + config-load helpers

**Files:**
- Create: `deep_quoridor/src/run_benchmarks_v2.py`
- Create: `deep_quoridor/test/test_run_benchmarks_v2.py`

- [ ] **Step 1: Write the failing tests**

Create `deep_quoridor/test/test_run_benchmarks_v2.py` with these tests:

```python
from pathlib import Path

import pytest
import yaml

from run_benchmarks_v2 import _derive_base_dir, _load_config


EXAMPLE_CONFIG = {
    "run_id": "test-run",
    "quoridor": {"board_size": 5, "max_walls": 3, "max_steps": 50},
    "alphazero": {"network": {"type": "mlp"}, "mcts_n": 300, "mcts_c_puct": 1.2},
    "self_play": {"num_processes": 2, "games_per_process": 16, "alphazero": {"mcts_noise_epsilon": 0.25}},
    "training": {
        "games_per_training_step": 25.0,
        "learning_rate": 0.001,
        "batch_size": 256,
        "weight_decay": 0.0001,
        "replay_buffer_size": 1000000,
    },
    "benchmarks": [
        {
            "every": "10 models",
            "jobs": [
                {"type": "tournament", "prefix": "raw", "times": 10, "opponents": ["random", "greedy"]},
            ],
        },
    ],
}


def _make_run_dir(tmp_path: Path, run_id: str = "test-run") -> Path:
    """Create a runs/<run_id>/ structure with a valid config.yaml inside."""
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    cfg = dict(EXAMPLE_CONFIG)
    cfg["run_id"] = run_id
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return run_dir


def test_derive_base_dir_uses_grandparent(tmp_path):
    run_dir = tmp_path / "runs" / "my-run"
    assert _derive_base_dir(run_dir) == str(tmp_path)


def test_load_config_returns_full_config(tmp_path):
    run_dir = _make_run_dir(tmp_path, run_id="my-run")
    config = _load_config(run_dir, overrides=None)
    assert config.run_id == "my-run"
    assert config.training.learning_rate == 0.001
    assert len(config.benchmarks) == 1
    # paths derived from the run dir
    assert config.paths.run_dir == run_dir


def test_load_config_applies_overrides(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    config = _load_config(run_dir, overrides=["training.learning_rate=0.05"])
    assert config.training.learning_rate == 0.05


def test_load_config_does_not_create_dirs(tmp_path):
    """Config.from_user(..., create_dirs=False) — no replay_buffers/, etc. spawned."""
    run_dir = _make_run_dir(tmp_path)
    _load_config(run_dir, overrides=None)
    assert not (run_dir / "replay_buffers").exists()
    assert not (run_dir / "models").exists()


def test_load_config_raises_when_config_yaml_missing(tmp_path):
    run_dir = tmp_path / "runs" / "my-run"
    run_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        _load_config(run_dir, overrides=None)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_run_benchmarks_v2.py -v
```

Expected: `ModuleNotFoundError: No module named 'run_benchmarks_v2'`.

- [ ] **Step 3: Create the script with the two helpers**

Create `deep_quoridor/src/run_benchmarks_v2.py`:

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_run_benchmarks_v2.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/run_benchmarks_v2.py deep_quoridor/test/test_run_benchmarks_v2.py
git commit -m "vibe: add run_benchmarks_v2 config-load helpers"
```

End with the Co-Authored-By trailer.

---

## Task 2: Startup checks + `main()` with empty-benchmarks early exit

**Files:**
- Modify: `deep_quoridor/src/run_benchmarks_v2.py`
- Modify: `deep_quoridor/test/test_run_benchmarks_v2.py`

- [ ] **Step 1: Write the failing tests**

First, update the imports at the top of `deep_quoridor/test/test_run_benchmarks_v2.py`:

- Add `from argparse import Namespace` near the other top-level imports.
- Extend the existing `from run_benchmarks_v2 import _derive_base_dir, _load_config` line to:
  ```python
  from run_benchmarks_v2 import _check_run_dir, _derive_base_dir, _load_config, main
  ```

Then append these test functions to the end of the file:

```python
def test_check_run_dir_raises_when_run_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run directory not found"):
        _check_run_dir(tmp_path / "does-not-exist")


def test_check_run_dir_raises_when_config_yaml_missing(tmp_path):
    run_dir = tmp_path / "runs" / "my-run"
    run_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        _check_run_dir(run_dir)


def test_check_run_dir_raises_when_latest_yaml_missing(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    with pytest.raises(FileNotFoundError, match="models/latest.yaml"):
        _check_run_dir(run_dir)


def test_check_run_dir_passes_when_all_present(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "models").mkdir()
    (run_dir / "models" / "latest.yaml").write_text("filename: /tmp/m.pt\nversion: 0\n")
    _check_run_dir(run_dir)  # no exception


def test_main_exits_zero_when_no_benchmarks(tmp_path, capsys):
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "models").mkdir()
    (run_dir / "models" / "latest.yaml").write_text("filename: /tmp/m.pt\nversion: 0\n")

    # Strip the benchmarks section from the config.yaml.
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    cfg["benchmarks"] = []
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    args = Namespace(run_dir=str(run_dir), overrides=None)
    exit_code = main(args)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "No benchmarks configured" in captured.out
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_run_benchmarks_v2.py -v
```

Expected: 5 new tests fail with `ImportError` on `_check_run_dir` and `main`.

- [ ] **Step 3: Add the startup-check helper and a minimal `main()`**

Append to `deep_quoridor/src/run_benchmarks_v2.py`:

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_run_benchmarks_v2.py -v
```

Expected: all 10 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/run_benchmarks_v2.py deep_quoridor/test/test_run_benchmarks_v2.py
git commit -m "vibe: add run_benchmarks_v2 startup checks + main"
```

End with the Co-Authored-By trailer.

---

## Task 3: Spawn benchmark processes + Ctrl-C shutdown + smoke test

**Files:**
- Modify: `deep_quoridor/src/run_benchmarks_v2.py`

No unit tests for this task — `mp.Process` spawning is covered end-to-end by the smoke test in Step 5.

- [ ] **Step 1: Add the spawn-and-wait body to `main()` and the `__main__` block**

Replace the entire contents of `deep_quoridor/src/run_benchmarks_v2.py` with:

```python
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
```

- [ ] **Step 2: Sanity-check the script imports + `--help`**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/python -c "import run_benchmarks_v2; print('ok')"
```

Expected: prints `ok`.

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/python deep_quoridor/src/run_benchmarks_v2.py --help 2>&1 | head -20
```

Expected: usage text showing `run_dir` positional and `-o/--overrides` option.

- [ ] **Step 3: Re-run the unit tests to confirm they still pass**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/pytest deep_quoridor/test/test_run_benchmarks_v2.py -v
```

Expected: all 10 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /home/jbinney/ws/deep_rabbit_hole
git add deep_quoridor/src/run_benchmarks_v2.py
git commit -m "vibe: spawn benchmark processes in run_benchmarks_v2"
```

End with the Co-Authored-By trailer.

- [ ] **Step 5: Smoke test against a fake run directory**

This step builds a tiny run dir with a valid `config.yaml` and `models/latest.yaml`, points the script at it, and verifies the script gets through preflight + spawn before being interrupted. The benchmark processes will themselves fail quickly (no real model file to load), which is fine for this smoke check — we're verifying entrypoint behavior, not benchmark execution.

The script writes to `/tmp/smoke_*` which is outside the sandbox; use `dangerouslyDisableSandbox: true` for these commands.

Prepare:

```bash
cd /home/jbinney/ws/deep_rabbit_hole
PYTHONPATH=deep_quoridor/src .venv/bin/python - <<'PY'
from pathlib import Path
import yaml
from pydantic_yaml import to_yaml_file
from v2.yaml_models import LatestModel

base = Path("/tmp/smoke_bench")
run_dir = base / "runs" / "smoke-bench-test"
(run_dir / "models").mkdir(parents=True, exist_ok=True)

cfg = {
    "run_id": "smoke-bench-test",
    "quoridor": {"board_size": 5, "max_walls": 3, "max_steps": 50},
    "alphazero": {"network": {"type": "mlp"}, "mcts_n": 25, "mcts_c_puct": 1.2},
    "self_play": {"num_processes": 1, "games_per_process": 4, "alphazero": {"mcts_noise_epsilon": 0.25}},
    "training": {
        "games_per_training_step": 1.0,
        "learning_rate": 0.001,
        "batch_size": 32,
        "weight_decay": 0.0001,
        "replay_buffer_size": 30,
    },
    "benchmarks": [
        {
            "every": "10 models",
            "jobs": [
                {"type": "dumb_score", "prefix": "raw"},
            ],
        },
    ],
}
(run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
to_yaml_file(
    run_dir / "models" / "latest.yaml",
    LatestModel(filename=str(run_dir / "models" / "model_0.pt"), version=0),
)
print(f"prepared {run_dir}")
PY
```

Run for ~5 seconds then Ctrl-C (use `timeout 5 ... || true` so the non-zero exit from the timeout doesn't fail the shell):

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src timeout 5 .venv/bin/python deep_quoridor/src/run_benchmarks_v2.py /tmp/smoke_bench/runs/smoke-bench-test 2>&1 | tee /tmp/smoke_bench.log || true
```

Verify expected output in the log:

```bash
grep -E "Started [0-9]+ benchmark processes|Waiting for|No benchmarks" /tmp/smoke_bench.log
```

Expected:
- `Started 1 benchmark processes` line present.
- At least one `Waiting for N benchmark processes` transition (the benchmark child process may have already crashed because `model_0.pt` doesn't exist on disk — that's expected; what matters is the script reached the spawn-and-wait stage).

Try the empty-benchmarks early-exit path against the same run:

```bash
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/python deep_quoridor/src/run_benchmarks_v2.py /tmp/smoke_bench/runs/smoke-bench-test -o benchmarks=[] 2>&1
```

Expected: prints `No benchmarks configured in /tmp/smoke_bench/runs/smoke-bench-test/config.yaml; nothing to run.` and exits 0.

Try the missing-latest.yaml abort path:

```bash
cd /home/jbinney/ws/deep_rabbit_hole && rm /tmp/smoke_bench/runs/smoke-bench-test/models/latest.yaml
cd /home/jbinney/ws/deep_rabbit_hole && PYTHONPATH=deep_quoridor/src .venv/bin/python deep_quoridor/src/run_benchmarks_v2.py /tmp/smoke_bench/runs/smoke-bench-test 2>&1; echo "exit=$?"
```

Expected: non-zero exit with `FileNotFoundError: No models/latest.yaml in ...`.

- [ ] **Step 6: Cleanup**

```bash
rm -rf /tmp/smoke_bench /tmp/smoke_bench.log
```

- [ ] **Step 7: Format pass**

```bash
cd /home/jbinney/ws/deep_rabbit_hole && .venv/bin/ruff format deep_quoridor/src/run_benchmarks_v2.py deep_quoridor/test/test_run_benchmarks_v2.py
cd /home/jbinney/ws/deep_rabbit_hole && git status --short
```

If anything changed, commit separately per AGENTS.md:

```bash
cd /home/jbinney/ws/deep_rabbit_hole && git add -u && git commit -m "vibe: ruff format"
```

End with the Co-Authored-By trailer. If nothing changed, skip.

---

## Done criteria

- All 10 unit tests in `test_run_benchmarks_v2.py` pass.
- `python run_benchmarks_v2.py --help` shows `run_dir` positional and `-o/--overrides` option.
- Smoke run prints `Started N benchmark processes` against a fake run dir; Ctrl-C / timeout doesn't crash the script before the spawn-and-wait loop is reached.
- Missing-latest.yaml path aborts with a clear error.
- Empty-benchmarks path prints the expected line and exits 0.
- Touched files clean under `ruff format`.
