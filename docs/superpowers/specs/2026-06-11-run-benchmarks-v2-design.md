# `run_benchmarks_v2.py` — design

**Date:** 2026-06-11
**Branch:** `jdb/train-on-existing-selfplay-games`
**Status:** approved (design), pending implementation plan

## Problem

`train_v2.py` orchestrates four kinds of work — training, self-play, benchmarks, AI report —
and there is no way to run only the benchmarks against an existing run. When training is done
or when iterating on benchmark configuration, the user wants to point at a run dir and start
just the benchmark schedules described in that run's `config.yaml`.

## Approach

A new script `deep_quoridor/src/run_benchmarks_v2.py`, parallel to `train_v2.py`. It takes a
run directory as its positional argument, loads the saved `config.yaml`, and spawns the same
`mp.Process` set that `train_v2.py` would spawn for benchmarks — using the existing
`benchmarks.create_benchmark_processes(config)` function unchanged. No changes to
`benchmarks.py`, the config schema, or any other module.

The script loops like `train_v2.py` does (the user picked "loop like train_v2" during
brainstorming): benchmark processes keep running on their `every:` triggers until
`ShutdownSignal` is set. Termination is via Ctrl-C — the script catches `KeyboardInterrupt`,
signals shutdown, and joins.

## CLI

```bash
python deep_quoridor/src/run_benchmarks_v2.py <run_dir> [-o key=val ...]
```

Positional argument:
- `<run_dir>` — path to an existing run directory, e.g. `/path/to/runs/<run_id>/`.

Options:
- `-o, --overrides key=val [key=val ...]` — same shape as `train_v2.py --overrides`. Applied
  to the loaded `UserConfig` before `Config.from_user`. Useful for adjusting benchmark
  frequency (`benchmarks.0.every=2 minutes`) or trying a different opponent list
  (`benchmarks.0.jobs.0.opponents=[random,greedy]`) without editing the saved yaml.

## Path & config loading

The standard run-dir layout is `base_dir/runs/<run_id>/`. Given `<run_dir>` the script
derives:

```python
run_dir = Path(args.run_dir).resolve()
base_dir = str(run_dir.parent.parent)   # the parent of "runs/"
# run_id comes from the saved config.yaml, not the directory name, so $DATETIME
# substitution doesn't cause a mismatch.
```

Loading uses the existing public `load_user_config(file, overrides=None)` which loads the
yaml and applies overrides in one call, then `Config.from_user`:

```python
config_yaml = run_dir / "config.yaml"
if not config_yaml.is_file():
    raise FileNotFoundError(f"No config.yaml in {run_dir}")

user_config = load_user_config(str(config_yaml), overrides=args.overrides)
config = Config.from_user(user_config, base_dir, create_dirs=False)
```

`create_dirs=False` so existing directories aren't disturbed and no `config.yaml` snapshot is
rewritten. `load_user_config` is already exported from `v2/config.py`; no schema or helper
visibility changes are needed.

We pull `run_id` from the loaded `UserConfig`, not from `run_dir.name`. `UserConfig`'s
`replace_datetime_placeholder` validator interpolates `$DATETIME` at load time, so the
in-memory `run_id` matches what was written to disk in the original run — which is also what
the run_dir is named.

## Startup checks

Before spawning any process:

1. `<run_dir>` must exist and be a directory.
2. `<run_dir>/config.yaml` must exist.
3. `<run_dir>/models/latest.yaml` must exist. Without it, `run_benchmark`'s
   `LatestModel.wait_for_creation` would block forever (no training is producing models in
   this script). Abort with a clear message naming the missing file.
4. `config.benchmarks` must be non-empty. If empty, log
   `"No benchmarks configured in <run_dir>/config.yaml; nothing to run."` and exit 0.
5. `ShutdownSignal.clear(config)` — a previous run may have left the signal set, which would
   make `run_benchmark`'s loop exit on its first iteration. Clear it before spawning.

## Orchestration

```python
mp.set_start_method("spawn", force=True)
ShutdownSignal.clear(config)

benchmark_processes = benchmarks.create_benchmark_processes(config)
[p.start() for p in benchmark_processes]
print(f"Started {len(benchmark_processes)} benchmark processes")

try:
    # Same shutdown-wait pattern train_v2.py uses, simplified to benchmarks only.
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
```

The benchmark loop in `run_benchmark` exits when `ShutdownSignal.is_set(config)` returns
True (inside `freq.wait(...)`). Ctrl-C triggers the signal; processes finish their current
iteration and exit cleanly.

## wandb behavior

Each benchmark process calls `wandb.init(id=f"{config.run_id}-benchmark-{idx}",
resume="allow", ...)`. Re-running this script appends to the existing benchmark wandb runs —
the right behavior for "I ran some benchmarks during training, now I'm running more after
training finished."

No new wandb config or behavior. Inherits whatever the saved `config.yaml` says.

## Failure modes

- `<run_dir>` not a directory → `FileNotFoundError` or our explicit check, abort.
- `config.yaml` missing → explicit check at startup.
- `latest.yaml` missing → explicit check at startup (avoids the blocking
  `wait_for_creation`).
- pydantic validation fails (e.g. a `-o` override is invalid) → propagate the validation
  error.
- `config.benchmarks` empty → log and exit 0.
- `<run_dir>` doesn't match the `base_dir/runs/<run_id>/` shape → the derived `base_dir` is
  still a valid path; `Config.from_user` succeeds because `create_dirs=False`. The benchmark
  processes still find `latest.yaml` via the absolute `paths.latest_model_yaml`. No special
  guard needed — non-standard layouts just work as long as `<run_dir>/models/latest.yaml`
  resolves.

## Testing

**Unit test (small):**

- A script-level entrypoint test that calls a refactored `main(args)` function with a
  mocked `mp.Process` (or by stubbing `benchmarks.create_benchmark_processes` to return
  `[]`), confirming that when `config.benchmarks` is empty the script exits 0 with the
  expected "nothing to run" log line.

**Manual smoke verification:**

- Take an existing run dir (from yesterday's smoke tests or any past training run that
  produced a `latest.yaml`). Invoke `run_benchmarks_v2.py <run_dir>`. Confirm:
  - Process starts without error.
  - It prints `"Started N benchmark processes"`.
  - At least one benchmark process logs its first iteration ("Running TournamentBenchmarkJob
    …" etc.).
  - Ctrl-C cleanly shuts everything down.

The script orchestration itself is otherwise covered by the existing `benchmarks.py` tests
(if any) and by train_v2's existing flow — we're not changing benchmark execution behavior,
only the entrypoint that triggers it.

## Out of scope (YAGNI)

- A `--once` flag that runs each benchmark schedule exactly once and exits. Skipped per the
  brainstorming decision.
- Running a subset of benchmarks (e.g. `--benchmark-index 1`). All configured benchmarks
  run.
- Changing the `wandb` group/run-id naming for benchmark-only invocations. We deliberately
  reuse the same ids so the rerun appends to the original benchmark run.
- A general-purpose "run any subset of train_v2 stages" mode. This script is
  benchmarks-only.
