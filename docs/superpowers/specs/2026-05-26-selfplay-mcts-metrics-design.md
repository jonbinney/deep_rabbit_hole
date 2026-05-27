# Self-play MCTS metrics → W&B — design

**Date:** 2026-05-26
**Branch:** `jdb/b9w10-performance`
**Status:** approved (design), pending implementation plan

## Problem

The Rust self-play binary (`deep_quoridor/rust`, driven by `train_v2.py`) runs leaf-parallel
MCTS to generate training games, but exposes almost no visibility into *how* the search is
behaving during a training run. We want diagnostics — terminal-hit rate, tree depth, how
"spread out" the search is, and how many distinct games are being produced — surfaced to
Weights & Biases in the **same W&B group as the other workers** (`train`, `benchmark`,
`ai_report`), and **reset each time the model is updated** so each metric reflects a single
model version.

## Constraint that shapes everything

The Rust self-play runs as a **subprocess** (`subprocess.Popen` in `train_v2.py:112`) with **no
W&B client** — only the Python processes call `wandb.init(group=config.run_id, …)`. There is
currently no self-play W&B run at all. So a Rust→Python bridge is required: Rust writes metric
records to files; a new Python process logs them to W&B. This reuses the existing file-based
Rust→Python channel already used for replay games.

## Architecture & data flow

```
Rust selfplay subprocess(es)              Python (spawned by train_v2.py)
┌──────────────────────────────┐         ┌─────────────────────────────┐
│ per-search SearchStats ──┐    │  JSON   │ run_selfplay_metrics(config)│
│ per-game hashes ─────────┤    │ records │  • polls metrics dir        │
│ SelfPlayMetrics accumulator   │ ──────► │  • aggregates across pids   │
│  (keyed by model_version)     │  files  │  • wandb.log(group=run_id)  │
│  flush+reset on version change│         │    x-axis = "Model version" │
└──────────────────────────────┘         └─────────────────────────────┘
```

Rust accumulates raw metric aggregates for the current `model_version`. The existing
model-reload poll task (`selfplay.rs:679-688`) already detects version changes and is the
natural flush+reset point. On a version change (and on shutdown) Rust writes one JSON record
per `(model_version, pid)` to a metrics directory and clears the accumulator. A new Python
process — spawned by `train_v2.py` like the benchmark processes — polls that directory and logs
each completed version to W&B once.

## Metrics (all per model version; x-axis = `Model version`)

| W&B metric | Definition | Version aggregation |
|---|---|---|
| `selfplay/terminal_sim_frac` | fraction of MCTS simulations whose selected leaf is a **win** terminal | Σ terminal-win sims / Σ sims |
| `selfplay/truncation_sim_frac` | fraction of simulations whose leaf hit the `max_steps` cap (code already distinguishes win vs truncation in `selfplay_mcts.rs:180-197`) | Σ trunc sims / Σ sims |
| `selfplay/max_tree_depth` | deepest selection path (`path.len()` from `select_leaf_with_vl`) | max |
| `selfplay/mean_tree_depth` | mean selection-path depth per sim | Σ depth / Σ sims |
| `selfplay/root_visit_entropy` | entropy (nats) of root child visit distribution | mean over moves |
| `selfplay/root_visit_perplexity` | `exp(entropy)` = effective number of moves considered | mean over moves |
| `selfplay/top_move_visit_frac` | max child visits / total root visits | mean over moves |
| `selfplay/mean_nodes_per_search` | arena node count per search | Σ nodes / moves |
| `selfplay/mean_branching` | tree edges / internal (expanded) nodes | pooled: (Σ nodes − Σ moves) / Σ internal_nodes |
| `selfplay/games_generated` | games completed this version | sum |
| `selfplay/unique_games_full` / `selfplay/unique_frac_full` | distinct full move-sequences; and unique/total | per-process dedup, summed (see Decisions) |
| `selfplay/unique_games_opening` / `selfplay/unique_frac_opening` | distinct first-K-ply prefixes; and unique/total | per-process dedup, summed |

"Simulation" = one leaf selection/backprop iteration of MCTS. "Move" = one root search (one
played move). Entropy/perplexity/top-frac are computed per move from the root child visit
counts, then averaged over the moves in the version.

## Rust side

- **`SearchStats`** — `search()` (`selfplay_mcts.rs:102`) returns this alongside its existing
  `(children, root_value)`. Fields: `sims`, `terminal_wins`, `truncations`, `max_depth`,
  `sum_depth`, `nodes`, `internal_nodes`, `root_visit_entropy`, `top_move_visit_frac`. All are
  derived from data the search already has: per-sim `path` length (depth), the existing
  terminal/truncation branch, the arena node count, and the root `children` visit counts.
- **Per-game folding** — `play_game_async` folds each move's `SearchStats` into a per-game
  total and, at game end, computes a full-game move-sequence hash and a first-K-ply hash.
- **`SelfPlayMetrics` accumulator** — one per process, `Arc<Mutex<…>>`, keyed by
  `model_version`. Holds running raw aggregates: summed counters, running `max_depth`, sums and
  counts for the per-move means, `games_generated`, and two `HashSet<u64>` (full + opening
  hashes) for within-process uniqueness. Game tasks fold their per-game results in under the
  current `model_version`.
- **Flush + reset** — at the version-change point in the poll task, and on shutdown: serialize
  the raw aggregates for the just-finished version to
  `<metrics-dir>/v{version}_pid{pid}.json`, then clear the accumulator. Writing raw aggregates
  (not final metrics) lets Python combine multiple processes correctly.
- **CLI** — new `--metrics-dir <path>` arg. When absent, metric collection is disabled entirely
  (no accumulator, no overhead) so the binary stays usable standalone. `train_v2.py` always
  passes it for Rust self-play.

### JSON record schema (one file per `(version, pid)`)
```json
{
  "model_version": 42, "pid": 700653,
  "sims": 12830000, "terminal_wins": 410000, "truncations": 90000,
  "max_depth": 37, "sum_depth": 41000000,
  "moves": 9800, "sum_root_entropy": 18000.0, "sum_top_move_frac": 6100.0,
  "sum_nodes": 9100000, "sum_internal_nodes": 5200000,
  "games_generated": 96,
  "unique_full": 96, "unique_opening": 71
}
```
(Per-move sums are divided by `moves` on the Python side; `sum_depth` is divided by `sims`.)

## Python side

- **`run_selfplay_metrics(config)`** — a new function/module spawned by `train_v2.py` as an
  `mp.Process`, alongside `create_benchmark_processes`, only when `config.self_play.program ==
  "rust"`. It `wandb.init(project=config.wandb.project, group=config.run_id,
  job_type="selfplay", name=f"{config.run_id}-selfplay", id=f"{config.run_id}-selfplay",
  resume="allow")` and `wandb.define_metric("*", "Model version")` (matching `benchmarks.py`).
- **Poll loop** — every few seconds: scan the metrics dir. A version `V` is **complete** when a
  record file for some version `> V` exists, or `ShutdownSignal` is set. For each complete,
  not-yet-logged `V`: read all `v{V}_pid*.json`, combine (sum the sums/counters, max the maxes,
  sum the unique/total counts), compute final metrics, `wandb.log({…metrics…, "Model version":
  V})`, and record `V` as logged. Exits on `ShutdownSignal`.
- Aggregation lives in a small pure helper (`aggregate_records(list[dict]) -> dict[str,float]`)
  so it is unit-testable without W&B or the filesystem.

## Decisions

- **Opening length `K = 8` plies** (4 moves each).
- **Uniqueness: per-process dedup, summed across processes by Python.** With `num_processes=1`
  (current setup) this is exact. With >1 it can slightly *over*count uniques (a game produced by
  two processes counts twice), which is acceptable for a collapse-detection diagnostic and
  avoids shipping large hash lists in the JSON.
- **No periodic heartbeat** — flush only at version boundaries and shutdown, one definitive
  record per version, logged once. Model updates are frequent (≈ every training step), so the
  in-progress version becoming visible only at the next update is acceptable.
- **Disabled unless `--metrics-dir` is passed**, so standalone/benchmark runs of the binary are
  unaffected and there is zero overhead when off.

## Edge cases

- A version that produces zero games/searches before the next update: skip (don't log an empty
  record, or log with zeros — implementation will log with whatever was accumulated; a version
  with `moves == 0` is skipped to avoid divide-by-zero).
- Shutdown mid-version: the shutdown flush writes the final (partial) version so it isn't lost.
- Multiple processes desynced across versions: Python only finalizes `V` once a `> V` record
  exists from *any* process; late records for an already-logged `V` are ignored (logged once).
- Metrics dir must be created by the Rust side on startup (like `tmp_dir`).

## Testing

- **Rust unit test**: drive `search()` with the existing stub coordinator and assert
  `SearchStats` is sane (`sims == n`, `max_depth ≥ 1`, `top_move_visit_frac ∈ (0,1]`,
  `entropy ≥ 0`). A game-hash test: two identical forced move-sequences hash equal; two
  different ones do not.
- **Python unit test**: feed synthetic per-pid JSON records to `aggregate_records` and assert
  weighted means, max-of-maxes, and summed unique/total counts are correct.

## Out of scope (YAGNI)

- Exact cross-process unique-game dedup (hash union).
- Periodic in-version heartbeat logging.
- Per-game or per-move time-series (only per-version aggregates).
- Logging from the legacy Python self-play path (`v2/self_play.py`) — Rust path only.
