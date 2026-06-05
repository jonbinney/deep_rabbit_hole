"""Self-play MCTS metrics: aggregate per-process Rust JSON records and log to W&B.

The Rust self-play binary writes one raw-aggregate record per (model_version, pid)
to a metrics directory. This module combines a version's records across processes,
computes the final metrics, and logs them to a W&B run in the training group.
"""
import glob
import json
import math
import os
import re
import time
from typing import Optional

import wandb

from v2.common import MockWandb, ShutdownSignal
from v2.config import Config

_FILE_RE = re.compile(r"v(\d+)_pid\d+\.json$")


def metrics_dir_for(config: Config) -> str:
    """Directory shared by the Rust writer and this reader."""
    return str(config.paths.run_dir / "selfplay_metrics")


def aggregate_records(records: list[dict]) -> Optional[dict]:
    """Combine per-process raw records for one model version into final metrics.

    Returns None if no searches happened (nothing to log).
    """
    sims = sum(r["sims"] for r in records)
    moves = sum(r["moves"] for r in records)
    if moves == 0 or sims == 0:
        return None

    sum_entropy = sum(r["sum_root_entropy"] for r in records)
    sum_nodes = sum(r["sum_nodes"] for r in records)
    sum_internal = sum(r["sum_internal_nodes"] for r in records)
    games = sum(r["games_generated"] for r in records)
    unique_full = sum(r["unique_full"] for r in records)
    unique_opening = sum(r["unique_opening"] for r in records)
    mean_entropy = sum_entropy / moves

    out = {
        "selfplay/terminal_sim_frac": sum(r["terminal_wins"] for r in records) / sims,
        "selfplay/truncation_sim_frac": sum(r["truncations"] for r in records) / sims,
        "selfplay/max_tree_depth": max(r["max_depth"] for r in records),
        "selfplay/mean_tree_depth": sum(r["sum_depth"] for r in records) / sims,
        "selfplay/root_visit_entropy": mean_entropy,
        "selfplay/root_visit_perplexity": math.exp(mean_entropy),
        "selfplay/top_move_visit_frac": sum(r["sum_top_move_frac"] for r in records) / moves,
        "selfplay/mean_nodes_per_search": sum_nodes / moves,
        "selfplay/mean_branching": (sum_nodes - moves) / sum_internal if sum_internal else 0.0,
        "selfplay/games_generated": games,
        "selfplay/unique_games_full": unique_full,
        "selfplay/unique_games_opening": unique_opening,
        "selfplay/unique_frac_full": unique_full / games if games else 0.0,
        "selfplay/unique_frac_opening": unique_opening / games if games else 0.0,
    }
    return out


def _scan(metrics_dir: str) -> dict[int, list[str]]:
    """Map model_version -> list of record file paths present on disk."""
    by_version: dict[int, list[str]] = {}
    for path in glob.glob(os.path.join(metrics_dir, "v*_pid*.json")):
        m = _FILE_RE.search(os.path.basename(path))
        if m:
            by_version.setdefault(int(m.group(1)), []).append(path)
    return by_version


def _load(paths: list[str]) -> list[dict]:
    records = []
    for p in paths:
        try:
            with open(p) as f:
                records.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue  # mid-write or transient; picked up on a later poll
    return records


def run_selfplay_metrics(config: Config, poll_seconds: float = 5.0):
    """Poll the metrics dir and log each completed model version to W&B once."""
    metrics_dir = metrics_dir_for(config)
    os.makedirs(metrics_dir, exist_ok=True)

    if config.wandb:
        run_id = f"{config.run_id}-selfplay"
        wandb_run = wandb.init(
            project=config.wandb.project,
            job_type="selfplay",
            group=config.run_id,
            name=run_id,
            id=run_id,
            resume="allow",
        )
        wandb.define_metric("Model version", hidden=True)
        wandb.define_metric("*", "Model version")
    else:
        wandb_run = MockWandb()

    logged: set[int] = set()

    def flush(finalize_all: bool):
        by_version = _scan(metrics_dir)
        if not by_version:
            return
        max_version = max(by_version)
        for version in sorted(by_version):
            if version in logged:
                continue
            # A version is complete once a newer version exists (the writer moved on)
            # or we're finalizing on shutdown.
            if not finalize_all and version >= max_version:
                continue
            agg = aggregate_records(_load(by_version[version]))
            if agg is not None:
                agg["Model version"] = version
                wandb_run.log(agg)
            logged.add(version)

    while not ShutdownSignal.is_set(config):
        flush(finalize_all=False)
        time.sleep(poll_seconds)

    flush(finalize_all=True)  # log the last in-progress version on shutdown
