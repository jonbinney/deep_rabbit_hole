"""Dump wandb metrics for a logical run group to a single JSON file.

A deep_quoridor training run corresponds to a wandb *group* (``config.run_id``)
containing one ``<run_id>-training`` run plus one ``<run_id>-benchmark-<idx>``
run per benchmark schedule, plus ``<run_id>-ai-report`` if AI reports are
enabled. The AI reporter calls :func:`dump_group_metrics` to snapshot the
whole group so the AI prompt can reason over raw metrics without having to
write wandb API code itself.

Patterns here mirror ``deep_quoridor/src/metrics_reader.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import wandb


def _summary_dict(run) -> dict:
    """Return ``run.summary`` as a plain dict (defensive against SDK variants)."""
    summary = run.summary
    raw = getattr(summary, "_json_dict", None)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    try:
        return dict(summary)
    except Exception:
        return {}


def _runs_path(project: str, entity: str | None) -> str:
    return f"{entity}/{project}" if entity else project


def _clean_summary(summary: dict) -> dict:
    """Strip wandb-internal and bookkeeping keys from a summary dict."""
    return {
        k: v
        for k, v in summary.items()
        if not k.startswith("_") and not k.startswith("time-")
    }


def dump_group_metrics(
    project: str,
    group: str,
    out_path: Path,
    entity: str | None = None,
    history_samples: int = 500,
) -> Path:
    """Snapshot every wandb run in ``group`` and write JSON to ``out_path``.

    The output JSON has the shape::

        {
          "project": "...",
          "group": "...",
          "runs": {
             "<wandb run name>": {
                "state": "...",
                "summary": { metric_name: latest_value, ... },
                "history": [ {x_axis: v, metric: v, ...}, ... ],  # downsampled
             },
             ...
          }
        }

    History is downsampled server-side to ``history_samples`` rows per run, so
    memory is bounded regardless of run length.
    """
    api = wandb.Api()
    runs = list(api.runs(path=_runs_path(project, entity), filters={"group": group}))

    payload: dict[str, Any] = {
        "project": project,
        "group": group,
        "runs": {},
    }

    for run in runs:
        summary = _clean_summary(_summary_dict(run))
        # Fetch all metric keys present in summary (one history call covers them all).
        metric_keys = [k for k in summary.keys() if not k.startswith("time-")]

        history: list[dict] = []
        try:
            # run.history with pandas=False returns a list of dicts.
            history = list(
                run.history(keys=metric_keys, samples=history_samples, pandas=False)
            )
        except Exception as exc:
            history = []
            print(f"[wandb_metrics] failed to fetch history for {run.name}: {exc}")

        # run.config is wandb's native hyperparameter store — equivalent to config.yaml
        # values. Including it makes the snapshot self-contained for old runs where
        # the local config.yaml may no longer be around.
        try:
            run_config = dict(run.config)
        except Exception:
            run_config = {}

        payload["runs"][run.name] = {
            "state": getattr(run, "state", None),
            "job_type": getattr(run, "job_type", None),
            "config": run_config,
            "summary": summary,
            "history": history,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path
