"""Generate an AI training report on demand for a wandb run group.

Useful for:
  - Generating reports on older / completed runs.
  - Iterating on the prompt without having to spin up training.
  - Asking the AI specific questions via ``--guidance``.

The report is printed to stdout. Nothing is saved to disk and nothing is
uploaded to wandb. Pipe to a file if you want to keep a copy:

    python deep_quoridor/src/ai_report_cli.py my-run-20260415-1340 \\
        --project B5W3 > report.md

A logical ``<run_id>`` (same name as the local ``runs/<run_id>/`` directory)
maps to a wandb group containing the training, benchmark, and ai-report runs.
"""

from __future__ import annotations

import argparse
import sys

from v2.ai_report import SUPPORTED_AIS, generate_on_demand_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an AI training report on demand for a wandb run group.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_id",
        help="Logical run_id (the wandb group name — same as the local runs/<id>/ dir).",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="wandb project containing the run.",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="wandb entity (user or team). Optional; defaults to your wandb default.",
    )
    parser.add_argument(
        "--ai",
        default="claude",
        choices=list(SUPPORTED_AIS),
        help="AI backend to use (default: claude).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model identifier passed to the AI backend. For Claude: 'sonnet', "
            "'opus', 'haiku', or a full model ID. Defaults to the backend's default."
        ),
    )
    parser.add_argument(
        "-g",
        "--guidance",
        default=None,
        help=(
            "Extra text appended to the prompt — use it to ask a specific question, "
            "focus on a particular metric, or steer the analysis."
        ),
    )
    args = parser.parse_args()

    print(
        f"Fetching wandb metrics for group '{args.run_id}' "
        f"from project '{args.project}'...",
        file=sys.stderr,
    )
    model_label = args.model or "default"
    print(f"Running {args.ai} (model={model_label})...", file=sys.stderr)

    try:
        text = generate_on_demand_report(
            project=args.project,
            group=args.run_id,
            ai=args.ai,
            entity=args.entity,
            guidance=args.guidance,
            model=args.model,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
