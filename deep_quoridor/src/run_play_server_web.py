"""Serve the Quoridor browser play app + model files + config API.

Read-only over an existing run directory (its config.yaml + models/checkpoints/).
Mirrors run_benchmarks_v2.py's CLI shape."""

import argparse
from pathlib import Path

import uvicorn

from v2.play_server_web.app import create_app


def main(args) -> int:
    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "config.yaml").is_file():
        print(f"error: {run_dir / 'config.yaml'} not found")
        return 1
    static_dir = Path(args.static_dir).resolve() if args.static_dir else None
    models_dir = Path(args.models_dir).resolve() if args.models_dir else None
    app = create_app(run_dir, static_dir=static_dir, models_dir=models_dir)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Serve the Quoridor browser play app + model files + config API."
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to a run directory containing config.yaml and models/checkpoints/.",
    )
    parser.add_argument(
        "--static-dir", type=str, default=None, help="Directory of the built SPA (Plan 3 output)."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Override the .onnx models dir (default: <run_dir>/models/checkpoints).",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    raise SystemExit(main(args))
