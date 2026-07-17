"""Serve the Quoridor browser play app + model files + config API.

Read-only over a play directory: a directory containing `config.yaml` and a
`models/` subdirectory of `.onnx` files. Every model uses the settings in
`config.yaml`."""

import argparse
from pathlib import Path

import uvicorn

from v2.play_server_web.app import create_app


def main(args) -> int:
    play_dir = Path(args.play_dir).resolve()
    if not (play_dir / "config.yaml").is_file():
        print(f"error: {play_dir / 'config.yaml'} not found")
        return 1
    static_dir = Path(args.static_dir).resolve() if args.static_dir else None
    models_dir = Path(args.models_dir).resolve() if args.models_dir else None
    app = create_app(play_dir, static_dir=static_dir, models_dir=models_dir)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Serve the Quoridor browser play app + model files + config API."
    )
    parser.add_argument(
        "play_dir",
        type=str,
        help="Directory containing config.yaml and a models/ dir of .onnx files.",
    )
    parser.add_argument(
        "--static-dir", type=str, default=None, help="Directory of the built SPA (frontend build output)."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Override the .onnx models dir (default: <play_dir>/models).",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    raise SystemExit(main(args))
