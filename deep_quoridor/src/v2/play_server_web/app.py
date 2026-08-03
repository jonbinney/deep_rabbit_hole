"""FastAPI app factory for the Quoridor browser play server.

Serves the built SPA + its .wasm and the trained model .onnx files, with
cross-origin isolation + correct wasm MIME, and exposes /api/config and
/api/models. Owns no AI code — the AI runs in the browser (quoridor-wasm)."""

import mimetypes
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from v2.config import load_user_config
from v2.play_server_web.config_view import build_config_view
from v2.play_server_web.model_listing import default_model, list_onnx_models

# Python's mimetypes doesn't always know .wasm; register it so StaticFiles serves
# wasm with the type browsers require for streaming compilation.
mimetypes.add_type("application/wasm", ".wasm")

_PLACEHOLDER = (
    "<!doctype html><title>Quoridor</title>"
    "<p>SPA build not found. Build the frontend and start the server "
    "with --static-dir pointing at the build output.</p>"
)


def create_app(
    play_dir: Path,
    static_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
) -> FastAPI:
    """`play_dir` is a directory containing `config.yaml` and a `models/`
    subdirectory of `.onnx` files. Every model uses the settings in
    `config.yaml` (served via /api/config); the models are just a pickable list.
    """
    play_dir = Path(play_dir)
    config_file = play_dir / "config.yaml"
    models_dir = Path(models_dir) if models_dir is not None else play_dir / "models"

    app = FastAPI(title="Quoridor play server")

    @app.middleware("http")
    async def add_cross_origin_isolation(request, call_next):
        # COOP+COEP enable SharedArrayBuffer (ORT's multi-threaded WASM-CPU
        # fallback). Harmless for the WebGPU path.
        response = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        return response

    @app.get("/api/config")
    def api_config():
        cfg = load_user_config(str(config_file))
        return build_config_view(cfg)

    @app.get("/api/models")
    def api_models():
        models = list_onnx_models(models_dir)
        return {"models": models, "default": default_model(models)}

    # Model files fetched by onnxruntime-web (same-origin).
    if models_dir.is_dir():
        app.mount("/models", StaticFiles(directory=str(models_dir)), name="models")

    # SPA catch-all, mounted LAST so /api and /models win. Placeholder until Plan 3.
    if static_dir is not None and Path(static_dir).is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="spa")
    else:

        @app.get("/", response_class=HTMLResponse)
        def placeholder():
            return _PLACEHOLDER

    return app
