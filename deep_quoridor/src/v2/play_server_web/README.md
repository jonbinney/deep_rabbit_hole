# play_server_web

Thin FastAPI server for the browser Quoridor play app. Serves the built SPA + its
`.wasm` and the trained model `.onnx` files (with cross-origin isolation + wasm
MIME), and exposes a small config/model API. The AI runs client-side (see
`rust/quoridor-wasm`). Design: `docs/superpowers/specs/2026-07-05-browser-wasm-play-server-design.md`.

## Run

```
pip install fastapi uvicorn
PYTHONPATH=deep_quoridor/src python deep_quoridor/src/run_play_server_web.py \
    /path/to/runs/<run_id> --static-dir /path/to/spa/dist --port 8080
```
- `run_dir` (positional): an existing run directory (`config.yaml` + `models/checkpoints/`).
- `--static-dir`: the built SPA (Plan 3). Omit to serve a placeholder page.
- `--models-dir`: override where `.onnx` files are read from (default `<run_dir>/models/checkpoints`).
- `--host` / `--port`: bind address (default `127.0.0.1:8080`).

## API
- `GET /api/config` → `{ board_size, max_walls, max_steps, defaults: { mcts_n, mcts_c_puct, temperature, mcts_noise_epsilon, mcts_noise_alpha, leaf_parallelism, virtual_loss, mcts_worker_threads } }`
- `GET /api/models` → `{ models: ["model_1.onnx", ...], default: "model_N.onnx" }`
- `GET /models/<file>.onnx` → the model file (for onnxruntime-web to fetch).
- `GET /` and other paths → the SPA (or a placeholder until Plan 3 is built).

All responses carry `Cross-Origin-Opener-Policy: same-origin` +
`Cross-Origin-Embedder-Policy: require-corp`. **Plan 3 must bundle onnxruntime-web
same-origin** — COEP `require-corp` blocks non-CORP cross-origin resources (so a
CDN `<script>` for ORT would be blocked unless served with the right headers).

## Tests
```
PYTHONPATH=deep_quoridor/src python -m pytest deep_quoridor/test/test_play_server_web.py -v
```
