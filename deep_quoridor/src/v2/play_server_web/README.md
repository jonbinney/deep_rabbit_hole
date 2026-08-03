# play_server_web

Thin FastAPI server for the browser Quoridor play app. Serves the built SPA + its
`.wasm` and the trained model `.onnx` files (with cross-origin isolation + wasm
MIME), and exposes a small config/model API. The AI runs client-side (see
`rust/quoridor-wasm`). Design: `docs/superpowers/specs/2026-07-05-browser-wasm-play-server-design.md`.

## Play directory layout

Point the server at a **play directory**:

```
<play-dir>/
  config.yaml          # board size + AlphaZero/MCTS default settings
  models/
    model_a.onnx       # one or more models; all use config.yaml's settings
    model_b.onnx
```

`config.yaml` provides the settings for **every** model in `models/` — the UI just
picks which `.onnx` to run. (`config.yaml` is the same schema `v2.config` uses, so
you can copy one from a training run.)

## Run

```
pip install fastapi uvicorn
PYTHONPATH=deep_quoridor/src python deep_quoridor/src/run_play_server_web.py \
    /path/to/play-dir --static-dir /path/to/spa/dist --port 8080
```
- `play_dir` (positional): a directory with `config.yaml` and a `models/` dir of `.onnx` files.
- `--static-dir`: the built SPA. Omit to serve a placeholder page.
- `--models-dir`: override where `.onnx` files are read from (default `<play_dir>/models`).
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
