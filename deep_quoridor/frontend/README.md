# Quoridor frontend (Svelte + Web Worker + onnxruntime-web)

The browser play app. A Web Worker runs the `quoridor-wasm` MCTS and evaluates the
net with onnxruntime-web (WebGPU, wasm-CPU fallback); the main thread renders the
board and streams the AI's "thinking" progress. Served by the Python play server
(`src/run_play_server_web.py`). Design: `docs/superpowers/specs/2026-07-05-browser-wasm-play-server-design.md`.

## Build

```
# 1. Build the wasm package (Plan 1):
wasm-pack build deep_quoridor/rust/quoridor-wasm --target web --release
# 2. Build the frontend:
npm --prefix deep_quoridor/frontend install
npm --prefix deep_quoridor/frontend run build   # -> deep_quoridor/frontend/dist/
```

## Run (play a game)

```
# Point the server at a run directory (config.yaml + models/checkpoints/*.onnx)
# and the built SPA:
PYTHONPATH=deep_quoridor/src python deep_quoridor/src/run_play_server_web.py \
    /path/to/runs/<run_id> --static-dir deep_quoridor/frontend/dist --port 8080
# open http://localhost:8080
```

Quick try with the bundled fixture model (5×5, 2 walls):
```
RUN=$(mktemp -d)
printf 'run_id: play\nquoridor:\n  board_size: 5\n  max_walls: 2\n  max_steps: 50\nalphazero:\n  mcts_n: 200\n  mcts_c_puct: 1.4\nself_play:\n  num_processes: 1\n  games_per_process: 1\ntraining:\n  games_per_training_step: 1.0\n  learning_rate: 0.001\n  batch_size: 64\n  weight_decay: 0.0001\n  replay_buffer_size: 1000\n' > "$RUN/config.yaml"
mkdir -p "$RUN/models/checkpoints"
cp deep_quoridor/rust/fixtures/alphazero_B5W2_mv1.onnx "$RUN/models/checkpoints/model_1.onnx"
PYTHONPATH=deep_quoridor/src python deep_quoridor/src/run_play_server_web.py "$RUN" \
    --static-dir deep_quoridor/frontend/dist --port 8080
# open http://localhost:8080  (needs a WebGPU-capable browser, or it falls back to wasm-CPU)
```

Dev mode with HMR: `npm --prefix deep_quoridor/frontend run dev`. Vite serves the
COOP/COEP headers itself, but the `/api/*` and `/models/*` routes come from the
Python server, so for full play run the built app behind the Python server (above)
or add a Vite dev proxy to it.

## How it fits together

```
Browser
  main thread (Svelte)  ── postMessage ─▶  Web Worker (ai.worker.ts)
    board, progress bar,  ◀─ state/progress ─   quoridor-wasm  (game + MCTS)
    undo, config drawer                          onnxruntime-web (WebGPU) ── eval_batch
Python server: /  (SPA)  ·  /api/config  ·  /api/models  ·  /models/*.onnx  ·  /ort/*.wasm
```

## Tests
```
npm --prefix deep_quoridor/frontend run test   # vitest: board model, eval marshalling, api client
```
Unit tests cover the pure logic (board derivation, eval tensor marshalling, API
client). End-to-end gameplay (WebGPU inference, progress bar, clicking) is a manual
browser check — the server's `/api/*`, `/models/*`, and `/ort/*` routes are
integration-verified via curl, but actually playing needs a real browser.
