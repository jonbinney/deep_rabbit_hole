# Python FastAPI play server — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A thin Python FastAPI server that serves the browser play app (the built SPA + its `.wasm`) and the trained model `.onnx` files, with cross-origin-isolation + correct wasm headers, and exposes `GET /api/models` and `GET /api/config` by reusing the project's existing `v2.config` loader.

**Architecture:** A new package `src/v2/play_server_web/` split by responsibility: pure, unit-testable helpers (`model_listing.py`, `config_view.py`) and a FastAPI **app factory** (`app.py`) wiring routes + COOP/COEP middleware + static mounts. A thin CLI entrypoint `src/run_play_server_web.py` (argparse + uvicorn) mirrors `run_benchmarks_v2.py`. The server touches no AI code — the AI runs in the browser (Plan 1's `quoridor-wasm`).

**Tech Stack:** FastAPI + uvicorn (new deps), httpx (TestClient, dev), pydantic (existing), reusing `v2.config.load_user_config`. Python 3.12.

**Spec:** `docs/superpowers/specs/2026-07-05-browser-wasm-play-server-design.md` (Component D). This is Plan 2 of 3 (Plan 1 = `quoridor-wasm`, done; Plan 3 = Svelte/worker frontend).

---

## Key decisions (design notes)

- **Input is a run directory, models come from `models/checkpoints/`.** The spec's Component D said `<play-dir>/models/*.onnx`, but the trainer actually writes `<run_dir>/models/checkpoints/model_<version>.onnx` (verified). So the server takes `--run-dir` (an existing run dir with `config.yaml` + `models/checkpoints/`) and lists/serves `.onnx` from there. A `--models-dir` flag can override the models location (e.g. to point at `rust/fixtures/` for a quick model). This reconciles the spec's stale assumption with reality.
- **Config is read-only via `load_user_config`** (returns `UserConfig`, no side effects, no path/run-layout assumptions) — NOT `load_config_and_setup_run` (which creates dirs) or `Config.from_user` (which assumes `<base>/runs/<run_id>/`).
- **`/api/models` and `/api/config` are separate** (per spec): models list is a filesystem glob; config view is parsed yaml.
- **COOP/COEP always sent.** `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp` — needed for ORT's multi-threaded WASM-CPU fallback (SharedArrayBuffer); harmless for the WebGPU path. Implication for Plan 3: onnxruntime-web must be bundled **same-origin** (COEP blocks non-CORP cross-origin resources) — a Plan 3 concern, noted here.
- **SPA is optional at this stage.** Plan 3 produces the SPA build; until then `--static-dir` is omitted and the server serves a placeholder page at `/` (API + models still work).

## File structure

**Created:**
- `deep_quoridor/src/v2/play_server_web/__init__.py`
- `deep_quoridor/src/v2/play_server_web/model_listing.py` — `list_onnx_models`, `default_model`.
- `deep_quoridor/src/v2/play_server_web/config_view.py` — `build_config_view`.
- `deep_quoridor/src/v2/play_server_web/app.py` — `create_app(...)` factory.
- `deep_quoridor/src/run_play_server_web.py` — CLI entrypoint.
- `deep_quoridor/test/test_play_server_web.py` — pytest (pure fns + TestClient).

**Modified:**
- `deep_quoridor/requirements.txt`, `deep_quoridor/ci_requirements.txt` — add `fastapi`, `uvicorn`, `httpx`.

## Conventions (how to run things here)

- Python imports use `from v2.x import ...` with `PYTHONPATH` = `deep_quoridor/src`. Run tests from the **repo root** (`/home/jbinney/ws/deep_rabbit_hole`):
  `PYTHONPATH=$(pwd)/deep_quoridor/src python -m pytest deep_quoridor/test/test_play_server_web.py -v`
- An active virtualenv is at `/home/jbinney/ws/deep_rabbit_hole/.venv` (writable). `pip`'s default cache (`~/.cache/pip`) is read-only in this environment, so install with `--no-cache-dir`.
- Never use `cd` in a compound shell command (permission prompt) — use absolute paths / `PYTHONPATH=...`.

---

## Task 1: Add and install the web dependencies

**Files:** Modify `deep_quoridor/requirements.txt`, `deep_quoridor/ci_requirements.txt`.

- [ ] **Step 1: Add deps to `deep_quoridor/requirements.txt`**

Append these three lines (unpinned, matching the file's existing style):
```
fastapi
uvicorn
httpx
```

- [ ] **Step 2: Add the same deps to `deep_quoridor/ci_requirements.txt`**

Append the same three lines:
```
fastapi
uvicorn
httpx
```

- [ ] **Step 3: Install into the active venv**

Run:
```bash
pip install --no-cache-dir fastapi uvicorn httpx 2>&1 | tail -5
```
Expected: successful install (downloads from PyPI). If PyPI is unreachable in this environment, STOP and report BLOCKED — the maintainer must `pip install fastapi uvicorn httpx` in their own environment before tests can run.

- [ ] **Step 4: Verify the imports resolve**

Run:
```bash
python -c "import fastapi, uvicorn, httpx; from fastapi.testclient import TestClient; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git -C deep_quoridor add requirements.txt ci_requirements.txt
git -C deep_quoridor commit -m "deps: add fastapi, uvicorn, httpx for the Python play server"
```

---

## Task 2: `model_listing.py` — list `.onnx` models

**Files:** Create `deep_quoridor/src/v2/play_server_web/__init__.py` (empty), `deep_quoridor/src/v2/play_server_web/model_listing.py`; Test in `deep_quoridor/test/test_play_server_web.py`.

- [ ] **Step 1: Create the empty package marker**

Create `deep_quoridor/src/v2/play_server_web/__init__.py` with a single line:
```python
"""FastAPI play server: serves the browser app + models, exposes config/model APIs."""
```

- [ ] **Step 2: Write the failing test**

Create `deep_quoridor/test/test_play_server_web.py`:
```python
from pathlib import Path

from v2.play_server_web.model_listing import default_model, list_onnx_models


def _touch(dir_: Path, *names: str) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for n in names:
        (dir_ / n).write_bytes(b"")


def test_list_onnx_models_natural_sorted(tmp_path):
    _touch(tmp_path, "model_2.onnx", "model_10.onnx", "model_1.onnx", "model_0.pt", "notes.txt")
    assert list_onnx_models(tmp_path) == ["model_1.onnx", "model_2.onnx", "model_10.onnx"]


def test_list_onnx_models_missing_dir_is_empty(tmp_path):
    assert list_onnx_models(tmp_path / "nope") == []


def test_default_model_is_highest_version_or_none():
    assert default_model(["model_1.onnx", "model_2.onnx", "model_10.onnx"]) == "model_10.onnx"
    assert default_model([]) is None
```

- [ ] **Step 3: Run it to confirm it fails**

Run:
```bash
PYTHONPATH=$(pwd)/deep_quoridor/src python -m pytest deep_quoridor/test/test_play_server_web.py -q 2>&1 | tail -10
```
Expected: import error / failure (`model_listing` not found).

- [ ] **Step 4: Implement `model_listing.py`**

Create `deep_quoridor/src/v2/play_server_web/model_listing.py`:
```python
"""List trained ONNX model files for the play server's model picker."""

import re
from pathlib import Path

_INT_RE = re.compile(r"\d+")


def _version_key(name: str) -> tuple[int, str]:
    ints = _INT_RE.findall(name)
    return (int(ints[-1]) if ints else -1, name)


def list_onnx_models(models_dir: Path) -> list[str]:
    """Basenames of ``*.onnx`` files in ``models_dir``, sorted by the trailing
    integer (so ``model_9`` precedes ``model_10``). Empty if the dir is absent."""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    names = [p.name for p in models_dir.glob("*.onnx")]
    return sorted(names, key=_version_key)


def default_model(models: list[str]) -> str | None:
    """The highest-version model (last, given the natural sort), or None."""
    return models[-1] if models else None
```

- [ ] **Step 5: Run the tests to confirm they pass**

Run:
```bash
PYTHONPATH=$(pwd)/deep_quoridor/src python -m pytest deep_quoridor/test/test_play_server_web.py -q 2>&1 | tail -10
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git -C deep_quoridor add src/v2/play_server_web/__init__.py src/v2/play_server_web/model_listing.py test/test_play_server_web.py
git commit -C deep_quoridor -m "feat(play-server): model listing helper" || git -C deep_quoridor commit -m "feat(play-server): model listing helper"
```

---

## Task 3: `config_view.py` — board dims + AlphaZero defaults from config

**Files:** Create `deep_quoridor/src/v2/play_server_web/config_view.py`; extend the test file.

- [ ] **Step 1: Write the failing test**

Append to `deep_quoridor/test/test_play_server_web.py`:
```python
from v2.config import load_user_config
from v2.play_server_web.config_view import build_config_view

MINIMAL_CONFIG_YAML = """\
run_id: test-run
quoridor:
  board_size: 5
  max_walls: 2
  max_steps: 50
alphazero:
  mcts_n: 123
  mcts_c_puct: 1.4
self_play:
  num_processes: 1
  games_per_process: 1
training:
  games_per_training_step: 1.0
  learning_rate: 0.001
  batch_size: 64
  weight_decay: 0.0001
  replay_buffer_size: 1000
"""


def _write_config(tmp_path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(MINIMAL_CONFIG_YAML)
    return p


def test_build_config_view_board_and_defaults(tmp_path):
    cfg = load_user_config(str(_write_config(tmp_path)))
    view = build_config_view(cfg)
    assert view["board_size"] == 5
    assert view["max_walls"] == 2
    assert view["max_steps"] == 50
    d = view["defaults"]
    assert d["mcts_n"] == 123
    assert d["mcts_c_puct"] == 1.4
    assert d["leaf_parallelism"] == 16  # SelfPlayConfig default
    assert d["virtual_loss"] == 3       # SelfPlayConfig default
    # No self_play.alphazero block in the minimal config:
    assert d["temperature"] is None
    assert d["mcts_noise_epsilon"] == 0.0
    assert d["mcts_worker_threads"] is None
```

- [ ] **Step 2: Run it to confirm it fails**

Run:
```bash
PYTHONPATH=$(pwd)/deep_quoridor/src python -m pytest deep_quoridor/test/test_play_server_web.py::test_build_config_view_board_and_defaults -q 2>&1 | tail -10
```
Expected: import/attribute failure (`config_view` not found).

- [ ] **Step 3: Implement `config_view.py`**

Create `deep_quoridor/src/v2/play_server_web/config_view.py`:
```python
"""Assemble the /api/config payload (board dims + AlphaZero defaults) from a
loaded UserConfig. The relevant fields live across the alphazero + self_play
sections, so we gather them into one flat 'defaults' block for the UI."""

from v2.config import UserConfig


def build_config_view(cfg: UserConfig) -> dict:
    sp = cfg.self_play
    spa = sp.alphazero  # Optional[AlphaZeroSelfPlayConfig]
    return {
        "board_size": cfg.quoridor.board_size,
        "max_walls": cfg.quoridor.max_walls,
        "max_steps": cfg.quoridor.max_steps,
        "defaults": {
            "mcts_n": cfg.alphazero.mcts_n,
            "mcts_c_puct": cfg.alphazero.mcts_c_puct,
            "temperature": spa.temperature if spa else None,
            "mcts_noise_epsilon": spa.mcts_noise_epsilon if spa else 0.0,
            "mcts_noise_alpha": spa.mcts_noise_alpha if spa else None,
            "leaf_parallelism": sp.leaf_parallelism,
            "virtual_loss": sp.virtual_loss,
            "mcts_worker_threads": sp.mcts_worker_threads,
        },
    }
```

- [ ] **Step 4: Run the test to confirm it passes**

Run:
```bash
PYTHONPATH=$(pwd)/deep_quoridor/src python -m pytest deep_quoridor/test/test_play_server_web.py -q 2>&1 | tail -10
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git -C deep_quoridor add src/v2/play_server_web/config_view.py test/test_play_server_web.py
git -C deep_quoridor commit -m "feat(play-server): config view (board dims + alphazero defaults)"
```

---

## Task 4: `app.py` — the FastAPI app factory

**Files:** Create `deep_quoridor/src/v2/play_server_web/app.py`; extend the test file.

- [ ] **Step 1: Write the failing tests (TestClient)**

Append to `deep_quoridor/test/test_play_server_web.py`:
```python
from fastapi.testclient import TestClient

from v2.play_server_web.app import create_app


def _make_run_dir(tmp_path):
    (tmp_path / "config.yaml").write_text(MINIMAL_CONFIG_YAML)
    models = tmp_path / "models" / "checkpoints"
    _touch(models, "model_1.onnx", "model_2.onnx")
    return tmp_path, models


def test_api_config_endpoint(tmp_path):
    run_dir, _ = _make_run_dir(tmp_path)
    client = TestClient(create_app(run_dir))
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body["board_size"] == 5
    assert body["defaults"]["mcts_n"] == 123


def test_api_models_endpoint(tmp_path):
    run_dir, _ = _make_run_dir(tmp_path)
    client = TestClient(create_app(run_dir))
    r = client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    assert body["models"] == ["model_1.onnx", "model_2.onnx"]
    assert body["default"] == "model_2.onnx"


def test_static_serving_headers_and_wasm_mime(tmp_path):
    run_dir, _ = _make_run_dir(tmp_path)
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>hi</title>")
    (static / "app_bg.wasm").write_bytes(b"\x00asm")
    client = TestClient(create_app(run_dir, static_dir=static))

    idx = client.get("/")
    assert idx.status_code == 200
    assert idx.headers["cross-origin-opener-policy"] == "same-origin"
    assert idx.headers["cross-origin-embedder-policy"] == "require-corp"

    wasm = client.get("/app_bg.wasm")
    assert wasm.status_code == 200
    assert wasm.headers["content-type"] == "application/wasm"


def test_models_are_served_as_static(tmp_path):
    run_dir, models = _make_run_dir(tmp_path)
    (models / "model_1.onnx").write_bytes(b"ONNXDATA")
    client = TestClient(create_app(run_dir))
    r = client.get("/models/model_1.onnx")
    assert r.status_code == 200
    assert r.content == b"ONNXDATA"


def test_placeholder_when_no_static_dir(tmp_path):
    run_dir, _ = _make_run_dir(tmp_path)
    client = TestClient(create_app(run_dir))
    r = client.get("/")
    assert r.status_code == 200
    assert "SPA build not found" in r.text
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
PYTHONPATH=$(pwd)/deep_quoridor/src python -m pytest deep_quoridor/test/test_play_server_web.py -q 2>&1 | tail -12
```
Expected: import failure (`app` not found).

- [ ] **Step 3: Implement `app.py`**

Create `deep_quoridor/src/v2/play_server_web/app.py`:
```python
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
    "<p>SPA build not found. Build the frontend (Plan 3) and start the server "
    "with --static-dir pointing at the build output.</p>"
)


def create_app(
    run_dir: Path,
    static_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
) -> FastAPI:
    run_dir = Path(run_dir)
    config_file = run_dir / "config.yaml"
    models_dir = Path(models_dir) if models_dir is not None else run_dir / "models" / "checkpoints"

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
```

- [ ] **Step 4: Run the full test file to confirm all pass**

Run:
```bash
PYTHONPATH=$(pwd)/deep_quoridor/src python -m pytest deep_quoridor/test/test_play_server_web.py -q 2>&1 | tail -12
```
Expected: 9 passed. If the `/models` static mount 404s because the directory is created after mounting, note that `_make_run_dir` creates it first (it does), so the mount sees it.

- [ ] **Step 5: Commit**

```bash
git -C deep_quoridor add src/v2/play_server_web/app.py test/test_play_server_web.py
git -C deep_quoridor commit -m "feat(play-server): FastAPI app factory (config/models APIs, static SPA+models, COOP/COEP)"
```

---

## Task 5: CLI entrypoint

**Files:** Create `deep_quoridor/src/run_play_server_web.py`; extend the test file.

- [ ] **Step 1: Write the failing test**

Append to `deep_quoridor/test/test_play_server_web.py`:
```python
import importlib


def test_main_returns_1_on_missing_config(tmp_path):
    mod = importlib.import_module("run_play_server_web")

    class Args:
        run_dir = str(tmp_path)  # no config.yaml here
        static_dir = None
        models_dir = None
        host = "127.0.0.1"
        port = 8080

    assert mod.main(Args()) == 1
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
PYTHONPATH=$(pwd)/deep_quoridor/src python -m pytest deep_quoridor/test/test_play_server_web.py::test_main_returns_1_on_missing_config -q 2>&1 | tail -10
```
Expected: `ModuleNotFoundError: No module named 'run_play_server_web'`.

- [ ] **Step 3: Implement `run_play_server_web.py`**

Create `deep_quoridor/src/run_play_server_web.py`:
```python
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
```

- [ ] **Step 4: Run the test to confirm it passes**

Run:
```bash
PYTHONPATH=$(pwd)/deep_quoridor/src python -m pytest deep_quoridor/test/test_play_server_web.py -q 2>&1 | tail -12
```
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git -C deep_quoridor add src/run_play_server_web.py test/test_play_server_web.py
git -C deep_quoridor commit -m "feat(play-server): uvicorn CLI entrypoint"
```

---

## Task 6: Docs + verification sweep

**Files:** Create `deep_quoridor/src/v2/play_server_web/README.md`.

- [ ] **Step 1: Write the README**

Create `deep_quoridor/src/v2/play_server_web/README.md`:
```markdown
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
- `--models-dir`: override where `.onnx` files are read from.

## API
- `GET /api/config` → `{ board_size, max_walls, max_steps, defaults: { mcts_n, mcts_c_puct, temperature, mcts_noise_epsilon, mcts_noise_alpha, leaf_parallelism, virtual_loss, mcts_worker_threads } }`
- `GET /api/models` → `{ models: ["model_1.onnx", ...], default: "model_N.onnx" }`
- `GET /models/<file>.onnx` → the model file (for onnxruntime-web to fetch).
- `GET /` and other paths → the SPA (or placeholder).

All responses carry `Cross-Origin-Opener-Policy: same-origin` +
`Cross-Origin-Embedder-Policy: require-corp`. Plan 3 must bundle onnxruntime-web
**same-origin** (COEP blocks non-CORP cross-origin resources).

## Tests
```
PYTHONPATH=deep_quoridor/src python -m pytest deep_quoridor/test/test_play_server_web.py -v
```
```

- [ ] **Step 2: Full verification sweep**

Run:
```bash
PYTHONPATH=$(pwd)/deep_quoridor/src python -m pytest deep_quoridor/test/test_play_server_web.py -v 2>&1 | tail -20
```
Expected: all tests pass (10).

- [ ] **Step 3: Confirm the entrypoint starts (smoke, then stop)**

Run (starts the server briefly against a placeholder, then kills it):
```bash
RUN=$(mktemp -d); printf 'run_id: s\nquoridor:\n  board_size: 5\n  max_walls: 2\n  max_steps: 50\nalphazero:\n  mcts_n: 100\n  mcts_c_puct: 1.4\nself_play:\n  num_processes: 1\n  games_per_process: 1\ntraining:\n  games_per_training_step: 1.0\n  learning_rate: 0.001\n  batch_size: 64\n  weight_decay: 0.0001\n  replay_buffer_size: 1000\n' > "$RUN/config.yaml"
PYTHONPATH=$(pwd)/deep_quoridor/src timeout 4 python deep_quoridor/src/run_play_server_web.py "$RUN" --port 8137 2>&1 | tail -5 &
sleep 2; curl -s http://127.0.0.1:8137/api/config | head -c 300; echo; wait
```
Expected: the `curl` prints the JSON config view (board_size 5, defaults…). (The `timeout 4` stops uvicorn.)

- [ ] **Step 4: Commit**

```bash
git -C deep_quoridor add src/v2/play_server_web/README.md
git -C deep_quoridor commit -m "docs(play-server): README with run instructions and API"
```

---

## Self-review checklist (run after writing all tasks)

- **Spec coverage:** static SPA+wasm serving ✓ (Task 4), model `.onnx` serving ✓ (Task 4 `/models` mount), COOP/COEP + wasm MIME ✓ (Task 4), `/api/models` ✓ (Task 4), `/api/config` ✓ (Task 3/4), reuse of Python config loader ✓ (`load_user_config`). Retiring the Rust `play_server` is a separate cleanup, tracked below.
- **Deferred / out of scope:** deleting the Rust `rust/src/play_server/` module (do it once Plan 3's SPA replaces its function end-to-end); serving a curated flat `models/` instead of all checkpoints; M4 REST (high scores).

## Follow-ups (not built here)
- Retire `rust/src/play_server/` + `bin/play_server.rs` + `tests/play_server_e2e.rs` once the browser path fully replaces it (Plan 3).
- Optional: a `/api/config` "play defaults" that overrides the run's self-play values with play-appropriate ones (temperature 0, no noise) — currently the UI is expected to choose those.
