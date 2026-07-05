# Browser-WASM play server — design

**Date:** 2026-07-05
**Branch:** TBD (new feature branch off `main`)
**Status:** approved (design), pending implementation plan
**Supersedes:** the server-side-AI architecture of
`2026-05-29-quoridor-play-server-design.md` (that server ran MCTS + ONNX
per-session on the host; this design moves the AI into the browser).

## Problem

The current play server (`rust/src/play_server/`) runs the AlphaZero agent
server-side: each browser session holds an `Arc<Mutex<GameSession>>` that owns
an `AlphaZeroAgent` and runs MCTS + ONNX inference on the host. That works but:

- **Requires a GPU host.** ONNX inference for a good-strength AI wants a graphics
  card on the server.
- **Scales poorly.** Every concurrent player consumes server CPU/GPU and RAM for
  their MCTS search.

We want to keep reusing the project's **Rust** AlphaZero implementation (game
mechanics + MCTS), which we like, but move the entire AI computation into the
player's browser so the server becomes a cheap static host.

## Goals

- Run the AI **entirely in the browser**, using the **client machine's GPU** for
  neural-network inference.
- Keep the server **low-resource**: serve the page, the WASM, and the models.
  (A REST API for things like high scores comes later.)
- A more **browser-heavy, responsive** UI (Svelte). Beyond current functionality,
  add an **Undo** button and a real-time **"AI thinking" progress bar** showing
  the fraction of MCTS simulations completed.
- Expose most AlphaZero parameters for **viewing and editing** in the UI
  (`mcts_n`, `temperature`, Dirichlet α/ε, `c_puct`, `leaf_parallelism`,
  `mcts_worker_threads`, …), presented in the same tree structure as the yaml
  config files in run directories.
- Work on **modern browsers** (Chrome, Edge, Safari, Firefox — no dev/nightly
  builds required). WebGPU is stable in all of these.
- Be fast enough not to frustrate the player, via the **leaf-parallel batched
  MCTS** used in self-play.

## Non-goals (deferred to later milestones)

This spec is **Milestone 1**: the smallest thing that fully replaces today's
server. Explicitly deferred:

- **M2 — Model & search visualization:** policy/value heatmap over the board;
  interactive MCTS tree that grows as the AI thinks.
- **M3 — Self-play & run exploration:** server exposes a folder of run
  directories; browse/replay self-play games; inspect models.
- **M4 — REST API:** high scores and other stateful features.

The design keeps hooks for these (see "Forward compatibility") but does not
build them.

## Architecture overview

```
Browser
  Main thread (Svelte app)
    board · legal-move hints · undo · progress bar · move history
    · slide-over "Advanced" config drawer · model picker
        │  start(state, config)                     ▲ progress {done/N}
        ▼                                            │ result {move, policy}
  Web Worker (one OS thread)
    quoridor-wasm (Rust→WASM): game mechanics + leaf-parallel MCTS
        │  evaluate_batch(features[K]) ──await──▶ onnxruntime-web
        ▼                                            (WebGPU EP; WASM-CPU fallback)
    collects K diverse leaves (virtual loss) → one batched eval → backprop K
        │
        ▼
    Client GPU (WebGPU)

Thin server (new Python app — FastAPI)
  serves: SPA + .wasm + model .onnx files
  sets:   COOP/COEP + correct wasm/wasm-streaming MIME headers
  API:    GET /api/models, GET /api/config (reuse existing Python config loaders)
  (the old Rust rust/src/play_server is retired: MCTS/ONNX/session move to WASM)
```

### Key decisions (with rationale)

1. **Inference: `onnxruntime-web` with the WebGPU execution provider**
   (CPU-WASM fallback automatic). *Why over a pure-Rust engine (`wonnx`/`burn`):*
   ORT-web loads `.onnx` files **at runtime over HTTP**, which matches
   "server serves the models" and lets the UI pick among models; it is mature
   with broad op coverage for a small ResNet; and it keeps the risky part
   (browser NN inference) on a well-trodden path. The trade-off — two WASM
   runtimes and a JS↔WASM seam per batch — is amortized by batching.

2. **Threading: single Web Worker + GPU batching** (self-play's leaf-parallel
   design, reused for one game). *Why not multi-threaded tree search:* on a small
   board the tree ops are microseconds; the cost is NN eval, which runs in
   parallel **on the GPU**. Batching K leaves into one WebGPU call is the real
   speedup and needs no `SharedArrayBuffer` for our code. True multi-threaded
   tree search (`wasm-bindgen-rayon` + worker pool + `SharedArrayBuffer`) is a
   deferred, profile-driven optimization.

3. **Framework: Svelte.** Tiny runtime, fine-grained reactivity suited to a
   live-updating board / progress bar / config tree, and clean SVG/canvas interop
   for the M2 visualizations.

4. **Crate boundary: a new `quoridor-wasm` crate** (`crate-type=["cdylib"]`) in
   the workspace, depending on `quoridor-rs` with a slim feature set. *Why a
   separate crate over a `wasm` feature flag on `quoridor-rs`:* keeps
   `wasm-bindgen` out of the core lib and avoids the cdylib/rlib target conflict
   already documented in `rust/Cargo.toml`.

5. **Server: a new Python app (FastAPI)** that fully replaces the Rust
   `play_server`. *Why Python, not the existing Rust server:* the server no
   longer touches the AI at all (MCTS + ONNX now run in WASM), so there is no
   reason to keep it in Rust. The team already uses Python, it has strong web
   libraries, and — decisively — the run directories, trainer, config, and
   yaml models are **already Python** (`src/v2/config.py`, `yaml_models.py`,
   etc.), so M3 (run-dir browsing) and M4 (REST high scores) reuse existing
   Python code instead of reimplementing it. FastAPI gives easy custom headers
   (COOP/COEP) via middleware, `.wasm` static serving, and a clean REST surface
   for later. (Flask/Starlette are viable; FastAPI is the recommendation.)

## Components

### A. `quoridor-wasm` crate (Rust → WASM)

New workspace crate `rust/quoridor-wasm` (or a `quoridor-wasm` member;
finalize location in the plan).

- **Depends on** `quoridor-rs` with `default-features = false` and a slim
  feature set exposing: `compact/*` game mechanics, `actions`, `grid_helpers`
  (feature encoding), `rotation`, and the MCTS **tree primitives** from
  `mcts.rs` / `selfplay_mcts.rs` (`NodeArena`, `select_leaf_with_vl`, virtual
  loss, `expand_node`, `backpropagate`, `promote_subtree`). None of `ort`,
  `tokio`, `tiny_http`, `pyo3`, `rayon`, or `dashmap` are pulled into this build.
  - This likely requires making some currently `pub(super)` tree primitives
    `pub`, and confirming the core modules are free of the native-only deps
    (the eval pipeline / ORT / tokio code stays behind `binary`).
- **Adds** `wasm-bindgen`, `wasm-bindgen-futures`, `js-sys`, `serde-wasm-bindgen`.
- **Exposes** `#[wasm_bindgen]` bindings, roughly:
  - `new_game(board_size, max_walls, max_steps, human_player) -> GameHandle`
  - `legal_actions(handle) -> JsValue` (enriched actions, reusing the
    `EnrichedAction` shape from `play_server::state`)
  - `apply_action(handle, action_index) -> StateView`
  - `undo(handle) -> StateView` (replays move history minus the last human ply)
  - `run_search(handle, config, eval_batch_cb, progress_cb) -> Promise<Result>`
- **New async MCTS driver** (the one genuinely new piece): the same
  leaf-parallel structure as `selfplay_mcts.rs`, but the eval transport is a
  **JS-provided batched-eval callback** instead of the tokio→ORT eval pipeline.
  Per outer iteration: select K diverse leaves (virtual loss) → `await`
  `eval_batch_cb(features)` (resolved by onnxruntime-web) → backprop all K →
  fire `progress_cb(done, total)` every few sims → repeat to `mcts_n`. Returns
  the chosen move plus the root policy (the policy field is already computed and
  is the hook M2 will use).

### B. Svelte app (main thread)

- **Board component:** renders `StateView` (pawns, walls, walls-remaining),
  highlights legal moves/wall slots, handles click-to-move / click-edge-to-wall.
- **Control rail:** AI-thinking card (progress bar: `done/total` sims + current
  leading move), **Undo**, **New game** (model dropdown from `/api/models`,
  who-goes-first), **move history** list.
- **Advanced config drawer** (slide-over): editable tree mirroring the yaml
  config structure of run directories; values feed the `config` argument to
  `run_search` and apply on the next search. (M1 exposes the AlphaZero search
  params; the tree shape is chosen so more of the yaml can be surfaced later.)
- **Worker client:** posts `start(state, config)`, receives `progress`/`result`
  messages, drives the progress bar and board updates.

### C. Web Worker

- Loads `quoridor-wasm` and `onnxruntime-web`; owns the ORT `InferenceSession`
  (created with the WebGPU EP, CPU-WASM fallback).
- Fetches/caches the selected `.onnx` (HTTP cache for M1; IndexedDB/Cache API is
  a later nicety).
- Bridges `quoridor-wasm`'s `eval_batch_cb` to `session.run(...)` and posts
  progress/result messages to the main thread.

### D. Thin server (new Python app — FastAPI)

- A new Python package (e.g. `src/play_server_web/` or a small FastAPI app under
  the existing `src/`), launched with uvicorn.
- Serves the built SPA + `.wasm` + `<play-dir>/models/*.onnx` (static mounts).
- Sends **COOP: `same-origin`** + **COEP: `require-corp`** (so ORT's CPU-fallback
  threads work) via middleware, and correct `application/wasm` MIME for streaming
  compile.
- Small JSON API:
  - **`GET /api/models`** — lists `<play-dir>/models/*.onnx`, reusing the
    project's existing Python config/yaml loaders (`src/v2/config.py`,
    `yaml_models.py`) rather than reimplementing the Rust `config.rs` listing.
  - **`GET /api/config`** — board dimensions + default AlphaZero params.
- The old Rust `rust/src/play_server/` module (HTTP, handlers, session,
  server-side MCTS/ONNX) is **retired**: game play, the `StateView` shape, and
  move/undo logic all move into `quoridor-wasm`. (The `EnrichedAction`/
  `StateView` types are a useful reference when porting the view shape to WASM.)

## Data flow: one AI move

1. Human clicks a legal cell/edge → Svelte calls `wasm.apply_action(idx)` →
   gets updated `StateView`, re-renders, pushes to move history.
2. If it's now the AI's turn, Svelte tells the worker `start(state, config)`.
3. Worker calls `wasm.run_search(handle, config, eval_batch_cb, progress_cb)`.
4. The Rust driver loops: select K leaves → `await eval_batch_cb(features)`
   (ORT/WebGPU) → backprop → `progress_cb(done, total)` (→ posted to main
   thread → progress bar) until `mcts_n` sims.
5. Driver returns `{ move, policy }`; worker posts `result`; Svelte applies the
   AI move via `wasm.apply_action`, re-renders, clears the progress bar.

## Build & toolchain

- **Vite + Svelte** for the app.
- **`wasm-pack`** builds `quoridor-wasm` into an npm-consumable package that Vite
  bundles (including the worker + `.wasm`).
- **`onnxruntime-web`** from npm.
- Dev: Vite dev server (with COOP/COEP dev headers to mirror prod). Prod: `vite
  build` output is what the Python (FastAPI/uvicorn) server serves as static
  files.

## Testing

- **Rust (`quoridor-wasm`) unit tests** on native target for the game/undo logic
  and the async driver against a **mock eval callback** (uniform priors, à la
  `UniformMockEvaluator`) — asserts sim counts, progress-callback cadence, and
  that N sims produce a legal move. Reuses existing tree-primitive tests.
- **Parity test:** for a fixed seed/model, the WASM driver's visit counts /
  chosen move match the native `selfplay_mcts` leaf-parallel search (guards the
  new eval transport, not new tree logic). Consider extending the existing
  cross-language consistency harness.
- **Svelte component tests** (Vitest): board rendering from `StateView`, legal-
  move highlighting, undo reducing history, progress-bar binding.
- **Server tests (Python, pytest + FastAPI `TestClient`):** static routes serve
  with correct `application/wasm` MIME + COOP/COEP headers; `/api/models` lists
  `.onnx`; `/api/config` returns board dims. (The Rust `play_server_e2e.rs` is
  retired with the module.)
- **Manual smoke:** real browser, WebGPU on and forced-off (CPU fallback), full
  game vs AI with visible progress bar and working undo.

## Forward compatibility (hooks, not built)

- `run_search`'s `result` already carries the **root policy** → M2 heatmap.
- The async driver can emit **tree snapshots** on `progress` → M2 live tree.
- The config drawer renders a **general yaml-shaped tree** → more params later.
- The Python server → M3 run-dir routes + M4 REST slot straight into existing
  Python run/config/training code.

## Risks / open questions for the plan

- **ORT-web op coverage / perf on WebGPU** for this exact ResNet — validate
  early with a spike (load a real `.onnx`, batched eval, measure).
- **Model download size** on first load — HTTP caching is fine for M1; revisit
  IndexedDB if models are large.
- **Exact `quoridor-rs` feature split** — confirm core modules compile to
  `wasm32-unknown-unknown` with the native-only deps fully gated out.
- **`mcts_worker_threads` in the config UI** — it is exposed for viewing/editing
  but is inert in M1's single-worker design; the plan should decide whether to
  show it as disabled/"self-play only" to avoid implying an effect it doesn't
  have.
