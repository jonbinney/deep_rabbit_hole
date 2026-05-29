# Quoridor play server — design

**Date:** 2026-05-29
**Branch:** `jdb/rust-self-play-logging`
**Status:** approved (design), pending implementation plan

## Problem

We want a locally-hosted web app so the author and friends can play Quoridor
against the project's AlphaZero agent in their browsers. Each person opens the
URL in their browser and gets their own independent game vs the AI; multiple
games run concurrently on the same server. Constraints:

- Use the project's **Rust** AlphaZero agent (the agent that actually has the
  trained models we want to play against).
- Keep it as simple as possible — no large web frameworks, no Python in the
  request path, no build step for the frontend.
- The player can pick which ONNX model the AI uses from a directory of models.

## Architecture

A single new Rust binary `bin/play_server.rs` in the existing `quoridor_rs`
crate:

- Reads `<play-dir>/config.yaml` once at startup for `board_size`, `max_walls`,
  `max_steps` (reusing the existing `selfplay_config` loader), and scans
  `<play-dir>/models/*.onnx` for the selectable models.
- Listens on TCP (default port 8080; default bind `127.0.0.1`, `0.0.0.0` for
  LAN).
- Serves a single `index.html` + `app.css` + `app.js` (embedded into the binary
  via `include_str!`) plus a small JSON API.
- Holds active games in `Arc<Mutex<HashMap<GameId, Mutex<GameSession>>>>`. Each
  `GameSession` owns its own `AlphaZeroAgent` (built from the chosen ONNX with
  the chosen `mcts_n`, `temperature=0`, deterministic tie-break) and the current
  `CompactState` / `QGameMechanics`.

### User flow
1. Friend opens `http://<host>:8080` in their browser.
2. Picks **model** (dropdown), **`mcts_n`** (slider), and **who goes first**
   (toggle); clicks "New Game".
3. The client POSTs `/api/games`; server creates a `GameSession`, returns a
   `game_id` + initial `state`.
4. On the human's turn, legal moves and legal wall slots are highlighted; the
   user clicks one, the client POSTs `/api/games/<id>/move` with the chosen
   action index, the server applies the human move and (if it's then the AI's
   turn) the AI response, and returns the new `state` in the same round-trip.
5. Repeat until `state.winner != null`. A game-over banner appears.

### Why single Rust binary, not Python + pyo3
"Use the Rust agent" + "no big frameworks" + "as simple as possible" point at
one binary: no venv, no maturin step, no two-language tooling for a small
feature. The only new dep is `tiny_http`, which is a small HTTP server library
(not a framework — no routing macros, no extractors, no async runtime).

## Folder layout consumed by the server

The user passes a directory to `--play-dir`. The expected layout is:
```
<play-dir>/
  config.yaml                # board_size, max_walls, max_steps, alphazero.*
  models/
    *.onnx                   # selectable models
```
The server treats the single board config as authoritative for the session;
every model in `models/` is assumed to match it. (Mixed-config setups are out
of scope.)

## CLI
```
play_server --play-dir <path>
            [--port 8080]
            [--bind 127.0.0.1]            # use 0.0.0.0 for LAN
            [--default-mcts-n 400]
```

## HTTP API (small, JSON)

- `GET /` → `index.html`
- `GET /static/<file>` → embedded `app.css`, `app.js`
- `GET /api/config` →
  ```json
  { "board_size": 9, "max_walls": 10, "max_steps": 100,
    "models": ["model_0.onnx", "model_100.onnx", ...],
    "default_mcts_n": 400 }
  ```
- `POST /api/games` body `{ "model": "model_100.onnx", "mcts_n": 400, "human_player": 0 }`
  → `{ "game_id": "8 hex chars", "state": <State> }`
- `GET  /api/games/<game_id>` → `{ "state": <State> }`
- `POST /api/games/<game_id>/move` body `{ "action_index": 17 }`
  → `{ "state": <State> }` (atomically applies the human move plus the AI's
  response if it then becomes the AI's turn).

**Errors:** invalid move → `400` with a `{ "error": "..." }` body; unknown
`game_id` → `404`; bad model name → `400`; ORT load failure → `500` with the
message; the server keeps running through any of these.

## The `State` object

Everything the client needs to render the board without knowing the action
encoding:
```json
{
  "board_size": 9, "max_walls": 10,
  "current_player": 0,
  "p1_pos": [0, 4], "p2_pos": [8, 4],
  "p1_walls": 10,   "p2_walls": 10,
  "walls": [{ "row": 3, "col": 2, "orientation": "h" }, ...],
  "legal_actions": [
    { "index": 3,  "kind": "move", "to": [4, 5] },
    { "index": 17, "kind": "wall", "row": 3, "col": 2, "orientation": "h" }
  ],
  "completed_steps": 5,
  "winner": null,
  "human_player": 0,
  "last_action": { "kind": "move", "to": [1, 4] },
  "move_history": [3, 17, ...]
}
```
`legal_actions` carrying the kind/coords is the key piece: the client never has
to mirror the action-encoding logic. The wall list and `last_action` use the
same shape so the frontend has one render path.

## Frontend

Vanilla HTML + CSS + JS, served from the binary's embedded strings. The board
is a `(2N-1) × (2N-1)` CSS grid:

- even row, even col → **pawn cell** (clickable if a `move` action lands here)
- odd row,  even col → **horizontal wall slot** (clickable if a horizontal wall
  is legal here)
- even row, odd col → **vertical wall slot** (clickable if a vertical wall is
  legal here)
- odd row,  odd col → wall post / spacer

To set up interactions, the client iterates `state.legal_actions`, attaches a
click handler to the appropriate cell/slot keyed by the bare `index`, and on
click POSTs `{ "action_index": index }` to the move endpoint. The same
iteration produces the legal-move highlights. Walls and `last_action` render
through the same mapping.

A right-side panel shows: whose turn it is, walls remaining per player,
completed steps, an "AI is thinking…" spinner while a move request is in
flight, and a "New Game" button. The new-game form has the model `<select>`,
the `mcts_n` `<input type="range">`, and the human-player toggle.

**Orientation.** Render with the human's home row at the bottom regardless of
`human_player`, so the player always sees "I'm advancing upward." The server
sends absolute coordinates; the client mirrors when `human_player == 1`.

**Game-over banner.** "You won" / "AI won" / "Draw (truncated)" based on
`state.winner` (vs `human_player`).

## Concurrency

- Outer `Arc<Mutex<HashMap<GameId, Arc<Mutex<GameSession>>>>>`: takes the outer
  lock briefly to look up the session, then holds the inner per-game lock for
  the duration of a move + AI response.
- AI inference is already single-threaded (`with_intra_threads(1)` is set on
  the session builders), so other concurrent games proceed in parallel without
  oversubscribing the CPU.
- Game IDs are 8 hex chars from a secure RNG.

## Cargo.toml additions
```toml
[[bin]]
name = "play_server"
path = "src/bin/play_server.rs"
required-features = ["binary"]

[dependencies]
tiny_http = { version = "0.12", optional = true }
```
And `tiny_http` is added to the `binary` feature list. `serde_json` is already
on `binary` from the metrics work.

## Testing

- **Rust unit tests** in `play_server.rs` for the pure helpers: `State`
  serialization, legal-action enrichment (`index → {kind, to/row/col/orient}`),
  the wall list / `last_action` mapping.
- **Rust integration test** using the existing `alphazero_B5W2_mv1.onnx`
  fixture: bind to port 0 (OS-assigned), `POST /api/games`, walk through a
  small game, assert state transitions and that `winner` is eventually set on a
  forced win. Uses `ureq` as a dev-dependency for the HTTP client.
- No frontend tests — manual.

## Out of scope (YAGNI)

- Auth or user accounts.
- HTTPS (localhost/LAN only).
- Game persistence — sessions are in-memory and lost on server restart.
- Undo / takeback.
- Spectator mode.
- Human-vs-human (matchmaking, lobbies).
- Replay save/load.
- Mobile-responsive layout (assume desktop browser).
- Per-session TTL or eviction — restart the server to clear.
- Model switching mid-game.
- Mixed-config model directories.
