# quoridor-wasm

WebAssembly bindings for the Quoridor game core + AlphaZero MCTS, so the AI can
run entirely in the browser. Part of Milestone 1 of the browser-WASM play server
— see `docs/superpowers/specs/2026-07-05-browser-wasm-play-server-design.md` and
the implementation plan `docs/superpowers/plans/2026-07-05-quoridor-wasm-crate.md`.

This crate is a thin `wasm-bindgen` layer over `quoridor-rs`: it owns the game
session and drives the leaf-parallel batched MCTS, but the neural-network forward
pass is injected as a JavaScript callback (wired to onnxruntime-web / WebGPU by
the frontend). No `tokio`/`ort` is pulled into the wasm build.

## Build

```
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build rust/quoridor-wasm --target web --release
```

Output lands in `rust/quoridor-wasm/pkg/`: import `quoridor_wasm.js`, which loads
`quoridor_wasm_bg.wasm`. Call the module's default `init()` once on startup
(routes Rust panics to `console.error`).

## JS API

- `new Game(board_size, max_walls, max_steps, human_player)` — create a session.
- `game.stateView()` → `StateView` object (board dims, pawn positions, walls,
  `legal_actions`, `winner`, `move_history`, …).
- `game.applyAction(actionIndex)` → `StateView` (throws on illegal/over).
- `game.undo(count)` → `StateView` (replays history minus `count` plies).
- `game.runSearch(mctsN, cPuct, leafParallelism, virtualLoss, evalBatch, progress)`
  → `Promise<{ action, rootValue, children: [{ actionIndex, visitCount }] }>`.
  Rejects if called on a finished game.

`evalBatch` is the NN forward pass, supplied by the frontend (Plan 3):

```
evalBatch(flat: Float32Array, n, c, h, w) =>
    Promise<{ values: Float32Array /*[n]*/, logits: Float32Array /*[n*policySize]*/ }>
```

`flat` is the batch of `n` feature tensors of shape `[n, c, h, w]` in row-major
order (`c == 5`, `h == w == board_size*2 + 3`). Return **raw** policy logits — the
Rust side applies masked softmax + un-rotation. `progress(done, total)` is called
after each search round (drive a progress bar); it is best-effort (a throwing
progress callback is ignored).

## Tests

- **Native unit tests** (game/undo logic, view serialization):
  `cargo test -p quoridor-wasm`
- **Wasm binding test** (`runSearch` end-to-end with a JS mock eval):
  `wasm-pack test --node`
  The committed `tests/web.rs` targets Node (CommonJS mock, no `run_in_browser`).
  To run it in a headless browser instead, drop the CJS/Node form for the ESM
  `export function` mock and add `wasm_bindgen_test_configure!(run_in_browser);`,
  then `wasm-pack test --headless --chrome` (see the note in `tests/web.rs`).

> Note: some sandboxed CI containers ship a Node + `wasm-bindgen-test-runner`
> combination that traps on any wasm test under `--node`. If `wasm-pack test
> --node` aborts with a V8 fatal error unrelated to your code, run the browser
> variant above (with a `chromedriver` whose major version matches the installed
> Chrome). The binding logic itself has been verified end-to-end under headless
> Chrome.
