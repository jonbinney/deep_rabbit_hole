# Svelte + Worker + onnxruntime-web frontend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The browser play app that ties Plans 1 + 2 together into a runnable site: a Svelte UI (board, legal-move hints, undo, live "AI thinking" progress bar, model/param controls) driven by a Web Worker that runs the `quoridor-wasm` MCTS and evaluates the neural net with **onnxruntime-web** (WebGPU, wasm-CPU fallback).

**Architecture:** A Vite + Svelte 5 app in `deep_quoridor/frontend/`. The AI lives in a **Web Worker** (`ai.worker.ts`) that loads the `quoridor-wasm` package and an onnxruntime-web `InferenceSession`, then drives `Game.runSearch(...)` — the search's `eval_batch` callback runs `session.run` on batched features; `progress` posts ticks to the main thread. The main thread renders `StateView` snapshots and sends move/undo/new-game messages. `onnxruntime-web`'s own `.wasm` files are copied into the build and served **same-origin** (COEP `require-corp`, set by the Plan 2 server, blocks cross-origin non-CORP resources). Built output (`frontend/dist/`) is served by the Plan 2 FastAPI server via `--static-dir`.

**Tech Stack:** Vite 5, Svelte 5, `@sveltejs/vite-plugin-svelte` 4, `onnxruntime-web` 1.27 (WebGPU), `vite-plugin-static-copy` 2, TypeScript, Vitest 2. Consumes Plan 1's `rust/quoridor-wasm/pkg/` (a `wasm-pack --target web` build) and Plan 2's `/api/config`, `/api/models`, `/models/*.onnx`.

**Spec:** `docs/superpowers/specs/2026-07-05-browser-wasm-play-server-design.md`. This is Plan 3 of 3 (Plan 1 = `quoridor-wasm`, done; Plan 2 = FastAPI server, done).

---

## Verified integration facts (build against these exactly)

- **`quoridor-wasm` JS API** (`pkg/quoridor_wasm.js`, ESM `--target web`):
  - default export `init(module_or_path?) => Promise` — call once (`await init()`), loads the `.wasm`.
  - `init()` named export — routes Rust panics to `console.error` (call after the default init).
  - `class Game`: `new Game(board_size, max_walls, max_steps, human_player)`, `applyAction(index) => StateView`, `stateView() => StateView`, `undo(count) => StateView`, `runSearch(mcts_n, c_puct, leaf_parallelism, virtual_loss, eval_batch: Function, progress: Function) => Promise<{action, rootValue, children}>`.
- **`StateView` shape** (fields the UI renders): `board_size, max_walls, max_steps, current_player, p1_pos:[r,c], p2_pos:[r,c], p1_walls, p2_walls, walls:[{row,col,orientation:"h"|"v"}], legal_actions:[EnrichedAction], completed_steps, winner:0|1|null, human_player, last_action, move_history:number[]`. `EnrichedAction` is `{kind:"move", index, to:[r,c]}` or `{kind:"wall", index, row, col, orientation:"h"|"v"}`.
- **`eval_batch(flat, n, c, h, w)`** (called by `runSearch`): `flat` is a `Float32Array` of shape `[n, c, h, w]` row-major; MUST return `{ values: Float32Array /*len n*/, logits: Float32Array /*len n*policy_size*/ }` with **raw** logits (the Rust side masks + softmaxes).
- **Model I/O** (verified on `rust/fixtures/alphazero_B5W2_mv1.onnx`): input tensor name `"input"`, dims `[N,5,13,13]` for a 5×5 board (`M = board_size*2+3`); outputs `"policy_logits"` `[N,57]` and `"value"` `[N,1]`.
- **`runSearch` rejects on a finished game** — the UI only calls it when it's the AI's turn and there's no winner.

## Environment / how to run things here

- node v18 + npm 9 (npm cache writable). Registry reachable.
- Work in `deep_quoridor/frontend/`. Run npm from there via `npm --prefix deep_quoridor/frontend <cmd>` (NEVER `cd` in a compound command — permission prompt).
- The `quoridor-wasm` package must be **built first**: `wasm-pack build rust/quoridor-wasm --target web --release` (read-only `~/.cargo` here → prefix with `CARGO_HOME=/tmp/claude-1000/-home-jbinney-ws-deep-rabbit-hole/691b6962-dbfd-48cf-9bea-c7141fc64ed1/scratchpad/cargo-home XDG_CACHE_HOME=/tmp/claude-1000/-home-jbinney-ws-deep-rabbit-hole/691b6962-dbfd-48cf-9bea-c7141fc64ed1/scratchpad/cache`). `pkg/` is gitignored — it's a build input, not committed.
- Pin versions for node-18 compatibility (below). If `npm install` fails an `engines` check on a newer transitive major, pin that dep down rather than upgrading node.

## File structure

**Created (all under `deep_quoridor/frontend/`):**
- `package.json`, `vite.config.ts`, `tsconfig.json`, `svelte.config.js`, `index.html`
- `src/main.ts` — mounts the app.
- `src/lib/types.ts` — `StateView`/`EnrichedAction`/worker-message TS types.
- `src/lib/board.ts` — `deriveBoard(view)` (pure: cells, pawns, walls, legal-move/legal-wall lookup) + Vitest.
- `src/lib/evalRunner.ts` — `runEval(session, flat, n, c, h, w)` (pure marshalling to/from an ORT-session-like object) + Vitest.
- `src/lib/api.ts` — `fetchConfig()`, `fetchModels()` + Vitest.
- `src/lib/aiClient.ts` — main-thread wrapper around the worker (promise-per-request + progress callback).
- `src/ai.worker.ts` — the Web Worker: loads wasm + ORT, message protocol, `eval_batch` via ORT.
- `src/App.svelte`, `src/lib/Board.svelte`, `src/lib/ControlRail.svelte`, `src/lib/ConfigDrawer.svelte`.
- `src/app.css`
- `README.md`
- `.gitignore` (`node_modules`, `dist`)

**Modified:** none in existing code. (Plan 2's server already serves `--static-dir`.)

---

## Task 1: Scaffold the Vite + Svelte project and prove it builds

**Files:** create `deep_quoridor/frontend/{package.json,vite.config.ts,tsconfig.json,svelte.config.js,index.html,.gitignore,src/main.ts,src/App.svelte,src/app.css,src/lib/smoke.test.ts}`.

- [ ] **Step 1: Build the wasm package (input dependency)**

Run (with the CARGO_HOME/XDG prefix from "Environment" above):
```bash
<PREFIX> wasm-pack build rust/quoridor-wasm --target web --release 2>&1 | tail -4
```
Expected: `pkg/` exists with `quoridor_wasm.js` + `quoridor_wasm_bg.wasm`.

- [ ] **Step 2: Create `deep_quoridor/frontend/package.json`**

```json
{
  "name": "quoridor-frontend",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest run"
  },
  "dependencies": {
    "onnxruntime-web": "^1.27.0",
    "quoridor-wasm": "file:../rust/quoridor-wasm/pkg"
  },
  "devDependencies": {
    "@sveltejs/vite-plugin-svelte": "^4.0.0",
    "svelte": "^5.0.0",
    "svelte-check": "^4.0.0",
    "typescript": "^5.5.0",
    "vite": "^5.4.0",
    "vite-plugin-static-copy": "^2.0.0",
    "vitest": "^2.0.0"
  }
}
```

- [ ] **Step 3: Create `deep_quoridor/frontend/vite.config.ts`**

```ts
import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { viteStaticCopy } from "vite-plugin-static-copy";

// Copy onnxruntime-web's wasm/mjs runtime into the build so it is served
// SAME-ORIGIN (COEP require-corp, set by the Python server, blocks cross-origin
// non-CORP resources — a CDN load of ORT would be blocked). We point
// ort.env.wasm.wasmPaths at "/ort/" (see ai.worker.ts).
export default defineConfig({
  plugins: [
    svelte(),
    viteStaticCopy({
      targets: [
        { src: "node_modules/onnxruntime-web/dist/*.wasm", dest: "ort" },
        { src: "node_modules/onnxruntime-web/dist/*.mjs", dest: "ort" },
      ],
    }),
  ],
  // Vite serves cross-origin-isolation headers in dev so SharedArrayBuffer /
  // the wasm-CPU fallback work locally the same as behind the Python server.
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  worker: { format: "es" },
});
```

- [ ] **Step 4: Create `svelte.config.js`, `tsconfig.json`, `.gitignore`, `index.html`, `src/app.css`**

`deep_quoridor/frontend/svelte.config.js`:
```js
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";
export default { preprocess: vitePreprocess() };
```

`deep_quoridor/frontend/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ESNext",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "types": ["svelte", "vite/client"],
    "skipLibCheck": true,
    "isolatedModules": true
  },
  "include": ["src"]
}
```

`deep_quoridor/frontend/.gitignore`:
```
node_modules
dist
```

`deep_quoridor/frontend/index.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Quoridor</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

`deep_quoridor/frontend/src/app.css`:
```css
:root { font-family: system-ui, sans-serif; }
body { margin: 0; padding: 1rem; }
```

- [ ] **Step 5: Create a placeholder `src/App.svelte`, `src/main.ts`, and a smoke test**

`deep_quoridor/frontend/src/App.svelte`:
```svelte
<script lang="ts">
  let ready = $state(false);
</script>

<main>
  <h1>Quoridor</h1>
  <p>{ready ? "ready" : "scaffold OK"}</p>
</main>
```

`deep_quoridor/frontend/src/main.ts`:
```ts
import "./app.css";
import { mount } from "svelte";
import App from "./App.svelte";

const app = mount(App, { target: document.getElementById("app")! });
export default app;
```

`deep_quoridor/frontend/src/lib/smoke.test.ts`:
```ts
import { describe, expect, it } from "vitest";

describe("scaffold", () => {
  it("runs vitest", () => {
    expect(1 + 1).toBe(2);
  });
});
```

- [ ] **Step 6: Install, test, and build**

Run:
```bash
npm --prefix deep_quoridor/frontend install 2>&1 | tail -8
npm --prefix deep_quoridor/frontend run test 2>&1 | tail -8
npm --prefix deep_quoridor/frontend run build 2>&1 | tail -12
```
Expected: install succeeds; vitest passes (1 test); `vite build` emits `deep_quoridor/frontend/dist/`. If `npm install` fails an `engines` check for node 18, pin the offending dep to a node-18-compatible major and retry (record what you pinned).

- [ ] **Step 7: Commit**

```bash
git -C deep_quoridor add frontend/package.json frontend/vite.config.ts frontend/tsconfig.json frontend/svelte.config.js frontend/index.html frontend/.gitignore frontend/src/main.ts frontend/src/App.svelte frontend/src/app.css frontend/src/lib/smoke.test.ts
git -C deep_quoridor commit -m "feat(frontend): scaffold Vite+Svelte app (builds, vitest, ORT wasm copy)"
```
(Do NOT commit `frontend/node_modules` or `frontend/dist` — gitignored. `package-lock.json` may be added if present.)

---

## Task 2: Pure logic — board model + eval marshalling (TDD)

**Files:** create `src/lib/types.ts`, `src/lib/board.ts`, `src/lib/board.test.ts`, `src/lib/evalRunner.ts`, `src/lib/evalRunner.test.ts`.

- [ ] **Step 1: Create `src/lib/types.ts`**

```ts
export type Orientation = "h" | "v";

export type EnrichedAction =
  | { kind: "move"; index: number; to: [number, number] }
  | { kind: "wall"; index: number; row: number; col: number; orientation: Orientation };

export interface WallEntry { row: number; col: number; orientation: Orientation }

export interface StateView {
  board_size: number;
  max_walls: number;
  max_steps: number;
  current_player: number;
  p1_pos: [number, number];
  p2_pos: [number, number];
  p1_walls: number;
  p2_walls: number;
  walls: WallEntry[];
  legal_actions: EnrichedAction[];
  completed_steps: number;
  winner: number | null;
  human_player: number;
  last_action: EnrichedAction | null;
  move_history: number[];
}

export interface SearchResult {
  action: number;
  rootValue: number;
  children: { actionIndex: number; visitCount: number }[];
}
```

- [ ] **Step 2: Write `src/lib/board.test.ts` (failing)**

```ts
import { describe, expect, it } from "vitest";
import { deriveBoard } from "./board";
import type { StateView } from "./types";

function baseView(over: Partial<StateView> = {}): StateView {
  return {
    board_size: 5, max_walls: 2, max_steps: 50, current_player: 0,
    p1_pos: [4, 2], p2_pos: [0, 2], p1_walls: 2, p2_walls: 2,
    walls: [], legal_actions: [], completed_steps: 0, winner: null,
    human_player: 0, last_action: null, move_history: [],
    ...over,
  };
}

describe("deriveBoard", () => {
  it("maps legal move destinations to their action index", () => {
    const view = baseView({
      legal_actions: [
        { kind: "move", index: 7, to: [3, 2] },
        { kind: "wall", index: 30, row: 1, col: 1, orientation: "v" },
      ],
    });
    const b = deriveBoard(view);
    expect(b.moveActionAt(3, 2)).toBe(7);
    expect(b.moveActionAt(0, 0)).toBeUndefined();
    expect(b.wallActionAt(1, 1, "v")).toBe(30);
  });

  it("exposes pawn positions and placed walls", () => {
    const view = baseView({ walls: [{ row: 2, col: 2, orientation: "h" }] });
    const b = deriveBoard(view);
    expect(b.pawns[0]).toEqual([4, 2]);
    expect(b.pawns[1]).toEqual([0, 2]);
    expect(b.walls).toHaveLength(1);
    expect(b.size).toBe(5);
  });
});
```

- [ ] **Step 3: Implement `src/lib/board.ts`**

```ts
import type { Orientation, StateView, WallEntry } from "./types";

export interface BoardModel {
  size: number;
  pawns: [[number, number], [number, number]];
  walls: WallEntry[];
  /** Action index for the legal move landing on (r,c), or undefined. */
  moveActionAt(r: number, c: number): number | undefined;
  /** Action index for the legal wall at (r,c,orientation), or undefined. */
  wallActionAt(r: number, c: number, o: Orientation): number | undefined;
}

export function deriveBoard(view: StateView): BoardModel {
  const moves = new Map<string, number>();
  const walls = new Map<string, number>();
  for (const a of view.legal_actions) {
    if (a.kind === "move") moves.set(`${a.to[0]},${a.to[1]}`, a.index);
    else walls.set(`${a.row},${a.col},${a.orientation}`, a.index);
  }
  return {
    size: view.board_size,
    pawns: [view.p1_pos, view.p2_pos],
    walls: view.walls,
    moveActionAt: (r, c) => moves.get(`${r},${c}`),
    wallActionAt: (r, c, o) => walls.get(`${r},${c},${o}`),
  };
}
```

- [ ] **Step 4: Write `src/lib/evalRunner.test.ts` (failing)**

```ts
import { describe, expect, it } from "vitest";
import { runEval } from "./evalRunner";

// A fake ORT-like session: records the input tensor and returns fixed outputs.
function fakeSession(n: number, policy: number) {
  return {
    lastInput: null as any,
    async run(feeds: any) {
      this.lastInput = feeds.input;
      return {
        value: { data: Float32Array.from({ length: n }, (_, i) => i * 0.1) },
        policy_logits: { data: new Float32Array(n * policy) },
      };
    },
  };
}

// Minimal Tensor stand-in so runEval doesn't need the real ort in unit tests.
class FakeTensor {
  constructor(public type: string, public data: Float32Array, public dims: number[]) {}
}

describe("runEval", () => {
  it("builds an [n,c,h,w] input tensor and splits value/logits", async () => {
    const n = 2, c = 5, h = 13, w = 13, policy = 57;
    const session = fakeSession(n, policy);
    const flat = new Float32Array(n * c * h * w);
    const out = await runEval(session as any, FakeTensor as any, flat, n, c, h, w);

    expect(session.lastInput.dims).toEqual([n, c, h, w]);
    expect(session.lastInput.data).toBe(flat);
    expect(out.values).toHaveLength(n);
    expect(out.values[1]).toBeCloseTo(0.1);
    expect(out.logits).toHaveLength(n * policy);
  });
});
```

- [ ] **Step 5: Implement `src/lib/evalRunner.ts`**

```ts
// Pure marshalling between quoridor-wasm's eval_batch(flat, n, c, h, w) contract
// and an onnxruntime-web InferenceSession. Kept free of a hard `ort` import so it
// is unit-testable with a fake session + tensor.

export interface OrtLikeSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: Float32Array }>>;
}
export type TensorCtor = new (type: "float32", data: Float32Array, dims: number[]) => unknown;

export async function runEval(
  session: OrtLikeSession,
  Tensor: TensorCtor,
  flat: Float32Array,
  n: number,
  c: number,
  h: number,
  w: number,
): Promise<{ values: Float32Array; logits: Float32Array }> {
  const input = new Tensor("float32", flat, [n, c, h, w]);
  const out = await session.run({ input });
  return {
    values: out.value.data,
    logits: out.policy_logits.data,
  };
}
```

- [ ] **Step 6: Run the tests**

```bash
npm --prefix deep_quoridor/frontend run test 2>&1 | tail -12
```
Expected: all pass (smoke + board 2 + evalRunner 1).

- [ ] **Step 7: Commit**

```bash
git -C deep_quoridor add frontend/src/lib/types.ts frontend/src/lib/board.ts frontend/src/lib/board.test.ts frontend/src/lib/evalRunner.ts frontend/src/lib/evalRunner.test.ts
git -C deep_quoridor commit -m "feat(frontend): pure board model + eval marshalling with vitest"
```

---

## Task 3: API client (TDD)

**Files:** create `src/lib/api.ts`, `src/lib/api.test.ts`.

- [ ] **Step 1: Write `src/lib/api.test.ts` (failing)**

```ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchConfig, fetchModels } from "./api";

afterEach(() => vi.unstubAllGlobals());

function stubFetch(body: unknown) {
  vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => body })));
}

describe("api", () => {
  it("fetchConfig returns the parsed config", async () => {
    stubFetch({ board_size: 5, max_walls: 2, max_steps: 50, defaults: { mcts_n: 100 } });
    const cfg = await fetchConfig();
    expect(cfg.board_size).toBe(5);
    expect(cfg.defaults.mcts_n).toBe(100);
  });

  it("fetchModels returns models + default", async () => {
    stubFetch({ models: ["model_1.onnx", "model_2.onnx"], default: "model_2.onnx" });
    const m = await fetchModels();
    expect(m.models).toEqual(["model_1.onnx", "model_2.onnx"]);
    expect(m.default).toBe("model_2.onnx");
  });
});
```

- [ ] **Step 2: Implement `src/lib/api.ts`**

```ts
export interface ConfigView {
  board_size: number;
  max_walls: number;
  max_steps: number;
  defaults: {
    mcts_n: number;
    mcts_c_puct: number;
    temperature: number | null;
    mcts_noise_epsilon: number;
    mcts_noise_alpha: number | null;
    leaf_parallelism: number;
    virtual_loss: number;
    mcts_worker_threads: number | null;
  };
}
export interface ModelsView { models: string[]; default: string | null }

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return (await r.json()) as T;
}

export const fetchConfig = () => getJson<ConfigView>("/api/config");
export const fetchModels = () => getJson<ModelsView>("/api/models");
```

- [ ] **Step 3: Run tests, then commit**

```bash
npm --prefix deep_quoridor/frontend run test 2>&1 | tail -8
git -C deep_quoridor add frontend/src/lib/api.ts frontend/src/lib/api.test.ts
git -C deep_quoridor commit -m "feat(frontend): API client for /api/config and /api/models"
```
Expected: tests pass.

---

## Task 4: The AI Web Worker + main-thread client

**Files:** create `src/ai.worker.ts`, `src/lib/aiClient.ts`. (Integration glue — verified by build + the end-to-end run in Task 6; the marshalling it relies on is already unit-tested in Task 2.)

- [ ] **Step 1: Create `src/ai.worker.ts`**

```ts
/// <reference lib="webworker" />
import initWasm, { Game, init as installPanicHook } from "quoridor-wasm";
import * as ort from "onnxruntime-web/webgpu";
import { runEval } from "./lib/evalRunner";
import type { StateView } from "./lib/types";

// Serve ORT's wasm/mjs from our own origin (copied there by vite-plugin-static-copy).
ort.env.wasm.wasmPaths = "/ort/";

type Params = {
  mctsN: number; cPuct: number; leafParallelism: number; virtualLoss: number;
};

let game: Game | null = null;
let session: ort.InferenceSession | null = null;
let params: Params = { mctsN: 200, cPuct: 1.4, leafParallelism: 8, virtualLoss: 1 };
let wasmReady: Promise<void> | null = null;

function ensureWasm(): Promise<void> {
  if (!wasmReady) wasmReady = initWasm().then(() => installPanicHook());
  return wasmReady;
}

async function loadSession(model: string) {
  session = await ort.InferenceSession.create(`/models/${model}`, {
    executionProviders: ["webgpu", "wasm"],
  });
}

function post(msg: unknown) { (self as unknown as Worker).postMessage(msg); }

async function evalBatch(flat: Float32Array, n: number, c: number, h: number, w: number) {
  if (!session) throw new Error("no model session");
  return runEval(session, ort.Tensor as never, flat, n, c, h, w);
}

function progress(done: number, total: number) {
  post({ type: "progress", done, total });
}

async function aiMoveIfNeeded(view: StateView): Promise<StateView> {
  if (view.winner !== null || view.current_player === view.human_player) return view;
  const res = await game!.runSearch(
    params.mctsN, params.cPuct, params.leafParallelism, params.virtualLoss,
    evalBatch as unknown as Function, progress as unknown as Function,
  );
  return game!.applyAction(res.action) as StateView;
}

self.onmessage = async (e: MessageEvent) => {
  const m = e.data;
  try {
    if (m.type === "newGame") {
      await ensureWasm();
      params = m.params;
      await loadSession(m.model);
      game = new Game(m.boardSize, m.maxWalls, m.maxSteps, m.humanPlayer);
      let view = game.stateView() as StateView;
      view = await aiMoveIfNeeded(view); // AI opens if human is player 2
      post({ type: "state", view });
    } else if (m.type === "move") {
      let view = game!.applyAction(m.index) as StateView;
      post({ type: "state", view });      // show the human move immediately
      view = await aiMoveIfNeeded(view);
      post({ type: "state", view });
    } else if (m.type === "undo") {
      const view = game!.undo(m.count) as StateView;
      post({ type: "state", view });
    } else if (m.type === "setParams") {
      params = m.params;
    }
  } catch (err) {
    post({ type: "error", message: String(err) });
  }
};
```

- [ ] **Step 2: Create `src/lib/aiClient.ts` (main-thread wrapper)**

```ts
import type { StateView } from "./types";

type Params = { mctsN: number; cPuct: number; leafParallelism: number; virtualLoss: number };

export class AiClient {
  private worker: Worker;
  onState?: (v: StateView) => void;
  onProgress?: (done: number, total: number) => void;
  onError?: (msg: string) => void;

  constructor() {
    this.worker = new Worker(new URL("../ai.worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e: MessageEvent) => {
      const m = e.data;
      if (m.type === "state") this.onState?.(m.view);
      else if (m.type === "progress") this.onProgress?.(m.done, m.total);
      else if (m.type === "error") this.onError?.(m.message);
    };
  }

  newGame(o: {
    model: string; boardSize: number; maxWalls: number; maxSteps: number;
    humanPlayer: number; params: Params;
  }) { this.worker.postMessage({ type: "newGame", ...o }); }

  move(index: number) { this.worker.postMessage({ type: "move", index }); }
  undo(count: number) { this.worker.postMessage({ type: "undo", count }); }
  setParams(params: Params) { this.worker.postMessage({ type: "setParams", params }); }
}
```

- [ ] **Step 3: Type-check + build (no new unit tests here — glue is covered by Task 6's run)**

```bash
npm --prefix deep_quoridor/frontend run build 2>&1 | tail -15
```
Expected: `vite build` succeeds (bundles the worker + wasm + ORT). If the worker import of `quoridor-wasm` fails to resolve, confirm Task 1 built `pkg/` and that `package.json`'s `file:../rust/quoridor-wasm/pkg` dependency installed (re-run `npm install`). If Vite errors on the `?worker`/`new URL` worker, ensure `worker.format: "es"` is set (Task 1 vite.config).

- [ ] **Step 4: Commit**

```bash
git -C deep_quoridor add frontend/src/ai.worker.ts frontend/src/lib/aiClient.ts
git -C deep_quoridor commit -m "feat(frontend): AI web worker (wasm+ORT) and main-thread client"
```

---

## Task 5: Svelte UI — board, control rail, config drawer, app shell

**Files:** create `src/lib/Board.svelte`, `src/lib/ControlRail.svelte`, `src/lib/ConfigDrawer.svelte`; rewrite `src/App.svelte`.

- [ ] **Step 1: Create `src/lib/Board.svelte`**

```svelte
<script lang="ts">
  import type { StateView } from "./types";
  import { deriveBoard } from "./board";
  let { view, disabled, onaction }: {
    view: StateView; disabled: boolean; onaction: (index: number) => void;
  } = $props();

  const board = $derived(deriveBoard(view));
  const cells = $derived(
    Array.from({ length: board.size }, (_, r) =>
      Array.from({ length: board.size }, (_, c) => ({ r, c })),
    ),
  );
  function pawnAt(r: number, c: number): number | null {
    if (board.pawns[0][0] === r && board.pawns[0][1] === c) return 0;
    if (board.pawns[1][0] === r && board.pawns[1][1] === c) return 1;
    return null;
  }
</script>

<div class="board" style="--n:{board.size}">
  {#each cells as row}
    {#each row as { r, c }}
      {@const move = board.moveActionAt(r, c)}
      {@const pawn = pawnAt(r, c)}
      <button
        class="cell"
        class:legal={move !== undefined}
        class:p1={pawn === 0}
        class:p2={pawn === 1}
        disabled={disabled || move === undefined}
        onclick={() => move !== undefined && onaction(move)}
      >{pawn === 0 ? "●" : pawn === 1 ? "○" : ""}</button>
    {/each}
  {/each}
</div>
<!-- Wall placement: list legal wall actions as buttons (functional M1 UI). -->
<div class="walls">
  {#each view.legal_actions.filter((a) => a.kind === "wall") as a}
    <button disabled={disabled} onclick={() => onaction(a.index)}>
      wall {a.kind === "wall" ? `${a.orientation} @${a.row},${a.col}` : ""}
    </button>
  {/each}
</div>

<style>
  .board { display: grid; grid-template-columns: repeat(var(--n), 44px); gap: 3px; }
  .cell { width: 44px; height: 44px; font-size: 1.4rem; }
  .cell.legal { outline: 2px dashed #2a7; }
  .cell.p1 { background: #c0392b; color: #fff; }
  .cell.p2 { background: #2980b9; color: #fff; }
  .walls { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 4px; max-width: 360px; }
  .walls button { font-size: 0.75rem; }
</style>
```

- [ ] **Step 2: Create `src/lib/ControlRail.svelte`**

```svelte
<script lang="ts">
  import type { StateView } from "./types";
  let { view, thinking, progress, onundo, onnewgame }: {
    view: StateView | null;
    thinking: boolean;
    progress: { done: number; total: number } | null;
    onundo: () => void;
    onnewgame: () => void;
  } = $props();
  const pct = $derived(progress && progress.total ? Math.round((100 * progress.done) / progress.total) : 0);
</script>

<div class="rail">
  {#if thinking}
    <div class="card">
      <strong>AI thinking…</strong>
      <div class="bar"><div class="fill" style="width:{pct}%"></div></div>
      <small>{progress?.done ?? 0} / {progress?.total ?? 0} sims</small>
    </div>
  {/if}
  {#if view?.winner != null}
    <div class="card"><strong>{view.winner === view.human_player ? "You win!" : "AI wins"}</strong></div>
  {/if}
  <button onclick={onundo} disabled={thinking || !view || view.move_history.length === 0}>↶ Undo</button>
  <button onclick={onnewgame} disabled={thinking}>New game</button>
  {#if view}
    <div class="card">
      <strong>Walls</strong> — You {view.human_player === 0 ? view.p1_walls : view.p2_walls}
      · AI {view.human_player === 0 ? view.p2_walls : view.p1_walls}
    </div>
    <div class="card"><strong>Moves</strong>: {view.move_history.length}</div>
  {/if}
</div>

<style>
  .rail { display: flex; flex-direction: column; gap: 10px; width: 240px; }
  .card { border: 1px solid #ccc; border-radius: 6px; padding: 8px; }
  .bar { height: 12px; background: #ddd; border-radius: 6px; overflow: hidden; margin: 6px 0; }
  .fill { height: 100%; background: #2a7; }
</style>
```

- [ ] **Step 3: Create `src/lib/ConfigDrawer.svelte`**

```svelte
<script lang="ts">
  import type { ConfigView, ModelsView } from "./api";
  let { config, models, model, params, onchange }: {
    config: ConfigView | null;
    models: ModelsView | null;
    model: string;
    params: { mctsN: number; cPuct: number; leafParallelism: number; virtualLoss: number };
    onchange: (o: { model: string; params: typeof params }) => void;
  } = $props();
</script>

<div class="drawer">
  <h3>Setup</h3>
  <label>Model
    <select value={model} onchange={(e) => onchange({ model: e.currentTarget.value, params })}>
      {#each models?.models ?? [] as m}<option value={m}>{m}</option>{/each}
    </select>
  </label>
  <label>MCTS sims: {params.mctsN}
    <input type="range" min="16" max="2000" step="16" value={params.mctsN}
      oninput={(e) => onchange({ model, params: { ...params, mctsN: +e.currentTarget.value } })} />
  </label>
  <label>c_puct: {params.cPuct}
    <input type="range" min="0.5" max="3" step="0.1" value={params.cPuct}
      oninput={(e) => onchange({ model, params: { ...params, cPuct: +e.currentTarget.value } })} />
  </label>
  <label>leaf parallelism: {params.leafParallelism}
    <input type="range" min="1" max="32" step="1" value={params.leafParallelism}
      oninput={(e) => onchange({ model, params: { ...params, leafParallelism: +e.currentTarget.value } })} />
  </label>
</div>

<style>
  .drawer { display: flex; flex-direction: column; gap: 10px; width: 240px; }
  label { display: flex; flex-direction: column; font-size: 0.85rem; gap: 2px; }
</style>
```

- [ ] **Step 4: Rewrite `src/App.svelte` wiring it all together**

```svelte
<script lang="ts">
  import { onMount } from "svelte";
  import Board from "./lib/Board.svelte";
  import ControlRail from "./lib/ControlRail.svelte";
  import ConfigDrawer from "./lib/ConfigDrawer.svelte";
  import { AiClient } from "./lib/aiClient";
  import { fetchConfig, fetchModels, type ConfigView, type ModelsView } from "./lib/api";
  import type { StateView } from "./lib/types";

  let config = $state<ConfigView | null>(null);
  let models = $state<ModelsView | null>(null);
  let view = $state<StateView | null>(null);
  let thinking = $state(false);
  let progress = $state<{ done: number; total: number } | null>(null);
  let error = $state<string | null>(null);
  let model = $state("");
  let humanPlayer = $state(0);
  let params = $state({ mctsN: 200, cPuct: 1.4, leafParallelism: 8, virtualLoss: 1 });

  const ai = new AiClient();
  ai.onState = (v) => { view = v; thinking = false; progress = null; };
  ai.onProgress = (done, total) => { thinking = true; progress = { done, total }; };
  ai.onError = (m) => { error = m; thinking = false; };

  onMount(async () => {
    config = await fetchConfig();
    models = await fetchModels();
    model = models.default ?? models.models[0] ?? "";
    params = { ...params, mctsN: config.defaults.mcts_n, cPuct: config.defaults.mcts_c_puct };
    newGame();
  });

  function newGame() {
    if (!config || !model) return;
    error = null; thinking = true; progress = null;
    ai.newGame({
      model, boardSize: config.board_size, maxWalls: config.max_walls,
      maxSteps: config.max_steps, humanPlayer, params,
    });
  }
  function act(index: number) { thinking = true; ai.move(index); }
</script>

<div class="layout">
  <div>
    {#if error}<p class="err">Error: {error}</p>{/if}
    {#if view}
      <Board {view} disabled={thinking || view.winner != null} onaction={act} />
    {:else}
      <p>Loading…</p>
    {/if}
  </div>
  <ControlRail {view} {thinking} {progress} onundo={() => ai.undo(1)} onnewgame={newGame} />
  <ConfigDrawer {config} {models} {model} {params}
    onchange={(o) => { model = o.model; params = o.params; ai.setParams(o.params); }} />
</div>

<style>
  .layout { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }
  .err { color: #c0392b; }
</style>
```

- [ ] **Step 5: Type-check + build**

```bash
npm --prefix deep_quoridor/frontend run build 2>&1 | tail -15
```
Expected: `vite build` succeeds and emits `dist/`. Fix any Svelte 5 `$props`/`$state` rune or TS errors it reports.

- [ ] **Step 6: Commit**

```bash
git -C deep_quoridor add frontend/src/App.svelte frontend/src/lib/Board.svelte frontend/src/lib/ControlRail.svelte frontend/src/lib/ConfigDrawer.svelte
git -C deep_quoridor commit -m "feat(frontend): board, control rail, config drawer, app shell"
```

---

## Task 6: End-to-end wiring + run instructions

**Files:** none new (integration + docs go in Task 7's README). This task proves the whole stack serves and the API contract lines up.

- [ ] **Step 1: Build both artifacts**

```bash
<PREFIX> wasm-pack build rust/quoridor-wasm --target web --release 2>&1 | tail -3
npm --prefix deep_quoridor/frontend run build 2>&1 | tail -6
```
Expected: `frontend/dist/` contains `index.html`, hashed JS/CSS, the worker chunk, `quoridor_wasm_bg.wasm`, and an `ort/` dir with onnxruntime `.wasm`/`.mjs`. Verify:
```bash
ls deep_quoridor/frontend/dist/ && ls deep_quoridor/frontend/dist/ort/ | head
```

- [ ] **Step 2: Prepare a run directory the server can read (config + a model)**

The server needs a `config.yaml` (5×5, to match the fixture model) and a `models/checkpoints/*.onnx`. Build a scratch run dir pointing the models dir at the fixture:
```bash
RUN=$(mktemp -d)
printf 'run_id: play\nquoridor:\n  board_size: 5\n  max_walls: 2\n  max_steps: 50\nalphazero:\n  mcts_n: 200\n  mcts_c_puct: 1.4\nself_play:\n  num_processes: 1\n  games_per_process: 1\ntraining:\n  games_per_training_step: 1.0\n  learning_rate: 0.001\n  batch_size: 64\n  weight_decay: 0.0001\n  replay_buffer_size: 1000\n' > "$RUN/config.yaml"
mkdir -p "$RUN/models/checkpoints"
cp deep_quoridor/rust/fixtures/alphazero_B5W2_mv1.onnx "$RUN/models/checkpoints/model_1.onnx"
echo "$RUN"
```

- [ ] **Step 3: Start the server and verify the API + static wiring line up (curl)**

```bash
PYTHONPATH=$(pwd)/deep_quoridor/src timeout 8 python deep_quoridor/src/run_play_server_web.py "$RUN" \
  --static-dir deep_quoridor/frontend/dist --port 8139 >/tmp/claude-1000/play.log 2>&1 &
sleep 4
echo "config:"; curl -s http://127.0.0.1:8139/api/config
echo; echo "models:"; curl -s http://127.0.0.1:8139/api/models
echo; echo "index served + COOP:"; curl -s -D - -o /dev/null http://127.0.0.1:8139/ | grep -iE "HTTP/|cross-origin"
echo "model file:"; curl -s -o /dev/null -w "%{http_code} %{content_type}\n" http://127.0.0.1:8139/models/model_1.onnx
echo "an ORT wasm asset:"; curl -s -o /dev/null -w "%{http_code}\n" "http://127.0.0.1:8139/ort/$(ls deep_quoridor/frontend/dist/ort/ | grep '\.wasm$' | head -1)"
wait 2>/dev/null
```
Expected: `/api/config` returns board_size 5; `/api/models` lists `model_1.onnx` (default `model_1.onnx`); `/` returns 200 with COOP/COEP headers; `/models/model_1.onnx` is 200; the ORT wasm asset is 200. (This confirms every URL the browser will hit exists and is same-origin. The actual gameplay — WebGPU inference, the progress bar, clicking moves — is a manual browser check, documented in the README, since it needs a real browser with WebGPU/WASM.)

- [ ] **Step 4: (No commit — this task creates no files.)** Record the curl results in your report.

---

## Task 7: README + final verification sweep

**Files:** create `deep_quoridor/frontend/README.md`.

- [ ] **Step 1: Write `deep_quoridor/frontend/README.md`**

```markdown
# Quoridor frontend (Svelte + Web Worker + onnxruntime-web)

The browser play app. A Web Worker runs the `quoridor-wasm` MCTS and evaluates the
net with onnxruntime-web (WebGPU, wasm-CPU fallback); the main thread renders the
board and streams the AI's "thinking" progress. Served by the Python play server
(`src/run_play_server_web.py`). Design: `docs/superpowers/specs/2026-07-05-browser-wasm-play-server-design.md`.

## Build

```
# 1. Build the wasm package (Plan 1):
wasm-pack build rust/quoridor-wasm --target web --release
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
For a quick try with the bundled fixture model, make a scratch run dir with a 5×5
`config.yaml` and copy `rust/fixtures/alphazero_B5W2_mv1.onnx` to
`<run>/models/checkpoints/model_1.onnx` (see the plan's Task 6).

Dev mode with HMR: `npm --prefix deep_quoridor/frontend run dev` (Vite serves the
COOP/COEP headers itself; the model/API still need the Python server, so use a Vite
proxy or run the built app behind the Python server for full play).

## Tests
```
npm --prefix deep_quoridor/frontend run test   # vitest: board model, eval marshalling, api client
```
Unit tests cover the pure logic (board derivation, eval tensor marshalling, API
client). End-to-end gameplay (WebGPU inference, progress bar, clicking) is a manual
browser check — modern browser with WebGPU or a wasm-CPU fallback.
```

- [ ] **Step 2: Full verification sweep**

```bash
npm --prefix deep_quoridor/frontend run test 2>&1 | tail -10
npm --prefix deep_quoridor/frontend run build 2>&1 | tail -6
```
Expected: all vitest tests pass; build succeeds.

- [ ] **Step 3: Commit**

```bash
git -C deep_quoridor add frontend/README.md
git -C deep_quoridor commit -m "docs(frontend): README with build + run instructions"
```

---

## Self-review checklist (run after writing all tasks)

- **Spec coverage:** browser AI via worker + wasm ✓ (Task 4); WebGPU inference via ORT ✓ (Task 4, `executionProviders:["webgpu","wasm"]`); responsive UI + undo + real-time progress bar ✓ (Task 5); model + AlphaZero params exposed/editable ✓ (Task 5 ConfigDrawer, from `/api/config`); ORT served same-origin under COEP ✓ (Task 1 static-copy + Task 4 `wasmPaths`); served by the Python server ✓ (Task 6).
- **Type consistency:** `StateView`/`EnrichedAction` (types.ts) match the wasm `stateView()` shape and Plan 1's `view.rs`; `runEval` output `{values,logits}` matches `search.rs`'s `split_outputs` expectation; `AiClient` message types match `ai.worker.ts`.
- **Deferred (out of M1 scope):** polished wall-placement interaction (M1 lists legal walls as buttons); the full yaml-tree config editor (M1 exposes model + a few key params); IndexedDB model caching; the M2 visualizations (policy heatmap, live MCTS tree) — the worker `state`/`progress` messages and `runSearch`'s `children` are the hooks.

## Follow-ups (not built here)
- Replace the wall-button list with click-on-edge wall placement.
- Vite dev proxy to the Python server so `npm run dev` gives full play with HMR.
- Retire the old Rust `play_server` now that the browser path is complete end-to-end.
