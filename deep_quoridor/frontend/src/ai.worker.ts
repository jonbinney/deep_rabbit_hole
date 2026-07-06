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
