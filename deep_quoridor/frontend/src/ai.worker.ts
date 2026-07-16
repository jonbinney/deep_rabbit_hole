/// <reference lib="webworker" />
import initWasm, { Game, init as installPanicHook } from "quoridor-wasm";
import * as ort from "onnxruntime-web/webgpu";
import { runEval, type OrtLikeSession } from "./lib/evalRunner";
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
// Serializes game-mutating commands: while a move is in flight (MCTS + NN eval),
// later move/undo/newGame messages are dropped rather than racing the shared
// `game`/`session` state. The UI also gates on turn+thinking; this is defense-in-depth.
let busy = false;

function ensureWasm(): Promise<void> {
  if (!wasmReady) wasmReady = initWasm().then(() => installPanicHook());
  return wasmReady;
}

async function loadSession(model: string) {
  // Release the old session first so repeated New Game / model switches don't
  // leak GPU/WASM memory (ORT has no FinalizationRegistry backstop).
  if (session) { await session.release(); session = null; }
  session = await ort.InferenceSession.create(`/models/${model}`, {
    executionProviders: ["webgpu", "wasm"],
  });
  console.log("[ai] session ready:", session.inputNames, "->", session.outputNames);
}

function post(msg: unknown) { (self as unknown as Worker).postMessage(msg); }
function postState(view: StateView, thinking: boolean) { post({ type: "state", view, thinking }); }

async function evalBatch(flat: Float32Array, n: number, c: number, h: number, w: number) {
  if (!session) throw new Error("no model session");
  return runEval(session as unknown as OrtLikeSession, ort.Tensor as never, flat, n, c, h, w);
}

function progress(done: number, total: number) {
  post({ type: "progress", done, total });
}

/** Play one AI move if it's the AI's turn and the game isn't over (turns alternate). */
async function aiMoveIfNeeded(view: StateView): Promise<StateView> {
  // `!= null` catches both null and undefined (a None winner must not read as "over").
  if (view.winner != null || view.current_player === view.human_player) return view;
  console.log(`[ai] searching: mctsN=${params.mctsN} currentPlayer=${view.current_player} human=${view.human_player}`);
  const t0 = performance.now();
  const res = await game!.runSearch(
    params.mctsN, params.cPuct, params.leafParallelism, params.virtualLoss,
    evalBatch, progress,
  );
  console.log(`[ai] search done in ${Math.round(performance.now() - t0)}ms -> action ${res.action}`);
  return game!.applyAction(res.action) as StateView;
}

self.onmessage = async (e: MessageEvent) => {
  const m = e.data;
  // Param edits are safe any time and don't touch the game.
  if (m.type === "setParams") { params = m.params; return; }
  if (busy) return;
  busy = true;
  try {
    if (m.type === "newGame") {
      console.log("[ai] newGame model=", m.model);
      await ensureWasm();
      params = m.params;
      await loadSession(m.model);
      game?.free();
      game = new Game(m.boardSize, m.maxWalls, m.maxSteps, m.humanPlayer);
      let view = game.stateView() as StateView;
      view = await aiMoveIfNeeded(view); // AI opens if the human is player 2
      postState(view, false);
    } else if (m.type === "move") {
      let view = game!.applyAction(m.index) as StateView;
      postState(view, true);             // show the human move immediately; AI is about to think
      view = await aiMoveIfNeeded(view);
      postState(view, false);
    } else if (m.type === "undo") {
      let view = game!.undo(m.count) as StateView;
      view = await aiMoveIfNeeded(view); // if undo landed on the AI's turn, let it reply
      postState(view, false);
    }
  } catch (err) {
    console.error("[ai] error handling", m?.type, err);
    post({ type: "error", message: String(err) });
  } finally {
    busy = false;
  }
};
