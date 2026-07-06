import type { StateView } from "./types";

type Params = { mctsN: number; cPuct: number; leafParallelism: number; virtualLoss: number };

export class AiClient {
  private worker: Worker;
  onState?: (v: StateView, thinking: boolean) => void;
  onProgress?: (done: number, total: number) => void;
  onError?: (msg: string) => void;

  constructor() {
    this.worker = new Worker(new URL("../ai.worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e: MessageEvent) => {
      const m = e.data;
      if (m.type === "state") this.onState?.(m.view, m.thinking);
      else if (m.type === "progress") this.onProgress?.(m.done, m.total);
      else if (m.type === "error") this.onError?.(m.message);
    };
  }

  newGame(o: {
    model: string; boardSize: number; maxWalls: number; maxSteps: number;
    humanPlayer: number; params: Params;
  }) {
    // `params` may be a Svelte $state proxy, which postMessage can't structure-
    // clone (DataCloneError). Spread into a plain object first.
    this.worker.postMessage({ type: "newGame", ...o, params: { ...o.params } });
  }

  move(index: number) { this.worker.postMessage({ type: "move", index }); }
  undo(count: number) { this.worker.postMessage({ type: "undo", count }); }
  setParams(params: Params) { this.worker.postMessage({ type: "setParams", params: { ...params } }); }
}
