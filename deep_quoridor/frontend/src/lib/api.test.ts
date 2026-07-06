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
