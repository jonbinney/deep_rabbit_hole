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
