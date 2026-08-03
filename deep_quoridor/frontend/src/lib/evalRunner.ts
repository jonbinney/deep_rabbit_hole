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
