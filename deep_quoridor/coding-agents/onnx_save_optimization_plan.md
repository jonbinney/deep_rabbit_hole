# ONNX Save Optimization Plan: Cache ONNX Graph, Update Weights Only

## Problem

`save_model_onnx()` calls `torch.onnx.export()` on every save, re-tracing the full
computation graph each time even though the architecture never changes.

## Proposed Fix

Run the full export once, cache the ONNX protobuf in memory, then for all subsequent
saves replace only the weight tensors (ONNX initializers) and re-serialize — skipping
graph tracing entirely.

## Steps

- [ ] Write plan to file — commit: `vibe: add onnx save optimization plan`
- [ ] Baseline benchmark — run `test_onnx_export.yaml` with `model_save_timing: true`, record per-save timings
- [ ] Verify `onnx` package is available — check `requirements.txt`, add if missing
- [ ] Implement cached ONNX save in `alphazero.py`:
  - In `__init__`: add `self._onnx_proto = None` and `self._onnx_init_name_to_idx: dict = {}`
  - First call to `save_model_onnx`: run existing `torch.onnx.export(...)`, then load proto and build name→index map
  - Subsequent calls: update initializers directly from `state_dict()`, then `onnx.save()`
- [ ] Commit functional change: `vibe: cache ONNX proto and update initializers on subsequent saves`
- [ ] Post-implementation benchmark — run same test YAML, record per-save timings
- [ ] Write timing results to `onnx_save_optimization_results.md` — commit: `vibe: add onnx save optimization benchmark results`
- [ ] Formatting/linting — run ruff, commit: `vibe: formatting for onnx save optimization`

## Implementation Notes

- `do_constant_folding=True` is kept; PyTorch's exporter preserves all learned parameters
  as explicit ONNX initializers regardless, so the name mapping is complete for all
  trainable weights.
- "Cache proto + update initializers" chosen over `torch.jit.trace` reuse — the latter
  still rebuilds the ONNX graph on each export; the former bypasses graph construction
  entirely after the first save.
- Existing `model_save_timing` flag used for benchmarking — no new standalone script needed.
- No strict numerical output comparison.
- Two separate commits per change (functional + formatting) as per spec.

## Verification

- Confirm `model_save_timing` logs show reduced time on saves 2+
- Load each saved `.onnx` with `onnx.checker.check_model(path)` to confirm structural validity
- Run one inference through `onnxruntime` per saved model to confirm no crash
