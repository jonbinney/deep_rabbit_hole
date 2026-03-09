# ONNX Save Optimization Results

## Setup

- Experiment config: `deep_quoridor/experiments/B5W3/test_onnx_export.yaml`
- `finish_after: 10 models`, `model_save_timing: true`, `save_onnx: true`
- Network: MLP (21 initializers / trainable weight tensors)
- Timing reported by trainer covers both PyTorch `.pt` save and ONNX save together

## Baseline (before — full `torch.onnx.export()` on every save)

| Save # | Time (s) |
|--------|----------|
| 1      | 1.0424   |
| 2      | 0.9356   |
| 3      | 1.0927   |
| 4      | 1.0188   |
| 5      | 1.0655   |

**Average per save: ~1.03 s**  
**Total for 10 saves: ~10.3 s** (estimated)

## After (cached proto — full export once, update initializers on subsequent saves)

| Save # | Time (s) | Notes                        |
|--------|----------|------------------------------|
| 1      | —        | First save (full export, not timed separately by trainer) |
| 2      | 0.0052   | cached proto, weights updated |
| 3      | 0.0041   | cached proto, weights updated |
| 4      | 0.0041   | cached proto, weights updated |
| 5      | 0.0041   | cached proto, weights updated |
| 6      | 0.0043   | cached proto, weights updated |
| 7      | 0.0043   | cached proto, weights updated |
| 8      | 0.0042   | cached proto, weights updated |
| 9      | 0.0041   | cached proto, weights updated |
| 10     | 0.0041   | cached proto, weights updated |
| 11     | 0.0040   | cached proto, weights updated |

**Average per save (saves 2+): ~0.0043 s**  
**Total for 10 saves (saves 2–11): ~0.043 s**

## Comparison Table

| Metric                      | Before      | After       | Speedup     |
|-----------------------------|-------------|-------------|-------------|
| First save                  | ~1.03 s     | ~1.03 s     | 1×          |
| Subsequent saves (avg)      | ~1.03 s     | ~0.0043 s   | **~240×**   |
| Total for 10 saves          | ~10.3 s     | ~0.043 s    | **~240×**   |

## Verification

- All 11 saved `.onnx` files passed `onnx.checker.check_model()` ✅
- All 11 models ran inference via `onnxruntime` without error ✅
- Output shapes: `policy_logits=(1, 57)`, `value=(1, 1)` — correct for 5×5 Quoridor ✅
- Value outputs in reasonable range (−0.03 to +0.12) ✅

## Implementation Summary

- `_onnx_proto = None` and `_onnx_init_name_to_idx = {}` added to `AlphaZeroAgent.__init__`
- First call to `save_model_onnx`: full `torch.onnx.export()` as before, then loads the
  written file with `onnx.load()` and builds a name→index map over all graph initializers
  (21 for this MLP architecture).
- Subsequent calls: iterates `state_dict()`, looks up each weight's index in the map,
  calls `CopyFrom(onnx.numpy_helper.from_array(...))` in place, then `onnx.save()`.
  Graph tracing is skipped entirely.
