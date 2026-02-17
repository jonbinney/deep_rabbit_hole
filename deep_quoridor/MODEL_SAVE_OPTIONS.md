# Model Save Format Options

## Summary

Added configuration options to save AlphaZero models in ONNX format during training. PyTorch format (.pt files) is always saved.

## Configuration Options

Add these parameters to the `training` section of your YAML configuration:

```yaml
training:
  # ... other training parameters ...
  model_save_timing: false  # Set to true to print timing information for model saving
  save_onnx: false          # Save models in ONNX format (.onnx files) - DEFAULT
```

## Default Behavior

- **PyTorch format**: Always enabled (cannot be disabled)
- **ONNX format**: Disabled by default (`save_onnx: false`)
- **Timing output**: Disabled by default (`model_save_timing: false`)

This ensures backward compatibility with existing configurations.

## Examples

### Example 1: Save only PyTorch format (default)
```yaml
training:
  save_onnx: false
```

### Example 2: Save both PyTorch and ONNX formats
```yaml
training:
  save_onnx: true
  model_save_timing: true  # See timing for both formats
```

## Test Configurations

Two test configurations are provided:

1. **`experiments/test_model_save_timing.yaml`** - Basic test with PyTorch only
2. **`experiments/test_onnx_export.yaml`** - Test with ONNX export enabled

## ONNX Export Details

The ONNX export includes:
- **Opset version**: 11 (good compatibility)
- **Dynamic batch size**: Models can handle variable batch sizes
- **Input name**: `input`
- **Output names**: `policy_logits` and `value`
- **Constant folding**: Enabled for optimization

## Files Modified

1. **`src/v2/config.py`**
   - Added `save_onnx` and `model_save_timing` to `TrainingConfig`

2. **`src/v2/trainer.py`**
   - Updated initial model save (model_0) to support ONNX format
   - Updated training loop model saves to support ONNX format
   - Enhanced timing output to show which formats were saved

3. **`src/agents/alphazero/alphazero.py`**
   - Added `save_model_onnx()` method for ONNX export

4. **Test configurations**
   - Created/updated test YAML files with examples

## Usage

Run training with any configuration:
```bash
python src/train_v2.py experiments/test_onnx_export.yaml
```

With timing enabled, you'll see output like:
```
Saving model (PyTorch and ONNX) took 0.1234s
```
