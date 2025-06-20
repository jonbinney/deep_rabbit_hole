# AlphaZero Quoridor Implementation: Issues and Solutions

This document summarizes the key challenges encountered while implementing AlphaZero for Quoridor using OpenSpiel, and how they were resolved.

## Core Implementation Challenges

### 1. Neural Network Architecture Mismatch

**Problem**: Initially implemented ResNet architecture, which is designed for 2D grid observations (like chess or Go), but Quoridor in OpenSpiel represents observations as flat tensors, not spatial grids.

**Solution**: Switched to MLP (Multi-Layer Perceptron) architecture by changing `nn_model` parameter to "mlp" instead of "resnet". MLPs are better suited for flat vector inputs.

### 2. Observation Shape Handling

**Problem**: TensorFlow was encountering TensorShape errors because we were trying to specify an explicit observation shape when creating the AlphaZero config.

**Solution**: Set `observation_shape=None` to let AlphaZero infer the shape automatically from the game. This allowed the model to properly adapt to Quoridor's observation representation.

### 3. JSON Serialization Errors

**Problem**: The script crashed with JSON serialization errors when trying to log training progress, due to NumPy data types (float32, int64, etc.) not being JSON-serializable.

**Solution**: Implemented a custom `NumpyEncoder` class and monkey-patched it into Python's json module to handle NumPy types properly:

```python
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Monkey patch the json module
json._default_encoder = NumpyEncoder()
```

### 4. Bot Creation and Evaluation Issues

**Problem**: The bot creation function had scoping issues with the `game` variable and was using incorrect API calls to load models from checkpoints.

**Solution**: 
1. Restructured the evaluation code to explicitly pass the game instance to all functions
2. Fixed model loading by using the correct method: `az_model.Model.from_checkpoint(checkpoint_path)`
3. Created proper AlphaZeroEvaluator instances for the MCTS bot

The corrected bot creation function:

```python
def create_bot_from_checkpoint(checkpoint_path, game):
    """Creates an AlphaZero MCTS bot using a checkpoint."""
    # Load the model from checkpoint
    model = az_model.Model.from_checkpoint(checkpoint_path)

    # Create the AlphaZero evaluator with the loaded model
    evaluator = az_evaluator.AlphaZeroEvaluator(game, model)

    # Create the MCTS bot with the AlphaZero evaluator
    bot = mcts.MCTSBot(
        game,
        2.0,  # uct_c parameter - reasonable default
        800,  # max_simulations parameter - reasonable default
        evaluator,
        random_state=np.random.RandomState(42),
        solve=False,
        verbose=FLAGS.verbose,
    )
    return bot
```

### 5. GPU/CUDA Initialization Errors

**Problem**: On systems without compatible GPUs or proper CUDA configuration, TensorFlow would crash with CUDA initialization errors.

**Solution**: Force TensorFlow to use CPU only by setting an environment variable at the start of the script:

```python
# Force TensorFlow to use CPU only to avoid CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

### 6. Import Path Errors

**Problem**: Several import errors were encountered, particularly with the `uniform_random` module.

**Solution**: Fixed import paths to use the correct module locations in OpenSpiel:
```python
from open_spiel.python.bots import uniform_random  # Not in algorithms
```

### 7. Evaluation Without Retraining

**Problem**: Each time we wanted to evaluate the model, it required retraining from scratch, which was time-consuming.

**Solution**: Added an `--eval_only` flag and modified the main function to support evaluation without training:

```python
# In FLAGS definition
flags.DEFINE_boolean("eval_only", False, "Skip training and only evaluate an existing model.")

# In main function
if not FLAGS.eval_only:
    # Run training
    with spawn.main_handler():
        alpha_zero.alpha_zero(config)
else:
    # Skip training, just evaluate
    print(f"Skipping training, evaluating existing model at {final_checkpoint}")
```

## Best Practices and Lessons Learned

1. **Observation Space Understanding**: Always check how the game represents observations (flat vs. spatial) before choosing a neural network architecture.

2. **Error Handling**: Add proper error handling for serialization of custom types, especially when working with numerical libraries.

3. **Resource Management**: Provide options to control GPU usage and offer CPU fallbacks.

4. **Separation of Concerns**: Separate training and evaluation to allow for faster testing cycles.

5. **API Understanding**: Thoroughly understand the OpenSpiel AlphaZero API, particularly how models are saved and loaded.

6. **Flexibility**: Design your code to be configurable with command-line flags for key parameters.

7. **Explicit Game Instance**: Always explicitly pass the game instance to functions instead of relying on global variables.

## Next Steps and Improvements

1. Implement more sophisticated evaluation metrics beyond win rates

2. Add visualization of the learned policy and value networks

3. Add support for different board sizes and wall counts

4. Implement a tournament mode to compare different versions of trained agents

5. Add progressive learning by starting with smaller board sizes and increasing complexity
