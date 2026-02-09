# Quoridor ONNX Inference in Rust

This Rust binary demonstrates how to run inference on a Quoridor AlphaZero model exported to ONNX format.

## Prerequisites

- Rust toolchain (cargo)
- CUDA-capable GPU
- ONNX Runtime with CUDA support

## Model

The binary expects a Quoridor model saved in ONNX format. By default, it looks for `model_6.onnx` in the parent directory (`../model_6.onnx`).

To generate a model:
1. Train a model using `deep_quoridor/src/train_v2.py` with ONNX export enabled
2. Copy one of the generated `.onnx` files from the training run to `experiments/onnx/`
3. Update the `model_path` in `src/bin/infer_quoridor.rs` if needed

## Building

```bash
cargo build --release --bin infer_quoridor
```

## Running

### Using the convenience script:
```bash
./run_quoridor.sh
```

### Manually:
```bash
cargo run --release --bin infer_quoridor
```

## What it Does

1. Loads the ONNX model with CUDA execution provider (GPU acceleration)
2. Creates an initial Quoridor game state (5x5 board, 1 wall per player)
3. Converts the game state to the neural network's input format
4. Runs inference 1000 times for benchmarking
5. Displays:
   - Average inference time
   - Position evaluation (value)
   - Top 5 recommended actions with probabilities
   - Action space breakdown

## Model Input Format

The model expects a flattened tensor with the following structure:
- Player board (25 elements for 5x5): 1.0 where current player is located, 0.0 elsewhere
- Opponent board (25 elements): 1.0 where opponent is located, 0.0 elsewhere  
- Walls (32 elements for 4x4x2): Wall placement grid with separate channels for horizontal (16) and vertical (16) walls
- Wall counts (2 elements): [current_player_walls_remaining, opponent_walls_remaining]

Total input size: 84 elements for a 5x5 board with 1 wall per player.

## Model Outputs

- **policy_logits**: Raw logits for all possible actions (57 actions for 5x5 board)
  - 25 movement actions
  - 16 horizontal wall placements  
  - 16 vertical wall placements
  
- **value**: Position evaluation from -1 to 1
  - Positive values favor the current player
  - Negative values favor the opponent

## Example Output

```
Loading Quoridor model from ../model_6.onnx...
Board size: 5x5, Max walls: 1

✓ Model loaded successfully!
✓ CUDA execution provider configured (GPU acceleration enabled)

=== Initial Game State ===
Current player: 0
Player 0 position: [0, 2]
Player 1 position: [4, 2]
Walls remaining: [1, 1]

Input tensor shape: [1, 84]
Input tensor size: 84

Running inference 1000 times...
✓ Completed 1000 inferences
Total time: 0.4904s
Average time per inference: 0.000490s (2039.15 inferences/sec)

=== Model Output ===
Policy logits length: 57
Value length: 1
Position value: -0.0188 (range: -1 to 1, positive favors current player)

=== Top 5 Recommended Actions ===
1. Action index: 8, Probability: 0.0204
2. Action index: 6, Probability: 0.0191
3. Action index: 3, Probability: 0.0191
4. Action index: 7, Probability: 0.0187
5. Action index: 1, Probability: 0.0185

=== Action Space ===
Total action space size: 57
- Movement actions: 25 (indices 0-24)
- Horizontal wall actions: 16 (indices 25-40)
- Vertical wall actions: 16 (indices 41-56)
```

## Configuration

The model configuration is hardcoded in the binary and must match the trained model:
- `board_size = 5`
- `max_walls = 1`

If you're using a different board configuration, update these constants in `src/bin/infer_quoridor.rs`.
