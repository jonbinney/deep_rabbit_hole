"""Standalone CLI script: prints step-by-step game trace for cross-language consistency testing.

Usage: python step_trace_reference.py <src_dir> <board_size> <max_walls> <action_idx_0> [action_idx_1 ...]

For each step (including step 0, the initial state, before any action), this script outputs:

  G,<step>,<grid_hex>        -- grid as hex-encoded int8 bytes (row-major)
  P,<step>,<p0r>,<p0c>,<p1r>,<p1c>  -- player positions
  W,<step>,<w0>,<w1>         -- walls remaining
  C,<step>,<current_player>  -- current player (0 or 1)
  M,<step>,<bitmask>         -- action mask as '1'/'0' string
  T,<step>,<tensor_hex>      -- ResNet input tensor (5,M,M) as hex-encoded float32 bytes

After all pre-action snapshots, one final snapshot is emitted for the state after
the last action (with step = len(actions)).

The rotation-related outputs (rotated mask, rotated tensor) are also emitted when
the current player is Player.TWO:

  RM,<step>,<bitmask>        -- action mask *after* rotating the board
  RT,<step>,<tensor_hex>     -- ResNet tensor *after* rotating the board
"""

import struct
import sys
from pathlib import Path

src_dir = Path(sys.argv[1])
board_size = int(sys.argv[2])
max_walls = int(sys.argv[3])
action_indices = [int(x) for x in sys.argv[4:]]

sys.path.insert(0, str(src_dir))

from quoridor import ActionEncoder, Board, Player, Quoridor
from agents.alphazero.resnet_network import ResnetNetwork, ResnetConfig

# We only need game_to_input_array which is a method on ResnetNetwork.
# Create a minimal instance (never used for inference).
encoder = ActionEncoder(board_size)

# Build a lightweight ResnetNetwork just for game_to_input_array.
# It doesn't need GPU or trained weights.
import torch

dummy_device = torch.device("cpu")
net = ResnetNetwork(encoder, dummy_device, ResnetConfig(num_blocks=1, num_channels=1))


def grid_to_hex(grid):
    """Encode a 2-D int ndarray as hex string of int8 bytes, row-major."""
    return grid.astype("int8").tobytes().hex()


def tensor_to_hex(arr):
    """Encode a float32 ndarray as hex string of its raw bytes, C-contiguous."""
    return arr.astype("float32").tobytes().hex()


def emit_snapshot(game, step):
    """Print all fields for the given step."""
    grid = game.board._grid
    p0 = game.board.get_player_position(Player.ONE)
    p1 = game.board.get_player_position(Player.TWO)
    w0 = int(game.board._walls_remaining[Player.ONE])
    w1 = int(game.board._walls_remaining[Player.TWO])
    cp = int(game.get_current_player())

    print(f"G,{step},{grid_to_hex(grid)}")
    print(f"P,{step},{int(p0[0])},{int(p0[1])},{int(p1[0])},{int(p1[1])}")
    print(f"W,{step},{w0},{w1}")
    print(f"C,{step},{cp}")

    mask = game.get_action_mask()
    mask_str = "".join("1" if x else "0" for x in mask)
    print(f"M,{step},{mask_str}")

    tensor = net.game_to_input_array(game)
    print(f"T,{step},{tensor_to_hex(tensor)}")

    # If current player is P1, also emit rotated versions
    if cp == 1:
        rotated = game.create_new()
        rotated.rotate_board()
        rmask = rotated.get_action_mask()
        rmask_str = "".join("1" if x else "0" for x in rmask)
        print(f"RM,{step},{rmask_str}")
        rtensor = net.game_to_input_array(rotated)
        print(f"RT,{step},{tensor_to_hex(rtensor)}")


game = Quoridor(Board(board_size, max_walls))

for step, aidx in enumerate(action_indices):
    emit_snapshot(game, step)
    action = encoder.index_to_action(aidx)
    game.step(action)

# Final snapshot after last action
emit_snapshot(game, len(action_indices))
