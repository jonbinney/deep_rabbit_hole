"""Standalone CLI script: prints action encoding and initial action mask for a given board config.

Usage: python action_reference.py <src_dir> <board_size> <max_walls>

Output format:
  A,<idx>,<row>,<col>,<action_type>  -- one line per action (type: 0=VERTICAL wall, 1=HORIZONTAL wall, 2=move)
  M,<bitmask>                        -- '1'/'0' string for the initial action mask
"""

import sys
from pathlib import Path

src_dir = Path(sys.argv[1])
board_size = int(sys.argv[2])
max_walls = int(sys.argv[3])
sys.path.insert(0, str(src_dir))

from quoridor import ActionEncoder, Board, Quoridor

encoder = ActionEncoder(board_size)
for idx in range(encoder.num_actions):
    action = encoder.index_to_action(idx)
    if action.__class__.__name__ == "MoveAction":
        row, col = action.destination
        action_type = 2
    else:
        row, col = action.position
        action_type = 0 if action.orientation.name == "VERTICAL" else 1
    print(f"A,{idx},{row},{col},{action_type}")

game = Quoridor(Board(board_size, max_walls))
mask = "".join("1" if x else "0" for x in game.get_action_mask())
print(f"M,{mask}")
