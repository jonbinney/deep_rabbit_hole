import cProfile
import time

import numpy as np
from quoridor import Player, WallOrientation
from quoridor_env import env

e = env(board_size=9)

walls = [
    ((3, 0), WallOrientation.HORIZONTAL),
    ((3, 2), WallOrientation.HORIZONTAL),
    ((3, 4), WallOrientation.HORIZONTAL),
    ((3, 6), WallOrientation.HORIZONTAL),
    ((2, 7), WallOrientation.VERTICAL),
    ((5, 7), WallOrientation.HORIZONTAL),
    ((4, 0), WallOrientation.VERTICAL),
    ((4, 2), WallOrientation.VERTICAL),
    ((4, 4), WallOrientation.VERTICAL),
]


def run():
    st = time.time()
    for position, orientation in walls:
        e.game.board.add_wall(Player.ONE, position, orientation)
        e.observe("player_0")
        e.observe("player_1")

    et = time.time()
    print(e.render())
    print((et - st) * 1000000)


# Run in Cucu's computer.  Just for relative reference:
#
# No checks at all:                                90 usec
# Wall overlap check:                           3,000 usec
# Wall overlap and block:                     200,000 usec
# Using _is_wall_potential_block optimization: 25,000 usec

# Switch the comments to do a profiling
# run()
cProfile.run("run()", sort="tottime")
