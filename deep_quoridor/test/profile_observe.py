import cProfile
import time

from deep_quoridor.src.quoridor_env import env

e = env()
walls = [
    ((3, 0), 1),
    ((3, 2), 1),
    ((3, 4), 1),
    ((3, 6), 1),
    ((2, 7), 0),
    ((5, 7), 1),
    ((4, 0), 0),
    ((4, 2), 0),
    ((4, 4), 0),
]


def run():
    st = time.time()
    for w in walls:
        e.place_wall("player_0", w[0], w[1])
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
