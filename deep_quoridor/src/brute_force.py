import sys

import quoridor_env
from agents import AgentRegistry

print("Starting")

MAX_STEPS = 999999

BOARD_SIZE = int(sys.argv[1])
MAX_WALLS = int(sys.argv[2])
MAX_DEPTH = int(sys.argv[3])
print(f"BOARD_SIZE:{BOARD_SIZE} MAX_WALLS:{MAX_WALLS} MAX_DEPTH:{MAX_DEPTH}")

simple_entry = AgentRegistry.get_registry_entry("simple")
SimpleAgent = simple_entry.agent_class
SimpleParams = simple_entry.params_class
assert SimpleAgent is not None
assert SimpleParams is not None
simple_params = SimpleParams(max_depth=MAX_DEPTH, branching_factor=99, discount_factor=1.0, heuristic=0)
simple_agent = SimpleAgent(params=simple_params, board_size=BOARD_SIZE, max_walls=MAX_WALLS)

env = quoridor_env.env(
    board_size=BOARD_SIZE,
    max_walls=MAX_WALLS,
    max_steps=MAX_STEPS,
    step_rewards=False,
)

env.reset()
simple_agent.start_game(None, "Bob")
observation, _, termination, truncation, _ = env.last()
action_index = simple_agent.get_action(observation)
