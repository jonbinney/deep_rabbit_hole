import random
import time

import numpy as np
from agents.alphazero.nn_evaluator import NNEvaluator
from prettytable import PrettyTable
from quoridor import ActionEncoder, Board, Quoridor
from utils import my_device

print(f"Using device: {my_device()}")

num_queries = np.array([1, 10, 20, 50, 100, 500, 1000])
board_size = 5
max_walls = 3
action_encoder = ActionEncoder(board_size)
evaluator = NNEvaluator(action_encoder, my_device())
game = Quoridor(Board(board_size, max_walls))

# Create a bunch of game states to use as inputs to the NN
games = [game.create_new()]
while len(games) < max(num_queries):
    actions = game.get_valid_actions()
    action = random.choice(actions)
    game.step(action)
    games.append(game.create_new())
    if game.is_game_over():
        game = Quoridor(Board(board_size, max_walls))
print(f"Created {len(games)} game states to evaluate")

batch_durations = []
individual_durations = []
for n in num_queries:
    games_this_batch = games[:n]

    batch_start = time.time()
    batch_results = evaluator.evaluate_batch(games_this_batch)
    batch_end = time.time()
    batch_duration = batch_end - batch_start
    print(f"Batch evaluation: Total time: {batch_duration}  Average time: {batch_duration / len(games_this_batch)}")

    individual_start = time.time()
    individual_results = []
    for game in games_this_batch:
        individual_results.append(evaluator.evaluate(game))
    individual_end = time.time()
    individual_duration = individual_end - individual_start
    print(
        f"Individual evaluation: Total time: {individual_duration}  Average time: {individual_duration / len(games_this_batch)}"
    )

    print(f"Batch evaluation is {(individual_duration) / batch_duration} times faster")

    batch_durations.append(batch_duration)
    individual_durations.append(individual_duration)

batch_durations = np.array(batch_durations)
individual_durations = np.array(individual_durations)
speedups = individual_durations / batch_durations

table = PrettyTable()
table.title = f"Batch vs individual inference for device={my_device()}"
table.field_names = ["N", "Individual (ms)", "Batch (ms)", "Speedup"]
table.float_format = ".3"
for n, bd, id, s in zip(num_queries, batch_durations, individual_durations, speedups):
    table.add_row((n, id * 1000, bd * 1000, s))
print(table)
