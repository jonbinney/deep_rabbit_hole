import time

import quoridor as q
import torch.multiprocessing as mp
from agents.alphazero.alphazero import AlphaZeroAgent, AlphaZeroParams
from agents.alphazero.multiprocess_evaluator import EvaluatorClient, EvaluatorServer
from agents.alphazero.nn_evaluator import NNEvaluator
from quoridor_env import make_observation
from utils import my_device


def run_self_play_games(
    board_size,
    max_walls,
    max_game_length,
    worker_id,
    request_queue,
    result_queue: mp.SimpleQueue,
    num_games: int,
):
    evaluator_client = EvaluatorClient(worker_id, request_queue, result_queue)
    alphazero_params = AlphaZeroParams()
    alphazero_params.training_mode = True
    alphazero_params.train_every = None
    alphazero_agent = AlphaZeroAgent(board_size, max_walls, params=alphazero_params, evaluator=evaluator_client)

    primary_player = q.Player.ONE  # The player who is actively training
    for _ in range(num_games):
        game = q.Quoridor(q.Board(board_size, max_walls))
        num_turns = 0

        while not game.is_game_over() and num_turns < max_game_length:
            # TODO: Use environment class to properly set the agent_id arg to make_observation
            if game.get_current_player() == primary_player:
                observation = make_observation(game, "player_0", game.get_current_player(), True)
            else:
                observation = make_observation(game, "player_1", game.get_current_player(), False)

            action_mask = game.get_action_mask()

            action = alphazero_agent.get_action({"observation": observation, "action_mask": action_mask})
            game.step(action)
            num_turns += 1

        # TODO: change primary player each game
        pass


def main():
    # Set multiprocessing start method to avoid tensor sharing issues and Mac bugs
    mp.set_start_method("spawn", force=True)

    # Game parameters
    board_size = 5
    max_walls = 3
    max_game_length = 200

    # Parallelism parameters
    batch_size = 5
    max_interbatch_time = 0.001
    num_workers = 5
    num_games = 10
    games_per_worker = int(round(num_games / num_workers))

    action_encoder = q.ActionEncoder(board_size)
    evaluator = NNEvaluator(action_encoder, my_device())

    request_queue = mp.SimpleQueue()
    result_queues = [mp.SimpleQueue() for _ in range(num_workers)]
    evaluator_server = EvaluatorServer(evaluator, batch_size, max_interbatch_time, request_queue, result_queues)
    evaluator_server.start()

    workers = []
    for worker_id in range(num_workers):
        workers.append(
            mp.Process(
                target=run_self_play_games,
                args=(
                    board_size,
                    max_walls,
                    max_game_length,
                    worker_id,
                    request_queue,
                    result_queues[worker_id],
                    games_per_worker,
                ),
            )
        )

    t0 = time.time()

    for worker in workers:
        worker.start()

    t1 = time.time()

    for worker in workers:
        worker.join()

    t2 = time.time()

    evaluator_server.shutdown()
    evaluator_server.join()

    print(f"Worker startup time: {t1 - t0}")
    print(f"Total processing time {t2 - t0}")
    print(f"Throughput = {(num_games * num_workers) / (t2 - t0)}")
    print(evaluator_server.get_statistics())


if __name__ == "__main__":
    main()
