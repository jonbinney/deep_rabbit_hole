import argparse
import copy
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import quoridor_env
from agents.alphazero.alphazero import AlphaZeroAgent, AlphaZeroParams
from agents.alphazero.multiprocess_evaluator import EvaluatorClient, EvaluatorServer
from agents.alphazero.nn_evaluator import NNEvaluator
from quoridor import ActionEncoder
from utils import my_device, parse_subargs


@dataclass
class Worker:
    worker_id: int
    process: mp.Process
    job_queue: mp.SimpleQueue
    result_queue: mp.SimpleQueue


@dataclass
class RunGamesJob:
    num_games: int  # If -1, worker shuts itself down.
    alphazero_params: Optional[AlphaZeroParams]


@dataclass
class RunGamesResult:
    replay_buffer: list[dict]


def self_play_worker(
    board_size: int,
    max_walls: int,
    max_game_length: int,
    evaluator: EvaluatorClient,
    worker_id: int,
    job_queue: mp.SimpleQueue,
    result_queue: mp.SimpleQueue,
):
    environment = quoridor_env.env(board_size=board_size, max_walls=max_walls, step_rewards=False)

    while True:
        job = job_queue.get()
        if job.num_games == -1:
            break

        # Use the updated model and parameters
        alphazero_agent = AlphaZeroAgent(board_size, max_walls, params=job.alphazero_params, evaluator=evaluator)

        for _ in range(job.num_games):
            environment.reset()
            num_turns = 0

            for _ in environment.agent_iter():
                # TODO: Use environment class to properly set the agent_id arg to make_observation
                observation, _, termination, truncation, _ = environment.last()
                if termination or truncation:
                    break

                action_index = alphazero_agent.get_action(observation)
                environment.step(action_index)
                num_turns += 1

                # TODO: Move max steps with proper truncation to the environment
                if num_turns >= max_game_length:
                    print("Truncating game")
                    print(environment.render())
                    break

            print(f"Game ended after {num_turns} turns")
            alphazero_agent.end_game(environment)

        result_queue.put(RunGamesResult(alphazero_agent.replay_buffer))
        alphazero_agent.replay_buffer = []

    print(f"Worker {worker_id} exiting")


def train_alphazero(args, evaluator_server: EvaluatorServer, workers: list[Worker]):
    # Create an agent that we'll use to do training. The self play games will happen with agents
    # created in each worker process.
    training_params = parse_subargs(args.params, AlphaZeroParams)
    training_params.training_mode = True  # We always want training mode, don't make the user specify it
    training_params.train_every = None  # We manually run training at the end of each epoch
    training_agent = AlphaZeroAgent(args.board_size, args.max_walls, params=training_params)

    # Create parameters used by the workers during self play
    self_play_params = copy.deepcopy(training_params)
    self_play_params.replay_buffer_size = None  # Keep all moves, we'll manually clear them later

    # Prep the evaluator for training. The training_agent doesn't do this itself since we set
    # the train_every parameter to None
    training_agent.evaluator.train_prepare(
        training_params.learning_rate, training_params.batch_size, training_params.optimizer_iterations
    )

    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch}")

        # Tell workers to run self play games for this epoch
        games_per_worker = int(np.ceil(args.games_per_epoch / len(workers)))
        games_remaining_to_allocate = args.games_per_epoch
        for worker in workers:
            this_worker_num_games = min(games_per_worker, games_remaining_to_allocate)
            worker.job_queue.put(RunGamesJob(this_worker_num_games, self_play_params))

        # Collect results from the workers
        for worker in workers:
            result = worker.result_queue.get()
            # NOTE: Make sure the replay buffer size for the training agent is large enough to hold
            # the replay buffer results from all agents each epoch or else we'll end up discarding
            # some results and we'll have wasted computation by playing those games.
            training_agent.replay_buffer.extend(result.replay_buffer)

        if len(training_agent.replay_buffer) > training_params.batch_size:
            print(f"Training with {len(training_agent.replay_buffer)} turns in replay buffer")
            metrics = training_agent.evaluator.train_iteration(training_agent.replay_buffer)
            print(f"Training done: f{metrics}")
        else:
            print(
                f"Not training; batch_size={training_params.batch_size} is less than "
                + f"replay buffer size ({len(training_agent.replay_buffer)})"
            )

    print(evaluator_server.get_statistics())


def setup(
    board_size: int, max_walls: int, max_game_length: int, num_workers: int
) -> tuple[EvaluatorServer, list[Worker]]:
    # Queues used for worker processes to send evaluation requests to the EvaluatorSerer, and for it
    # to send the resulting (value, policy) back.
    evaluator_request_queue = mp.SimpleQueue()
    evaluator_result_queues = [mp.SimpleQueue() for _ in range(num_workers)]

    # Create the evaluator server and start its processing thread.
    action_encoder = ActionEncoder(board_size)
    nn_evaluator = NNEvaluator(action_encoder, my_device())
    evaluator_server = EvaluatorServer(
        nn_evaluator,
        batch_size=10,
        max_interbatch_time=0.0001,
        input_queue=evaluator_request_queue,
        output_queues=evaluator_result_queues,
    )
    evaluator_server.start()

    # Create the worker processes
    workers = []
    for worker_id in range(num_workers):
        evaluator_client = EvaluatorClient(
            worker_id, evaluator_request_queue, evaluator_result_queues[worker_id], board_size
        )
        job_queue = mp.SimpleQueue()
        result_queue = mp.SimpleQueue()
        process = mp.Process(
            target=self_play_worker,
            args=(
                board_size,
                max_walls,
                max_game_length,
                evaluator_client,
                worker_id,
                job_queue,
                result_queue,
            ),
        )
        workers.append(Worker(worker_id, process, job_queue, result_queue))

    # Start the worker processes
    for worker in workers:
        worker.process.start()

    return evaluator_server, workers


def shutdown(evaluator_server: EvaluatorServer, workers: list[Worker]):
    # Stop the worker processes
    for worker in workers:
        worker.job_queue.put(RunGamesJob(-1, None))

    for worker in workers:
        worker.process.join()

    # Stop the evaluator server
    evaluator_server.shutdown()
    evaluator_server.join()


def main(args):
    # Set multiprocessing start method to avoid tensor sharing issues and Mac bugs
    mp.set_start_method("spawn", force=True)

    t0 = time.time()

    evaluator_server, workers = setup(args.board_size, args.max_walls, args.max_game_length, args.num_workers)

    t1 = time.time()

    try:
        train_alphazero(args, evaluator_server, workers=workers)
    finally:
        shutdown(evaluator_server, workers)

    t2 = time.time()

    print(f"Worker startup time: {t1 - t0}")
    print(f"Total processing time {t2 - t0}")
    print(f"Time per game: {(t2 - t0) / (args.games_per_epoch * args.epochs)}")
    print(f"Throughput = {(args.games_per_epoch * args.epochs) / (t2 - t0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent for Quoridor")
    parser.add_argument("-p", "--params", type=str, default="", help="Alphazero agent params in subargs form")
    parser.add_argument("-N", "--board-size", type=int, default=5, help="Board Size")
    parser.add_argument("-W", "--max-walls", type=int, default=3, help="Max walls per player")
    parser.add_argument(
        "-g",
        "--games-per-epoch",
        type=int,
        default=100,
        help="Number of self play games to do between each model training",
    )
    parser.add_argument("-e", "--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of worker processes")
    parser.add_argument(
        "--max-game-length", type=int, default=200, help="Max number of turns before game is called a tie"
    )
    args = parser.parse_args()

    main(args)
