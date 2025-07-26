import argparse
import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np
import quoridor as q
import quoridor_env
from agents.alphazero.alphazero import AlphaZeroAgent, AlphaZeroParams


@dataclass
class Worker:
    worker_id: int
    process: mp.Process
    job_queue: mp.SimpleQueue
    result_queue: mp.SimpleQueue


@dataclass
class RunGamesJob:
    num_games: int  # If -1, worker shuts itself down.
    model_state: dict


@dataclass
class RunGamesResult:
    replay_buffer: list[dict]


def self_play_worker(
    board_size: int,
    max_walls: int,
    max_game_length: int,
    worker_id: int,
    job_queue: mp.SimpleQueue,
    result_queue: mp.SimpleQueue,
):
    alphazero_params = AlphaZeroParams()
    alphazero_params.training_mode = True
    alphazero_params.train_every = None  # Never actually train
    alphazero_params.replay_buffer_size = None  # Keep all moves, we'll manually clear them later
    alphazero_agent = AlphaZeroAgent(board_size, max_walls, params=alphazero_params)

    environment = quoridor_env.env(board_size=board_size, max_walls=max_walls, step_rewards=False)

    print(f"Worker {worker_id} ready for jobs")
    while True:
        job = job_queue.get()
        if job.num_games == -1:
            break

        # Use the updated model
        alphazero_agent.set_model_state(job.model_state)

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


def train_alphazero(args, workers: list[Worker]):
    # Create an agent that we'll use to do training. The self play games will happen with agents
    # created in each worker process.
    alphazero_params = AlphaZeroParams()
    alphazero_params.training_mode = True
    alphazero_params.train_every = None
    training_agent = AlphaZeroAgent(args.board_size, args.max_walls, params=alphazero_params)

    model_state = training_agent.get_model_state()

    # Start games on the workers
    games_per_worker = int(np.ceil(args.episodes / len(workers)))
    games_remaining_to_allocate = args.episodes
    for worker in workers:
        this_worker_num_games = min(games_per_worker, games_remaining_to_allocate)
        worker.job_queue.put(RunGamesJob(this_worker_num_games, model_state))

    # Collect results from the workers
    for worker in workers:
        result = worker.result_queue.get()
        # NOTE: Make sure the replay buffer size for the training agent is large enough to hold
        # the replay buffer results from all agents each epoch or else we'll end up discarding
        # some results and we'll have wasted computation by playing those games.
        training_agent.replay_buffer.extend(result.replay_buffer)
    print(f"Replay buffer now has {len(training_agent.replay_buffer)} items")


def create_workers(board_size: int, max_walls: int, max_game_length: int, num_workers: int) -> list[Worker]:
    workers = []
    for worker_id in range(num_workers):
        job_queue = mp.SimpleQueue()
        result_queue = mp.SimpleQueue()
        process = mp.Process(
            target=self_play_worker,
            args=(
                board_size,
                max_walls,
                max_game_length,
                worker_id,
                job_queue,
                result_queue,
            ),
        )
        workers.append(Worker(worker_id, process, job_queue, result_queue))

    for worker in workers:
        worker.process.start()

    return workers


def stop_workers(workers: list[Worker]):
    # Tell workers to stop
    for worker in workers:
        worker.job_queue.put(RunGamesJob(-1, ""))

    for worker in workers:
        worker.process.join()


def main(args):
    # Set multiprocessing start method to avoid tensor sharing issues and Mac bugs
    mp.set_start_method("spawn", force=True)

    t0 = time.time()

    workers = create_workers(args.board_size, args.max_walls, args.max_game_length, args.num_workers)

    t1 = time.time()

    train_alphazero(args, workers=workers)
    stop_workers(workers)

    t2 = time.time()

    print(f"Worker startup time: {t1 - t0}")
    print(f"Total processing time {t2 - t0}")
    print(f"Time per game: {(t2 - t0) / args.episodes}")
    print(f"Throughput = {args.episodes / (t2 - t0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent for Quoridor")
    parser.add_argument("-N", "--board-size", type=int, default=5, help="Board Size")
    parser.add_argument("-W", "--max-walls", type=int, default=3, help="Max walls per player")
    parser.add_argument("-e", "--episodes", type=int, default=1000, help="Number of episodes to train for")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of worker processes")
    parser.add_argument(
        "--max-game-length", type=int, default=200, help="Max number of turns before game is called a tie"
    )
    args = parser.parse_args()

    main(args)
