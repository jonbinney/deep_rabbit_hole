import argparse
import copy
import multiprocessing as mp
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import quoridor_env
from agents.alphazero.alphazero import AlphaZeroAgent, AlphaZeroParams
from agents.alphazero.multiprocess_evaluator import EvaluatorClient, EvaluatorServer, EvaluatorStatistics
from agents.alphazero.nn_evaluator import NNEvaluator
from agents.core.agent import AgentRegistry
from plugins.wandb_train import WandbParams, WandbTrainPlugin
from quoridor import ActionEncoder
from utils import my_device, parse_subargs, set_deterministic


@dataclass
class WorkerParams:
    worker_id: int
    random_seed: int
    job_queue: mp.Queue
    result_queue: mp.Queue
    evaluator: EvaluatorClient


@dataclass
class Worker:
    params: WorkerParams
    process: mp.Process


@dataclass
class RunGamesJob:
    num_games: int  # If -1, worker shuts itself down.
    alphazero_params: Optional[AlphaZeroParams]
    clear_evaluator_cache: bool


@dataclass
class RunGamesResult:
    worker_id: int
    replay_buffer: list[dict]
    evaluator_statistics: EvaluatorStatistics


def self_play_worker(
    board_size: int,
    max_walls: int,
    max_game_length: int,
    params: WorkerParams,
):
    # Each worker process uses its own random seed to make sure they don't make the exact same moves during
    # their self-play moves.
    set_deterministic(params.random_seed)

    environment = quoridor_env.env(board_size=board_size, max_walls=max_walls, step_rewards=False)

    while True:
        job = params.job_queue.get()
        if job.num_games == -1:
            break

        if job.clear_evaluator_cache:
            params.evaluator.clear_cache()

        # Use the updated model and parameters
        alphazero_agent = AlphaZeroAgent(board_size, max_walls, params=job.alphazero_params, evaluator=params.evaluator)

        for game_i in range(job.num_games):
            environment.reset()
            num_turns = 0

            for _ in environment.agent_iter():
                observation, _, termination, truncation, _ = environment.last()
                if termination or truncation:
                    break

                action_index = alphazero_agent.get_action(observation)
                environment.step(action_index)
                num_turns += 1

                # TODO: Move max steps with proper truncation to the environment and agent
                if num_turns >= max_game_length:
                    print("Truncating game")
                    print(environment.render())
                    break

            print(f"Worker {params.worker_id}: Game {game_i + 1}/{job.num_games} ended after {num_turns} turns")
            alphazero_agent.end_game(environment)

        params.result_queue.put(
            RunGamesResult(params.worker_id, alphazero_agent.replay_buffer, params.evaluator.get_statistics())
        )
        alphazero_agent.replay_buffer = []

    print(f"Worker {params.worker_id} exiting")


def train_alphazero(
    args: argparse.Namespace,
    evaluator_server: EvaluatorServer,
    workers: list[Worker],
    wandb_train_plugin: WandbTrainPlugin,
):
    # Create an agent that we'll use to do training. The self play games will happen with agents
    # created in each worker process.
    training_params = parse_subargs(args.params, AlphaZeroParams)
    training_params.training_mode = True  # We always want training mode, don't make the user specify it
    training_params.train_every = None  # We manually run training at the end of each epoch
    training_agent = AlphaZeroAgent(
        args.board_size, args.max_walls, params=training_params, evaluator=evaluator_server.evaluator
    )
    replay_buffer = deque(maxlen=training_params.replay_buffer_size)

    # Create parameters used by the workers during self play
    self_play_params = copy.deepcopy(training_params)
    self_play_params.replay_buffer_size = None  # Keep all moves, we'll manually clear them later

    evaluator_server.train_prepare(
        training_params.learning_rate, training_params.batch_size, training_params.optimizer_iterations
    )

    if wandb_train_plugin is not None:
        # HACK: the start_game method only cares that "game" has board_size and max_walls
        # members, so we pass in our arguments object. We need to call this method though,
        # because it calls the plugin's internal _intialize method which sets up metrics.
        wandb_train_plugin.start_game(game=args, agent1=training_agent, agent2=training_agent)

    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch}")

        # Tell workers to run self play games for this epoch
        games_per_worker = int(np.ceil(args.games_per_epoch / len(workers)))
        games_remaining_to_allocate = args.games_per_epoch
        for worker in workers:
            this_worker_num_games = min(games_per_worker, games_remaining_to_allocate)
            worker.params.job_queue.put(
                RunGamesJob(this_worker_num_games, self_play_params, clear_evaluator_cache=True)
            )

        # Collect results from the workers
        results = []
        for worker in workers:
            results.append(worker.params.result_queue.get())

        # Merge results into the primary replay buffer. We sort them to try and
        # make the results more deterministic across multiple runs.
        results = sorted(results, key=lambda r: r.worker_id)
        for r in results:
            # NOTE: Make sure the replay buffer size for the training agent is large enough to hold
            # the replay buffer results from all agents each epoch or else we'll end up discarding
            # some results and we'll have wasted computation by playing those games.
            replay_buffer.extend(r.replay_buffer)

        if len(replay_buffer) > training_params.batch_size:
            print(f"Training with {len(replay_buffer)} turns in replay buffer")
            metrics = evaluator_server.train_iteration(replay_buffer)
            print(f"Training done: {metrics}")
        else:
            print(
                f"Not training; batch_size={training_params.batch_size} is less than "
                + f"replay buffer size ({len(replay_buffer)})"
            )

        save_and_test_model(training_agent, f"_epoch_{epoch}", wandb_train_plugin, epoch * args.games_per_epoch)

        print(evaluator_server.get_statistics())
        for r in results:
            print(r.evaluator_statistics)


def save_and_test_model(agent: AlphaZeroAgent, model_suffix: str, wandb_train_plugin: WandbTrainPlugin, game_num: int):
    if wandb_train_plugin is not None and wandb_train_plugin.params.upload_model:
        model_dir = Path(agent.params.wandb_dir)
    else:
        model_dir = Path(agent.params.model_dir)

    # Save the model
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = model_dir / agent.resolve_filename(model_suffix)
    print(f"Saving checkpoint model to {model_path}")
    agent.save_model(model_path)

    if wandb_train_plugin is not None:
        # A bit of a hack.... the wandb plugin uses episode_count as the X axis for
        # graphs, but that isn't automatically set since we aren't calling the end_game
        # method after games in self-play (since those are run in other processes).
        wandb_train_plugin.episode_count = game_num
        wandb_train_plugin.compute_tournament_metrics(model_path)


def setup(args) -> tuple[EvaluatorServer, list[Worker]]:
    set_deterministic(args.seed)

    if args.wandb is None:
        wandb_train_plugin = None
    else:
        if args.wandb == "":
            wandb_params = WandbParams()
        else:
            wandb_params = parse_subargs(args.wandb, WandbParams)
            assert isinstance(wandb_params, WandbParams)

        agent_encoded_name = "alphazero:" + args.params
        wandb_train_params = WandbParams()
        wandb_train_plugin = WandbTrainPlugin(
            wandb_train_params, args.epochs * args.games_per_epoch, agent_encoded_name, args.benchmarks
        )

    # Queues used for worker processes to send evaluation requests to the EvaluatorSerer, and for it
    # to send the resulting (value, policy) back.
    evaluator_request_queue = mp.Queue()
    evaluator_result_queues = [mp.Queue() for _ in range(args.num_workers)]

    # Create the evaluator server and start its processing thread.
    action_encoder = ActionEncoder(args.board_size)
    nn_evaluator = NNEvaluator(action_encoder, my_device())
    evaluator_server = EvaluatorServer(
        nn_evaluator,
        input_queue=evaluator_request_queue,
        output_queues=evaluator_result_queues,
    )
    evaluator_server.start()

    # Create the worker processes. We hide the CUDA devices
    # in the worker processes since they're only doing CPU work,
    # and we don't want pytorch to waste GPU memory with the allocations
    # it automatically does on startup
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    workers = []
    for worker_id in range(args.num_workers):
        evaluator_client = EvaluatorClient(
            args.board_size,
            worker_id,
            evaluator_request_queue,
            evaluator_result_queues[worker_id],
        )
        job_queue = mp.Queue()
        result_queue = mp.Queue()
        worker_params = WorkerParams(worker_id, args.seed + worker_id, job_queue, result_queue, evaluator_client)
        worker_process = mp.Process(
            target=self_play_worker,
            args=(
                args.board_size,
                args.max_walls,
                args.max_steps,
                worker_params,
            ),
        )
        workers.append(Worker(worker_params, worker_process))

    # Start the worker processes
    for worker in workers:
        worker.process.start()

    return evaluator_server, workers, wandb_train_plugin


def shutdown(evaluator_server: EvaluatorServer, workers: list[Worker]):
    # Stop the worker processes
    for worker in workers:
        worker.params.job_queue.put(RunGamesJob(-1, None, False))

    for worker in workers:
        worker.process.join()

    # Stop the evaluator server
    evaluator_server.shutdown()
    evaluator_server.join()


def main(args):
    # Set multiprocessing start method to avoid tensor sharing issues and Mac bugs
    mp.set_start_method("spawn", force=True)

    t0 = time.time()

    evaluator_server, workers, wandb_train_plugin = setup(args)

    t1 = time.time()

    try:
        train_alphazero(args, evaluator_server, workers=workers, wandb_train_plugin=wandb_train_plugin)
    finally:
        shutdown(evaluator_server, workers)

    t2 = time.time()

    print(f"Worker startup time: {t1 - t0}")
    print(f"Total processing time {t2 - t0}")
    print(f"Time per game: {(t2 - t0) / (args.games_per_epoch * args.epochs)}")
    print(f"Throughput = {(args.games_per_epoch * args.epochs) / (t2 - t0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an alphazero agent for Quoridor")
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
    parser.add_argument("--max-game-length", type=int, help="Deprecated; use --max-steps instead")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max number of turns before game is called a tie")
    parser.add_argument(
        "-i",
        "--seed",
        type=int,
        default=42,
        help="Initializes the random seed for the training. Default is 42",
    )
    parser.add_argument(
        "-b",
        "--benchmarks",
        nargs="+",
        type=str,
        default=["random", "simple"],
        help=f"List of players to benchmark against. Can include parameters in parentheses. Allowed types {AgentRegistry.names()}",
    )
    parser.add_argument("-w", "--wandb", nargs="?", const="", default=None, type=str)
    args = parser.parse_args()

    # Handle deprecated --max-game-length argument
    if args.max_game_length is not None:
        if args.max_steps != parser.get_default("max-steps"):  # Check if --max-steps was also provided (not default)
            print("Warning: Both --max-game-length and --max-steps provided. Using --max-steps value.")
        else:
            print("Warning: --max-game-length is deprecated. Please use --max-steps instead.")
            args.max_steps = args.max_game_length

    main(args)
