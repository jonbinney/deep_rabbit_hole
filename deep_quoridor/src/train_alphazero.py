import argparse
import copy
import multiprocessing as mp
import time
from collections import deque
from pathlib import Path
from typing import Optional

from agents.alphazero.alphazero import AlphaZeroAgent, AlphaZeroParams
from agents.alphazero.self_play_manager import GameParams, SelfPlayManager
from agents.core.agent import AgentRegistry
from plugins.wandb_train import WandbParams, WandbTrainPlugin
from utils import parse_subargs, set_deterministic


def train_alphazero(
    args: argparse.Namespace,
    wandb_train_plugin: Optional[WandbTrainPlugin],
):
    game_params = GameParams(args.board_size, args.max_walls, args.max_steps)

    # Create an agent that we'll use to do training.
    training_params = parse_subargs(args.params, AlphaZeroParams)
    training_params.training_mode = True  # We only use this agent for training
    training_params.train_every = None  # We manually run training at the end of each epoch
    training_agent = AlphaZeroAgent(
        args.board_size,
        args.max_walls,
        params=training_params,
    )
    replay_buffer = deque(maxlen=training_params.replay_buffer_size)

    # Create parameters used by the workers during self play
    self_play_params = copy.deepcopy(training_params)
    self_play_params.replay_buffer_size = None  # Keep all moves, we'll manually clear them later

    training_agent.evaluator.train_prepare(
        training_params.learning_rate,
        training_params.batch_size,
        training_params.optimizer_iterations,
        training_params.weight_decay,
    )

    if wandb_train_plugin is not None:
        # HACK: the start_game method only cares that "game" has board_size and max_walls
        # members, so we pass in a GameParams object. We have to call start_game
        # because it calls the plugin's internal _intialize method which sets up metrics.
        wandb_train_plugin.start_game(game=args, agent1=training_agent, agent2=training_agent)

    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch}")

        # We use a different random seed each epoch to make sure we don't get correlated games, but
        # we want the runs to be repeatable so we use a deterministic scheme to generate the seeds.
        random_seed = args.seed + args.num_workers * epoch

        # Create a self-play manager to run self-play games across multiple processes. The
        # worker processes get re-spawned each epoch to make sure all cached values get freed.
        self_play_manager = SelfPlayManager(
            args.num_workers, random_seed, args.games_per_epoch, game_params, self_play_params
        )
        self_play_manager.start()
        new_replay_buffer_items = None
        while new_replay_buffer_items is None:
            new_replay_buffer_items = self_play_manager.get_results(timeout=0.1)
        replay_buffer.extend(new_replay_buffer_items)
        self_play_manager.join()

        # Do training if we have enough samples in the replay buffer.
        if len(replay_buffer) > training_params.batch_size:
            print(f"Training with {len(replay_buffer)} turns in replay buffer")
            metrics = training_agent.evaluator.train_iteration(replay_buffer)
            print(f"Training done: {metrics}")
        else:
            print(
                f"Not training; batch_size={training_params.batch_size} is less than "
                + f"replay buffer size ({len(replay_buffer)})"
            )

        game_num = epoch * args.games_per_epoch
        model_suffix = f"_epoch_{epoch}"
        model_dir = Path(training_agent.params.wandb_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / training_agent.resolve_filename(model_suffix)
        training_agent.save_model(model_path)
        if wandb_train_plugin is not None:
            # Save the model where the plugin wants it and use the plugin to compute metrics.
            wandb_train_plugin.episode_count = game_num
            wandb_train_plugin.compute_tournament_metrics(model_path)


def main(args):
    # Set multiprocessing start method to avoid tensor sharing issues and Mac bugs
    mp.set_start_method("spawn", force=True)

    set_deterministic(args.seed)

    t0 = time.time()

    if args.wandb is None:
        wandb_train_plugin = None
    else:
        if args.wandb == "":
            wandb_params = WandbParams()
        else:
            wandb_params = parse_subargs(args.wandb, WandbParams)
            assert isinstance(wandb_params, WandbParams)

        agent_encoded_name = "alphazero:" + args.params
        wandb_train_plugin = WandbTrainPlugin(
            wandb_params, args.epochs * args.games_per_epoch, agent_encoded_name, args.benchmarks
        )

    t0 = time.time()

    train_alphazero(args, wandb_train_plugin=wandb_train_plugin)

    t1 = time.time()

    print(f"Total processing time {t1 - t0}")
    print(f"Time per game: {(t1 - t0) / (args.games_per_epoch * args.epochs)}")
    print(f"Throughput = {(args.games_per_epoch * args.epochs) / (t1 - t0)}")


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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max number of turns before game is called a tie (pass -1 for no limit)",
    )
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
