import argparse
import copy
import multiprocessing as mp
import time
from typing import Optional

from agents.alphazero.alphazero import AlphaZeroAgent, AlphaZeroParams
from agents.alphazero.self_play_manager import GameParams, SelfPlayManager
from agents.core.agent import AgentRegistry
from metrics import Metrics
from plugins.wandb_train import WandbParams, WandbTrainPlugin
from utils import parse_subargs, set_deterministic


def train_alphazero(
    args: argparse.Namespace,
    wandb_train_plugin: Optional[WandbTrainPlugin],
):
    game_params = GameParams(args.board_size, args.max_walls, args.max_steps)

    # Create an agent that we'll use to do training.
    training_params = parse_subargs(args.params, AlphaZeroParams)
    assert isinstance(training_params, AlphaZeroParams)
    training_params.training_mode = True  # We only use this agent for training
    training_params.train_every = None  # We manually run training at the end of each epoch
    training_agent = AlphaZeroAgent(
        args.board_size,
        args.max_walls,
        params=training_params,
    )

    current_filename = training_agent.save_model_with_suffix("_initial")

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
        training_agent.set_wandb_run(wandb_train_plugin.run)

        # Compute the tournament metrics with the initial model, possibly random initialized, to
        # be able to see how it evolves from there
        wandb_train_plugin.episode_count = 0
        wandb_train_plugin.compute_tournament_metrics(str(current_filename))

    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch}")

        # When using per process evaluation, set the filename so that each process loads the most recent model.
        if args.per_process_evaluation:
            self_play_params.model_filename = str(current_filename)

        # Create a self-play manager to run self-play games across multiple processes. The
        # worker processes get re-spawned each epoch to make sure all cached values get freed.
        self_play_manager = SelfPlayManager(
            args.num_workers,
            args.seed,
            epoch,
            args.games_per_epoch,
            game_params,
            self_play_params,
            args.per_process_evaluation,
        )
        self_play_manager.start()
        new_replay_buffer_items = None
        while new_replay_buffer_items is None:
            new_replay_buffer_items = self_play_manager.get_results(timeout=0.1)
        training_agent.replay_buffer.extend(new_replay_buffer_items)
        self_play_manager.join()

        # Do training if we have enough samples in the replay buffer.
        # training_agent.episode_count = game_num
        training_occured = training_agent.train_iteration(epoch=epoch)
        if not training_occured:
            print("Not enough samples - skipping training")

        game_num = (epoch + 1) * args.games_per_epoch
        current_filename = training_agent.save_model_with_suffix(f"_epoch_{epoch}")
        if wandb_train_plugin is not None:
            # Save the model where the plugin wants it and use the plugin to compute metrics.
            wandb_train_plugin.episode_count = game_num
            wandb_train_plugin.compute_tournament_metrics(str(current_filename))

    # Close the arena so the best model and the final model are uploaded to wandb
    if wandb_train_plugin is not None:
        wandb_train_plugin.end_arena(None, None)


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

        metrics = Metrics(args.board_size, args.max_walls, args.benchmarks, args.benchmarks_t, args.max_steps)
        agent_encoded_name = "alphazero:" + args.params
        wandb_train_plugin = WandbTrainPlugin(
            wandb_params, args.epochs * args.games_per_epoch, agent_encoded_name, metrics
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
    parser.add_argument(
        "-bt",
        "--benchmarks_t",
        type=int,
        default=10,
        help="How many time to play against each opponent during benchmarks",
    )
    parser.add_argument("-w", "--wandb", nargs="?", const="", default=None, type=str)
    parser.add_argument(
        "--per-process-evaluation",
        action="store_true",
        default=False,
        help="Each process will do NN evaluations.  Otherwise, just one process will handle them all",
    )
    args = parser.parse_args()

    # Handle deprecated --max-game-length argument
    if args.max_game_length is not None:
        if args.max_steps != parser.get_default("max-steps"):  # Check if --max-steps was also provided (not default)
            print("Warning: Both --max-game-length and --max-steps provided. Using --max-steps value.")
        else:
            print("Warning: --max-game-length is deprecated. Please use --max-steps instead.")
            args.max_steps = args.max_game_length

    main(args)
