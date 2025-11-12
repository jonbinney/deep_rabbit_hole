import argparse
import copy
import gzip
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time
from dataclasses import asdict
from typing import BinaryIO, Optional, cast

from agent_evolution_tournament import AgentEvolutionTournament, AgentEvolutionTournamentParams
from agents.alphazero.alphazero import AlphaZeroAgent, AlphaZeroBenchmarkOverrideParams, AlphaZeroParams
from agents.alphazero.self_play_manager import GameParams, SelfPlayManager
from agents.core.agent import AgentRegistry
from metrics import Metrics
from plugins.wandb_train import WandbParams, WandbTrainPlugin
from utils import Timer, parse_subargs, set_deterministic
from utils.subargs import override_subargs


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
    initial_epoch = 0
    initial_artifact = training_agent.wandb_artifact()
    if initial_artifact:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = initial_artifact.download(root=tmpdir)
            filename = os.path.join(artifact_dir, "training_state.gz")
            if not os.path.exists(filename):
                raise RuntimeError(
                    "The specified wandb checkpoint doesn't have training_state.gz, so it's not possible to continue training from this point"
                )
            with gzip.open(filename, "rb") as f:
                data = pickle.load(f)

        initial_epoch = data["epoch"]
        training_agent.replay_buffer.extend(data["replay_buffer"])
        # TO DO: this is a bit tricky, so not sure if it's worth implementing, because the agent evolution tournament
        # relies on local model files that could not exist because they were deleted or created in another computer,
        # or they could even belong to a different training.  If we want to allow agent_evolution to continue, maybe
        # a solution would be to use a wandb alias rather than a local file name
        # if wandb_train_plugin is not None and wandb_train_plugin.agent_evolution_tournament is not None:
        #    wandb_train_plugin.agent_evolution_tournament.elos = data["agent_evolution_elos"]

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
        wandb_train_plugin.episode_count = initial_epoch * args.games_per_epoch
        if not args.benchmarks_skip_initial:
            wandb_train_plugin.compute_tournament_metrics(str(current_filename))

    last_epoch = initial_epoch + args.epochs
    for epoch in range(initial_epoch, last_epoch):
        print(f"Starting epoch {epoch}")

        # Set the filename so that each process loads the most recent model.
        self_play_params.model_filename = str(current_filename)

        Timer.start("self-play")

        # Create a self-play manager to run self-play games across multiple processes. The
        # worker processes get re-spawned each epoch to make sure all cached values get freed.
        wandb_params = None if wandb_train_plugin is None else wandb_train_plugin.params
        self_play_manager = SelfPlayManager(
            args.num_workers,
            args.seed,
            epoch,
            args.games_per_epoch,
            game_params,
            self_play_params,
            args.parallel_games,
            wandb_params=wandb_params,
        )
        self_play_manager.start()
        new_replay_buffer_items = None
        while new_replay_buffer_items is None:
            new_replay_buffer_items = self_play_manager.get_results(timeout=0.1)
        training_agent.replay_buffer.extend(new_replay_buffer_items)
        self_play_manager.join()
        game_num = (epoch + 1) * args.games_per_epoch

        Timer.finish("self-play", game_num)

        # Do training if we have enough samples in the replay buffer.
        training_occured = training_agent.train_iteration(epoch=epoch, episode=game_num)
        if not training_occured:
            print("Not enough samples - skipping training")

        current_filename = training_agent.save_model_with_suffix(f"_epoch_{epoch}")
        if wandb_train_plugin is not None:
            wandb_train_plugin.episode_count = game_num

            # Compute the metrics and upload the model periodically and in the last epoch
            if (epoch + 1) % args.benchmarks_every == 0 or epoch == last_epoch - 1:
                # Upload the model and training state
                with tempfile.TemporaryDirectory() as tmpdir:
                    training_state_filename = os.path.join(tmpdir, "training_state.gz")
                    save_training_state(
                        training_state_filename, training_agent, wandb_train_plugin, epoch + 1, game_num
                    )
                    wandb_train_plugin.upload_model(str(current_filename), [training_state_filename])

                wandb_train_plugin.compute_tournament_metrics(str(current_filename))

    Timer.log_totals()

    # Close the arena to finish wandb run
    if wandb_train_plugin is not None:
        wandb_train_plugin.end_arena(None, [])


def save_training_state(
    path: str, agent: AlphaZeroAgent, wandb_train_plugin: WandbTrainPlugin, epoch: int, episode: int
):
    aet = wandb_train_plugin.agent_evolution_tournament
    agent_evolution_elos = aet.elos if aet else {}

    state = {
        "replay_buffer": list(agent.replay_buffer),
        "epoch": epoch,
        "episode": episode,
        "agent_evolution_elos": agent_evolution_elos,
    }
    with gzip.open(path, "wb") as f:
        pickle.dump(state, cast(BinaryIO, f))


def main(args):
    # Set multiprocessing start method to avoid tensor sharing issues and Mac bugs
    mp.set_start_method("spawn", force=True)

    # MCTS can hit recursion limits during backpropagation sometimes
    sys.setrecursionlimit(10000)

    set_deterministic(args.seed)

    t0 = time.time()

    if args.wandb is None:
        wandb_train_plugin = None
    else:
        wandb_params = parse_subargs(args.wandb, WandbParams)
        assert isinstance(wandb_params, WandbParams)

        metrics = Metrics(
            args.board_size, args.max_walls, args.benchmarks, args.benchmarks_t, args.max_steps, args.num_workers
        )
        agent_encoded_name = "alphazero:" + args.params

        if args.benchmarks_params:
            benchmarks_params = parse_subargs(args.benchmarks_params, AlphaZeroBenchmarkOverrideParams)
            assert isinstance(benchmarks_params, AlphaZeroBenchmarkOverrideParams)
            override_args = {k: v for k, v in asdict(benchmarks_params).items() if v is not None}

            agent_encoded_name = "alphazero:" + override_subargs(args.params, override_args)

        if args.agent_evolution is not None:
            agent_evolution_params = parse_subargs(args.agent_evolution, AgentEvolutionTournamentParams)
            assert isinstance(agent_evolution_params, AgentEvolutionTournamentParams)
            agent_evolution_tournament = AgentEvolutionTournament(
                args.board_size,
                args.max_walls,
                args.max_steps,
                args.num_workers,
                agent_evolution_params,
            )
        else:
            agent_evolution_tournament = None

        wandb_train_plugin = WandbTrainPlugin(
            wandb_params,
            args.epochs * args.games_per_epoch,
            agent_encoded_name,
            metrics,
            agent_evolution_tournament,
            include_raw_metrics=True,
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
    parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes")
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
        "--benchmarks-t",
        type=int,
        default=10,
        help="How many time to play against each opponent during benchmarks",
    )
    parser.add_argument(
        "-be",
        "--benchmarks-every",
        type=int,
        default=1,
        help="Every how many epochs to compute the benchmark",
    )
    parser.add_argument("--benchmarks-params", nargs="?", const="", default=None, type=str)
    parser.add_argument(
        "--benchmarks-skip-initial", action="store_true", default=False, help="Skip the initial benchmark"
    )
    parser.add_argument("-w", "--wandb", nargs="?", const="", default=None, type=str)
    parser.add_argument(
        "-pg",
        "--parallel-games",
        type=int,
        default=32,
        help="How many games to play in parallel per process",
    )
    parser.add_argument(
        "-a",
        "--agent-evolution",
        nargs="?",
        const="",
        default=None,
        type=str,
        help="Parameters for the Agent Evolution Tournament",
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
