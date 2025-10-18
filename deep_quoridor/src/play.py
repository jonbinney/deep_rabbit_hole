import argparse
import sys

from agents import AgentRegistry
from arena import Arena
from plugins.arena_yaml_recorder import ArenaYAMLRecorder
from renderers import Renderer
from utils import set_deterministic, yargs


def player_with_params(arg):
    if not AgentRegistry.is_valid_encoded_name(arg):
        raise argparse.ArgumentTypeError(f"Invalid player name: {arg}")
    return arg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Quoridor")
    parser.add_argument("-N", "--board_size", type=int, default=None, help="Board Size")
    parser.add_argument("-W", "--max_walls", type=int, default=None, help="Max walls per player")
    parser.add_argument(
        "-r",
        "--renderers",
        nargs="+",
        choices=Renderer.names(),
        default=["progressbar", "arenaresults"],
        help="Render modes to be used",
    )
    parser.add_argument("--step_rewards", action="store_true", default=False, help="Enable step rewards")

    parser.add_argument(
        "-p",
        "--players",
        nargs="+",
        type=player_with_params,
        default=["random", "simple"],
        help=f"List of players to compete against each other. Can include parameters in parentheses. Allowed types {AgentRegistry.names()}",
    )
    parser.add_argument(
        "-A",
        "--all",
        action="store_true",
        default=False,
        help="Plays a tournament of all agents against each other",
    )
    parser.add_argument(
        "-t",
        "--times",
        type=int,
        default=10,
        help="Number of times each player will play with each opponent",
    )
    parser.add_argument(
        "--games_output_filename",
        type=str,
        default="game_recording.yaml",
        help="Save the played games to a file. Use 'None' to disable saving.",
    )
    parser.add_argument(
        "-i",
        "--seed",
        type=int,
        default=42,
        help="Initializes the random seed for the training. Default is 42",
    )

    parser.add_argument(
        "-mx",
        "--max_steps",
        type=int,
        default=10000,
        help="Maximum number of steps per game. Default is 10000",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Use cProfile to profile the game.",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes")

    args_dict = yargs(parser, "yargs/play")

    sys.setrecursionlimit(10000)  # Increase recursion limit for deep games
    for id, args in args_dict.items():
        if id:
            print(f"====< Running {id} >====")
        set_deterministic(args.seed)

        renderers = [Renderer.create(r) for r in args.renderers]

        saver = None
        if args.games_output_filename != "None":
            saver = ArenaYAMLRecorder(args.games_output_filename)

        players = AgentRegistry.names() if args.all else args.players

        arena_args = {
            "board_size": args.board_size,
            "max_walls": args.max_walls,
            "step_rewards": args.step_rewards,
            "renderers": renderers,
            "saver": saver,
            "max_steps": args.max_steps,
        }

        arena_args = {k: v for k, v in arena_args.items() if v is not None}
        arena = Arena(**arena_args)

        def make_call():
            arena.play_games(players, args.times, num_workers=args.num_workers)

        if args.profile:
            import cProfile

            print("Profiling the game...")
            cProfile.run("make_call()", sort="tottime")
        else:
            make_call()
