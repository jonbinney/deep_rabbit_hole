import argparse
from arena_yaml_recorder import ArenaYAMLRecorder
from arena import Arena
from renderers import Renderer
from agents import Agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Quoridor")
    parser.add_argument("-N", "--board_size", type=int, default=None, help="Board Size")
    parser.add_argument("-W", "--max_walls", type=int, default=None, help="Max walls per player")
    parser.add_argument("-r", "--renderer", choices=Renderer.names(), default="results", help="Render mode")
    parser.add_argument("--step_rewards", action="store_true", default=False, help="Enable step rewards")
    parser.add_argument(
        "-p",
        "--players",
        type=str,
        default="random,simple",
        help="Name of the players separated by comma. Each player will play with each one of the rest t times",
    )
    parser.add_argument(
        "-A", "--all", action="store_true", default=False, help="Plays a tournament of all agents against each other"
    )
    parser.add_argument(
        "-t", "--times", type=int, default=10, help="Number of times each player will play with each opponent"
    )
    parser.add_argument(
        "--games_output_filename",
        type=str,
        default="game_recording.yaml",
        help="Save the played games to a file. Use 'None' to disable saving.",
    )

    args = parser.parse_args()

    renderer = Renderer.create(args.renderer)

    saver = None
    if args.games_output_filename != "None":
        saver = ArenaYAMLRecorder(args.games_output_filename)

    if args.all:
        players = Agent.names()
    else:
        players = args.players.split(",")

    times = args.times

    args = {
        "board_size": args.board_size,
        "max_walls": args.max_walls,
        "step_rewards": args.step_rewards,
        "renderer": renderer,
        "saver": saver,
    }

    args = {k: v for k, v in args.items() if v is not None}
    arena = Arena(**args)

    arena.play_games(players, times)
