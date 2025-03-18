import argparse
from arena_yaml_recorder import ArenaYAMLRecorder
from arena import Arena
from agents import SimpleAgent, RandomAgent
from renderers import Renderer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Quoridor")
    parser.add_argument("-N", "--board_size", type=int, default=None, help="Board Size")
    parser.add_argument("-W", "--max_walls", type=int, default=None, help="Max walls per player")
    parser.add_argument("-r", "--renderer", choices=Renderer.names(), default="results", help="Render mode")
    parser.add_argument("--step_rewards", action="store_true", default=False, help="Enable step rewards")
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

    args = {
        "board_size": args.board_size,
        "max_walls": args.max_walls,
        "step_rewards": args.step_rewards,
        "renderer": renderer,
        "saver": saver,
    }

    args = {k: v for k, v in args.items() if v is not None}
    arena = Arena(**args)
    arena.play_game(RandomAgent(), SimpleAgent())
