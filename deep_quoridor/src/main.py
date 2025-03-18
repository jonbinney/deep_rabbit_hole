import argparse
from arena_yaml_recorder import ArenaYAMLRecorder
from arena import Arena
from simple_agent import SimpleAgent
from random_agents import RandomAgent
from renderers import ResultsRenderer, TextRenderer, CursesRenderer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Quoridor")
    parser.add_argument("-N", "--board_size", type=int, default=None, help="Board Size")
    parser.add_argument("-W", "--max_walls", type=int, default=None, help="Max walls per player")
    parser.add_argument("-r", "--render", choices=["result", "text", "curses"], default="result", help="Render mode")
    parser.add_argument("--step_rewards", action="store_true", default=False, help="Enable step rewards")
    parser.add_argument(
        "--games_output_filename",
        type=str,
        default="game.yaml",
        help="Save the played games to a file. Use 'None' to disable saving.",
    )

    args = parser.parse_args()

    # TO DO: this should be automatically detected from the environment
    renderer = None
    if args.render == "result":
        renderer = ResultsRenderer()
    elif args.render == "text":
        renderer = TextRenderer()
    elif args.render == "curses":
        renderer = CursesRenderer()

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
