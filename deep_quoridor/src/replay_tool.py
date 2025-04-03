import argparse
import time

from arena import Arena, ArenaPlugin
from plugins import ArenaYAMLRecorder
from renderers import Renderer

"""Deep Quoridor Game Replay Tool

This script allows replaying recorded Quoridor games from YAML files. It provides command-line
options to customize the replay experience, including renderer selection and specific game filtering.

Command-line Arguments:
    -r, --renderer: Render mode for game replay visualization (default: "results")
    -t, --time_delay: Time delay between moves in milliseconds. Only applied if > 0 (default: -1)
    -g, --game_ids: List of specific game IDs to replay. If not set, replays all games
    -f, --games_input_filename: Path to YAML file containing recorded games (default: "game_recording.yaml")

Example Usage:
    python replay_tool.py -r text -g game_0008 game_0009 -f my_games.yaml
    python replay_tool.py -r text -t 500 -f my_games.yaml  # Replay with 500ms delay between moves
"""


class ActionDelayPlugin(ArenaPlugin):
    def __init__(self, time_delay: int):
        self.time_delay = time_delay

    def after_action(self, game, step, agent, action):
        time.sleep(self.time_delay / 1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Quoridor replay tool")
    parser.add_argument(
        "-r",
        "--renderers",
        nargs="+",
        choices=Renderer.names(),
        default=["progressbar", "arenaresults"],
        help="Render modes to be used",
    )
    parser.add_argument(
        "-t",
        "--time_delay",
        type=int,
        default=-1,
        help="Time delay between moves in ms, > 0 or ignored (default: -1)",
    )
    parser.add_argument(
        "-g",
        "--game_ids",
        nargs="+",
        type=str,
        default=[],
        help="Game IDs to replay, if not set it will replay all games",
    )
    parser.add_argument(
        "-f",
        "--games_input_filename",
        type=str,
        default="game_recording.yaml",
        help="Load the played games from the file",
    )

    args = parser.parse_args()

    renderers = [Renderer.create(r) for r in args.renderers]

    arena_data = ArenaYAMLRecorder.load_recorded_arena_data(args.games_input_filename)

    arena_args = {
        "board_size": arena_data["config"]["board_size"],
        "max_walls": arena_data["config"]["max_walls"],
        "step_rewards": arena_data["config"]["step_rewards"],
        "renderers": renderers,
        "plugins": [ActionDelayPlugin(args.time_delay)] if args.time_delay > 0 else [],
    }

    arena_args = {k: v for k, v in arena_args.items() if v is not None}
    arena = Arena(**arena_args)

    arena.replay_games(arena_data, args.game_ids)
