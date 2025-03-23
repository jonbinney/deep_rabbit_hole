import argparse
import os
from arena_yaml_recorder import ArenaYAMLRecorder
from arena import Arena
from agents import RandomAgent, DQNAgent
from renderers import Renderer
from agents import Agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Quoridor")
    parser.add_argument("-N", "--board_size", type=int, default=None, help="Board Size")
    parser.add_argument(
        "-W", "--max_walls", type=int, default=None, help="Max walls per player"
    )
    parser.add_argument(
        "-r",
        "--renderers",
        nargs="+",
        choices=Renderer.names(),
        default=["progressbar", "arenaresults"],
        help="Render modes to be used",
    )
    parser.add_argument(
        "--step_rewards", action="store_true", default=False, help="Enable step rewards"
    )
    parser.add_argument(
        "-p",
        "--players",
        nargs="+",
        choices=Agent.names(),
        default=["random", "simple"],
        help="List of players to compete against each other",
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

    args = parser.parse_args()

    renderers = [Renderer.create(r) for r in args.renderers]

    saver = None
    if args.games_output_filename != "None":
        saver = ArenaYAMLRecorder(args.games_output_filename)

    players = Agent.names() if args.all else args.players

    arena_args = {
        "board_size": args.board_size,
        "max_walls": args.max_walls,
        "step_rewards": args.step_rewards,
        "renderers": renderers,
        "saver": saver,
    }

    arena_args = {k: v for k, v in arena_args.items() if v is not None}
    arena = Arena(**arena_args)

    arena.play_games(players, args.times)
    args = {k: v for k, v in args.items() if v is not None}
    # Calculate action space size for DQN agent
    board_size = args.get("board_size", 9)  # Default to 9 if not specified
    action_size = board_size**2 + ((board_size - 1) ** 2) * 2

    # Create agents
    random_agent = RandomAgent()

    # Initialize DQN agent
    dqn_agent = DQNAgent(board_size, action_size)

    # Load the pre-trained model
    model_path = "/home/julian/aaae/deep-rabbit-hole/code/deep_rabbit_hole/models/dqn_flat_nostep_final.pt"
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        dqn_agent.load_model(model_path)
    else:
        print(f"Warning: Model file {model_path} not found, using untrained agent")

    # Set up arena and play game
    arena = Arena(**args)

    # Play with DQN agent as player 0 (first player) against a random agent
    print("Playing: DQN Agent vs Random Agent")
    arena.play_game(dqn_agent, random_agent)

    # Optionally, play another game with reversed roles
    # print("\nPlaying: Random Agent vs DQN Agent")
    # arena.play_game(random_agent, dqn_agent)
