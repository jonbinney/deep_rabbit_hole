import argparse

import torch
from agents.core.agent import AgentRegistry
from arena import Arena
from arena_utils import ArenaPlugin
from play import player_with_params
from plugins import SaveModelEveryNEpisodesPlugin, WandbTrainPlugin
from renderers import Renderer, TrainingStatusRenderer
from utils.misc import set_deterministic


def train_dqn(
    episodes: int,
    board_size: int,
    max_walls: int,
    save_frequency: int = 100,
    step_rewards: bool = True,
    use_wandb: bool = True,
    players: list | None = None,
    renderers: list[ArenaPlugin] = [],
):
    plugins = []

    if use_wandb:
        plugins.append(WandbTrainPlugin(update_every=10, total_episodes=episodes))

    plugins.append(
        SaveModelEveryNEpisodesPlugin(
            update_every=save_frequency,
            board_size=board_size,
            max_walls=max_walls,
            save_final=not use_wandb,
        )
    )

    print_plugin = TrainingStatusRenderer(
        update_every=1,
        total_episodes=episodes,
    )
    arena = Arena(
        board_size=board_size,
        max_walls=max_walls,
        step_rewards=step_rewards,
        renderers=[print_plugin] + renderers,
        plugins=plugins,
        swap_players=True,
    )

    arena.play_games(players=players, times=episodes)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent for Quoridor")
    parser.add_argument("-N", "--board_size", type=int, default=9, help="Board Size")
    parser.add_argument("-W", "--max_walls", type=int, default=10, help="Max walls per player")
    parser.add_argument("-e", "--episodes", type=int, default=1000, help="Number of episodes to train for")
    parser.add_argument("-s", "--step_rewards", action="store_true", default=False, help="Enable step rewards")
    parser.add_argument("-sp", "--save_path", type=str, default="models", help="Directory to save models")
    parser.add_argument(
        "-f", "--save_frequency", type=int, default=500, help="How often to save the model (in episodes)"
    )
    parser.add_argument(
        "-i",
        "--seed",
        type=int,
        default=42,
        help="Initializes the random seed for the training. Default is 42",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_false",
        default=True,
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "-p",
        "--players",
        nargs="+",
        type=player_with_params,
        help=f"List of players to compete against each other. Can include parameters in parentheses. Allowed types {AgentRegistry.names()}",
    )
    parser.add_argument(
        "-r",
        "--renderers",
        nargs="+",
        choices=Renderer.names(),
        default=["arenaresults"],
        help="Render modes to be used. Note that TrainingStatusRenderer is always included",
    )

    args = parser.parse_args()

    renderers = [Renderer.create(r) for r in args.renderers]

    print("Starting DQN training...")
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Max walls: {args.max_walls}")
    print(f"Training for {args.episodes} episodes")
    print(f"Using step rewards: {args.step_rewards}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Initializing random seed {args.seed}")

    # Set random seed for reproducibility
    set_deterministic(args.seed)

    train_dqn(
        episodes=args.episodes,
        board_size=args.board_size,
        max_walls=args.max_walls,
        save_frequency=args.save_frequency,
        step_rewards=args.step_rewards,
        use_wandb=args.no_wandb,
        players=args.players,
        renderers=renderers,
    )

    print("Training completed!")
