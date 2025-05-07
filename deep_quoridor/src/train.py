import argparse
import datetime
import random

from agents.core.agent import Agent, AgentRegistry
from arena import Arena, PlayMode
from arena_utils import ArenaPlugin
from gymnasium import spaces
from play import player_with_params
from plugins import SaveModelEveryNEpisodesPlugin, WandbTrainPlugin
from renderers import Renderer, TrainingStatusRenderer
from utils import my_device, set_deterministic


def train_dqn(
    episodes: int,
    board_size: int,
    max_walls: int,
    save_frequency: int,
    step_rewards: bool = True,
    use_wandb: bool = True,
    players: list = [],
    renderers: list[ArenaPlugin] = [],
    run_id: str = "",
):
    plugins = []
    total_episodes = episodes * (len(players) - 1)

    after_save_method = None
    if use_wandb:
        wandb_plugin = WandbTrainPlugin(
            update_every=10, total_episodes=total_episodes, run_id=run_id, agent_encoded_name=players[0]
        )
        plugins.append(wandb_plugin)
        after_save_method = wandb_plugin.compute_tournament_metrics

    plugins.append(
        SaveModelEveryNEpisodesPlugin(
            update_every=save_frequency,
            board_size=board_size,
            max_walls=max_walls,
            save_final=not use_wandb,
            run_id=run_id,
            after_save=after_save_method,
        )
    )

    print_plugin = TrainingStatusRenderer(total_episodes=total_episodes)
    arena = Arena(
        board_size=board_size,
        max_walls=max_walls,
        step_rewards=step_rewards,
        renderers=[print_plugin] + renderers,
        plugins=plugins,
        swap_players=True,
        max_steps=1000,
    )

    agents = []
    for p in players:
        if isinstance(p, Agent):
            agents.append(p)
        else:
            agents.append(
                AgentRegistry.create_from_encoded_name(
                    p,
                    board_size=board_size,
                    max_walls=max_walls,
                    action_space=spaces.Discrete(
                        board_size**2 + ((board_size - 1) ** 2) * 2, seed=random.randint(0, 2**32 - 1)
                    ),
                )
            )

    agents.append(agents[0].new_mimic_model())

    arena.play_games(players=agents, times=episodes, mode=PlayMode.FIRST_VS_RANDOM)
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
    parser.add_argument(
        "-rp",
        "--run_prefix",
        required=True,
        type=str,
        help="Run prefix to use for this run. This will be used for naming, and tagging artifacts and files",
    )
    parser.add_argument(
        "-rs",
        "--run_suffix",
        type=str,
        default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Run suffix. Default is current date and time. This will be used for naming, and tagging artifacts and files",
    )

    args = parser.parse_args()

    renderers = [Renderer.create(r) for r in args.renderers]

    run_id = args.run_prefix + "-" + args.run_suffix
    print("Starting DQN training...")
    print(f"Run Id: {run_id}")
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Max walls: {args.max_walls}")
    print(f"Training for {args.episodes} episodes")
    print(f"Using step rewards: {args.step_rewards}")
    print(f"Device: {my_device()}")
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
        run_id=run_id,
    )

    print("Training completed!")
