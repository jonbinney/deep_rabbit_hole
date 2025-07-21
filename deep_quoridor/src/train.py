import argparse
from typing import Optional

from agents.core.agent import AgentRegistry
from arena import Arena, PlayMode
from arena_utils import ArenaPlugin
from play import player_with_params
from plugins import SaveModelEveryNEpisodesPlugin, WandbTrainPlugin
from plugins.wandb_train import WandbParams
from renderers import Renderer, TrainingStatusRenderer
from utils import my_device, set_deterministic
from utils.subargs import parse_subargs


def train_dqn(
    episodes: int,
    board_size: int,
    max_walls: int,
    save_frequency: int,
    step_rewards: bool = True,
    wandb_params: Optional[WandbParams] = None,
    players: list = [],
    benchmarks: list = [],
    renderers: list[ArenaPlugin] = [],
    run_id: str = "",
    trigger_metrics: Optional[tuple[int, int]] = None,
):
    plugins = []
    total_episodes = episodes * (len(players) - 1) if len(players) > 1 else episodes

    after_save_method = None
    if wandb_params is not None:
        wandb_plugin = WandbTrainPlugin(
            wandb_params, total_episodes=total_episodes, agent_encoded_name=players[0], benchmarks=benchmarks
        )
        plugins.append(wandb_plugin)
        after_save_method = wandb_plugin.compute_tournament_metrics

    plugins.append(
        SaveModelEveryNEpisodesPlugin(
            update_every=save_frequency,
            board_size=board_size,
            max_walls=max_walls,
            save_final=wandb_params is None,
            run_id=run_id,
            after_save=after_save_method,
            trigger_metrics=trigger_metrics,
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
        max_steps=100,
    )

    # Self play
    if len(players) == 1:
        agent = AgentRegistry.create_from_encoded_name(players[0], arena.game)
        players = [agent, agent]

    arena.play_games(players=players, times=episodes, mode=PlayMode.FIRST_VS_RANDOM)
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
    parser.add_argument("-w", "--wandb", nargs="?", const="", default=None, type=str)

    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Use cProfile to profile the game.",
    )

    parser.add_argument(
        "--trigger-metrics",
        nargs=2,
        type=int,
        metavar=("wins", "last_episodes"),
        help="Trigger tournament metrics computation and save the model if there were 'wins' wins in the last 'last_episodes'",
    )

    parser.add_argument(
        "-b",
        "--benchmarks",
        nargs="+",
        type=player_with_params,
        default=["random", "simple"],
        help=f"List of players to benchmark against. Can include parameters in parentheses. Allowed types {AgentRegistry.names()}",
    )

    args = parser.parse_args()

    renderers = [Renderer.create(r) for r in args.renderers]

    if args.wandb is None:
        wandb_params = None
    elif args.wandb == "":
        wandb_params = WandbParams()
    else:
        wandb_params = parse_subargs(args.wandb, WandbParams)
        assert isinstance(wandb_params, WandbParams)

    print("Starting DQN training...")
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Max walls: {args.max_walls}")
    print(f"Training for {args.episodes} episodes")
    print(f"Using step rewards: {args.step_rewards}")
    print(f"Device: {my_device()}")
    print(f"Initializing random seed {args.seed}")

    # Set random seed for reproducibility
    set_deterministic(args.seed)

    def make_call():
        train_dqn(
            episodes=args.episodes,
            board_size=args.board_size,
            max_walls=args.max_walls,
            save_frequency=args.save_frequency,
            step_rewards=args.step_rewards,
            players=args.players,
            benchmarks=args.benchmarks,
            renderers=renderers,
            wandb_params=wandb_params,
            trigger_metrics=args.trigger_metrics,
        )

    if args.profile:
        import cProfile

        cProfile.run("make_call()", sort="tottime")
    else:
        make_call()
    print("Training completed!")
