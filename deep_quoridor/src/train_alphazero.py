import argparse
import time
from typing import Optional

from agents.alphazero import AlphaZeroAgent, AlphaZeroParams
from arena import Arena, PlayMode
from plugins import SaveModelEveryNEpisodesPlugin, WandbTrainPlugin
from plugins.wandb_train import WandbParams
from renderers import TrainingStatusRenderer
from utils import my_device, set_deterministic
from utils.subargs import parse_subargs


def train_alphazero(
    episodes: int,
    board_size: int,
    max_walls: int,
    save_frequency: int,
    alphazero_params: AlphaZeroParams,
    wandb_params: Optional[WandbParams] = None,
    run_id: str = "",
    trigger_metrics: Optional[tuple[int, int]] = None,
):
    """Train AlphaZero agent using self-play."""

    plugins = []

    # Setup WandB plugin if specified
    after_save_method = None
    if wandb_params is not None:
        wandb_plugin = WandbTrainPlugin(wandb_params, total_episodes=episodes, agent_encoded_name="alphazero")
        plugins.append(wandb_plugin)
        after_save_method = wandb_plugin.compute_tournament_metrics

    # Setup model saving plugin
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

    # Create arena for self-play
    print_plugin = TrainingStatusRenderer(total_episodes=episodes)
    arena = Arena(
        board_size=board_size,
        max_walls=max_walls,
        step_rewards=False,  # AlphaZero uses terminal rewards only
        renderers=[print_plugin],
        plugins=plugins,
        swap_players=True,
        max_steps=1000,
    )

    # Create AlphaZero agents for self-play
    # Both agents share the same network but one is in training mode
    agent1 = AlphaZeroAgent(
        board_size=board_size,
        max_walls=max_walls,
        observation_space=arena.game.observation_space,
        action_space=arena.game.action_space,
        params=alphazero_params,
        load_model_if_needed=False,
    )

    # Second agent shares the network but doesn't train
    alphazero_params_inference = create_alphazero_params_from_args(
        type(
            "Args",
            (),
            {
                "learning_rate": alphazero_params.learning_rate,
                "batch_size": alphazero_params.batch_size,
                "train_every": alphazero_params.train_every,
                "temperature": alphazero_params.temperature,
                "mcts_simulations": alphazero_params.mcts_simulations,
                "c_puct": alphazero_params.c_puct,
                "replay_buffer_size": alphazero_params.replay_buffer_size,
            },
        )(),
        training_mode=False,
    )

    agent2 = AlphaZeroAgent(
        board_size=board_size,
        max_walls=max_walls,
        observation_space=arena.game.observation_space,
        action_space=arena.game.action_space,
        training_instance=agent1,
        params=alphazero_params_inference,
        load_model_if_needed=False,
    )

    print("Starting AlphaZero self-play training...")
    print(f"Board size: {board_size}x{board_size}")
    print(f"Max walls: {max_walls}")
    print(f"Training for {episodes} episodes")
    print(f"MCTS simulations: {alphazero_params.mcts_simulations}")
    print(f"Temperature: {alphazero_params.temperature}")
    print(f"Train every: {alphazero_params.train_every} episodes")
    print(f"Device: {my_device()}")

    # Run self-play training
    start_time = time.time()

    # Use FIRST_VS_RANDOM mode but with identical agents for true self-play
    players = [agent1, agent2]
    arena.play_games(players=players, times=episodes, mode=PlayMode.ALL_VS_ALL)

    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training completed in {training_time:.2f} seconds!")
    print(f"Average time per episode: {training_time / episodes:.2f} seconds")

    return agent1


def create_alphazero_params_from_args(args, training_mode=False) -> AlphaZeroParams:
    """Create AlphaZeroParams from command line arguments."""
    return AlphaZeroParams(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_every=args.train_every,
        temperature=args.temperature,
        mcts_simulations=args.mcts_simulations,
        c_puct=args.c_puct,
        replay_buffer_size=args.replay_buffer_size,
        training_mode=training_mode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AlphaZero agent for Quoridor using self-play")

    # Game parameters
    parser.add_argument("-N", "--board_size", type=int, default=9, help="Board size")
    parser.add_argument("-W", "--max_walls", type=int, default=10, help="Max walls per player")
    parser.add_argument("-e", "--episodes", type=int, default=1000, help="Number of episodes to train for")

    # AlphaZero specific parameters
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for neural network")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--train_every", type=int, default=100, help="Train network every N episodes")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for action selection")
    parser.add_argument("--mcts_simulations", type=int, default=800, help="Number of MCTS simulations per move")
    parser.add_argument("--c_puct", type=float, default=1.0, help="UCB exploration constant")
    parser.add_argument("--replay_buffer_size", type=int, default=10000, help="Size of replay buffer")

    # Training parameters
    parser.add_argument(
        "-f", "--save_frequency", type=int, default=500, help="How often to save the model (in episodes)"
    )
    parser.add_argument("-i", "--seed", type=int, default=42, help="Random seed for training")

    # WandB parameters
    parser.add_argument("-w", "--wandb", nargs="?", const="", default=None, type=str, help="Enable WandB logging")

    # Other parameters
    parser.add_argument("--profile", action="store_true", default=False, help="Use cProfile to profile the training")
    parser.add_argument(
        "--trigger-metrics",
        nargs=2,
        type=int,
        metavar=("wins", "last_episodes"),
        help="Trigger tournament metrics computation if there were 'wins' wins in the last 'last_episodes'",
    )

    args = parser.parse_args()

    # Parse WandB parameters
    if args.wandb is None:
        wandb_params = None
    elif args.wandb == "":
        wandb_params = WandbParams()
    else:
        wandb_params = parse_subargs(args.wandb, WandbParams)
        assert isinstance(wandb_params, WandbParams)

    print("Starting AlphaZero training...")
    print(f"Device: {my_device()}")
    print(f"Initializing random seed {args.seed}")

    # Set random seed for reproducibility
    set_deterministic(args.seed)

    def make_call():
        # Create AlphaZero parameters for training agent
        alphazero_params_training = create_alphazero_params_from_args(args, training_mode=True)

        train_alphazero(
            episodes=args.episodes,
            board_size=args.board_size,
            max_walls=args.max_walls,
            save_frequency=args.save_frequency,
            alphazero_params=alphazero_params_training,
            wandb_params=wandb_params,
            trigger_metrics=args.trigger_metrics,
        )

    if args.profile:
        import cProfile

        cProfile.run("make_call()", sort="tottime")
    else:
        make_call()

    print("AlphaZero training completed!")
