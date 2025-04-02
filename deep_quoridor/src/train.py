import argparse
import os

import torch
from agents import DExpAgent, GreedyAgent, SimpleAgent
from agents.dexp import DExpPlayParams
from arena import Arena
from plugins import SaveModelEveryNEpisodesPlugin
from renderers import TrainingStatusRenderer


def train_dqn(
    episodes,
    batch_size,
    update_target_every,
    board_size,
    max_walls,
    epsilon_decay=0.9999,
    save_path="models",
    save_frequency=100,
    step_rewards=True,
    assign_negative_reward=False,
):
    """
    Train a DQN agent to play Quoridor.

    Julian notes:
    - This is for now working for a trivial 3x3 board with no walls
    - It teaches the agent to use black (player 2) only, against a random agent
      Note that in a 3x3 board with no walls, black always wins (if it wants)
    - It's currently not assigning negative rewards for losing

    Args:
        episodes: Number of episodes to train for
        batch_size: Size of batches to sample from replay buffer
        update_target_every: Number of episodes between target network updates
        board_size: Size of the Quoridor board
        max_walls: Maximum number of walls per player
        save_path: Directory to save trained models
        save_frequency: How often to save the model (in episodes)
        step_rewards: Whether to use step rewards
    """
    # Create directory for saving models if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    agent1 = SimpleAgent()
    agent2 = DExpAgent(
        board_size=board_size,
        max_walls=max_walls,
        epsilon=1,
        epsilon_decay=epsilon_decay,
        gamma=0.9,
        batch_size=batch_size,
        update_target_every=update_target_every,
        assign_negative_reward=assign_negative_reward,
        params=DExpPlayParams(use_rotate_board=True, split_board=False, include_turn=True),
    )
    agent2.training_mode = True
    agent2.final_reward_multiplier = 2
    # agent2.use_opponentns_actions = False
    # agent2.load_model("models/dexp_B5W0_base.pt")
    agent3 = GreedyAgent()  # noqa: F841

    save_plugin = SaveModelEveryNEpisodesPlugin(
        update_every=save_frequency, path=save_path, agents=[agent2], board_size=board_size, max_walls=max_walls
    )
    print_plugin = TrainingStatusRenderer(
        update_every=1,
        total_episodes=episodes,
        agents=[agent2],
    )
    arena = Arena(
        board_size=board_size,
        max_walls=max_walls,
        step_rewards=step_rewards,
        renderers=[print_plugin],
        plugins=[save_plugin],
        swap_players=True,
    )

    agents = [agent2, agent1]
    print("Agent configurations:")
    for a in agents:
        print("-----------------------------------------------------------------")
        print(f"Agent: {a.name()}")
        print("Configuration")
        print(a.yaml_config())

    arena.play_games(players=agents, times=episodes)
    # agent2.epsilon = 0.8
    # arena.play_games(players=[agent2, agent3], times=episodes)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent for Quoridor")
    parser.add_argument("-N", "--board_size", type=int, default=9, help="Board Size")
    parser.add_argument("-W", "--max_walls", type=int, default=10, help="Max walls per player")
    parser.add_argument("-e", "--episodes", type=int, default=1000, help="Number of episodes to train for")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("-u", "--update_target", type=int, default=100, help="Episodes between target network updates")
    parser.add_argument("-s", "--step_rewards", action="store_true", default=False, help="Enable step rewards")
    parser.add_argument("-p", "--save_path", type=str, default="models", help="Directory to save models")
    parser.add_argument(
        "-f", "--save_frequency", type=int, default=500, help="How often to save the model (in episodes)"
    )
    parser.add_argument("-d", "--epsilon_decay", type=float, default=0.9999, help="Epsilon decay rate for exploration")
    parser.add_argument(
        "-n",
        "--assign_negative_reward",
        action="store_true",
        default=False,
        help="Assign negative reward when agent loses",
    )
    parser.add_argument(
        "-i",
        "--seed",
        type=int,
        default=42,
        help="Initializes the random seed for the training. Default is 42",
    )

    args = parser.parse_args()

    print("Starting DQN training...")
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Max walls: {args.max_walls}")
    print(f"Training for {args.episodes} episodes")
    print(f"Epsilon decay: {args.epsilon_decay}")
    print(f"Using step rewards: {args.step_rewards}")
    print(f"Assign negative reward: {args.assign_negative_reward}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Initializing random seed {args.seed}")

    # Set random seed for reproducibility
    set_deterministic(args.seed)
    train_dqn(
        episodes=args.episodes,
        batch_size=args.batch_size,
        update_target_every=args.update_target,
        board_size=args.board_size,
        max_walls=args.max_walls,
        epsilon_decay=args.epsilon_decay,
        save_path=args.save_path,
        save_frequency=args.save_frequency,
        step_rewards=args.step_rewards,
        assign_negative_reward=args.assign_negative_reward,
    )

    print("Training completed!")
