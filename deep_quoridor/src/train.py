import argparse
import os
from typing import Optional

import torch
import utils
from agents.dexp import DExpAgent
from agents.flat_dqn import AbstractTrainableAgent
from agents.random import RandomAgent
from arena import Arena
from arena_utils import ArenaPlugin
from renderers import Renderer

import wandb


def compute_loss_and_reward(agent: AbstractTrainableAgent, length: int):
    avg_reward = (
        sum(agent.episodes_rewards[-length:]) / min(length, len(agent.episodes_rewards))
        if agent.episodes_rewards
        else 0.0
    )
    avg_loss = (
        sum(agent.train_call_losses[-length:]) / min(length, len(agent.train_call_losses))
        if agent.train_call_losses
        else 0.0
    )

    return avg_loss, avg_reward


class TrainingStatusRenderer(Renderer):
    def __init__(self, update_every: int, total_episodes: int, agents: list[AbstractTrainableAgent]):
        self.agents = agents
        self.update_every = update_every
        self.total_episodes = total_episodes
        self.episode_count = 0

    def end_game(self, game, result):
        if self.episode_count % self.update_every == 0 or self.episode_count == (self.total_episodes - 1):
            for agent in self.agents:
                agent_name = agent.name()
                avg_reward, avg_loss = compute_loss_and_reward(agent, self.update_every)
                print(
                    f"{agent_name} Episode {self.episode_count + 1}/{self.total_episodes}, Avg Reward: {avg_reward:.2f}, "
                    f"Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}"
                )
        self.episode_count += 1
        return


class TrainingStatusWandb(ArenaPlugin):
    def __init__(self, update_every: int, total_episodes: int, agents: list[AbstractTrainableAgent]):
        self.agents = agents
        self.update_every = update_every
        self.total_episodes = total_episodes
        self.episode_count = 0
        self.tags = [agent.model_id() for agent in agents]

    def start_arena(self, game, total_games: int):
        config = {
            "board_size": game.board_size,
            "max_walls": game.max_walls,
        }
        for agent in self.agents:
            config.update({f"{agent.name()} hyperparameters": agent.model_hyperparameters()})

        self.run = wandb.init(
            project="deep_quoridor",
            job_type="train",
            config=config,
            tags=self.tags,
        )

    def end_game(self, game, result):
        if self.episode_count % self.update_every == 0:
            for agent in self.agents:
                avg_reward, avg_loss = compute_loss_and_reward(agent, self.update_every)

                self.run.log(
                    {"loss": avg_loss, "reward": avg_reward, "epsilon": agent.epsilon},
                    step=self.episode_count,
                )
        self.episode_count += 1

    def save(self, agent: AbstractTrainableAgent, filename: str):
        artifact = wandb.Artifact(f"{agent.model_id()}", type="model", metadata={"episodes": self.total_episodes})
        artifact.add_file(local_path=filename)
        artifact.save()


class SaveModelEveryNEpisodesPlugin(ArenaPlugin):
    def __init__(
        self,
        update_every: int,
        path: str,
        board_size: int,
        max_walls: int,
        agents: list[AbstractTrainableAgent],
        wandb_plugin: Optional[TrainingStatusWandb],
    ):
        self.agents = agents
        self.update_every = update_every
        self.path = path
        self.episode_count = 0
        self.board_size = board_size
        self.max_walls = max_walls
        self.wandb_plugin = wandb_plugin

    def end_game(self, game, result):
        if self.episode_count % self.update_every == 0 and self.episode_count > 0:
            self._save_models(f"_episode_{self.episode_count}")
        self.episode_count += 1

    def end_arena(self, game, results):
        self._save_models("_final", final=True)

    def _save_models(self, suffix: str, final: bool = False):
        for agent in self.agents:
            save_file = os.path.join(self.path, f"{agent.model_id()}{suffix}.pt")
            agent.save_model(save_file)
            print(f"{agent.name()} Model saved to {save_file}")

            if self.wandb_plugin is not None and final:
                self.wandb_plugin.save(agent, save_file)


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
    use_wandb=False,
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

    agent1 = RandomAgent()
    agent2 = DExpAgent(
        board_size=board_size,
        max_walls=max_walls,
        epsilon=1,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        update_target_every=update_target_every,
        assing_negative_reward=assign_negative_reward,
        training_mode=True,
    )
    if use_wandb:
        wandb_plugin = TrainingStatusWandb(update_every=10, total_episodes=episodes, agents=[agent2])
    else:
        wandb_plugin = None

    save_plugin = SaveModelEveryNEpisodesPlugin(
        update_every=save_frequency,
        path=save_path,
        board_size=board_size,
        max_walls=max_walls,
        agents=[agent2],
        wandb_plugin=wandb_plugin,
    )
    print_plugin = TrainingStatusRenderer(
        update_every=100,
        total_episodes=episodes,
        agents=[agent2],
    )
    arena = Arena(
        board_size=board_size,
        max_walls=max_walls,
        step_rewards=step_rewards,
        renderers=[print_plugin],
        plugins=[save_plugin, wandb_plugin],
    )

    arena.play_games(players=[agent1, agent2], times=episodes)
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
    parser.add_argument(
        "--no-wandb",
        action="store_false",
        default=True,
        help="Disable Weights & Biases logging",
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
    utils.set_deterministic(args.seed)

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
        use_wandb=args.no_wandb,
    )

    print("Training completed!")
