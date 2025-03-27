import argparse
import os
import random

import numpy as np
import torch
from agents import FlatDQNAgent
from agents.flat_dqn import AbstractTrainableAgent
from agents.random import RandomAgent
from quoridor_env import env


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
    # Set random seed for reproducibility
    random.seed(42)

    game = env(board_size=board_size, max_walls=max_walls, step_rewards=step_rewards)

    agent1 = RandomAgent()
    agent2 = FlatDQNAgent(board_size, epsilon_decay=epsilon_decay)

    agents = [agent1, agent2]

    # Create directory for saving models if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    total_rewards = {a.name(): [] for a in agents}
    losses = {a.name(): [] for a in agents}

    # Enable to allow agents to play as P0 and P1 alternatively
    switch_players = False
    for episode in range(episodes):
        game.reset()

        agents_by_playerid = (
            {"player_0": agent2, "player_1": agent1}
            if switch_players and episode % 2 == 1
            else {"player_0": agent1, "player_1": agent2}
        )

        episode_reward = {a.name(): 0 for a in agents}
        episode_losses = {a.name(): [] for a in agents}

        # Agent iteration loop
        for player_id in game.agent_iter():
            observation, reward, termination, truncation, _ = game.last()

            agent = agents_by_playerid[player_id]
            agent_name = agent.name()

            # If the game is over, update negative reward and break the loop
            if termination or truncation:
                # Update the reward on the last element of the replay buffer (if there is any)
                # marking it as final and with a large negative reward
                if (
                    assign_negative_reward
                    and isinstance(agent, AbstractTrainableAgent)
                    and len(agent.replay_buffer) > 0
                ):
                    last = agent.replay_buffer.get_last()
                    last[2] = game.rewards[player_id]
                    last[4] = 1.0
                break

            # If it's the DQN agent's turn
            if isinstance(agent, AbstractTrainableAgent):
                # Get current state
                state = agent.observation_to_tensor(observation)

                # Select action using epsilon-greedy
                action = agent.get_action(game)

                # Execute action
                game.step(action)

                # Get the observation and rewards for THIS agent (not the opponent)
                # NOTE: If we used game.last() it will return the observation and rewards for the currently active agent
                # which, since we already did game.step(), is now the opponent
                next_observation = game.observe(player_id)

                # Make the reward much larger than 1, to make it stand out
                reward = game.rewards[player_id]

                # See if the game is over
                # TODO: Understand what is truncation and if either of these values are player dependent
                _, _, termination, truncation, _ = game.last()

                # Add to episode reward
                episode_reward[agent_name] += reward

                # Store transition in replay buffer
                next_state = agent.observation_to_tensor(next_observation)
                done = 1 if termination or truncation else 0
                agent.replay_buffer.add(
                    state.cpu().numpy(),
                    action,
                    reward,
                    next_state.cpu().numpy() if next_state is not None else np.zeros_like(state.cpu().numpy()),
                    done,
                )

                # Train the agent
                if len(agent.replay_buffer) > batch_size:
                    loss = agent.train(batch_size)
                    if loss is not None:
                        episode_losses[agent_name].append(loss)

            # If it's the random opponent's turn
            else:
                # Get action from random agent
                action = agent.get_action(game)

                # Execute action
                game.step(action)

        # Aggregate episode statistics
        for agent in agents:
            agent_name = agent.name()
            total_rewards[agent_name].append(episode_reward[agent_name])
            if episode_losses[agent_name]:
                avg_loss = sum(episode_losses[agent_name]) / len(episode_losses[agent_name])
                losses[agent_name].append(avg_loss)

        # Update target network periodically
        if episode % update_target_every == 0:
            for agent in agents:
                if isinstance(agent, AbstractTrainableAgent):
                    agent_name = agent.name()
                    agent.update_target_network()
                    avg_reward = (
                        sum(total_rewards[agent_name][-1 * update_target_every :])
                        / min(update_target_every, len(total_rewards[agent_name]))
                        if total_rewards[agent_name]
                        else 0.0
                    )
                    avg_loss = (
                        sum(losses[agent_name][-1 * update_target_every :])
                        / min(update_target_every, len(losses[agent_name]))
                        if losses[agent_name]
                        else 0.0
                    )
                    print(
                        f"{agent_name} Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}, "
                        f"Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}"
                    )

        # Save model periodically
        if episode % save_frequency == 0 and episode > 0:
            for agent in agents:
                if isinstance(agent, AbstractTrainableAgent):
                    agent_name = agent.name()
                    save_file = os.path.join(save_path, f"{agent.name()}_episode_{episode}.pt")
                    agent.save_model(save_file)
                    print(f"{agent_name} Model saved to {save_file}")

    # Save final model
    for agent in agents:
        if isinstance(agent, AbstractTrainableAgent):
            agent_name = agent.name()
            final_save_file = os.path.join(save_path, f"{agent.name()}_final.pt")
            agent.save_model(final_save_file)
            print(f"{agent_name} Final model saved to {final_save_file}")

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

    args = parser.parse_args()

    print("Starting DQN training...")
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Max walls: {args.max_walls}")
    print(f"Training for {args.episodes} episodes")
    print(f"Epsilon decay: {args.epsilon_decay}")
    print(f"Using step rewards: {args.step_rewards}")
    print(f"Assign negative reward: {args.assign_negative_reward}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

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
