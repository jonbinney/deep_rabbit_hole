import argparse
import numpy as np
import os
from agents import DQNAgent, RandomAgent
from quoridor_env import env


def train_dqn(
    episodes,
    batch_size,
    update_target_every,
    board_size,
    max_walls,
    save_path="models",
    model_name="dqn_agent",
    save_frequency=100,
    step_rewards=True,
):
    """
    Train a DQN agent to play Quoridor.

    Args:
        episodes: Number of episodes to train for
        batch_size: Size of batches to sample from replay buffer
        update_target_every: Number of episodes between target network updates
        board_size: Size of the Quoridor board
        max_walls: Maximum number of walls per player
        save_path: Directory to save trained models
        model_name: Base name for saved models
        save_frequency: How often to save the model (in episodes)
        step_rewards: Whether to use step rewards
    """
    game = env(board_size=board_size, max_walls=max_walls, step_rewards=step_rewards)

    # Calculate action space size
    action_size = board_size**2 + ((board_size - 1) ** 2) * 2

    # Create the DQN agent
    dqn_agent = DQNAgent(board_size, action_size)

    # Create a random opponent
    random_agent = RandomAgent()

    # Create directory for saving models if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    total_rewards = []
    losses = []

    for episode in range(episodes):
        game.reset()

        # Reset episode-specific variables
        episode_reward = 0
        episode_losses = []

        # Agent iteration loop
        for agent_name in game.agent_iter():
            observation, reward, termination, truncation, _ = game.last()

            # If the game is over, break the loop
            if termination or truncation:
                break

            # If it's the DQN agent's turn
            if agent_name == "player_0":
                # Get current state
                state = dqn_agent.preprocess_observation(observation)

                # Select action using epsilon-greedy
                action = dqn_agent.get_action(game)

                # Execute action
                game.step(action)

                # Get new state, reward, etc.
                next_observation, reward, termination, truncation, _ = game.last()

                # Add to episode reward
                episode_reward += reward

                # Store transition in replay buffer
                next_state = (
                    dqn_agent.preprocess_observation(next_observation) if not (termination or truncation) else None
                )
                done = 1.0 if (termination or truncation) else 0.0
                dqn_agent.replay_buffer.add(
                    state.cpu().numpy(),
                    action,
                    reward,
                    next_state.cpu().numpy() if next_state is not None else np.zeros_like(state.cpu().numpy()),
                    done,
                )

                # Train the agent
                if len(dqn_agent.replay_buffer) > batch_size:
                    loss = dqn_agent.train(batch_size)
                    if loss is not None:
                        episode_losses.append(loss)

            # If it's the random opponent's turn
            else:
                # Get action from random agent
                action = random_agent.get_action(game)

                # Execute action
                game.step(action)

        # Aggregate episode statistics
        total_rewards.append(episode_reward)
        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
            losses.append(avg_loss)

        # Update target network periodically
        if episode % update_target_every == 0:
            dqn_agent.update_target_network()
            avg_reward = sum(total_rewards[-100:]) / min(100, len(total_rewards)) if total_rewards else 0.0
            avg_loss = sum(losses[-100:]) / min(100, len(losses)) if losses else 0.0
            print(
                f"Episode {episode}/{episodes}, Avg Reward: {avg_reward:.2f}, "
                f"Avg Loss: {avg_loss:.4f}, Epsilon: {dqn_agent.epsilon:.4f}"
            )

        # Save model periodically
        if episode % save_frequency == 0 and episode > 0:
            save_file = os.path.join(save_path, f"{model_name}_episode_{episode}.pt")
            dqn_agent.save_model(save_file)
            print(f"Model saved to {save_file}")

    # Save final model
    final_save_file = os.path.join(save_path, f"{model_name}_final.pt")
    dqn_agent.save_model(final_save_file)
    print(f"Final model saved to {final_save_file}")

    return dqn_agent, total_rewards, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent for Quoridor")
    parser.add_argument("-N", "--board_size", type=int, default=9, help="Board Size")
    parser.add_argument("-W", "--max_walls", type=int, default=10, help="Max walls per player")
    parser.add_argument("-e", "--episodes", type=int, default=1000, help="Number of episodes to train for")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("-u", "--update_target", type=int, default=100, help="Episodes between target network updates")
    parser.add_argument("--step_rewards", action="store_true", default=False, help="Enable step rewards")
    parser.add_argument("--save_path", type=str, default="models", help="Directory to save models")
    parser.add_argument("--model_name", type=str, default="dqn_agent", help="Base name for saved models")
    parser.add_argument("--save_frequency", type=int, default=500, help="How often to save the model (in episodes)")

    args = parser.parse_args()

    print("Starting DQN training...")
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Max walls: {args.max_walls}")
    print(f"Training for {args.episodes} episodes")
    print(f"Using step rewards: {args.step_rewards}")

    agent, rewards, losses = train_dqn(
        episodes=args.episodes,
        batch_size=args.batch_size,
        update_target_every=args.update_target,
        board_size=args.board_size,
        max_walls=args.max_walls,
        save_path=args.save_path,
        model_name=args.model_name,
        save_frequency=args.save_frequency,
        step_rewards=args.step_rewards,
    )

    print("Training completed!")
