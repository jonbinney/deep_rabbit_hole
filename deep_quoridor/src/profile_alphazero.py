import cProfile
import os
import time
from typing import Tuple

from agents.alphazero_os import AlphaZeroOSAgent, AlphaZeroOSParams
from agents.random import RandomAgent
from gymnasium import spaces
from quoridor_env import env


def run_games(n_games: int = 10) -> Tuple[int, int, int]:
    """
    Run n_games of AlphaZero against a random agent and return the win statistics.

    Args:
        n_games: Number of games to run

    Returns:
        Tuple of (alphazero_wins, random_wins, draws)
    """
    # Track wins
    alphazero_wins = 0
    random_wins = 0
    draws = 0

    # Create a new environment for each game
    quoridor_env = env(board_size=5, max_walls=3)

    # Setup action space
    action_space = spaces.Discrete(quoridor_env.board_size**2 + (quoridor_env.wall_size**2) * 2)

    # Initialize agents
    alphazero_agent = AlphaZeroOSAgent(
        board_size=quoridor_env.board_size,
        max_walls=quoridor_env.max_walls,
        action_space=action_space,
        params=AlphaZeroOSParams(
            n=100,  # Number of MCTS simulations
            c=1.4,  # Exploration constant
            checkpoint_path=os.path.join("models", "osaz", "alphazero_os_B5W3_mv1_best"),
        ),
    )

    random_agent = RandomAgent(action_space=action_space)

    for game_idx in range(n_games):
        print(f"Starting game {game_idx+1}/{n_games}")

        # Assign agents
        agents = {
            "player_0": alphazero_agent,
            "player_1": random_agent,
        }

        # Reset environment
        quoridor_env.reset()

        alphazero_agent.start_game(quoridor_env, "player_0")

        # For each agent, play until done
        while not quoridor_env.terminations[quoridor_env.agent_selection]:
            agent_id = quoridor_env.agent_selection
            agent = agents[agent_id]

            # Get observation
            observation = quoridor_env.observe(agent_id)

            # Get action from agent
            action = int(agent.get_action(observation))

            # Step environment
            quoridor_env.step(action)

            if agent_id == "player_1":
                alphazero_agent.handle_opponent_step_outcome(observation, None, None, None, action)

        # Check winner
        winner = quoridor_env.winner()

        if winner == 0:
            alphazero_wins += 1
            print(f"Game {game_idx+1}: AlphaZero wins!")
        elif winner == 1:
            random_wins += 1
            print(f"Game {game_idx+1}: Random agent wins!")
        else:
            draws += 1
            print(f"Game {game_idx+1}: Draw!")

        print(f"Current stats - AlphaZero: {alphazero_wins}, Random: {random_wins}, Draws: {draws}")

    return alphazero_wins, random_wins, draws


def main():
    """
    Main function to profile running 10 games of AlphaZero vs random agent.
    """
    start_time = time.time()

    # Run the games
    alphazero_wins, random_wins, draws = run_games(n_games=10)

    end_time = time.time()

    # Print results
    print("\n===== Results =====")
    print(f"AlphaZero wins: {alphazero_wins}")
    print(f"Random agent wins: {random_wins}")
    print(f"Draws: {draws}")
    print(f"Win rate: {alphazero_wins / 10 * 100:.1f}%")
    print(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # Run with cProfile
    cProfile.run("main()", sort="tottime")
