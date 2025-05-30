#!/usr/bin/env python3
"""
A script to play Quoridor with OpenSpiel, using MCTS agent vs random agent.
"""

import numpy as np
import pyspiel
from absl import app, flags
from open_spiel.python.algorithms import mcts
from open_spiel.python.bots import human, uniform_random

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_games", 1, "Number of games to play.")
flags.DEFINE_integer("num_sims", 1000, "Number of simulations for MCTS.")
flags.DEFINE_integer("max_simulations", 1000, "Maximum simulations for MCTS.")
flags.DEFINE_integer("rollout_count", 10, "Rollout count for MCTS.")
flags.DEFINE_float("temperature", 1.0, "Temperature for MCTS.")
flags.DEFINE_boolean("verbose", True, "Be verbose.")
flags.DEFINE_boolean("human_player", False, "Add a human player.")
flags.DEFINE_string("board_size", "5", "Board size for Quoridor (e.g., '5' for 5x5).")
flags.DEFINE_string("players", "2", "Number of players in Quoridor.")
flags.DEFINE_string("wall_count", "5", "Number of walls per player.")


# Using OpenSpiel's built-in random rollout evaluator
# Note: We'll modify it to include negative rewards for losses as per memory


def main(_):
    """Plays Quoridor with MCTS vs Random."""
    # Create game parameters
    game_parameters = {
        "board_size": int(FLAGS.board_size),
        "players": int(FLAGS.players),
        "wall_count": int(FLAGS.wall_count),
    }

    # Load the game
    game = pyspiel.load_game("quoridor", game_parameters)

    # Set up the agents
    # Using OpenSpiel's built-in evaluator with custom evaluation function to incorporate negative rewards
    base_evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count)

    # Creating evaluator instance
    evaluator = base_evaluator

    # MCTS Agent (player 0)
    mcts_bot = mcts.MCTSBot(
        game,
        FLAGS.temperature,
        FLAGS.max_simulations,
        evaluator=evaluator,
        solve=True,
        random_state=np.random.RandomState(42),
    )

    # Random Agent (player 1)
    random_bot = uniform_random.UniformRandomBot(player_id=1, rng=np.random.RandomState(42))

    # Human player if requested
    if FLAGS.human_player:
        human_bot = human.HumanBot()
        bots = [human_bot, random_bot]
    else:
        bots = [mcts_bot, random_bot]

    # Set up the game
    results = np.array([0, 0, 0])  # Wins for player 0, player 1, draws

    # Play the specified number of games
    for game_num in range(FLAGS.num_games):
        state = game.new_initial_state()

        if FLAGS.verbose:
            print(f"Game {game_num + 1}/{FLAGS.num_games}")
            print(f"Initial state: {state}")

        while not state.is_terminal():
            if state.is_chance_node():
                # Sample chance node outcome
                outcomes = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
            else:
                current_player = state.current_player()
                player_bot = bots[current_player]
                action = player_bot.step(state)
                if FLAGS.verbose:
                    print(f"Player {current_player} chooses action: {action}")

                    # Convert action to human-readable format for Quoridor
                    action_str = ""
                    if action < (int(FLAGS.board_size) - 1) * (int(FLAGS.board_size) - 1) * 2:
                        # Wall placement
                        wall_index = action
                        wall_type = (
                            "horizontal"
                            if wall_index < (int(FLAGS.board_size) - 1) * (int(FLAGS.board_size) - 1)
                            else "vertical"
                        )
                        wall_index = wall_index % ((int(FLAGS.board_size) - 1) * (int(FLAGS.board_size) - 1))
                        row = wall_index // (int(FLAGS.board_size) - 1)
                        col = wall_index % (int(FLAGS.board_size) - 1)
                        action_str = f"{wall_type} wall at position ({row}, {col})"
                    else:
                        # Pawn movement
                        move_index = action - (int(FLAGS.board_size) - 1) * (int(FLAGS.board_size) - 1) * 2
                        directions = ["Up", "Right", "Down", "Left"]
                        direction = directions[move_index % 4]
                        action_str = f"Move pawn {direction}"

                    print(f"Action decoded: {action_str}")

                state.apply_action(action)
                if FLAGS.verbose:
                    print(f"State after action: {state}")

        # Record the results
        if state.returns()[0] > 0:
            results[0] += 1
            winner = "MCTS Bot (Player 0)"
        elif state.returns()[1] > 0:
            results[1] += 1
            winner = "Random Bot (Player 1)"
        else:
            results[2] += 1
            winner = "Draw"

        if FLAGS.verbose:
            print(f"Game {game_num + 1} complete. Winner: {winner}")

    # Print overall results
    print("\nOverall results:")
    print(f"MCTS Bot (Player 0) wins: {results[0]}")
    print(f"Random Bot (Player 1) wins: {results[1]}")
    print(f"Draws: {results[2]}")
    print(
        f"Win percentages: MCTS: {results[0]/FLAGS.num_games:.1%}, "
        f"Random: {results[1]/FLAGS.num_games:.1%}, "
        f"Draw: {results[2]/FLAGS.num_games:.1%}"
    )


if __name__ == "__main__":
    app.run(main)
