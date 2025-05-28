#!/usr/bin/env python3
"""
Quoridor game with MCTS agent vs Random agent using OpenSpiel framework.
"""

import sys
import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import random_agent
from open_spiel.python.observation import make_observation


def print_board(state):
    """
    Print the current Quoridor board.
    
    Args:
        state: The current game state.
    """
    print(state)


def play_game(mcts_bot, random_bot):
    """
    Play a game of Quoridor between an MCTS agent and a Random agent.
    
    Args:
        mcts_bot: The MCTS agent.
        random_bot: The Random agent.
    
    Returns:
        The final state of the game.
    """
    game = pyspiel.load_game("quoridor")
    state = game.new_initial_state()
    
    bots = [mcts_bot, random_bot]
    
    print("Initial state:")
    print_board(state)
    print()
    
    while not state.is_terminal():
        current_player = state.current_player()
        current_bot = bots[current_player]
        action = current_bot.step(state)
        
        action_str = state.action_to_string(current_player, action)
        print(f"Player {current_player} chose action: {action_str}")
        state.apply_action(action)
        
        print("New state:")
        print_board(state)
        print()
    
    # Print game outcome
    if state.returns()[0] > 0:
        print("Player 0 won!")
    elif state.returns()[0] < 0:
        print("Player 1 won!")
    else:
        print("Game ended in a draw!")
    
    return state


def main():
    """Main function to run the Quoridor game."""
    game = pyspiel.load_game("quoridor")
    
    # Create MCTS bot (Player 0)
    mcts_solver = mcts.MCTSBot(
        game,
        uct_c=2,
        max_simulations=1000,
        solve=False,  # Quoridor is too complex to solve completely
        random_state=np.random.RandomState(seed=42))
    
    # Create Random bot (Player 1)
    random_solver = random_agent.RandomAgent(
        game,
        player_id=1,
        random_state=np.random.RandomState(seed=42))
    
    # Play game
    final_state = play_game(mcts_solver, random_solver)
    
    print(f"Returns: {final_state.returns()}")


if __name__ == "__main__":
    main()
