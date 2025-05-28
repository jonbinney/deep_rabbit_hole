#!/usr/bin/env python3
"""
TicTacToe game with MCTS agent vs Random agent using OpenSpiel framework.
"""

import sys
import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import random_agent


def print_board(state):
    """
    Print the current game board.
    
    Args:
        state: The current game state.
    """
    board_str = ""
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            if state.observation_string()[idx] == ".":
                board_str += "."
            elif state.observation_string()[idx] == "o":
                board_str += "O"
            else:
                board_str += "X"
            if col < 2:
                board_str += " "
        if row < 2:
            board_str += "\n"
    
    print(board_str)


def play_game(mcts_bot, random_bot):
    """
    Play a game of TicTacToe between an MCTS agent and a Random agent.
    
    Args:
        mcts_bot: The MCTS agent.
        random_bot: The Random agent.
    
    Returns:
        The final state of the game.
    """
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    
    bots = [mcts_bot, random_bot]
    
    print("Initial state:")
    print_board(state)
    print()
    
    while not state.is_terminal():
        current_player = state.current_player()
        current_bot = bots[current_player]
        action = current_bot.step(state)
        
        print(f"Player {current_player} ({('X' if current_player == 0 else 'O')}) chose action: {action}")
        state.apply_action(action)
        
        print("New state:")
        print_board(state)
        print()
    
    # Print game outcome
    if state.returns()[0] > 0:
        print("Player 0 (X) won!")
    elif state.returns()[0] < 0:
        print("Player 1 (O) won!")
    else:
        print("Game ended in a draw!")
    
    return state


def main():
    """Main function to run the TicTacToe game."""
    game = pyspiel.load_game("tic_tac_toe")
    
    # Create MCTS bot (Player 0)
    mcts_solver = mcts.MCTSBot(
        game,
        uct_c=2,
        max_simulations=1000,
        solve=True,
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
