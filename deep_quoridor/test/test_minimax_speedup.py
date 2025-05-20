import time
import argparse
import sys
import os
import numpy as np
from numba import njit, int32, float32

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import qgrid
from quoridor import Board, MoveAction, Player, Quoridor, WallAction, WallOrientation
from agents.simple import choose_action, SimpleParams, distance_to_row

# Use shared constants from qgrid
from qgrid import (
    WALL_ORIENTATION_VERTICAL,
    WALL_ORIENTATION_HORIZONTAL,
    CELL_FREE,
    CELL_PLAYER1,
    CELL_PLAYER2,
    CELL_WALL,
)


def test_minimax_speed(board_size=5, max_walls=3, max_depth=3, branching_factor=25, num_trials=5):
    """Test the speed of the minimax algorithm with Numba optimization"""
    
    # Create a new board with specified size
    board = Board(board_size=board_size, max_walls=max_walls)
    game = Quoridor(board)
    
    # First run to compile the Numba functions
    print("Compiling Numba functions...")
    _ = choose_action(
        game,
        Player.ONE,
        Player.TWO,
        max_depth=1,
        branching_factor=5,
        wall_sigma=0.5,
        discount_factor=0.99
    )
    print("Compilation complete")
    
    # Measure performance
    total_time = 0
    for i in range(num_trials):
        # Reset the game
        board = Board(board_size=board_size, max_walls=max_walls)
        game = Quoridor(board)
        
        # Add some walls to make it more interesting
        if i > 0:  # Skip adding walls for the first trial
            for _ in range(min(3, max_walls)):
                row = np.random.randint(0, board_size - 1)
                col = np.random.randint(0, board_size - 1)
                orientation = WallOrientation.VERTICAL if np.random.randint(0, 2) == 0 else WallOrientation.HORIZONTAL
                wall_orientation_qgrid = WALL_ORIENTATION_VERTICAL if orientation == WallOrientation.VERTICAL else WALL_ORIENTATION_HORIZONTAL
                
                # Use qgrid functions to check if wall placement is valid
                grid = game.board._grid
                player_positions = np.zeros((2, 2), dtype=np.int32)
                player_positions[0] = game.board.get_player_position(Player.ONE)
                player_positions[1] = game.board.get_player_position(Player.TWO)
                
                walls_remaining = np.zeros(2, dtype=np.int32)
                walls_remaining[0] = game.board.get_walls_remaining(Player.ONE)
                walls_remaining[1] = game.board.get_walls_remaining(Player.TWO)
                
                goal_rows = np.zeros(2, dtype=np.int32)
                goal_rows[0] = game.get_goal_row(Player.ONE)
                goal_rows[1] = game.get_goal_row(Player.TWO)
                
                current_player = 0  # Player.ONE is 0
                
                # Use qgrid's is_wall_action_valid function
                if qgrid.is_wall_action_valid(
                    grid, 
                    player_positions, 
                    walls_remaining, 
                    goal_rows, 
                    current_player, 
                    row, 
                    col, 
                    wall_orientation_qgrid
                ):
                    # Place the wall using the Board's method
                    game.board.add_wall(Player.ONE, (row, col), orientation, check_if_valid=False)
                    # Decrement walls remaining for Player.ONE
                    game.board._walls_remaining[Player.ONE] -= 1
        
        print(f"\nTrial {i+1}:")
        print(f"Board state:\n{game}")
        
        # Demonstrate the use of qgrid's distance_to_row function
        grid = game.board._grid
        player_positions_qgrid = np.zeros((2, 2), dtype=np.int32)
        player_positions_qgrid[0] = game.board.get_player_position(Player.ONE)
        player_positions_qgrid[1] = game.board.get_player_position(Player.TWO)
        
        p1_row, p1_col = player_positions_qgrid[0]
        goal_row_p1 = game.get_goal_row(Player.ONE)
        
        # Use qgrid's distance_to_row instead of the one in simple.py
        distance = qgrid.distance_to_row(grid, p1_row, p1_col, goal_row_p1)
        print(f"Distance to goal for Player 1: {distance}")
        
        # Time the minimax search
        start_time = time.time()
        action, value = choose_action(
            game,
            Player.ONE,
            Player.TWO,
            max_depth,
            branching_factor,
            wall_sigma=0.5,
            discount_factor=0.99,
        )
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        print(f"Took {elapsed_time:.4f}s to find action with value: {value:.2f}")
        
        # Apply the action
        if action:
            game.step(action)
            print(f"Chosen action: {action}")
            print(f"Updated board:\n{game}")
    
    avg_time = total_time / num_trials
    print(f"\nAverage time per search: {avg_time:.4f}s with depth={max_depth}, branching={branching_factor}")
    return avg_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the speed of minimax with Numba optimizations")
    parser.add_argument("--board_size", type=int, default=5, help="Board size")
    parser.add_argument("--max_walls", type=int, default=3, help="Maximum number of walls per player")
    parser.add_argument("--depth", type=int, default=3, help="Maximum search depth")
    parser.add_argument("--branching", type=int, default=25, help="Branching factor")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials to run")
    args = parser.parse_args()
    
    print(f"Testing minimax speed with:")
    print(f"- Board size: {args.board_size}x{args.board_size}")
    print(f"- Max walls: {args.max_walls}")
    print(f"- Search depth: {args.depth}")
    print(f"- Branching factor: {args.branching}")
    print(f"- Number of trials: {args.trials}")
    print("-" * 50)
    
    avg_time = test_minimax_speed(
        board_size=args.board_size,
        max_walls=args.max_walls,
        max_depth=args.depth,
        branching_factor=args.branching,
        num_trials=args.trials
    )
    
    print("-" * 50)
    print(f"Completed {args.trials} trials with average time: {avg_time:.4f}s")