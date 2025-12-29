#!/usr/bin/env python3
"""
Create a policy database from minimax evaluations.

This script initializes a Quoridor game and uses the Rust minimax implementation
to evaluate all possible actions from the initial state, logging the results to
a parquet file for later analysis or training.
"""

import argparse
import os
import sys

import numpy as np

# Import the Rust module
import quoridor_rs
from quoridor import Board, Player, Quoridor


def create_policy_database(
    board_size: int = 9,
    max_walls: int = 10,
    max_depth: int = 4,
    branching_factor: int = 100,
    wall_sigma: float = 0.5,
    discount_factor: float = 0.99,
    heuristic: int = 1,
    output_file: str = "policy_db.parquet",
) -> int:
    """
    Create a policy database by evaluating actions from the initial game state.

    Args:
        board_size: Size of the Quoridor board (default: 9 for standard game)
        max_walls: Maximum walls each player can place (default: 10)
        max_depth: How many moves to look ahead in minimax search
        branching_factor: How many actions to consider at each minimax stage
        wall_sigma: Standard deviation for Gaussian sampling of wall placements
        discount_factor: Discount factor for future rewards
        heuristic: Which heuristic to use (0=none, 1=distance+walls)
        output_file: Path to save the parquet file

    Returns:
        Number of entries written to the database
    """
    print("Initializing Quoridor game...")
    print(f"  Board size: {board_size}x{board_size}")
    print(f"  Max walls: {max_walls} per player")
    print(f"  Max depth: {max_depth}")
    print(f"  Branching factor: {branching_factor}")
    print(f"  Wall sigma: {wall_sigma}")
    print(f"  Discount factor: {discount_factor}")
    print(f"  Heuristic: {heuristic}")
    print(f"  Number of threads: {os.environ['RAYON_NUM_THREADS']}")

    # Create initial game state
    board = Board(board_size=board_size, max_walls=max_walls)
    game = Quoridor(board=board, current_player=Player.ONE)

    # Extract game state for Rust function
    # Ensure correct dtypes for Rust function
    grid = board.get_grid().astype(np.int8)
    player_positions = np.ascontiguousarray(board._player_positions, dtype=np.int32)
    walls_remaining = np.ascontiguousarray(board._walls_remaining, dtype=np.int32)
    goal_rows = np.ascontiguousarray(game._goal_rows, dtype=np.int32)
    current_player = int(game.current_player)

    print("\nGame state:")
    print(f"  Grid shape: {grid.shape}")
    print(f"  Player positions: {player_positions}")
    print(f"  Walls remaining: {walls_remaining}")
    print(f"  Goal rows: {goal_rows}")
    print(f"  Current player: {current_player}")

    # Call Rust function to create policy database
    print("\nEvaluating actions and creating policy database...")
    print(f"  Output file: {output_file}")

    num_entries = quoridor_rs.create_policy_db(
        grid,
        player_positions,
        walls_remaining,
        goal_rows,
        current_player,
        max_depth,
        branching_factor,
        wall_sigma,
        discount_factor,
        heuristic,
        output_file,
    )

    print(f"\n✓ Successfully created policy database with {num_entries} entries")
    print(f"  Saved to: {output_file}")

    return num_entries


def main():
    parser = argparse.ArgumentParser(
        description="Create a policy database from Quoridor minimax evaluations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--board-size",
        type=int,
        default=3,
        help="Size of the Quoridor board",
    )

    parser.add_argument(
        "--max-walls",
        type=int,
        default=0,
        help="Maximum walls each player can place",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum steps before game is terminated",
    )

    parser.add_argument(
        "--branching-factor",
        type=int,
        default=10000,
        help="How many actions to consider at each minimax stage",
    )

    parser.add_argument(
        "--wall-sigma",
        type=float,
        default=0.5,
        help="Standard deviation for Gaussian sampling of wall placements",
    )

    parser.add_argument(
        "--discount-factor",
        type=float,
        default=1.0,
        help="Discount factor for future rewards",
    )

    parser.add_argument(
        "--heuristic",
        type=int,
        default=0,
        choices=[0, 1],
        help="Heuristic to use (0=none, 1=distance+walls)",
    )

    parser.add_argument("--num-threads", type=int, default=1, help="Number of threads to use for computation")

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="policy_db.parquet",
        help="Output parquet file path",
    )

    args = parser.parse_args()

    os.environ["RAYON_NUM_THREADS"] = str(args.num_threads)
    try:
        num_entries = create_policy_database(
            board_size=args.board_size,
            max_walls=args.max_walls,
            max_depth=args.max_steps,
            branching_factor=args.branching_factor,
            wall_sigma=args.wall_sigma,
            discount_factor=args.discount_factor,
            heuristic=args.heuristic,
            output_file=args.output,
        )
        print(f"Saved policy database with {num_entries} entries to {args.output}")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
