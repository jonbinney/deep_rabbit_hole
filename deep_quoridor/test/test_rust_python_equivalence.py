#!/usr/bin/env python3
"""
Test script to verify that the Rust implementation matches the Python/Numba implementation.

This script compares the output of qgrid.distance_to_row (Numba) with
quoridor_rs.distance_to_row (Rust) on various test cases.
"""

import sys

try:
    import qgrid  # Python/Numba implementation
except ImportError:
    print("ERROR: Could not import qgrid. Make sure you're in the deep_quoridor directory.")
    sys.exit(1)

try:
    import quoridor_rs  # Rust implementation
except ImportError:
    print("ERROR: Could not import quoridor_rs. You need to build the Rust module first.")
    print("Run: cd rust && maturin develop")
    sys.exit(1)

from quoridor import Board, Player, WallOrientation


def test_distance_to_row_equivalence():
    """Test that Rust and Python implementations give identical results."""
    print("Testing distance_to_row equivalence...")

    test_cases = []

    # Test 1: Empty board, straight path
    board = Board(board_size=9, max_walls=10)
    test_cases.append(
        {
            "name": "Empty board - Player 1 at start",
            "grid": board._grid,
            "start_row": 0,
            "start_col": 4,
            "target_row": 8,
        }
    )

    # Test 2: Same position
    test_cases.append(
        {
            "name": "Same position",
            "grid": board._grid,
            "start_row": 4,
            "start_col": 4,
            "target_row": 4,
        }
    )

    # Test 3: Board with walls
    board_with_walls = Board(board_size=9, max_walls=10)
    board_with_walls.add_wall(Player.ONE, (4, 4), WallOrientation.HORIZONTAL, check_if_valid=False)
    board_with_walls.add_wall(Player.ONE, (4, 3), WallOrientation.HORIZONTAL, check_if_valid=False)
    test_cases.append(
        {
            "name": "Board with horizontal walls",
            "grid": board_with_walls._grid,
            "start_row": 0,
            "start_col": 4,
            "target_row": 8,
        }
    )

    # Test 4: Multiple positions on same board
    for start_row in [0, 2, 4, 6, 8]:
        for start_col in [0, 4, 8]:
            for target_row in [0, 4, 8]:
                test_cases.append(
                    {
                        "name": f"Empty board - ({start_row},{start_col}) to row {target_row}",
                        "grid": board._grid,
                        "start_row": start_row,
                        "start_col": start_col,
                        "target_row": target_row,
                    }
                )

    # Test 5: Complex wall configuration
    complex_board = Board(board_size=9, max_walls=10)
    complex_board.add_wall(Player.ONE, (2, 4), WallOrientation.VERTICAL, check_if_valid=False)
    complex_board.add_wall(Player.ONE, (3, 3), WallOrientation.HORIZONTAL, check_if_valid=False)
    complex_board.add_wall(Player.ONE, (5, 5), WallOrientation.VERTICAL, check_if_valid=False)
    test_cases.append(
        {
            "name": "Complex wall configuration",
            "grid": complex_board._grid,
            "start_row": 1,
            "start_col": 4,
            "target_row": 7,
        }
    )

    # Run all test cases
    passed = 0
    failed = 0

    for test_case in test_cases:
        grid = test_case["grid"]
        start_row = test_case["start_row"]
        start_col = test_case["start_col"]
        target_row = test_case["target_row"]

        # Call both implementations
        python_result = qgrid.distance_to_row(grid, start_row, start_col, target_row)
        rust_result = quoridor_rs.distance_to_row(grid, start_row, start_col, target_row)

        # Compare results
        if python_result == rust_result:
            passed += 1
            print(f"✓ {test_case['name']}: {python_result}")
        else:
            failed += 1
            print(f"✗ {test_case['name']}: Python={python_result}, Rust={rust_result}")

    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")

    if failed == 0:
        print("SUCCESS: All tests passed! ✓")
        return True
    else:
        print("FAILURE: Some tests failed. ✗")
        return False


def benchmark_performance():
    """Compare performance of Rust vs Python implementations."""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    import time

    # Create a board with some walls
    board = Board(board_size=9, max_walls=10)
    board.add_wall(Player.ONE, (2, 4), WallOrientation.VERTICAL, check_if_valid=False)
    board.add_wall(Player.ONE, (3, 3), WallOrientation.HORIZONTAL, check_if_valid=False)
    board.add_wall(Player.ONE, (5, 5), WallOrientation.VERTICAL, check_if_valid=False)
    grid = board._grid

    num_iterations = 10000

    # Benchmark Python/Numba
    start = time.time()
    for i in range(num_iterations):
        qgrid.distance_to_row(grid, 0, 4, 8)
    python_time = time.time() - start

    # Benchmark Rust
    start = time.time()
    for i in range(num_iterations):
        quoridor_rs.distance_to_row(grid, 0, 4, 8)
    rust_time = time.time() - start

    print(f"Python/Numba: {python_time:.4f}s for {num_iterations} iterations")
    print(f"Rust:         {rust_time:.4f}s for {num_iterations} iterations")
    print(f"Speedup:      {python_time / rust_time:.2f}x")


if __name__ == "__main__":
    success = test_distance_to_row_equivalence()

    if success:
        try:
            benchmark_performance()
        except Exception as e:
            print(f"\nBenchmark failed: {e}")

    sys.exit(0 if success else 1)
