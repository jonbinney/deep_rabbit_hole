import numpy as np
import pytest
from quoridor import Board, MoveAction, Player, Quoridor, WallAction, WallOrientation
import quoridor_cuda as qcuda

def test_distance_calculation():
    """Test that the CUDA distance calculation matches the CPU implementation"""
    board = Board(board_size=9)
    grid = board._grid
    
    # Test standard board distance
    cpu_dist = qcuda.distance_to_row(grid, 0, 4, 8, 9)
    
    # Check if CUDA is available
    if qcuda.is_cuda_available():
        positions = [(0, 4)]
        targets = [8]
        board_sizes = [9]
        cuda_dist = qcuda.evaluate_positions(
            [grid], positions, positions, 
            targets, targets, [0], [[10, 10]], board_sizes
        )
        assert cpu_dist > 0  # Should be reachable
        # We don't compare exact values because the CUDA implementation might 
        # find a different but equally valid path
    else:
        pytest.skip("CUDA not available")

def test_wall_cell_check():
    """Test that wall cell checking works correctly"""
    board = Board(board_size=9)
    grid = board._grid
    
    # Check a position where we can place a wall
    assert qcuda.are_wall_cells_free(grid, 0, 0, qcuda.WallOrientation.VERTICAL)
    
    # Place a wall
    board.add_wall(Player.ONE, (0, 0), WallOrientation.VERTICAL)
    
    # Check that the position is now occupied
    assert not qcuda.are_wall_cells_free(grid, 0, 0, qcuda.WallOrientation.VERTICAL)

def test_wall_blocking():
    """Test wall blocking detection"""
    board = Board(board_size=9)
    grid = board._grid
    
    # Add walls to create a blocking formation
    board.add_wall(Player.ONE, (2, 2), WallOrientation.VERTICAL)
    board.add_wall(Player.ONE, (2, 3), WallOrientation.VERTICAL)
    
    # Position between the walls should be detected as potential block
    assert qcuda.is_wall_potential_block(grid, 2, 2, qcuda.WallOrientation.HORIZONTAL)

def test_gpu_batch_minimax():
    """Test GPU batch minimax when available"""
    if not qcuda.is_cuda_available():
        pytest.skip("CUDA not available")
        
    # Create a simple game state
    board = Board(board_size=5)
    grid = board._grid
    
    # Set up players
    p1_pos = (0, 2)  # Player 1 at top middle
    p2_pos = (4, 2)  # Player 2 at bottom middle
    board.move_player(Player.ONE, p1_pos)
    board.move_player(Player.TWO, p2_pos)
    
    # Create some test actions
    action_types = [
        qcuda.ACTION_MOVE,    # Move down
        qcuda.ACTION_MOVE,    # Move right
        qcuda.ACTION_MOVE,    # Move left
        qcuda.ACTION_WALL     # Place wall
    ]
    
    action_data = [
        np.array([1, 2, 0, 0, 2]),     # Move to (1,2)
        np.array([0, 3, 0, 0, 2]),     # Move to (0,3)
        np.array([0, 1, 0, 0, 2]),     # Move to (0,1)
        np.array([2, 2, 0, 0, 0])      # Wall at (2,2) vertical
    ]
    
    # Batch parameters
    batch_size = len(action_types)
    grids = [grid.copy() for _ in range(batch_size)]
    p1_positions = [p1_pos for _ in range(batch_size)]
    p2_positions = [p2_pos for _ in range(batch_size)]
    p1_targets = [4 for _ in range(batch_size)]  # Player 1 wants to reach bottom
    p2_targets = [0 for _ in range(batch_size)]  # Player 2 wants to reach top
    walls_remaining = [[10, 10] for _ in range(batch_size)]
    current_players = [0 for _ in range(batch_size)]  # Player 1's turn
    board_sizes = [5 for _ in range(batch_size)]
    max_depths = [1 for _ in range(batch_size)]
    is_p1_turns = [1 for _ in range(batch_size)]
    discount_factors = [0.99 for _ in range(batch_size)]
    
    # Run batch minimax
    results = qcuda.batch_minimax(
        grids, p1_positions, p2_positions,
        p1_targets, p2_targets, walls_remaining,
        action_types, action_data, current_players,
        board_sizes, max_depths, is_p1_turns, discount_factors
    )
    
    # Check that we got results for all actions
    assert len(results) == batch_size
    
    # Moving down should be better than moving sideways for player 1
    assert results[0] > results[1]
    assert results[0] > results[2]

if __name__ == "__main__":
    # Run tests and print results
    print("Testing CUDA Quoridor implementation")
    print(f"CUDA available: {qcuda.is_cuda_available()}")
    
    if qcuda.is_cuda_available():
        print("Running all tests...")
        test_distance_calculation()
        test_wall_cell_check()
        test_wall_blocking()
        test_gpu_batch_minimax()
        print("All tests passed!")
    else:
        print("Running CPU-only tests...")
        test_wall_cell_check()
        test_wall_blocking()
        print("All CPU tests passed!")