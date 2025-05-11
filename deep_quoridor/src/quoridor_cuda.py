import numpy as np
from numba import cuda, int32, njit
from enum import IntEnum
from typing import List, Optional, Tuple

# Constants matching the original quoridor.py
FREE = -1
PLAYER1 = 0
PLAYER2 = 1
WALL = 10

class WallOrientation(IntEnum):
    VERTICAL = 0
    HORIZONTAL = 1

# Action type constants
ACTION_MOVE = 0
ACTION_WALL = 1

# Define device functions for reusable CUDA kernels
@cuda.jit(device=True)
def _is_position_on_board_device(row: int, col: int, board_size: int) -> bool:
    return 0 <= row < board_size and 0 <= col < board_size

@cuda.jit(device=True)
def _player_cell_to_grid_index_device(row: int, col: int) -> Tuple[int, int]:
    return row * 2 + 2, col * 2 + 2

@cuda.jit(device=True)
def _wall_position_to_grid_index_device(row: int, col: int, orientation: int) -> Tuple[int, int]:
    if orientation == WallOrientation.VERTICAL:
        return row * 2 + 2, col * 2 + 3
    else:  # HORIZONTAL
        return row * 2 + 3, col * 2 + 2

@cuda.jit(device=True)
def _get_wall_slice_device(grid, row: int, col: int, orientation: int):
    grid_row, grid_col = _wall_position_to_grid_index_device(row, col, orientation)
    if orientation == WallOrientation.VERTICAL:
        return grid_row, grid_row + 3, grid_col, grid_col + 1
    else:  # HORIZONTAL
        return grid_row, grid_row + 1, grid_col, grid_col + 3

@cuda.jit(device=True)
def _are_wall_cells_free_device(grid, row: int, col: int, orientation: int) -> bool:
    r_start, r_end, c_start, c_end = _get_wall_slice_device(grid, row, col, orientation)
    
    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            if grid[r, c] != FREE:
                return False
    return True

@cuda.jit(device=True)
def _is_wall_between_positions_device(grid, row1: int, col1: int, row2: int, col2: int) -> bool:
    # Calculate grid indices for the player positions
    grid_row1, grid_col1 = _player_cell_to_grid_index_device(row1, col1)
    grid_row2, grid_col2 = _player_cell_to_grid_index_device(row2, col2)
    
    # Calculate the midpoint (wall position)
    wall_row = (grid_row1 + grid_row2) // 2
    wall_col = (grid_col1 + grid_col2) // 2
    
    # Check if there's a wall
    if 0 <= wall_row < grid.shape[0] and 0 <= wall_col < grid.shape[1]:
        return grid[wall_row, wall_col] == WALL
    
    # By default, edges are treated as walls
    return True

@cuda.jit(device=True)
def _is_wall_potential_block_device(grid, row: int, col: int, orientation: int) -> bool:
    grid_row, grid_col = _wall_position_to_grid_index_device(row, col, orientation)
    
    touches = 0
    if orientation == WallOrientation.VERTICAL:
        # Check top of wall
        if (grid[grid_row-1, grid_col-1] == WALL or 
            grid[grid_row-2, grid_col] == WALL or 
            grid[grid_row-1, grid_col+1] == WALL):
            touches += 1
            
        # Check middle of wall
        if grid[grid_row+1, grid_col-1] == WALL or grid[grid_row+1, grid_col+1] == WALL:
            touches += 1
            
        # Check bottom of wall
        if (grid[grid_row+3, grid_col-1] == WALL or 
            grid[grid_row+4, grid_col] == WALL or 
            grid[grid_row+3, grid_col+1] == WALL):
            touches += 1
    else:  # HORIZONTAL
        # Check left of wall
        if (grid[grid_row-1, grid_col-1] == WALL or 
            grid[grid_row, grid_col-2] == WALL or 
            grid[grid_row+1, grid_col-1] == WALL):
            touches += 1
            
        # Check middle of wall
        if grid[grid_row-1, grid_col+1] == WALL or grid[grid_row+1, grid_col+1] == WALL:
            touches += 1
            
        # Check right of wall
        if (grid[grid_row-1, grid_col+3] == WALL or 
            grid[grid_row, grid_col+4] == WALL or 
            grid[grid_row+1, grid_col+3] == WALL):
            touches += 1
    
    return touches >= 2

@cuda.jit(device=True)
def _can_place_wall_device(grid, row: int, col: int, orientation: int, walls_remaining: int) -> bool:
    if walls_remaining < 1:
        return False
    
    return _are_wall_cells_free_device(grid, row, col, orientation)

# BFS implementation for finding shortest path to target row
@cuda.jit(device=True)
def _distance_to_row_device(grid, start_row: int, start_col: int, target_row: int, board_size: int,
                           visited, queue, queue_size) -> int:
    """
    Device function for breadth-first search to find the shortest path to the target row.
    
    Args:
        grid: The game grid
        start_row, start_col: Starting position
        target_row: Target row to reach
        board_size: Size of the game board
        visited: Pre-allocated array for tracking visited cells
        queue: Pre-allocated queue for BFS
        queue_size: Size of the queue array
        
    Returns:
        Number of steps to reach the target or -1 if unreachable
    """
    # Reset visited array
    for r in range(board_size):
        for c in range(board_size):
            visited[r, c] = 0
    
    # Initialize BFS
    if start_row == target_row:
        return 0
    
    # Initialize queue with start position and distance
    queue[0, 0] = start_row  # row
    queue[0, 1] = start_col  # col
    queue[0, 2] = 0  # steps
    
    queue_front = 0
    queue_back = 1
    visited[start_row, start_col] = 1
    
    while queue_front < queue_back:
        curr_row = int(queue[queue_front, 0])
        curr_col = int(queue[queue_front, 1])
        steps = int(queue[queue_front, 2])
        queue_front += 1
        
        # Convert to grid coordinates
        grid_row, grid_col = _player_cell_to_grid_index_device(curr_row, curr_col)
        
        # Check all four directions
        # Down
        new_row = curr_row + 1
        if (new_row < board_size and 
            visited[new_row, curr_col] == 0 and 
            grid[grid_row + 1, grid_col] != WALL):
            
            if new_row == target_row:
                return steps + 1
                
            visited[new_row, curr_col] = 1
            if queue_back < queue_size:
                queue[queue_back, 0] = new_row
                queue[queue_back, 1] = curr_col
                queue[queue_back, 2] = steps + 1
                queue_back += 1
        
        # Up
        new_row = curr_row - 1
        if (new_row >= 0 and 
            visited[new_row, curr_col] == 0 and 
            grid[grid_row - 1, grid_col] != WALL):
            
            if new_row == target_row:
                return steps + 1
                
            visited[new_row, curr_col] = 1
            if queue_back < queue_size:
                queue[queue_back, 0] = new_row
                queue[queue_back, 1] = curr_col
                queue[queue_back, 2] = steps + 1
                queue_back += 1
        
        # Right
        new_col = curr_col + 1
        if (new_col < board_size and 
            visited[curr_row, new_col] == 0 and 
            grid[grid_row, grid_col + 1] != WALL):
            
            visited[curr_row, new_col] = 1
            if queue_back < queue_size:
                queue[queue_back, 0] = curr_row
                queue[queue_back, 1] = new_col
                queue[queue_back, 2] = steps + 1
                queue_back += 1
        
        # Left
        new_col = curr_col - 1
        if (new_col >= 0 and 
            visited[curr_row, new_col] == 0 and 
            grid[grid_row, grid_col - 1] != WALL):
            
            visited[curr_row, new_col] = 1
            if queue_back < queue_size:
                queue[queue_back, 0] = curr_row
                queue[queue_back, 1] = new_col
                queue[queue_back, 2] = steps + 1
                queue_back += 1
    
    # If we get here, there's no path to the target row
    return -1

@cuda.jit(device=True)
def _can_reach_target_device(grid, start_row: int, start_col: int, target_row: int, board_size: int,
                            visited, queue, queue_size) -> bool:
    """
    Device function to check if a player can reach the target row.
    Uses BFS, but just needs to determine reachability, not distance.
    """
    return _distance_to_row_device(grid, start_row, start_col, target_row, board_size, visited, queue, queue_size) != -1

@cuda.jit(device=True)
def _evaluate_board_device(grid, p1_row: int, p1_col: int, p2_row: int, p2_col: int, 
                          p1_target: int, p2_target: int, board_size: int,
                          visited, queue, queue_size) -> float:
    """
    Evaluates the current board state from player 1's perspective.
    
    Returns player 1's advantage as a float:
    - Positive values favor player 1
    - Negative values favor player 2
    - Larger magnitudes indicate stronger advantages
    """
    # Check for immediate win conditions
    if p1_row == p1_target:
        return 1_000_000.0  # Player 1 wins
    
    if p2_row == p2_target:
        return -1_000_000.0  # Player 2 wins
    
    # Find distances to goal for both players
    p1_dist = _distance_to_row_device(grid, p1_row, p1_col, p1_target, board_size, visited, queue, queue_size)
    
    # Reset visited array before second search
    for r in range(board_size):
        for c in range(board_size):
            visited[r, c] = 0
    
    p2_dist = _distance_to_row_device(grid, p2_row, p2_col, p2_target, board_size, visited, queue, queue_size)
    
    # If either player can't reach their goal, they lose
    if p1_dist == -1:
        return -1_000_000.0  # Player 1 can't reach goal, loses
    
    if p2_dist == -1:
        return 1_000_000.0  # Player 2 can't reach goal, loses
    
    # Otherwise, compare path distances
    # Player 1 wants: (p2's distance) - (p1's distance) to be large
    return float(p2_dist - p1_dist)

@cuda.jit(device=True)
def _apply_action_device(grid, action_type: int, action_data, 
                        p1_row: int, p1_col: int, p2_row: int, p2_col: int,
                        walls_remaining, current_player: int, board_size: int) -> bool:
    """
    Apply an action to the board.
    
    Args:
        grid: The game board
        action_type: 0 for move, 1 for wall placement
        action_data: Move or wall placement parameters
        p1_row, p1_col, p2_row, p2_col: Current player positions
        walls_remaining: Number of walls remaining for each player
        current_player: The player making the move (0 or 1)
        board_size: Size of the game board
        
    Returns:
        True if the action was applied successfully, False otherwise
    """
    if action_type == ACTION_MOVE:
        # Move action
        dest_row, dest_col = action_data[0], action_data[1]
        
        # Check if the destination is on the board
        if not _is_position_on_board_device(dest_row, dest_col, board_size):
            return False
        
        # Get grid coordinates
        grid_row, grid_col = _player_cell_to_grid_index_device(dest_row, dest_col)
        
        # Check if the destination is free
        if grid[grid_row, grid_col] != FREE:
            return False
        
        # Move the player
        old_grid_row, old_grid_col = _player_cell_to_grid_index_device(
            p1_row if current_player == 0 else p2_row,
            p1_col if current_player == 0 else p2_col
        )
        
        # Update grid
        grid[old_grid_row, old_grid_col] = FREE
        grid[grid_row, grid_col] = current_player
        
        # Update player position
        if current_player == 0:
            p1_row, p1_col = dest_row, dest_col
        else:
            p2_row, p2_col = dest_row, dest_col
            
        return True
        
    elif action_type == ACTION_WALL:
        # Wall action
        wall_row, wall_col, orientation = action_data[0], action_data[1], action_data[2]
        
        # Check if we can place the wall
        if not _can_place_wall_device(grid, wall_row, wall_col, orientation, walls_remaining[current_player]):
            return False
        
        # Place the wall
        r_start, r_end, c_start, c_end = _get_wall_slice_device(grid, wall_row, wall_col, orientation)
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                grid[r, c] = WALL
                
        # Decrement wall count
        walls_remaining[current_player] -= 1
        
        return True
    
    return False

@cuda.jit(device=True)
def _remove_action_device(grid, action_type: int, action_data,
                         p1_row: int, p1_col: int, p2_row: int, p2_col: int,
                         walls_remaining, current_player: int):
    """
    Remove an action from the board (undo).
    
    Args:
        grid: The game board
        action_type: 0 for move, 1 for wall placement
        action_data: Move or wall placement parameters
        p1_row, p1_col, p2_row, p2_col: Current player positions after the move
        walls_remaining: Number of walls remaining for each player
        current_player: The player who made the move (0 or 1)
    """
    if action_type == ACTION_MOVE:
        # Restore previous position
        prev_row, prev_col = action_data[3], action_data[4]
        new_row, new_col = action_data[0], action_data[1]
        
        # Get grid coordinates
        old_grid_row, old_grid_col = _player_cell_to_grid_index_device(new_row, new_col)
        new_grid_row, new_grid_col = _player_cell_to_grid_index_device(prev_row, prev_col)
        
        # Restore grid
        grid[old_grid_row, old_grid_col] = FREE
        grid[new_grid_row, new_grid_col] = current_player
        
    elif action_type == ACTION_WALL:
        # Wall action
        wall_row, wall_col, orientation = action_data[0], action_data[1], action_data[2]
        
        # Remove the wall
        r_start, r_end, c_start, c_end = _get_wall_slice_device(grid, wall_row, wall_col, orientation)
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                grid[r, c] = FREE
                
        # Increment wall count
        walls_remaining[current_player] += 1

# CUDA kernel for batch evaluation of board positions
@cuda.jit
def _evaluate_positions_kernel(grids, p1_positions, p2_positions, 
                              p1_targets, p2_targets, current_players,
                              walls_remaining, board_sizes, results):
    """
    CUDA kernel to evaluate multiple board positions in parallel.
    
    Args:
        grids: Array of game grids [batch_size, height, width]
        p1_positions: Positions of player 1 [batch_size, 2]
        p2_positions: Positions of player 2 [batch_size, 2]
        p1_targets: Target rows for player 1 [batch_size]
        p2_targets: Target rows for player 2 [batch_size]
        current_players: Current player for each position [batch_size]
        walls_remaining: Walls remaining for each player [batch_size, 2]
        board_sizes: Board sizes [batch_size]
        results: Output evaluation scores [batch_size]
    """
    # Get thread ID
    i = cuda.grid(1)
    
    # Check bounds
    if i < grids.shape[0]:
        # Get the grid and positions for this thread
        grid = grids[i]
        p1_row, p1_col = p1_positions[i, 0], p1_positions[i, 1]
        p2_row, p2_col = p2_positions[i, 0], p2_positions[i, 1]
        p1_target = p1_targets[i]
        p2_target = p2_targets[i]
        board_size = board_sizes[i]
        
        # Allocate local memory for BFS
        visited = cuda.local.array((32, 32), dtype=int32)  # Max board size of 32
        queue = cuda.local.array((1024, 3), dtype=int32)   # Queue for BFS (row, col, steps)
        queue_size = 1024
        
        # Evaluate the position
        evaluation = _evaluate_board_device(grid, p1_row, p1_col, p2_row, p2_col,
                                         p1_target, p2_target, board_size,
                                         visited, queue, queue_size)
        
        # Store result
        results[i] = evaluation

# CUDA kernel for batch minimax search
@cuda.jit
def _batch_minimax_kernel(grids, p1_positions, p2_positions, 
                         p1_targets, p2_targets, walls_remaining, 
                         action_types, action_data, current_players,
                         board_sizes, max_depths, is_p1_turn,
                         discount_factors, results):
    """
    CUDA kernel to perform minimax search for multiple actions in parallel.
    
    Args:
        grids: Array of game boards [batch_size, height, width]
        p1_positions: Positions of player 1 [batch_size, 2]
        p2_positions: Positions of player 2 [batch_size, 2]
        p1_targets, p2_targets: Target rows for each player [batch_size]
        walls_remaining: Walls remaining for each player [batch_size, 2]
        action_types: Type of each action (0 = move, 1 = wall) [batch_size]
        action_data: Data for each action [batch_size, 5]
        current_players: Current player for each board [batch_size]
        board_sizes: Size of each board [batch_size]
        max_depths: Maximum search depth for each board [batch_size]
        is_p1_turn: Whether it's player 1's turn [batch_size]
        discount_factors: Discount factors [batch_size]
        results: Output evaluation scores [batch_size]
    """
    # Get thread ID
    i = cuda.grid(1)
    
    # Check bounds
    if i < grids.shape[0]:
        # Get the board state for this thread
        grid = grids[i]
        p1_row, p1_col = p1_positions[i, 0], p1_positions[i, 1]
        p2_row, p2_col = p2_positions[i, 0], p2_positions[i, 1]
        p1_target = p1_targets[i]
        p2_target = p2_targets[i]
        walls = walls_remaining[i]
        board_size = board_sizes[i]
        max_depth = max_depths[i]
        discount = discount_factors[i]
        
        # Allocate local memory for BFS
        visited = cuda.local.array((32, 32), dtype=int32)
        queue = cuda.local.array((1024, 3), dtype=int32)
        queue_size = 1024
        
        # Create work arrays for storing intermediate state
        local_p1_row, local_p1_col = p1_row, p1_col
        local_p2_row, local_p2_col = p2_row, p2_col
        local_walls = cuda.local.array(2, dtype=int32)
        local_walls[0] = walls[0]
        local_walls[1] = walls[1]
        
        # Apply the action
        action_type = action_types[i]
        action = action_data[i]
        current_player = current_players[i]
        
        # Apply action
        success = _apply_action_device(grid, action_type, action, 
                                     local_p1_row, local_p1_col, 
                                     local_p2_row, local_p2_col,
                                     local_walls, current_player, board_size)
        
        if not success:
            # Action couldn't be applied
            results[i] = -1000000.0 if is_p1_turn[i] else 1000000.0
            return
        
        # Check terminal conditions
        if max_depth == 0 or local_p1_row == p1_target or local_p2_row == p2_target:
            # Terminal node - evaluate the board
            results[i] = _evaluate_board_device(grid, local_p1_row, local_p1_col,
                                             local_p2_row, local_p2_col,
                                             p1_target, p2_target, board_size,
                                             visited, queue, queue_size)
        else:
            # We would normally do recursive minimax here, but CUDA doesn't support recursion
            # For the CUDA implementation, we only evaluate one step ahead
            results[i] = _evaluate_board_device(grid, local_p1_row, local_p1_col,
                                             local_p2_row, local_p2_col,
                                             p1_target, p2_target, board_size,
                                             visited, queue, queue_size)
        
        # Apply discount factor
        results[i] *= discount
        
        # Undo the action (restore board state)
        _remove_action_device(grid, action_type, action,
                            local_p1_row, local_p1_col,
                            local_p2_row, local_p2_col,
                            local_walls, current_player)

# Host functions that launch the CUDA kernels
def evaluate_positions(grids, p1_positions, p2_positions, p1_targets, p2_targets, 
                      current_players, walls_remaining, board_sizes):
    """
    Evaluate multiple board positions in parallel using CUDA.
    
    Args:
        grids: List of game grids
        p1_positions, p2_positions: Lists of player positions
        p1_targets, p2_targets: Lists of target rows
        current_players: List of current players
        walls_remaining: List of walls remaining per player
        board_sizes: List of board sizes
        
    Returns:
        Array of evaluation scores
    """
    batch_size = len(grids)
    
    # Convert inputs to numpy arrays
    d_grids = cuda.to_device(np.array(grids, dtype=np.int32))
    d_p1_positions = cuda.to_device(np.array(p1_positions, dtype=np.int32))
    d_p2_positions = cuda.to_device(np.array(p2_positions, dtype=np.int32))
    d_p1_targets = cuda.to_device(np.array(p1_targets, dtype=np.int32))
    d_p2_targets = cuda.to_device(np.array(p2_targets, dtype=np.int32))
    d_current_players = cuda.to_device(np.array(current_players, dtype=np.int32))
    d_walls_remaining = cuda.to_device(np.array(walls_remaining, dtype=np.int32))
    d_board_sizes = cuda.to_device(np.array(board_sizes, dtype=np.int32))
    
    # Allocate output array
    d_results = cuda.device_array(batch_size, dtype=np.float32)
    
    # Launch kernel
    threads_per_block = 128
    blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
    
    _evaluate_positions_kernel[blocks_per_grid, threads_per_block](
        d_grids, d_p1_positions, d_p2_positions,
        d_p1_targets, d_p2_targets, d_current_players,
        d_walls_remaining, d_board_sizes, d_results
    )
    
    # Copy results back to host
    results = d_results.copy_to_host()
    return results

def batch_minimax(grids, p1_positions, p2_positions, p1_targets, p2_targets,
                 walls_remaining, action_types, action_data, current_players,
                 board_sizes, max_depths, is_p1_turn, discount_factors):
    """
    Perform minimax search for multiple actions in parallel using CUDA.
    
    Args:
        grids: List of game grids
        p1_positions, p2_positions: Lists of player positions
        p1_targets, p2_targets: Lists of target rows
        walls_remaining: List of walls remaining per player
        action_types: List of action types (0 = move, 1 = wall)
        action_data: List of action data
        current_players: List of current players
        board_sizes: List of board sizes
        max_depths: List of maximum search depths
        is_p1_turn: List of flags indicating if it's player 1's turn
        discount_factors: List of discount factors
        
    Returns:
        Array of evaluation scores for each action
    """
    batch_size = len(grids)
    
    # Convert inputs to numpy arrays
    d_grids = cuda.to_device(np.array(grids, dtype=np.int32))
    d_p1_positions = cuda.to_device(np.array(p1_positions, dtype=np.int32))
    d_p2_positions = cuda.to_device(np.array(p2_positions, dtype=np.int32))
    d_p1_targets = cuda.to_device(np.array(p1_targets, dtype=np.int32))
    d_p2_targets = cuda.to_device(np.array(p2_targets, dtype=np.int32))
    d_walls_remaining = cuda.to_device(np.array(walls_remaining, dtype=np.int32))
    d_action_types = cuda.to_device(np.array(action_types, dtype=np.int32))
    d_action_data = cuda.to_device(np.array(action_data, dtype=np.int32))
    d_current_players = cuda.to_device(np.array(current_players, dtype=np.int32))
    d_board_sizes = cuda.to_device(np.array(board_sizes, dtype=np.int32))
    d_max_depths = cuda.to_device(np.array(max_depths, dtype=np.int32))
    d_is_p1_turn = cuda.to_device(np.array(is_p1_turn, dtype=np.int32))
    d_discount_factors = cuda.to_device(np.array(discount_factors, dtype=np.float32))
    
    # Allocate output array
    d_results = cuda.device_array(batch_size, dtype=np.float32)
    
    # Launch kernel
    threads_per_block = 128
    blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
    
    _batch_minimax_kernel[blocks_per_grid, threads_per_block](
        d_grids, d_p1_positions, d_p2_positions,
        d_p1_targets, d_p2_targets, d_walls_remaining,
        d_action_types, d_action_data, d_current_players,
        d_board_sizes, d_max_depths, d_is_p1_turn,
        d_discount_factors, d_results
    )
    
    # Copy results back to host
    results = d_results.copy_to_host()
    return results

# CPU fallback functions
@njit
def distance_to_row(grid, start_row, start_col, target_row, board_size):
    """CPU implementation of distance calculation for when CUDA is unavailable"""
    if start_row == target_row:
        return 0
    
    visited = np.zeros((board_size, board_size), dtype=np.int32)
    queue = np.zeros((board_size * board_size, 3), dtype=np.int32)
    
    # Initialize queue
    queue[0, 0] = start_row
    queue[0, 1] = start_col
    queue[0, 2] = 0
    queue_front = 0
    queue_back = 1
    visited[start_row, start_col] = 1
    
    while queue_front < queue_back:
        curr_row = queue[queue_front, 0]
        curr_col = queue[queue_front, 1]
        steps = queue[queue_front, 2]
        queue_front += 1
        
        # Grid coordinates
        grid_row = curr_row * 2 + 2
        grid_col = curr_col * 2 + 2
        
        # Check all four directions
        # Down
        new_row = curr_row + 1
        if (new_row < board_size and 
            visited[new_row, curr_col] == 0 and 
            grid[grid_row + 1, grid_col] != WALL):
            
            if new_row == target_row:
                return steps + 1
                
            visited[new_row, curr_col] = 1
            queue[queue_back, 0] = new_row
            queue[queue_back, 1] = curr_col
            queue[queue_back, 2] = steps + 1
            queue_back += 1
        
        # Up
        new_row = curr_row - 1
        if (new_row >= 0 and 
            visited[new_row, curr_col] == 0 and 
            grid[grid_row - 1, grid_col] != WALL):
            
            if new_row == target_row:
                return steps + 1
                
            visited[new_row, curr_col] = 1
            queue[queue_back, 0] = new_row
            queue[queue_back, 1] = curr_col
            queue[queue_back, 2] = steps + 1
            queue_back += 1
        
        # Right
        new_col = curr_col + 1
        if (new_col < board_size and 
            visited[curr_row, new_col] == 0 and 
            grid[grid_row, grid_col + 1] != WALL):
            
            visited[curr_row, new_col] = 1
            queue[queue_back, 0] = curr_row
            queue[queue_back, 1] = new_col
            queue[queue_back, 2] = steps + 1
            queue_back += 1
        
        # Left
        new_col = curr_col - 1
        if (new_col >= 0 and 
            visited[curr_row, new_col] == 0 and 
            grid[grid_row, grid_col - 1] != WALL):
            
            visited[curr_row, new_col] = 1
            queue[queue_back, 0] = curr_row
            queue[queue_back, 1] = new_col
            queue[queue_back, 2] = steps + 1
            queue_back += 1
    
    return -1

@njit
def is_wall_potential_block(grid, row, col, orientation):
    """CPU implementation of wall blocking check"""
    if orientation == WallOrientation.VERTICAL:
        grid_row = row * 2 + 2
        grid_col = col * 2 + 3
        
        touches = 0
        # Check top of wall
        if (grid[grid_row-1, grid_col-1] == WALL or 
            grid[grid_row-2, grid_col] == WALL or 
            grid[grid_row-1, grid_col+1] == WALL):
            touches += 1
            
        # Check middle of wall
        if grid[grid_row+1, grid_col-1] == WALL or grid[grid_row+1, grid_col+1] == WALL:
            touches += 1
            
        # Check bottom of wall
        if (grid[grid_row+3, grid_col-1] == WALL or 
            grid[grid_row+4, grid_col] == WALL or 
            grid[grid_row+3, grid_col+1] == WALL):
            touches += 1
    else:  # HORIZONTAL
        grid_row = row * 2 + 3
        grid_col = col * 2 + 2
        
        touches = 0
        # Check left of wall
        if (grid[grid_row-1, grid_col-1] == WALL or 
            grid[grid_row, grid_col-2] == WALL or 
            grid[grid_row+1, grid_col-1] == WALL):
            touches += 1
            
        # Check middle of wall
        if grid[grid_row-1, grid_col+1] == WALL or grid[grid_row+1, grid_col+1] == WALL:
            touches += 1
            
        # Check right of wall
        if (grid[grid_row-1, grid_col+3] == WALL or 
            grid[grid_row, grid_col+4] == WALL or 
            grid[grid_row+1, grid_col+3] == WALL):
            touches += 1
    
    return touches >= 2

@njit
def are_wall_cells_free(grid, row, col, orientation):
    """CPU implementation of wall cell check"""
    if orientation == WallOrientation.VERTICAL:
        grid_row = row * 2 + 2
        grid_col = col * 2 + 3
        
        return (grid[grid_row, grid_col] == FREE and
                grid[grid_row + 1, grid_col] == FREE and
                grid[grid_row + 2, grid_col] == FREE)
    else:  # HORIZONTAL
        grid_row = row * 2 + 3
        grid_col = col * 2 + 2
        
        return (grid[grid_row, grid_col] == FREE and
                grid[grid_row, grid_col + 1] == FREE and
                grid[grid_row, grid_col + 2] == FREE)

# Function to check if CUDA is available
def is_cuda_available():
    try:
        return cuda.is_available()
    except:
        return False