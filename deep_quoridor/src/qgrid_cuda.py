import numpy as np
from numba import cuda, njit, int32, float32
import math

# Constants matching the original qgrid.py
WALL_ORIENTATION_VERTICAL = 0
WALL_ORIENTATION_HORIZONTAL = 1

CELL_FREE = -1
CELL_PLAYER1 = 0
CELL_PLAYER2 = 1
CELL_WALL = 10

# CUDA kernel for calculating distances to target row in parallel
@cuda.jit
def _distance_to_row_kernel(grid, start_positions, target_rows, results):
    """
    CUDA kernel that computes distances from multiple start positions to their target rows in parallel.
    
    Args:
        grid: The game grid
        start_positions: Array of (row, col) start positions
        target_rows: Array of target rows for each start position
        results: Output array for distances
    """
    # Get the thread ID
    i = cuda.grid(1)
    
    # Check if the thread is within bounds
    if i < start_positions.shape[0]:
        start_row = start_positions[i, 0]
        start_col = start_positions[i, 1]
        target_row = target_rows[i]
        
        grid_width = grid.shape[0]
        grid_height = grid.shape[1]
        start_i = start_row * 2 + 2
        start_j = start_col * 2 + 2
        target_i = target_row * 2 + 2
        
        if target_i == start_i:
            results[i] = 0
            return
            
        # We'll use a custom BFS implementation since CUDA doesn't handle dynamic data structures well
        # Initialize visited array
        visited = cuda.local.array(shape=(25, 25), dtype=int32)  # Assuming max grid size of 11x11 (25x25 with walls)
        
        # Initialize visited array to zeros
        for r in range(grid_width):
            for c in range(grid_height):
                visited[r, c] = 0
                
        # Mark start position as visited
        visited[start_i, start_j] = 1
        
        # Initialize queue for BFS (fixed size queue implementation)
        queue_max_size = 200  # Large enough for most boards
        queue = cuda.local.array(shape=(200, 3), dtype=int32)  # (i, j, steps)
        queue_start = 0
        queue_end = 1
        
        # Add start position to queue
        queue[0, 0] = start_i
        queue[0, 1] = start_j
        queue[0, 2] = 0
        
        while queue_start < queue_end:
            # Dequeue
            i, j, steps = queue[queue_start, 0], queue[queue_start, 1], queue[queue_start, 2]
            queue_start += 1
            
            # Check all four directions
            
            # Down
            new_i = i + 2
            wall_i = i + 1
            if new_i < grid_width and visited[new_i, j] == 0 and grid[wall_i, j] != CELL_WALL:
                visited[new_i, j] = 1
                if target_i == new_i:
                    results[i] = steps + 1
                    return
                if queue_end < queue_max_size:
                    queue[queue_end, 0] = new_i
                    queue[queue_end, 1] = j
                    queue[queue_end, 2] = steps + 1
                    queue_end += 1
            
            # Up
            new_i = i - 2
            wall_i = i - 1
            if new_i >= 0 and visited[new_i, j] == 0 and grid[wall_i, j] != CELL_WALL:
                visited[new_i, j] = 1
                if target_i == new_i:
                    results[i] = steps + 1
                    return
                if queue_end < queue_max_size:
                    queue[queue_end, 0] = new_i
                    queue[queue_end, 1] = j
                    queue[queue_end, 2] = steps + 1
                    queue_end += 1
            
            # Right
            new_j = j + 2
            wall_j = j + 1
            if new_j < grid_height and visited[i, new_j] == 0 and grid[i, wall_j] != CELL_WALL:
                visited[i, new_j] = 1
                if target_i == i:
                    results[i] = steps + 1
                    return
                if queue_end < queue_max_size:
                    queue[queue_end, 0] = i
                    queue[queue_end, 1] = new_j
                    queue[queue_end, 2] = steps + 1
                    queue_end += 1
            
            # Left
            new_j = j - 2
            wall_j = j - 1
            if new_j >= 0 and visited[i, new_j] == 0 and grid[i, wall_j] != CELL_WALL:
                visited[i, new_j] = 1
                if target_i == i:
                    results[i] = steps + 1
                    return
                if queue_end < queue_max_size:
                    queue[queue_end, 0] = i
                    queue[queue_end, 1] = new_j
                    queue[queue_end, 2] = steps + 1
                    queue_end += 1
        
        # If we get here, there's no path
        results[i] = -1


@cuda.jit
def _evaluate_actions_kernel(grid, player_positions, target_rows, actions, action_types, action_params, results):
    """
    CUDA kernel that evaluates multiple actions in parallel.
    
    Args:
        grid: The game grid
        player_positions: Array of (row, col) positions for each player
        target_rows: Array of target rows for each player
        actions: Array of action types (0 for move, 1 for wall)
        action_params: Parameters for each action (destination for move, position and orientation for wall)
        results: Output array for action evaluation results
    """
    # Get the thread ID
    i = cuda.grid(1)
    
    # Check if the thread is within bounds
    if i < actions.shape[0]:
        # Copy the grid to make modifications
        local_grid = cuda.local.array(shape=(25, 25), dtype=int32)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                local_grid[r, c] = grid[r, c]
        
        # Copy player positions
        local_positions = cuda.local.array(shape=(2, 2), dtype=int32)
        for p in range(2):
            for d in range(2):
                local_positions[p, d] = player_positions[p, d]
        
        # Apply the action
        action_type = actions[i]
        if action_type == 0:  # Move action
            player = 0  # Current player
            dest_row = action_params[i, 0]
            dest_col = action_params[i, 1]
            
            # Clear old position
            old_row, old_col = local_positions[player, 0], local_positions[player, 1]
            grid_row = old_row * 2 + 2
            grid_col = old_col * 2 + 2
            local_grid[grid_row, grid_col] = CELL_FREE
            
            # Set new position
            local_positions[player, 0] = dest_row
            local_positions[player, 1] = dest_col
            grid_row = dest_row * 2 + 2
            grid_col = dest_col * 2 + 2
            local_grid[grid_row, grid_col] = player
            
        elif action_type == 1:  # Wall action
            pos_row = action_params[i, 0]
            pos_col = action_params[i, 1]
            orientation = action_params[i, 2]
            
            # Place wall
            if orientation == WALL_ORIENTATION_VERTICAL:
                start_i = pos_row * 2 + 2
                start_j = pos_col * 2 + 3
                local_grid[start_i, start_j] = CELL_WALL
                local_grid[start_i + 1, start_j] = CELL_WALL
                local_grid[start_i + 2, start_j] = CELL_WALL
            else:  # HORIZONTAL
                start_i = pos_row * 2 + 3
                start_j = pos_col * 2 + 2
                local_grid[start_i, start_j] = CELL_WALL
                local_grid[start_i, start_j + 1] = CELL_WALL
                local_grid[start_i, start_j + 2] = CELL_WALL
        
        # Evaluate the position for both players
        p0_dist = -1
        p1_dist = -1
        
        # Calculate player distances using a simplified BFS for each player
        for p in range(2):
            start_row = local_positions[p, 0]
            start_col = local_positions[p, 1]
            target_row = target_rows[p]
            
            grid_width = grid.shape[0]
            grid_height = grid.shape[1]
            start_i = start_row * 2 + 2
            start_j = start_col * 2 + 2
            target_i = target_row * 2 + 2
            
            if target_i == start_i:
                if p == 0:
                    p0_dist = 0
                else:
                    p1_dist = 0
                continue
                
            # Initialize visited array
            visited = cuda.local.array(shape=(25, 25), dtype=int32)
            for r in range(grid_width):
                for c in range(grid_height):
                    visited[r, c] = 0
            
            # Mark start position as visited
            visited[start_i, start_j] = 1
            
            # Initialize queue for BFS
            queue_max_size = 200
            queue = cuda.local.array(shape=(200, 3), dtype=int32)
            queue_start = 0
            queue_end = 1
            
            # Add start position to queue
            queue[0, 0] = start_i
            queue[0, 1] = start_j
            queue[0, 2] = 0
            
            found = False
            while queue_start < queue_end and not found:
                # Dequeue
                curr_i, curr_j, steps = queue[queue_start, 0], queue[queue_start, 1], queue[queue_start, 2]
                queue_start += 1
                
                # Check all four directions
                directions = cuda.local.array(shape=(4, 3), dtype=int32)
                # Down: new_i, new_j, wall_i, wall_j
                directions[0, 0] = curr_i + 2
                directions[0, 1] = curr_j
                directions[0, 2] = curr_i + 1
                
                # Up
                directions[1, 0] = curr_i - 2
                directions[1, 1] = curr_j
                directions[1, 2] = curr_i - 1
                
                # Right
                directions[2, 0] = curr_i
                directions[2, 1] = curr_j + 2
                directions[2, 2] = curr_j + 1
                
                # Left
                directions[3, 0] = curr_i
                directions[3, 1] = curr_j - 2
                directions[3, 2] = curr_j - 1
                
                for d in range(4):
                    new_i = directions[d, 0]
                    new_j = directions[d, 1]
                    
                    # Vertical movements (up/down)
                    if d < 2:
                        wall_i = directions[d, 2]
                        wall_j = curr_j
                        check_target = (new_i == target_i)
                    else:  # Horizontal movements (left/right)
                        wall_i = curr_i
                        wall_j = directions[d, 2]
                        check_target = (curr_i == target_i)
                    
                    if new_i >= 0 and new_i < grid_width and new_j >= 0 and new_j < grid_height:
                        if visited[new_i, new_j] == 0 and local_grid[wall_i, wall_j] != CELL_WALL:
                            visited[new_i, new_j] = 1
                            if check_target:
                                if p == 0:
                                    p0_dist = steps + 1
                                else:
                                    p1_dist = steps + 1
                                found = True
                                break
                            
                            if queue_end < queue_max_size:
                                queue[queue_end, 0] = new_i
                                queue[queue_end, 1] = new_j
                                queue[queue_end, 2] = steps + 1
                                queue_end += 1
        
        # Calculate the position value - opponent distance minus player distance
        # with a large reward for winning positions
        if p0_dist == 0:  # Player 0 has won
            results[i] = 1000000
        elif p1_dist == 0:  # Player 1 has won
            results[i] = -1000000
        elif p0_dist == -1:  # Player 0 can't reach goal
            results[i] = -1000000
        elif p1_dist == -1:  # Player 1 can't reach goal
            results[i] = 1000000
        else:
            results[i] = p1_dist - p0_dist


def distance_to_row_batch(grid, start_positions, target_rows):
    """
    Computes distances from multiple start positions to their target rows in parallel.
    
    Args:
        grid: The game grid
        start_positions: Array of (row, col) start positions
        target_rows: Array of target rows for each start position
    
    Returns:
        Array of distances
    """
    # Prepare input data
    n = len(start_positions)
    d_grid = cuda.to_device(grid)
    d_start_positions = cuda.to_device(np.array(start_positions, dtype=np.int32))
    d_target_rows = cuda.to_device(np.array(target_rows, dtype=np.int32))
    d_results = cuda.device_array(n, dtype=np.int32)
    
    # Configure CUDA grid
    threads_per_block = 128
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    _distance_to_row_kernel[blocks_per_grid, threads_per_block](d_grid, d_start_positions, d_target_rows, d_results)
    
    # Retrieve results
    results = d_results.copy_to_host()
    return results


def distance_to_row(grid, start_row, start_col, target_row):
    """
    Computes distance from a single start position to target row.
    
    Args:
        grid: The game grid
        start_row: Starting row
        start_col: Starting column
        target_row: Target row
    
    Returns:
        Distance to target row or -1 if unreachable
    """
    # Just call the batch version with a single position
    start_positions = np.array([[start_row, start_col]], dtype=np.int32)
    target_rows = np.array([target_row], dtype=np.int32)
    results = distance_to_row_batch(grid, start_positions, target_rows)
    return results[0]


def evaluate_actions_batch(grid, player_positions, target_rows, actions, action_params):
    """
    Evaluates multiple potential actions in parallel.
    
    Args:
        grid: The game grid
        player_positions: Positions of both players
        target_rows: Target rows for both players
        actions: Array of action types (0 for move, 1 for wall)
        action_params: Parameters for each action
    
    Returns:
        Array of position evaluations after each action
    """
    # Prepare input data
    n = len(actions)
    d_grid = cuda.to_device(grid)
    d_player_positions = cuda.to_device(np.array(player_positions, dtype=np.int32))
    d_target_rows = cuda.to_device(np.array(target_rows, dtype=np.int32))
    d_actions = cuda.to_device(np.array(actions, dtype=np.int32))
    d_action_params = cuda.to_device(np.array(action_params, dtype=np.int32))
    d_results = cuda.device_array(n, dtype=np.float32)
    
    # Configure CUDA grid
    threads_per_block = 128
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    _evaluate_actions_kernel[blocks_per_grid, threads_per_block](
        d_grid, d_player_positions, d_target_rows, d_actions, d_action_params, d_results
    )
    
    # Retrieve results
    results = d_results.copy_to_host()
    return results


def are_wall_cells_free(grid, start_row, start_col, orientation):
    """CPU fallback for the wall cell check"""
    grid_width = grid.shape[0]
    grid_height = grid.shape[1]

    if orientation == WALL_ORIENTATION_VERTICAL:
        start_i = start_row * 2 + 2
        start_j = start_col * 2 + 3
        if not (start_i >= 0 and start_i + 2 < grid_height and start_j >= 0 and start_j < grid_width):
            return False

        return (
            grid[start_i, start_j] == CELL_FREE
            and grid[start_i + 1, start_j] == CELL_FREE
            and grid[start_i + 2, start_j] == CELL_FREE
        )
    else:
        start_i = start_row * 2 + 3
        start_j = start_col * 2 + 2
        if not (start_i >= 0 and start_i < grid_height and start_j >= 0 and start_j + 3 < grid_width):
            return False

        return (
            grid[start_i, start_j] == CELL_FREE
            and grid[start_i, start_j + 1] == CELL_FREE
            and grid[start_i, start_j + 2] == CELL_FREE
        )


def is_wall_potential_block(grid, start_row, start_col, orientation):
    """CPU fallback for the wall block check"""
    grid_width = grid.shape[0]
    grid_height = grid.shape[1]

    if orientation == WALL_ORIENTATION_VERTICAL:
        start_i = start_row * 2 + 2
        start_j = start_col * 2 + 3
        if not (start_i >= 0 and start_i + 2 < grid_height and start_j >= 0 and start_j < grid_width):
            return False

        touches = 0
        if (
            grid[start_i - 1, start_j - 1] == CELL_WALL
            or grid[start_i - 2, start_j] == CELL_WALL
            or grid[start_i - 1, start_j + 1] == CELL_WALL
        ):
            touches += 1
        if grid[start_i + 1, start_j - 1] == CELL_WALL or grid[start_i + 1, start_j + 1] == CELL_WALL:
            touches += 1
        if (
            grid[start_i + 3, start_j - 1] == CELL_WALL
            or grid[start_i + 4, start_j] == CELL_WALL
            or grid[start_i + 3, start_j + 1] == CELL_WALL
        ):
            touches += 1
        return touches >= 2

    else:  # HORIZONTAL
        start_i = start_row * 2 + 3
        start_j = start_col * 2 + 2
        if not (start_i >= 0 and start_i < grid_height and start_j >= 0 and start_j + 3 < grid_width):
            return False

        touches = 0
        if (
            grid[start_i - 1, start_j - 1] == CELL_WALL
            or grid[start_i, start_j - 2] == CELL_WALL
            or grid[start_i + 1, start_j - 1] == CELL_WALL
        ):
            touches += 1
        if grid[start_i - 1, start_j + 1] == CELL_WALL or grid[start_i + 1, start_j + 1] == CELL_WALL:
            touches += 1
        if (
            grid[start_i - 1, start_j + 3] == CELL_WALL
            or grid[start_i, start_j + 4] == CELL_WALL
            or grid[start_i + 1, start_j + 3] == CELL_WALL
        ):
            touches += 1
        return touches >= 2