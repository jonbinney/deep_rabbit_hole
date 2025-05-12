import numpy as np
from numba import njit

WALL_ORIENTATION_VERTICAL = 0
WALL_ORIENTATION_HORIZONTAL = 1

# Possible values for each cell in the grid.
CELL_FREE = -1
# The player numbers start at 0 so that they can also be used as indices into the player positions array.
CELL_PLAYER1 = 0
CELL_PLAYER2 = 1
CELL_WALL = 10


@njit
def distance_to_row(grid: np.ndarray, start_row: int, start_col: int, target_row: int) -> int:
    """
    Args:
        row (int): The current row of the pawn
        col (int): The current column of the pawn
        target_row (int): The target row to reach
        visited (numpy.array): A 2D boolean array with the same shape as the board,
            indicating which positions have been visited

    Returns:
        int: Number of steps to reach the target or -1 if it's unreachable
    """
    grid_width = grid.shape[0]
    grid_height = grid.shape[1]
    start_i = start_row * 2 + 2
    start_j = start_col * 2 + 2
    target_i = target_row * 2 + 2

    if target_i == start_i:
        return 0

    visited = np.zeros(grid.shape, dtype="bool")
    visited[start_i, start_j] = True

    queue = [(start_i, start_j, 0)]

    while queue:
        i, j, steps = queue.pop(0)
        # Iterate in the 4 directions, and if we can move to that position and we haven't already visited it, add it to the queue.
        # This was done in a for loop before, but making everything explicit makes it significantly faster, and this method is called
        # very often.

        # Down
        new_i = i + 2
        wall_i = i + 1
        if new_i < grid_height and not visited[new_i, j] and grid[wall_i, j] != CELL_WALL:
            visited[new_i, j] = True
            if target_i == new_i:
                return steps + 1
            queue.append((new_i, j, steps + 1))

        # Up
        new_i = i - 2
        wall_i = i - 1
        if new_i >= 0 and not visited[new_i, j] and grid[wall_i, j] != CELL_WALL:
            visited[new_i, j] = True
            if target_i == new_i:
                return steps + 1
            queue.append((new_i, j, steps + 1))

        # Right
        new_j = j + 2
        wall_j = j + 1
        if new_j < grid_width and not visited[i, new_j] and grid[i, wall_j] != CELL_WALL:
            visited[i, new_j] = True
            if target_i == i:
                return steps + 1
            queue.append((i, new_j, steps + 1))

        # Left
        new_j = j - 2
        wall_j = j - 1
        if new_j >= 0 and not visited[i, new_j] and grid[i, wall_j] != CELL_WALL:
            visited[i, new_j] = True
            if target_i == i:
                return steps + 1
            queue.append((i, new_j, steps + 1))

    return -1


@njit
def are_wall_cells_free(grid: np.ndarray, start_row: int, start_col: int, orientation: int) -> bool:
    grid_width = grid.shape[0]
    grid_height = grid.shape[1]

    if orientation == WALL_ORIENTATION_VERTICAL:
        start_i = start_row * 2 + 2
        start_j = start_col * 2 + 3
        assert start_i >= 0 and start_i + 2 < grid_height and start_j >= 0 and start_j < grid_width

        return (
            grid[start_i, start_j] == CELL_FREE
            and grid[start_i + 1, start_j] == CELL_FREE
            and grid[start_i + 2, start_j] == CELL_FREE
        )
    else:
        start_i = start_row * 2 + 3
        start_j = start_col * 2 + 2
        assert start_i >= 0 and start_i < grid_height and start_j >= 0 and start_j + 2 < grid_width

        return (
            grid[start_i, start_j] == CELL_FREE
            and grid[start_i, start_j + 1] == CELL_FREE
            and grid[start_i, start_j + 2] == CELL_FREE
        )


@njit
def is_wall_potential_block(grid, start_row, start_col, orientation):
    grid_width = grid.shape[0]
    grid_height = grid.shape[1]

    if orientation == WALL_ORIENTATION_VERTICAL:
        start_i = start_row * 2 + 2
        start_j = start_col * 2 + 3
        assert start_i >= 0 and start_i + 2 < grid_height and start_j >= 0 and start_j < grid_width

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
        assert start_i >= 0 and start_i < grid_height and start_j >= 0 and start_j + 2 < grid_width

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


@njit
def set_wall_cells(grid, start_row, start_col, orientation, value):
    grid_width = grid.shape[0]
    grid_height = grid.shape[1]

    if orientation == WALL_ORIENTATION_VERTICAL:
        start_i = start_row * 2 + 2
        start_j = start_col * 2 + 3
        assert start_i >= 0 and start_i + 2 < grid_height and start_j >= 0 and start_j < grid_width

        grid[start_i, start_j] = value
        grid[start_i + 1, start_j] = value
        grid[start_i + 2, start_j] = value
    else:
        start_i = start_row * 2 + 3
        start_j = start_col * 2 + 2
        assert start_i >= 0 and start_i < grid_height and start_j >= 0 and start_j + 2 < grid_width

        grid[start_i, start_j] = value
        grid[start_i, start_j + 1] = value
        grid[start_i, start_j + 2] = value


@njit
def check_wall_cells(grid, wall_position, wall_orientation, cell_value):
    """
    Return True iff all the grid cells for the wall equal the given value.
    """
    grid_width = grid.shape[0]
    grid_height = grid.shape[1]

    if wall_orientation == WALL_ORIENTATION_VERTICAL:
        start_i = wall_position[0] * 2 + 2
        start_j = wall_position[1] * 2 + 3
        assert start_i >= 0 and start_i + 2 < grid_height and start_j >= 0 and start_j < grid_width
        return (
            grid[start_i, start_j] == cell_value
            and grid[start_i + 1, start_j] == cell_value
            and grid[start_i + 2, start_j] == cell_value
        )
    else:
        start_i = wall_position[0] * 2 + 3
        start_j = wall_position[1] * 2 + 2
        assert start_i >= 0 and start_i < grid_height and start_j >= 0 and start_j + 2 < grid_width

        return (
            grid[start_i, start_j] == cell_value
            and grid[start_i, start_j + 1] == cell_value
            and grid[start_i, start_j + 2] == cell_value
        )


@njit
def is_wall_action_valid(
    grid, player_positions, walls_remaining, goal_rows, current_player, wall_position, wall_orientation
):
    is_valid = True
    if walls_remaining[current_player] <= 0:
        is_valid = False
    elif not check_wall_cells(grid, wall_position, wall_orientation, CELL_FREE):
        is_valid = False
    elif is_wall_potential_block(grid, wall_position[0], wall_position[1], wall_orientation):
        # Temprorarily add the wall.
        set_wall_cells(grid, wall_position[0], wall_position[1], wall_orientation, CELL_WALL)

        # Make sure all players still have some path to their goal row.
        for i in range(len(player_positions)):
            if distance_to_row(grid, player_positions[i][0], player_positions[i][1], goal_rows[i]) == -1:
                is_valid = False
                break

        # Restore the grid to its previous state.
        set_wall_cells(grid, wall_position[0], wall_position[1], wall_orientation, CELL_FREE)

    return is_valid
