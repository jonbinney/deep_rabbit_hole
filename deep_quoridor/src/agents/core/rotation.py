import numpy as np


def rotate_action_mask(board_size, mask):
    total_actions = board_size * board_size  # Movement actions
    wall_actions = (board_size - 1) ** 2  # Actions for each wall type

    # Split the mask into board moves and wall placements
    indices = np.array([total_actions, total_actions + wall_actions])
    board_mask, walls_v, walls_h = np.split(mask, indices)

    # Rotate board moves (first part of mask)
    board_mask = board_mask.reshape(board_size, board_size)
    board_mask = np.rot90(board_mask, k=2).flatten()

    # Rotate wall placements
    walls_v = walls_v.reshape(board_size - 1, board_size - 1)
    walls_h = walls_h.reshape(board_size - 1, board_size - 1)
    walls_v = np.rot90(walls_v, k=2).flatten()
    walls_h = np.rot90(walls_h, k=2).flatten()

    # Combine rotated masks back together
    rotated_mask = np.concatenate([board_mask, walls_v, walls_h])

    return rotated_mask


def _map_rotated_index_to_original(index, grid_size, offset=0):
    """
    Helper method to convert rotated indices back to original space.

    Args:
        index: The index within the current section (board/walls)
        grid_size: Size of the grid for this section (board_size or board_size-1)
        offset: Offset to add to final result (0 for board moves, total_actions for walls)
    """
    row, col = divmod(index, grid_size)
    rotated_row = grid_size - 1 - row
    rotated_col = grid_size - 1 - col
    return offset + rotated_row * grid_size + rotated_col


def convert_rotated_action_index_to_original(board_size, rotated_action_index):
    """Convert action index from rotated tensor back to original action space.

    This function maps an action index from a rotated board representation back to
    the original action space. It handles three types of actions:
    1. Board movement actions (0 to board_size^2 - 1)
    2. Vertical wall placements (board_size^2 to board_size^2 + (board_size-1)^2 - 1)
    3. Horizontal wall placements (board_size^2 + (board_size-1)^2 to board_size^2 + 2*(board_size-1)^2 - 1)

    Args:
        board_size (int): Size of the game board (N for an NxN board)
        rotated_action_index (int): Action index in the rotated space to be converted

    Returns:
        int: Corresponding action index in the original action space

    Example:
        >>> convert_rotated_action_index_to_original(9, 45)
        72
    """
    """Convert action index from rotated tensor back to original action space."""

    total_actions = board_size * board_size
    wall_actions = (board_size - 1) ** 2

    # Determine which section of the action space we're in
    if rotated_action_index < total_actions:
        # Board movement action
        return _map_rotated_index_to_original(rotated_action_index, board_size)

    elif rotated_action_index < total_actions + wall_actions:
        # Vertical wall action
        wall_index = rotated_action_index - total_actions
        return _map_rotated_index_to_original(wall_index, board_size - 1, total_actions)

    else:
        # Horizontal wall action
        wall_index = rotated_action_index - (total_actions + wall_actions)
        return _map_rotated_index_to_original(wall_index, board_size - 1, total_actions + wall_actions)


def _map_original_index_to_rotated(index, grid_size, offset=0):
    """
    Helper method to convert original indices to rotated space.

    Args:
        index: The index within the original section (board/walls)
        grid_size: Size of the grid for this section (board_size or board_size-1)
        offset: Offset to add to final result (0 for board moves, total_actions for walls)
    """
    row, col = divmod(index - offset, grid_size)
    rotated_row = grid_size - 1 - row
    rotated_col = grid_size - 1 - col
    return offset + rotated_row * grid_size + rotated_col


def convert_original_action_index_to_rotated(board_size, original_action_index):
    """Convert action index from original tensor to rotated action space.

    This function maps an action index from the original board representation to
    the rotated action space. It handles three types of actions:
    1. Board movement actions (0 to board_size^2 - 1)
    2. Vertical wall placements (board_size^2 to board_size^2 + (board_size-1)^2 - 1)
    3. Horizontal wall placements (board_size^2 + (board_size-1)^2 to board_size^2 + 2*(board_size-1)^2 - 1)

    Args:
        board_size (int): Size of the game board (N for an NxN board)
        original_action_index (int): Action index in the original space to be converted

    Returns:
        int: Corresponding action index in the rotated action space
    """

    total_actions = board_size * board_size
    wall_actions = (board_size - 1) ** 2

    # Determine which section of the action space we're in
    if original_action_index < total_actions:
        # Board movement action
        return _map_original_index_to_rotated(original_action_index, board_size)

    elif original_action_index < total_actions + wall_actions:
        # Vertical wall action
        return _map_original_index_to_rotated(original_action_index, board_size - 1, total_actions)

    else:
        # Horizontal wall action
        return _map_original_index_to_rotated(original_action_index, board_size - 1, total_actions + wall_actions)


def rotate_board(board):
    return np.rot90(board, k=2)


def rotate_walls(walls):
    rotated = np.zeros_like(walls)
    for i in range(walls.shape[2]):
        rotated[:, :, i] = np.rot90(walls[:, :, i], k=2)
    return rotated
