import numpy as np
from agents.core.rotation import (
    convert_rotated_action_index_to_original,
    rotate_action_mask,
    rotate_board,
    rotate_walls,
)


def test_rotate_action_mask_simple():
    """
    For board_size = 3, the action mask represents:

    Board positions (0-8):    Wall positions:
    0 1 2                     Vertical(9-12):   Horizontal(13-16):
    3 4 5                     v v              h h
    6 7 8                     v v              h h
                              v v              h h

    Initial mask:             Expected after 180° rotation:
    1 0 0                     0 0 0
    0 0 0                     0 0 0
    0 0 0                     0 0 1
    """
    board_size = 3
    total_actions = board_size * board_size
    wall_actions = (board_size - 1) ** 2

    mask = np.zeros(total_actions + 2 * wall_actions)
    mask[0] = 1  # First board position (top-left)
    mask[total_actions] = 1  # First vertical wall
    mask[total_actions + wall_actions] = 1  # First horizontal wall

    print("\nInput mask visualization:")
    print(mask[:9].reshape(3, 3))  # Print board positions

    rotated = rotate_action_mask(board_size, mask)

    print("\nRotated mask visualization:")
    print(rotated[:9].reshape(3, 3))  # Print rotated board positions

    assert rotated[8] == 1
    assert rotated[12] == 1
    assert rotated[16] == 1
    assert np.sum(rotated) == 3


def test_rotate_action_mask_identity():
    """
    Test that rotating twice (360°) returns the original mask
    Board size = 3
    """
    board_size = 3
    total_actions = board_size * board_size
    wall_actions = (board_size - 1) ** 2

    # Create random mask
    mask = np.random.randint(0, 2, total_actions + 2 * wall_actions)

    print("\nOriginal random mask (board positions):")
    print(mask[:9].reshape(3, 3))

    rotated_once = rotate_action_mask(board_size, mask)
    rotated_twice = rotate_action_mask(board_size, rotated_once)

    print("\nRotated twice mask (board positions):")
    print(rotated_twice[:9].reshape(3, 3))

    np.testing.assert_array_equal(mask, rotated_twice)


def test_rotate_action_mask_shape():
    """
    Test with larger board (5x5)
    Board positions: 25 (0-24)
    Vertical walls: 16 (25-40)
    Horizontal walls: 16 (41-56)
    Total length: 57
    """
    board_size = 5
    total_actions = board_size * board_size
    wall_actions = (board_size - 1) ** 2

    mask = np.ones(total_actions + 2 * wall_actions)
    rotated = rotate_action_mask(board_size, mask)

    print("\nInput 5x5 board positions:")
    print(mask[:25].reshape(5, 5))

    assert rotated.shape == mask.shape


def test_convert_rotated_action_index():
    """Test conversion of rotated action indices back to original space"""
    board_size = 3

    # Test board movement actions
    assert convert_rotated_action_index_to_original(board_size, 8) == 0
    assert convert_rotated_action_index_to_original(board_size, 0) == 8

    # Test vertical wall actions
    assert convert_rotated_action_index_to_original(board_size, 9) == 12
    assert convert_rotated_action_index_to_original(board_size, 12) == 9

    # Test horizontal wall actions
    assert convert_rotated_action_index_to_original(board_size, 13) == 16
    assert convert_rotated_action_index_to_original(board_size, 16) == 13


def test_rotate_board():
    """Test board rotation"""
    board = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    expected = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
    rotated = rotate_board(board)
    np.testing.assert_array_equal(rotated, expected)


def test_rotate_walls():
    """Test wall rotation"""
    walls = np.zeros((3, 3, 2))
    # Set some walls
    walls[0, 0, 0] = 1  # Vertical wall
    walls[1, 1, 1] = 1  # Horizontal wall

    rotated = rotate_walls(walls)

    # Check rotated positions
    assert rotated[2, 2, 0] == 1  # Rotated vertical wall
    assert rotated[1, 1, 1] == 1  # Center horizontal wall unchanged
