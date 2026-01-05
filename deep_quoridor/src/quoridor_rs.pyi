"""Type stubs for quoridor_rs module (PyO3 Rust extension)."""

from typing import Tuple
import numpy as np
import numpy.typing as npt

# Constants
CELL_FREE: int
CELL_PLAYER1: int
CELL_PLAYER2: int
CELL_WALL: int

WALL_ORIENTATION_VERTICAL: int
WALL_ORIENTATION_HORIZONTAL: int

ACTION_WALL_VERTICAL: int
ACTION_WALL_HORIZONTAL: int
ACTION_MOVE: int

# Core functions
def distance_to_row(
    grid: npt.NDArray[np.int8],
    start_row: int,
    start_col: int,
    target_row: int,
) -> int:
    """Calculate the shortest distance from a position to a target row.

    Args:
        grid: The game grid as a 2D numpy array
        start_row: Starting row on the Quoridor board (0-8 for standard board)
        start_col: Starting column on the Quoridor board (0-8 for standard board)
        target_row: Target row to reach

    Returns:
        The minimum number of moves to reach the target row, or -1 if unreachable
    """
    ...

# Grid functions
def are_wall_cells_free(
    grid: npt.NDArray[np.int8],
    wall_row: int,
    wall_col: int,
    wall_orientation: int,
) -> bool:
    """Check if wall cells are free."""
    ...

def set_wall_cells(
    grid: npt.NDArray[np.int8],
    wall_row: int,
    wall_col: int,
    wall_orientation: int,
    cell_value: int,
) -> None:
    """Set wall cells to a specific value."""
    ...

def check_wall_cells(
    grid: npt.NDArray[np.int8],
    wall_row: int,
    wall_col: int,
    wall_orientation: int,
    cell_value: int,
) -> bool:
    """Check if wall cells equal a specific value."""
    ...

def is_wall_potential_block(
    grid: npt.NDArray[np.int8],
    wall_row: int,
    wall_col: int,
    wall_orientation: int,
) -> bool:
    """Check if a wall could potentially block a player's path."""
    ...

# Validation functions
def is_move_action_valid(
    grid: npt.NDArray[np.int8],
    player_positions: npt.NDArray[np.int32],
    current_player: int,
    destination_row: int,
    destination_col: int,
) -> bool:
    """Validate if a move action is legal."""
    ...

def is_wall_action_valid(
    grid: npt.NDArray[np.int8],
    player_positions: npt.NDArray[np.int32],
    walls_remaining: npt.NDArray[np.int32],
    goal_rows: npt.NDArray[np.int32],
    current_player: int,
    wall_row: int,
    wall_col: int,
    wall_orientation: int,
) -> bool:
    """Validate if a wall placement is legal."""
    ...

# Action functions
def compute_move_action_mask(
    grid: npt.NDArray[np.int8],
    player_positions: npt.NDArray[np.int32],
    current_player: int,
    action_mask: npt.NDArray[np.bool_],
) -> None:
    """Compute a mask of valid move actions."""
    ...

def compute_wall_action_mask(
    grid: npt.NDArray[np.int8],
    player_positions: npt.NDArray[np.int32],
    walls_remaining: npt.NDArray[np.int32],
    goal_rows: npt.NDArray[np.int32],
    current_player: int,
    action_mask: npt.NDArray[np.bool_],
) -> None:
    """Compute a mask of valid wall actions."""
    ...

def get_valid_move_actions(
    grid: npt.NDArray[np.int8],
    player_positions: npt.NDArray[np.int32],
    current_player: int,
) -> npt.NDArray[np.int32]:
    """Get all valid move actions."""
    ...

def get_valid_wall_actions(
    grid: npt.NDArray[np.int8],
    player_positions: npt.NDArray[np.int32],
    walls_remaining: npt.NDArray[np.int32],
    goal_rows: npt.NDArray[np.int32],
    current_player: int,
) -> npt.NDArray[np.int32]:
    """Get all valid wall actions."""
    ...

# Game state functions
def check_win(
    player_positions: npt.NDArray[np.int32],
    goal_rows: npt.NDArray[np.int32],
    player: int,
) -> bool:
    """Check if a player has won."""
    ...

def apply_action(
    grid: npt.NDArray[np.int8],
    player_positions: npt.NDArray[np.int32],
    walls_remaining: npt.NDArray[np.int32],
    current_player: int,
    action: npt.NDArray[np.int32],
) -> None:
    """Apply an action to the game state."""
    ...

def undo_action(
    grid: npt.NDArray[np.int8],
    player_positions: npt.NDArray[np.int32],
    walls_remaining: npt.NDArray[np.int32],
    player_that_took_action: int,
    action: npt.NDArray[np.int32],
    previous_position: npt.NDArray[np.int32],
) -> None:
    """Undo a previously applied action."""
    ...

# Minimax evaluation functions
def evaluate_actions(
    grid: npt.NDArray[np.int8],
    player_positions: npt.NDArray[np.int32],
    walls_remaining: npt.NDArray[np.int32],
    goal_rows: npt.NDArray[np.int32],
    current_player: int,
    max_steps: int,
    branching_factor: int,
    wall_sigma: float,
    discount_factor: float,
    heuristic: int,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Evaluate all actions for the current player using the minimax algorithm.

    This is parallelized using Rayon for better performance.

    Returns:
        A tuple of (actions, values) where:
        - actions: 2D array of shape (N, 3) with [row, col, action_type] for each action
        - values: 1D array of shape (N,) with the minimax value for each action
    """
    ...

def q_evaluate_actions(
    grid: npt.NDArray[np.int8],
    player_positions: npt.NDArray[np.int32],
    walls_remaining: npt.NDArray[np.int32],
    goal_rows: npt.NDArray[np.int32],
    current_player: int,
    max_steps: int,
    branching_factor: int,
    wall_sigma: float,
    discount_factor: float,
    heuristic: int,
    board_size: int,
    max_walls: int,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Evaluate all actions using QBitRepr-based minimax (more efficient).

    Takes the same inputs as evaluate_actions but converts to QBitRepr internally.

    Returns:
        A tuple of (actions, values) where:
        - actions: 2D array of shape (N, 3) with [row, col, action_type] for each action
        - values: 1D array of shape (N,) with the minimax value for each action
    """
    ...

def create_policy_db(
    grid: npt.NDArray[np.int8],
    player_positions: npt.NDArray[np.int32],
    walls_remaining: npt.NDArray[np.int32],
    goal_rows: npt.NDArray[np.int32],
    current_player: int,
    max_steps: int,
    branching_factor: int,
    wall_sigma: float,
    discount_factor: float,
    heuristic: int,
    filename: str,
) -> int:
    """Create a policy database by evaluating actions and saving to a SQLite database.

    Returns:
        The number of entries written to the database
    """
    ...
