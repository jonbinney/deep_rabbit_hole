"""
Matplotlib visualization tools for Quoridor games.

This module provides standalone functions for visualizing Quoridor game states,
designed for easy use from debug consoles when analyzing AlphaZero behavior.
"""

from typing import Optional

import matplotlib.figure
import matplotlib.pyplot as plt
from quoridor import Board, MoveAction, Player, Quoridor, WallAction, WallOrientation

# Configure matplotlib to avoid memory warnings
plt.rcParams["figure.max_open_warning"] = 0  # Disable warning
plt.rcParams["figure.figsize"] = (8, 8)  # Set default figure size


def visualize_board(
    game: Quoridor,
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8),
    title: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Visualize a Quoridor game board with players and walls.

    Args:
        game: Quoridor game instance
        show: Whether to display the plot interactively
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height)
        title: Optional title for the plot

    Returns:
        matplotlib Figure object
    """
    board_size = game.board.board_size

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Set up the grid
    ax.set_xlim(-0.5, board_size - 0.5)
    ax.set_ylim(-0.5, board_size - 0.5)
    ax.set_aspect("equal")

    # Invert y-axis so (0,0) is at top-left like the game board
    ax.invert_yaxis()

    # Draw grid lines
    for i in range(board_size + 1):
        ax.axhline(y=i - 0.5, color="lightgray", linewidth=8, alpha=0.7)
        ax.axvline(x=i - 0.5, color="lightgray", linewidth=8, alpha=0.7)

    # Get player positions and draw pawns
    p1_pos = game.board.get_player_position(Player.ONE)
    p2_pos = game.board.get_player_position(Player.TWO)

    # Player 1 (Player.ONE = 0) - white with black border
    ax.plot(
        p1_pos[1],
        p1_pos[0],
        "o",
        color="white",
        markersize=50,
        markeredgecolor="black",
        markeredgewidth=3,
    )

    # Player 2 (Player.TWO = 1) - black
    ax.plot(
        p2_pos[1],
        p2_pos[0],
        "o",
        color="black",
        markersize=50,
    )

    # Draw walls
    walls = game.board.get_old_style_walls()
    # Calculate wall thickness as 1/5th of cell width
    # Use a more direct approach: scale based on figure size and board size
    wall_thickness = max(10, (figsize[0] * 72) / board_size / 10)  # Minimum 10 points, or 1/10th of cell

    for row in range(board_size - 1):
        for col in range(board_size - 1):
            # Vertical walls
            if walls[row, col, WallOrientation.VERTICAL]:
                # Wall between columns col and col+1, spanning exactly 2 cells
                wall_x = col + 0.5
                wall_y_start = row - 0.35
                wall_y_end = row + 1.35
                ax.plot([wall_x, wall_x], [wall_y_start, wall_y_end], color="gray", linewidth=wall_thickness)

            # Horizontal walls
            if walls[row, col, WallOrientation.HORIZONTAL]:
                # Wall between rows row and row+1, spanning exactly 2 cells
                wall_y = row + 0.5
                wall_x_start = col - 0.35
                wall_x_end = col + 1.35
                ax.plot([wall_x_start, wall_x_end], [wall_y, wall_y], color="gray", linewidth=wall_thickness)

    # Add current player indicator
    current_player_name = "Player 1" if game.current_player == Player.ONE else "Player 2"

    # Set labels and title
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    if title is None:
        title = f"{current_player_name} to move.  P1: {game.board.get_walls_remaining(Player.ONE)}.  P2: {game.board.get_walls_remaining(Player.TWO)}"
    ax.set_title(title)

    # Set integer ticks
    ax.set_xticks(range(board_size))
    ax.set_yticks(range(board_size))

    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    # Show if requested
    if show:
        plt.show()
    else:
        # Close the figure if not showing to free memory
        plt.close(fig)

    return fig


if __name__ == "__main__":
    game = Quoridor(Board(board_size=5, max_walls=3))
    game.step(MoveAction((1, 2)))
    game.step(WallAction((0, 1), WallOrientation.HORIZONTAL))
    game.step(WallAction((0, 3), WallOrientation.HORIZONTAL))
    game.step(WallAction((0, 2), WallOrientation.VERTICAL))

    visualize_board(game)
