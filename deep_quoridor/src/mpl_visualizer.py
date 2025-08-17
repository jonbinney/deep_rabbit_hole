"""
Matplotlib visualization tools for Quoridor games.

This module provides standalone functions for visualizing Quoridor game states,
designed for easy use from debug consoles when analyzing AlphaZero behavior.
"""

from typing import Optional

import matplotlib.figure
import matplotlib.pyplot as plt
from agents import ActionLog
from quoridor import Board, MoveAction, Player, Quoridor, WallAction, WallOrientation

# Configure matplotlib to avoid memory warnings
plt.rcParams["figure.max_open_warning"] = 0  # Disable warning
plt.rcParams["figure.figsize"] = (8, 8)  # Set default figure size

# Color palettes converted from pygame RGB values (normalized to 0-1)
PALETTE_TEAL = [
    (8 / 255, 127 / 255, 91 / 255),
    (9 / 255, 146 / 255, 104 / 255),
    (12 / 255, 166 / 255, 120 / 255),
    (18 / 255, 184 / 255, 134 / 255),
    (32 / 255, 201 / 255, 151 / 255),
    (56 / 255, 217 / 255, 169 / 255),
    (99 / 255, 230 / 255, 190 / 255),
    (150 / 255, 242 / 255, 215 / 255),
    (195 / 255, 250 / 255, 232 / 255),
    (230 / 255, 252 / 255, 245 / 255),
]
PALETTE_RED = [
    (201 / 255, 42 / 255, 42 / 255),
    (224 / 255, 49 / 255, 49 / 255),
    (240 / 255, 62 / 255, 62 / 255),
    (250 / 255, 82 / 255, 82 / 255),
    (255 / 255, 107 / 255, 107 / 255),
    (255 / 255, 135 / 255, 135 / 255),
    (255 / 255, 168 / 255, 168 / 255),
    (255 / 255, 201 / 255, 201 / 255),
    (255 / 255, 227 / 255, 227 / 255),
    (255 / 255, 245 / 255, 245 / 255),
]
PALETTE_ORANGE = [
    (217 / 255, 72 / 255, 15 / 255),
    (232 / 255, 89 / 255, 12 / 255),
    (247 / 255, 103 / 255, 7 / 255),
    (253 / 255, 126 / 255, 20 / 255),
    (255 / 255, 146 / 255, 43 / 255),
    (255 / 255, 169 / 255, 77 / 255),
    (255 / 255, 192 / 255, 120 / 255),
    (255 / 255, 216 / 255, 168 / 255),
    (255 / 255, 232 / 255, 204 / 255),
    (255 / 255, 244 / 255, 230 / 255),
]
PALETTE_BLUE = [
    (24 / 255, 100 / 255, 171 / 255),
    (25 / 255, 113 / 255, 194 / 255),
    (28 / 255, 126 / 255, 214 / 255),
    (34 / 255, 139 / 255, 230 / 255),
    (51 / 255, 154 / 255, 240 / 255),
    (77 / 255, 171 / 255, 247 / 255),
    (116 / 255, 192 / 255, 252 / 255),
    (165 / 255, 216 / 255, 255 / 255),
    (208 / 255, 235 / 255, 255 / 255),
    (231 / 255, 245 / 255, 255 / 255),
]
PALETTE_GRAY = [
    (33 / 255, 37 / 255, 41 / 255),
    (52 / 255, 58 / 255, 64 / 255),
    (73 / 255, 80 / 255, 87 / 255),
    (134 / 255, 142 / 255, 150 / 255),
    (173 / 255, 181 / 255, 189 / 255),
    (206 / 255, 212 / 255, 218 / 255),
    (222 / 255, 226 / 255, 230 / 255),
    (233 / 255, 236 / 255, 239 / 255),
    (241 / 255, 243 / 255, 245 / 255),
    (248 / 255, 249 / 255, 250 / 255),
]

PALETTES = [PALETTE_TEAL, PALETTE_RED, PALETTE_ORANGE, PALETTE_BLUE]


def _draw_log_action(ax, action, text, color):
    """Draw an action with text label on the matplotlib plot."""
    if isinstance(action, MoveAction):
        row, col = action.destination
        # Draw a circle at the destination position
        ax.plot(col, row, "o", color=color, markersize=40, alpha=0.7)
        # Add text label
        ax.text(
            col,
            row,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
        )

    elif isinstance(action, WallAction):
        row, col = action.position
        if action.orientation == WallOrientation.VERTICAL:
            # Vertical wall - draw line between columns
            x = col + 0.5
            y_start = row - 0.35
            y_end = row + 1.35
            ax.plot([x, x], [y_start, y_end], color=color, linewidth=8, alpha=0.7)
            ax.text(
                x,
                y_start + 0.3,
                text,
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
            )
        else:
            # Horizontal wall - draw line between rows
            y = row + 0.5
            x_start = col - 0.35
            x_end = col + 1.35
            ax.plot([x_start, x_end], [y, y], color=color, linewidth=8, alpha=0.7)
            # Add text at center
            ax.text(
                x_start + 0.3,
                y,
                text,
                ha="center",
                va="center",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
            )


def _interpolate_color(color1, color2, t):
    """Linear interpolation between two colors."""
    return (
        color1[0] + t * (color2[0] - color1[0]),
        color1[1] + t * (color2[1] - color1[1]),
        color1[2] + t * (color2[2] - color1[2]),
    )


def _get_color_from_score(score, min_score, max_score, palette):
    """Get interpolated color based on score using linear interpolation."""
    if min_score == max_score:
        # All scores are the same, use middle of palette
        return palette[len(palette) // 2]

    # Normalize score to [0, 1] range
    normalized_score = (score - min_score) / (max_score - min_score)

    # Invert the mapping: higher scores -> lower indices (darker colors)
    # Map to palette range [palette_size - 1, 0]
    palette_position = (1 - normalized_score) * (len(palette) - 1)

    # Get indices for interpolation
    lower_idx = int(palette_position)
    upper_idx = min(lower_idx + 1, len(palette) - 1)

    # Calculate interpolation factor
    t = palette_position - lower_idx

    # Interpolate between the two colors
    if lower_idx == upper_idx:
        return palette[lower_idx]
    else:
        return _interpolate_color(palette[lower_idx], palette[upper_idx], t)


def _draw_log_action_score_ranking(ax, entry, palette_id):
    """Draw ActionScoreRanking with score-based color interpolation."""
    palette = PALETTES[palette_id % len(PALETTES)]

    # Get min and max scores for normalization
    scores = [score for _, _, score in entry.ranking]
    min_score = min(scores)
    max_score = max(scores)

    for ranking, action, score in entry.ranking:
        # Format score text
        if ranking <= 5:
            text = f"#{ranking}: {score:0.2f}" if score < 10 else f"#{ranking}: {int(score)}"
        else:
            text = ""

        # Get interpolated color based on score
        color = _get_color_from_score(score, min_score, max_score, palette)

        _draw_log_action(ax, action, text, color)

    return (palette_id + 1) % len(PALETTES)


def _draw_board_base(
    ax,
    game: Quoridor,
    figsize: tuple = (8, 8),
    selected_player: Optional[Player] = None,
) -> None:
    """
    Draw the base board elements (grid, walls, players) on the given axis.

    Args:
        ax: matplotlib axis to draw on
        game: Quoridor game instance
        figsize: Figure size for wall thickness calculation
        selected_player: Optional player to highlight with selection indicator
    """
    board_size = game.board.board_size

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

    # Draw selection indicator if specified
    if selected_player is not None:
        if selected_player == Player.ONE:
            selected_pos = p1_pos
        else:
            selected_pos = p2_pos

        # Draw bright colored circle around selected player
        ax.plot(
            selected_pos[1],
            selected_pos[0],
            "o",
            color="none",
            markersize=70,
            markeredgecolor="lime",
            markeredgewidth=5,
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


def _draw_action_log(ax, action_log: Optional[ActionLog]) -> None:
    """
    Draw ActionLog visualizations on the given axis.

    Args:
        ax: matplotlib axis to draw on
        action_log: ActionLog instance to visualize
    """
    if action_log is None:
        return

    palette_id = 0
    for record in action_log.records:
        if isinstance(record, ActionLog.ActionScoreRanking):
            palette_id = _draw_log_action_score_ranking(ax, record, palette_id)
        elif isinstance(record, ActionLog.ActionText):
            _draw_log_action(ax, record.action, record.text, PALETTE_GRAY[4])


def visualize_board(
    game: Quoridor,
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8),
    title: Optional[str] = None,
    action_log: Optional[ActionLog] = None,
) -> matplotlib.figure.Figure:
    """
    Visualize a Quoridor game board with players and walls.

    Args:
        game: Quoridor game instance
        show: Whether to display the plot interactively
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height)
        title: Optional title for the plot
        action_log: Optional ActionLog to visualize action records

    Returns:
        matplotlib Figure object
    """
    board_size = game.board.board_size

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Draw base board elements
    _draw_board_base(ax, game, figsize)

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

    # Draw ActionLog if provided
    _draw_action_log(ax, action_log)

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

    action_log = ActionLog()
    action_log.set_enabled(True)
    action_log.action_text(MoveAction((1, 2)), "0.42")
    action_log.action_text(MoveAction((4, 2)), "0.1")
    action_log.action_score_ranking(
        {
            WallAction((1, 1), WallOrientation.HORIZONTAL): 0.1,
            WallAction((1, 3), WallOrientation.HORIZONTAL): 0.3,
            WallAction((1, 2), WallOrientation.VERTICAL): 0.6,
        }
    )
    visualize_board(game, action_log=action_log)
