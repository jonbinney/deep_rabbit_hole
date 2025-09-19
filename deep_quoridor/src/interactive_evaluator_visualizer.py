"""
Interactive Evaluator Visualization Utility

This module provides an interactive matplotlib-based utility for visualizing and editing
Quoridor board states while displaying real-time ActionLog information from evaluators.
"""

import argparse
import re
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
from agents.alphazero.nn_evaluator import NNEvaluator
from agents.core.agent import ActionLog
from mpl_visualizer import _draw_action_log, _draw_board_base
from quoridor import ActionEncoder, Board, MoveAction, Player, Quoridor, WallOrientation
from utils.misc import my_device


class InteractiveEvaluatorVisualizer:
    """
    Interactive utility for visualizing evaluator internals on editable Quoridor boards.

    Allows real-time editing of board states in "god mode":
    - Teleport pawns anywhere (no movement restrictions)
    - Place/remove walls with simplified conflict resolution
    - Wall counts displayed in title and updated in real-time
    - ActionLog visualizations from evaluator objects
    """

    def __init__(
        self,
        initial_game: Quoridor,
        evaluator: Any,
        figsize: tuple = (10, 10),
        title: Optional[str] = None,
    ):
        """
        Initialize the interactive visualizer.

        Args:
            initial_game: Initial Quoridor game state
            evaluator: The evaluate method will be called on it to evaluate the game
            figsize: Figure size as (width, height)
            title: Optional title for the plot
        """
        self.game = initial_game.create_new()  # Create a copy to avoid modifying original
        self.evaluator = evaluator
        self.figsize = figsize
        self.title = title
        self.selected_player = self.game.current_player
        self.board_size = self.game.board.board_size

        # Create the matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        # Initial render
        self._update_visualization()
        plt.show()

    def _update_visualization(self) -> None:
        """Update the entire visualization with current game state and ActionLog."""
        # Clear the axis
        self.ax.clear()

        # Draw base board with selected player indicator
        _draw_board_base(self.ax, self.game, self.figsize, self.selected_player)

        # Get ActionLog from evaluator
        action_log = self.action_log_for_game(self.game)
        _draw_action_log(self.ax, action_log)

        # Set labels and title
        self._update_labels_and_title()

        # Refresh the display
        self.fig.canvas.draw()

    def _update_labels_and_title(self) -> None:
        """Update axis labels, title, and ticks."""
        self.ax.set_xlabel("Column")
        self.ax.set_ylabel("Row")

        # Set integer ticks
        self.ax.set_xticks(range(self.board_size))
        self.ax.set_yticks(range(self.board_size))

        # Generate title with wall counts
        if self.title is None:
            current_player_name = "Player 1" if self.game.current_player == Player.ONE else "Player 2"
            selected_player_name = "Player 1" if self.selected_player == Player.ONE else "Player 2"
            p1_walls = self.game.board.get_walls_remaining(Player.ONE)
            p2_walls = self.game.board.get_walls_remaining(Player.TWO)
            title = (
                f"Current: {current_player_name} | Selected: {selected_player_name} | "
                f"P1 Walls: {p1_walls} | P2 Walls: {p2_walls}"
            )
        else:
            # Show wall counts even with custom title
            p1_walls = self.game.board.get_walls_remaining(Player.ONE)
            p2_walls = self.game.board.get_walls_remaining(Player.TWO)
            title = f"{self.title} | P1 Walls: {p1_walls} | P2 Walls: {p2_walls}"

        self.ax.set_title(title)
        plt.tight_layout()

    def _on_click(self, event) -> None:
        """Handle mouse click events on the plot."""
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Convert matplotlib coordinates to game coordinates
        col = round(x)
        row = round(y)

        # Check if click is on a player (switch selection)
        if self._is_click_on_player(row, col):
            self._handle_player_click(row, col)
        # Check if click is on board cell (move player)
        elif self._is_click_on_cell(x, y):
            self._handle_cell_click(row, col)
        # Check if click is on wall area (place/remove wall)
        else:
            self._handle_wall_click(x, y)

    def _is_click_on_player(self, row: int, col: int) -> bool:
        """Check if click is on a player position."""
        p1_pos = self.game.board.get_player_position(Player.ONE)
        p2_pos = self.game.board.get_player_position(Player.TWO)
        return (row, col) == p1_pos or (row, col) == p2_pos

    def _is_click_on_cell(self, x: float, y: float) -> bool:
        """Check if click is on a valid board cell (not on grid lines)."""
        # Check if we're close to the center of a cell (not near grid lines)
        col_frac = x - int(x)
        row_frac = y - int(y)

        # If click is close to grid lines (±0.25 from 0.5), it's a wall click
        is_near_vertical_line = abs(col_frac - 0.5) < 0.25
        is_near_horizontal_line = abs(row_frac - 0.5) < 0.25

        # It's a cell click if we're not near grid lines and within bounds
        row, col = round(y), round(x)
        in_bounds = 0 <= row < self.board_size and 0 <= col < self.board_size

        return in_bounds and not (is_near_vertical_line or is_near_horizontal_line)

    def _handle_player_click(self, row: int, col: int) -> None:
        """Handle click on a player (switch selection)."""
        p1_pos = self.game.board.get_player_position(Player.ONE)
        p2_pos = self.game.board.get_player_position(Player.TWO)

        if (row, col) == p1_pos and self.selected_player != Player.ONE:
            self.selected_player = Player.ONE
            self.game.set_current_player(Player.ONE)
            self._update_visualization()
        elif (row, col) == p2_pos and self.selected_player != Player.TWO:
            self.selected_player = Player.TWO
            self.game.set_current_player(Player.TWO)
            self._update_visualization()

    def _handle_cell_click(self, row: int, col: int) -> None:
        """Handle click on a board cell (move selected player) - GOD MODE."""
        # God mode: Allow placing pawns anywhere except where the other player is
        p1_pos = self.game.board.get_player_position(Player.ONE)
        p2_pos = self.game.board.get_player_position(Player.TWO)

        # Check if the destination is valid (not occupied by other player)
        if self.selected_player == Player.ONE and (row, col) == p2_pos:
            return  # Can't move to opponent's position
        if self.selected_player == Player.TWO and (row, col) == p1_pos:
            return  # Can't move to opponent's position

        # Check if the destination is on the board
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return  # Out of bounds

        # God mode: Simply place the pawn at the clicked position
        try:
            self.game.board.move_player(self.selected_player, (row, col))
            self._update_visualization()
        except Exception as e:
            print(f"Failed to move player to ({row}, {col}): {e}")
            pass

    def _handle_wall_click(self, x: float, y: float) -> None:
        """Handle click on wall area (place/remove wall)."""
        # Determine if this is a vertical or horizontal wall click
        col_frac = x - int(x)
        row_frac = y - int(y)
        # If click is close to vertical grid line (x.5), it's a vertical wall
        # If click is close to horizontal grid line (y.5), it's a horizontal wall
        is_vertical_line = abs(col_frac - 0.5) < 0.05
        is_horizontal_line = abs(row_frac - 0.5) < 0.05

        if is_vertical_line and not is_horizontal_line:
            self._place_wall(int(y + 0.5), int(x), WallOrientation.VERTICAL)
        elif is_horizontal_line and not is_vertical_line:
            self._place_wall(int(y), int(x + 0.5), WallOrientation.HORIZONTAL)
        # If click is on intersection, do nothing (as specified)

    def _place_wall(self, row: int, col: int, orientation: WallOrientation) -> None:
        """Place or remove a wall, handling conflicts as specified."""
        # Check bounds
        if not (0 <= row < self.board_size - 1 and 0 <= col < self.board_size - 1):
            return

        # Get current wall state
        walls = self.game.board.get_old_style_walls()
        wall_exists = walls[row, col, orientation]

        if wall_exists:
            self._remove_wall_simple(row, col, orientation)
        elif self.game.board.can_place_wall(self.selected_player, (row, col), orientation):
            self._add_wall(row, col, orientation, self.selected_player)

        self._update_visualization()

    def _add_wall(self, row: int, col: int, orientation: WallOrientation, player: Player) -> bool:
        """Add a wall for the specified player if they have walls remaining."""
        if self.game.board.get_walls_remaining(player) <= 0:
            return False

        # Use the board's proper wall placement method
        try:
            self.game.board.add_wall(player, (row, col), orientation, check_if_valid=False)
            return True
        except Exception as e:
            print(f"Error adding wall: {e}")
            return False

    def _remove_wall_simple(self, row: int, col: int, orientation: WallOrientation) -> None:
        """Remove a wall and return it to the selected player."""
        try:
            # Use the board's proper wall removal method
            self.game.board.remove_wall(self.selected_player, (row, col), orientation)
        except Exception as e:
            print(f"Error removing wall: {e}")

    def get_current_game(self) -> Quoridor:
        """Get the current game state."""
        return self.game

    def reset_to_initial_state(self, initial_game: Quoridor) -> None:
        """Reset the visualizer to a new initial state."""
        self.game = initial_game.create_new()
        self.selected_player = self.game.current_player
        self._update_visualization()

    def action_log_for_game(self, game: Quoridor) -> ActionLog:
        """Generate ActionLog showing neural network evaluation for the current game state."""
        al = ActionLog()
        al.set_enabled(True)

        # Get neural network evaluation
        value, policy = self.evaluator.evaluate(game)
        # Get valid actions and their scores
        valid_actions = game.get_valid_actions()
        action_scores = {}

        for action in valid_actions:
            action_index = game.action_encoder.action_to_index(action)
            score = policy[action_index]
            action_scores[action] = float(score)

        if action_scores:
            al.action_score_ranking(action_scores)

        # Add game value as text on current player position
        current_pos = game.board.get_player_position(game.current_player)
        move_to_current = MoveAction(current_pos)
        al.action_text(move_to_current, f"V:{value:.2f}")

        return al


def create_interactive_visualizer(
    initial_game: Quoridor,
    evaluator: Any,
    figsize: tuple = (10, 10),
    title: Optional[str] = None,
) -> InteractiveEvaluatorVisualizer:
    """
    Create and display an interactive evaluator visualizer.

    Args:
        initial_game: Initial Quoridor game state
        evaluator: The evaluate method will be called on it to evaluate the game
        figsize: Figure size as (width, height)
        title: Optional title for the plot

    Returns:
        InteractiveEvaluatorVisualizer instance
    """
    return InteractiveEvaluatorVisualizer(initial_game, evaluator, figsize, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize how the model evaluates a state")
    parser.add_argument("-N", "--board_size", type=int, default=9, help="Board Size")
    parser.add_argument("-W", "--max_walls", type=int, default=10, help="Max walls per player")
    parser.add_argument(
        "model",
        nargs="?",
        type=str,
        help="Path to the model file or nothing to use a random initialized network",
    )

    args = parser.parse_args()
    board_size = args.board_size
    max_walls = args.max_walls

    # Infer from model name
    if args.model:
        match = re.search(r"_B(\d+)W(\d+)", args.model)
        if match:
            board_size = int(match.group(1))
            max_walls = int(match.group(2))

    game = Quoridor(Board(board_size=board_size, max_walls=max_walls))
    evaluator = NNEvaluator(ActionEncoder(board_size), my_device())

    if args.model:
        model_state = torch.load(args.model, map_location=my_device())
        evaluator.network.load_state_dict(model_state["network_state_dict"])

    visualizer = create_interactive_visualizer(
        initial_game=game, evaluator=evaluator, figsize=(10, 10), title="Interactive Evaluator Visualizer"
    )
