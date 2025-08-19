#!/usr/bin/env python3
"""
Replay Buffer Visualization Tool

This tool visualizes the contents of saved replay buffers and optionally compares them
with trained model predictions. It provides an interactive matplotlib GUI for navigating
through replay buffer entries sorted by input array frequency.

Usage:
    python replay_buffer_visualizer.py <replay_buffer_file> [model_file]
"""

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from agents import ActionLog
from agents.alphazero.nn_evaluator import NNEvaluator
from quoridor import ActionEncoder, Board, MoveAction, Player, Quoridor, WallOrientation
from utils.misc import my_device


class ReplayBufferVisualizer:
    """Interactive visualization tool for replay buffer contents."""

    def __init__(self, replay_buffer_path: str, model_path: Optional[str] = None, figsize: tuple = (12, 6)):
        """
        Initialize the replay buffer visualizer.

        Args:
            replay_buffer_path: Path to the replay buffer pickle file
            model_path: Optional path to the trained model file
            figsize: Figure size as (width, height)
        """
        self.replay_buffer_path = replay_buffer_path
        self.model_path = model_path
        self.figsize = figsize

        # Load and process replay buffer data
        print("Loading replay buffer...")
        self.replay_buffer = self._load_replay_buffer(replay_buffer_path)
        print(f"Loaded {len(self.replay_buffer)} entries")

        # Sort by input array frequency (descending)
        print("Sorting entries by input array frequency...")
        self.sorted_entries = self._sort_entries_by_frequency()

        # Count unique game states for reporting
        unique_inputs = set()
        for entry, _ in self.sorted_entries:
            hashable_array = tuple(entry["input_array"].flatten())
            unique_inputs.add(hashable_array)

        print(f"Sorted {len(self.sorted_entries)} total entries representing {len(unique_inputs)} unique game states")

        # Load model if provided
        self.model = None
        if model_path:
            print("Loading trained model...")
            self.model = self._load_model(model_path)
            print("Model loaded successfully")

        # Navigation state
        self.current_index = 0

        # Create matplotlib GUI
        self._create_gui()

    def _load_replay_buffer(self, filepath: str) -> list[dict]:
        """Load replay buffer from pickle file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def _sort_entries_by_frequency(self) -> list[tuple[dict, int]]:
        """Sort replay buffer entries by input array frequency (descending)."""
        # Count input array frequencies
        input_array_counts = Counter()
        for entry in self.replay_buffer:
            input_array = entry.get("input_array")
            if input_array is not None:
                hashable_array = tuple(input_array.flatten())
                input_array_counts[hashable_array] += 1

        # Group entries by input array and sort by frequency
        entries_by_input = {}
        for entry in self.replay_buffer:
            input_array = entry.get("input_array")
            if input_array is not None:
                hashable_array = tuple(input_array.flatten())
                if hashable_array not in entries_by_input:
                    entries_by_input[hashable_array] = []
                entries_by_input[hashable_array].append(entry)

        # Sort by frequency (descending) and return ALL entries with their frequency counts
        sorted_entries = []
        for hashable_array, count in input_array_counts.most_common():
            entries = entries_by_input[hashable_array]
            # Include ALL entries for each input array, each tagged with the frequency count
            for entry in entries:
                sorted_entries.append((entry, count))

        return sorted_entries

    def _load_model(self, model_path: str) -> NNEvaluator:
        """Load trained model for comparison."""
        # Determine board size from first entry
        sample_entry = self.sorted_entries[0][0]
        input_array = sample_entry["input_array"]
        board_size = self._infer_board_size(input_array)

        # Create evaluator
        action_encoder = ActionEncoder(board_size)
        device = my_device()
        evaluator = NNEvaluator(action_encoder, device)

        # Load model state
        model_state = torch.load(model_path, map_location=device)
        evaluator.network.load_state_dict(model_state["network_state_dict"])
        evaluator.network.eval()

        return evaluator

    def _infer_board_size(self, input_array: np.ndarray) -> int:
        """Infer board size from input array dimensions."""
        # Input array structure: player_board + opponent_board + walls + [my_walls, opponent_walls]
        # For board_size B: 2*B*B + 2*(B-1)*(B-1) + 2 = total length
        # Solving: 2*B^2 + 2*(B-1)^2 + 2 = len
        # 2*B^2 + 2*(B^2 - 2*B + 1) + 2 = len
        # 4*B^2 - 4*B + 4 = len
        # 4*B^2 - 4*B + (4 - len) = 0
        # B^2 - B + (1 - len/4) = 0

        total_len = len(input_array)
        # Try common board sizes
        for board_size in range(3, 20):
            expected_len = 2 * board_size**2 + 2 * (board_size - 1) ** 2 + 2
            if expected_len == total_len:
                return board_size

        raise ValueError(f"Cannot infer board size from input array of length {total_len}")

    def _input_array_to_game(self, input_array: np.ndarray, player: Player) -> Quoridor:
        """Reconstruct Quoridor game from input array."""
        board_size = self._infer_board_size(input_array)

        # Parse the input array components
        offset = 0

        # Player boards (flattened)
        player_board_flat = input_array[offset : offset + board_size**2]
        offset += board_size**2
        opponent_board_flat = input_array[offset : offset + board_size**2]
        offset += board_size**2

        # Walls (flattened)
        wall_size = board_size - 1
        walls_flat = input_array[offset : offset + 2 * wall_size**2]
        offset += 2 * wall_size**2

        # Wall counts
        my_walls = int(input_array[offset])
        opponent_walls = int(input_array[offset + 1])

        # Reshape boards
        player_board = player_board_flat.reshape(board_size, board_size)
        opponent_board = opponent_board_flat.reshape(board_size, board_size)
        walls = walls_flat.reshape(wall_size, wall_size, 2)

        # Find player positions
        player_pos = tuple(np.unravel_index(np.argmax(player_board), player_board.shape))
        opponent_pos = tuple(np.unravel_index(np.argmax(opponent_board), opponent_board.shape))

        # Create game with exact max_walls_per_player calculation
        # Count walls already placed on the board
        walls_placed = int(walls.sum())  # Each wall contributes 1, so sum gives total count

        max_walls_per_player = (my_walls + opponent_walls + walls_placed) // 2
        board = Board(board_size=board_size, max_walls=max_walls_per_player)
        game = Quoridor(board)

        # Set player positions
        if player == Player.ONE:
            board.move_player(Player.ONE, player_pos)
            board.move_player(Player.TWO, opponent_pos)
        else:
            board.move_player(Player.ONE, opponent_pos)
            board.move_player(Player.TWO, player_pos)

        current_player = player
        my_walls_remaining_to_place = max_walls_per_player - my_walls
        # Set walls using proper board method
        for row in range(wall_size):
            for col in range(wall_size):
                if my_walls_remaining_to_place == 0 and current_player == player:
                    current_player = Player(1 - current_player)
                if walls[row, col, 0] == 1:  # Vertical wall
                    board.add_wall(current_player, (row, col), WallOrientation.VERTICAL, check_if_valid=False)
                    my_walls_remaining_to_place -= 1
                if walls[row, col, 1] == 1:  # Horizontal wall
                    board.add_wall(current_player, (row, col), WallOrientation.HORIZONTAL, check_if_valid=False)
                    my_walls_remaining_to_place -= 1

        # Set current player
        game.set_current_player(player)
        return game

    def _create_replay_buffer_action_log(self, entry: dict, game: Quoridor) -> ActionLog:
        """Create ActionLog from replay buffer entry."""
        action_log = ActionLog()
        action_log.set_enabled(True)

        # Get MCTS policy and create action scores
        mcts_policy = entry["mcts_policy"]
        valid_actions = game.get_valid_actions()
        action_scores = {}

        # Map policy indices to actions
        for action in valid_actions:
            action_index = game.action_encoder.action_to_index(action)
            if action_index < len(mcts_policy):
                score = float(mcts_policy[action_index])
                action_scores[action] = score

        if action_scores:
            action_log.action_score_ranking(action_scores)

        # Add value at current player position
        value = entry["value"]
        current_pos = game.board.get_player_position(game.current_player)
        move_to_current = MoveAction(current_pos)
        action_log.action_text(move_to_current, f"V:{value}")

        return action_log

    def _create_model_action_log(self, game: Quoridor) -> ActionLog:
        """Create ActionLog from trained model prediction."""
        if self.model is None:
            return None

        action_log = ActionLog()
        action_log.set_enabled(True)

        # Get model evaluation
        value, policy = self.model.evaluate(game)
        valid_actions = game.get_valid_actions()
        action_scores = {}

        for action in valid_actions:
            action_index = game.action_encoder.action_to_index(action)
            score = float(policy[action_index])
            action_scores[action] = score

        if action_scores:
            action_log.action_score_ranking(action_scores)

        # Add value at current player position
        current_pos = game.board.get_player_position(game.current_player)
        move_to_current = MoveAction(current_pos)
        action_log.action_text(move_to_current, f"V:{value:.3f}")

        return action_log

    def _create_gui(self):
        """Create the matplotlib GUI."""
        # Create figure with subplots
        if self.model_path:
            self.fig, (self.ax_rb, self.ax_model) = plt.subplots(1, 2, figsize=self.figsize)
        else:
            self.fig, self.ax_rb = plt.subplots(1, 1, figsize=(self.figsize[0] // 2, self.figsize[1]))
            self.ax_model = None

        # Connect keyboard events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        # Initial render
        self._update_visualization()

        # Show plot
        plt.tight_layout()
        plt.show()

    def _update_visualization(self):
        """Update the visualization with current entry."""
        if not self.sorted_entries:
            return

        entry, frequency = self.sorted_entries[self.current_index]

        # Reconstruct game from input array
        player = entry["player"]
        game = self._input_array_to_game(entry["input_array"], player)

        # Clear axes
        self.ax_rb.clear()
        if self.ax_model:
            self.ax_model.clear()

        # Replay buffer panel
        rb_action_log = self._create_replay_buffer_action_log(entry, game)
        self._visualize_game_on_axis(
            self.ax_rb,
            game,
            rb_action_log,
            f"Replay Buffer Entry {self.current_index + 1}/{len(self.sorted_entries)}\nFrequency: {frequency}, Player: {player.name}",
        )

        # Model panel (if available)
        if self.ax_model and self.model:
            model_action_log = self._create_model_action_log(game)
            self._visualize_game_on_axis(self.ax_model, game, model_action_log, "Trained Model Prediction")

        # Refresh display
        self.fig.canvas.draw()

    def _visualize_game_on_axis(self, ax, game: Quoridor, action_log: ActionLog, title: str):
        """Visualize a game on a specific axis."""
        from mpl_visualizer import _draw_action_log, _draw_board_base

        # Draw base board
        _draw_board_base(ax, game, self.figsize)

        # Draw action log
        _draw_action_log(ax, action_log)

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        # Set ticks
        board_size = game.board.board_size
        ax.set_xticks(range(board_size))
        ax.set_yticks(range(board_size))

    def _on_key_press(self, event):
        """Handle keyboard navigation."""
        if event.key == "right" or event.key == "n":
            self._next_entry()
        elif event.key == "left" or event.key == "p":
            self._previous_entry()
        elif event.key == "shift+right" or event.key == "N":
            self._next_different_state()
        elif event.key == "shift+left" or event.key == "P":
            self._previous_different_state()
        elif event.key == "q":
            plt.close("all")

    def _next_entry(self):
        """Go to next entry."""
        if self.current_index < len(self.sorted_entries) - 1:
            self.current_index += 1
            self._update_visualization()

    def _previous_entry(self):
        """Go to previous entry."""
        if self.current_index > 0:
            self.current_index -= 1
            self._update_visualization()

    def _next_different_state(self):
        """Go to next entry with different state (skipping duplicates)."""
        current_entry, _ = self.sorted_entries[self.current_index]
        current_input = tuple(current_entry["input_array"].flatten())

        for i in range(self.current_index + 1, len(self.sorted_entries)):
            entry, _ = self.sorted_entries[i]
            input_array = tuple(entry["input_array"].flatten())
            if input_array != current_input:
                self.current_index = i
                self._update_visualization()
                break

    def _previous_different_state(self):
        """Go to previous entry with different state (skipping duplicates)."""
        current_entry, _ = self.sorted_entries[self.current_index]
        current_input = tuple(current_entry["input_array"].flatten())

        for i in range(self.current_index - 1, -1, -1):
            entry, _ = self.sorted_entries[i]
            input_array = tuple(entry["input_array"].flatten())
            if input_array != current_input:
                self.current_index = i
                self._update_visualization()
                break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize replay buffer contents")
    parser.add_argument("replay_buffer", help="Path to the replay buffer pickle file")
    parser.add_argument("model", nargs="?", help="Optional path to the trained model file")

    args = parser.parse_args()

    # Validate files exist
    if not Path(args.replay_buffer).exists():
        print(f"Error: Replay buffer file not found: {args.replay_buffer}")
        return 1

    if args.model and not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return 1

    try:
        # Create and run visualizer
        ReplayBufferVisualizer(args.replay_buffer, args.model)

        # Print usage instructions
        print("\nNavigation Controls:")
        print("  Right Arrow / 'n': Next entry")
        print("  Left Arrow / 'p':  Previous entry")
        print("  Shift+Right / 'N':  Next different state")
        print("  Shift+Left / 'P':   Previous different state")
        print("  'q':                Quit")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
