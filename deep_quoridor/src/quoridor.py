from dataclasses import dataclass
from enum import IntEnum, unique

import networkx as nx
import numpy as np


@unique
class WallOrientation(IntEnum):
    VERTICAL = 0
    HORIZONTAL = 1


class Action:
    pass


@dataclass
class MoveAction(Action):
    destination: tuple[int, int]  # (row, col)


@dataclass
class WallAction(Action):
    position: tuple[int, int]  # Start of wall (row, col)
    orientation: WallOrientation


def create_board_graph(board_size: int):
    """
    Create a graph representation of the board using networkx library.
    Each node represents a position on the board and each edge represents a possible move.
    """

    graph = nx.Graph()
    for row in range(board_size):
        for col in range(board_size):
            # Add node for each position on the board
            graph.add_node((row, col))

            # Add edges for possible moves
            if row < board_size - 1:
                graph.add_edge((row, col), (row + 1, col))  # Down
            if row > 0:
                graph.add_edge((row, col), (row - 1, col))  # Up
            if col < board_size - 1:
                graph.add_edge((row, col), (row, col + 1))  # Right
            if col > 0:
                graph.add_edge((row, col), (row, col - 1))  # Left

    return graph


class CombinedGrid:
    """
    Grid which has both walls and possible pawn positions.

    Pawn positions are on even rows and columns, walls are on odd rows and columns.  Occupied pawn and
    wall positions are marked as True.
    """

    def __init__(self, board_size: int):
        self._grid = np.zeros((board_size * 2 - 1, board_size * 2 - 1), dtype=np.bool)

    def set_player_cell(self, position: tuple[int, int], value: bool):
        self._grid[position[0] * 2, position[1] * 2] = value

    def get_player_cell(self, position: tuple[int, int]) -> bool:
        self._grid[position[0] * 2, position[1] * 2]

    def set_wall_cells(self, position: tuple[int, int], orientation: WallOrientation, value: bool):
        if orientation == WallOrientation.VERTICAL:
            self._grid[position[0] * 2 + 1 : position[0] * 2 + 4, position[1] * 2 + 1] = value
        else:
            self._grid[position[0] * 2 + 1, position[1] * 2 + 1 : position[1] * 2 + 4] = value

    def get_wall_cells(self, position: tuple[int, int], orientation: WallOrientation) -> bool:
        if orientation == WallOrientation.VERTICAL:
            return self._grid[position[0] * 2 + 1 : position[0] * 2 + 4, position[1] * 2 + 1]
        else:
            return self._grid[position[0] * 2 + 1, position[1] * 2 + 1 : position[1] * 2 + 4]

    def get_walls_between(self, position1: tuple[int, int], position2: tuple[int, int]) -> bool:
        """
        Return a sequence of occupancy values (both wall and player positions) between two player positions

        The first element of the sequence will be player, then wall, etc.
        The last element of the sequence will be the player position at position2.
        """
        start = (position1[0] * 2, position1[1] * 2)
        destination = (position2[0] * 2, position2[1] * 2)
        return self._grid[start[0] : destination[0] + 1, start[1] : destination[1] + 1]

    def __str__(self):
        return "\n".join("".join("X" if cell else "." for cell in row) for row in self._grid)


class Quoridor:
    def __init__(self, board_size: int, max_walls: int):
        self.board_size = board_size
        self.max_walls = max_walls
        self.walls = np.zeros((board_size - 1, board_size - 1, 2), dtype=np.int8)
        self.player_positions = [(0, board_size // 2), (board_size - 1, board_size // 2)]
        self.walls_remaining = [self.max_walls, self.max_walls]
        self.current_player = 0

        # Occupancy grid which has both walls and possible pawn positions.
        # Occupied pawn and wall positions are marked as True.
        self._combined_occ_grid = CombinedGrid(board_size)

    def step(self, action: Action, validate: bool = True):
        """
        Execute the given action and update the game state.

        Turning off validation can save some computation if you're sure the action is valid,
        and also makes it easier to teleport players around the board for testing purposes.
        """
        if validate and not self.is_action_valid(action):
            raise ValueError("Invalid action")

        if isinstance(action, MoveAction):
            # Move the pawn, mark previous position as empty, and mark new position as occupied.
            self._combined_occ_grid.set_player_cell(self.player_positions[self.current_player], False)
            self.player_positions[self.current_player] = action.destination
            self._combined_occ_grid.set_player_cell(self.player_positions[self.current_player], True)
        elif isinstance(action, WallAction):
            self._combined_occ_grid.set_wall_cells(action.position, action.orientation, True)
            # TODO: Check that the wall doesn't block any player from reaching their goal.
            self.walls_remaining[self.current_player] -= 1
        else:
            raise ValueError("Invalid action type")

        self.current_player = 1 - self.current_player

    def is_action_valid(self, action: Action):
        """
        Check whether the given action is valid given the current game state.
        """
        if isinstance(action, MoveAction):
            # Is somone in the destination cell?
            if self._combined_occ_grid.get_player_cell(action.destination):
                return False

            delta = tuple(np.subtract(action.destination, self.player_positions[self.current_player]))
            path_walls = self._combined_occ_grid.get_walls_between(
                self.player_positions[self.current_player], action.destination
            )

            # This is a valid normal one space move. Just make sure there are no walls in the way.
            if np.abs(delta).sum() == 1:
                return True

            # Straight jump. The must be an opponent in between, and no wall.
            if delta in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
                middle_position = np.mean([action.destination, self.player_positions[self.current_player]])
                if not self._combined_occ_grid.get_player_cell((action.destination[0] - 1, action.destination[1])):
                    return False

                path_walls = self._combined_occ_grid.get_walls_between(
                    self.player_positions[self.current_player], action.destination
                )
                if path_walls.any():
                    # There is a wall in between. Can't jump.
                    return False

        elif isinstance(action, WallAction):
            if self.walls_remaining[self.current_player] < 1:
                return False

            if self._combined_occ_grid.get_wall_cells(action.position, action.orientation).any():
                return False
        else:
            raise ValueError("Invalid action type")

        return True

    def __str__(self):
        return str(self._combined_occ_grid)


if __name__ == "__main__":
    game = Quoridor(9, 10)
    print(str(game))
    print("\n")
    game.step(MoveAction((1, 4)))
    game.step(WallAction((0, 4), WallOrientation.HORIZONTAL))
    game.step(WallAction((4, 2), WallOrientation.VERTICAL))
    print(str(game))
