from dataclasses import dataclass
from enum import IntEnum, unique

import networkx as nx
import numpy as np

type Position = np.ndarray[tuple[int, int], np.dtype[np.uint8]]  # (row, col)


@unique
class WallOrientation(IntEnum):
    VERTICAL = 0
    HORIZONTAL = 1


class Action:
    pass


@dataclass
class MoveAction(Action):
    destination: Position


@dataclass
class WallAction(Action):
    position: Position  # Start of wall
    orientation: WallOrientation


class Board:
    """
    Grid which has both walls and possible pawn positions.

    Pawn positions are on even rows and columns, walls are on odd rows and columns. We use negative
    values to represent free cells and walls, 0 for player 1, and 1 for player 2.
    integers.
    """

    # Possible values for each cell in the grid.
    OUT_OF_BOUNDS = -2  # Returned if a move is off of the board.
    FREE = -1
    # The player numbers start at 0 so that they can also be used as indices into the player positions array.
    PLAYER1 = 0
    PLAYER2 = 1
    # Walls are stored as 10 + player number.We leave 2-9 free for implementing more than 2 players in the future.
    PLAYER1_WALL = 10
    PLAYER2_WALL = 11

    def __init__(self, board_size: int, max_walls: int):
        self.board_size = board_size
        self.max_walls = max_walls

        self._grid = np.full((board_size * 2 - 1, board_size * 2 - 1), Board.FREE, dtype=np.int8)
        self._player_positions = np.array([(0, board_size // 2), (board_size - 1, board_size // 2)])
        for player, position in enumerate(self._player_positions):
            self._grid[*position * 2] = player
        self._walls_remaining = [self.max_walls, self.max_walls]

    def get_player_position(self, player: int) -> Position:
        """
        Get the position of the player's pawn.
        """
        try:
            return self._player_positions[player]
        except IndexError:
            raise ValueError(f"Invalid player {player}")

    def move_player(self, player: int, new_position: Position):
        """
        Move the player's pawn. Doesn't check if the move is valid according to game rules.
        """
        new_position = np.asarray(new_position, dtype=np.uint8)

        try:
            old_position = self._player_positions[player]
        except IndexError:
            raise ValueError(f"Invalid player {player}")

        try:
            self._grid[*old_position * 2] = Board.FREE
            self._grid[*new_position * 2] = player
            self._player_positions[player] = new_position
        except IndexError:
            return Board.OUT_OF_BOUNDS

    def get_cell(self, position: Position) -> int:
        """
        Get the value of a "player cell" in the grid.
        """
        position = np.asarray(position, dtype=np.uint8)
        try:
            return self._grid[*position * 2]
        except IndexError:
            return Board.OUT_OF_BOUNDS

    def add_wall(self, player: int, position: Position, orientation: WallOrientation):
        """
        Mark the grid cells corresponding to the wall as occupied.
        """
        position = np.asarray(position, dtype=np.uint8)

        if self._walls_remaining[player] < 1:
            raise RuntimeError("Cannot place wall, no walls remaining for player {player}")
        try:
            self._grid[self._get_wall_slices(position, orientation)] = 10 + player
        except IndexError:
            return Board.OUT_OF_BOUNDS

        self._walls_remaining[player] -= 1

    def can_place_wall(self, position: Position, orientation: WallOrientation) -> bool:
        """
        Returns True if the wall can be placed at the given position and orientation.

        Checks that the player has walls remaining, the wall is within the bounds of the board, and doesn't overlap with other walls.
        """
        position = np.asarray(position, dtype=np.uint8)
        try:
            return (self._grid[self._get_wall_slices(position, orientation)] == Board.FREE).all()
        except IndexError:
            return False

    def is_wall_between(self, position1: Position, position2: Position) -> bool:
        """
        Check if there is a wall between two positions.
        """
        position1 = np.asarray(position1, dtype=np.uint8)
        position2 = np.asarray(position2, dtype=np.uint8)
        wall_position = position1 + position2
        return self._grid[*wall_position] == Board.PLAYER1_WALL

    def _get_wall_slices(self, position: Position, orientation: WallOrientation) -> tuple[slice, slice]:
        """
        Get a tuple of slices that correspond to the wall's cells in the grid.
        """
        if orientation == WallOrientation.VERTICAL:
            return (slice(position[0] * 2, position[0] * 2 + 3), position[1] * 2 + 1)
        elif orientation == WallOrientation.HORIZONTAL:
            return (position[0] * 2 + 1, slice(position[1] * 2, position[1] * 2 + 3))
        else:
            raise ValueError("Invalid orientation: {orientation}")

    def __str__(self):
        """
        Return a pretty-printed string representation of the grid.
        """
        display_grid = np.full(self._grid.shape, " ", dtype=str)
        display_grid[::2, ::2] = "."
        display_grid[self._grid == Board.PLAYER1_WALL] = "w"
        display_grid[self._grid == Board.PLAYER2_WALL] = "W"
        for player, _ in enumerate(self._player_positions):
            display_grid[*self._player_positions[player] * 2] = str(player + 1)
        return "\n".join([" ".join(row) for row in display_grid]) + "\n"


class Quoridor:
    def __init__(self, board: Board):
        """
        If you want to start from the initial game state, pass in board=Board(board_size, max_walls).
        """
        self.board = board
        self.current_player = 0

        self._jump_checks = create_jump_checks()

    def step(self, action: Action, validate: bool = True):
        """
        Execute the given action and update the game state.

        Turning off validation can save some computation if you're sure the action is valid,
        and also makes it easier to teleport players around the board for testing purposes.
        """
        if validate and not self.is_action_valid(action):
            raise ValueError("Invalid action")

        if isinstance(action, MoveAction):
            self.board.move_player(self.current_player, action.destination)
        elif isinstance(action, WallAction):
            self.board.add_wall(self.current_player, action.position, action.orientation)
            # TODO: Check that the wall doesn't block any player from reaching their goal.
        else:
            raise ValueError("Invalid action type")

        self.current_player = 1 - self.current_player

    def is_action_valid(self, action: Action):
        """
        Check whether the given action is valid given the current game state.
        """
        if isinstance(action, MoveAction):
            player = self.get_current_player()
            current_position = self.board.get_player_position(player)
            opponent = 1 - player
            opponent_position = self.board.get_player_position(opponent)
            opponent_offset = opponent_position - current_position
            destination_position = np.asarray(action.destination, dtype=np.uint8)
            position_delta = tuple(destination_position - current_position)  # Tuple so we can use it as a key.

            # Destination cell must be free
            is_valid = True
            if self.board.get_cell(destination_position) != Board.FREE:
                is_valid = False

            elif np.sum(np.abs(position_delta)) == 1:
                # Moving to an adjacent cell, just make sure no wall is in the way.
                if self.board.is_wall_between(current_position, destination_position):
                    is_valid = False

            elif (tuple(opponent_offset), tuple(position_delta)) in self._jump_checks:
                jump_checks = self._jump_checks[(tuple(opponent_offset), tuple(position_delta))]

                for check_delta_1, check_delta_2 in jump_checks["wall"]:
                    if not self.board.is_wall_between(
                        current_position + check_delta_1, current_position + check_delta_2
                    ):
                        is_valid = False
                for check_delta_1, check_delta_2 in jump_checks["nowall"]:
                    if self.board.is_wall_between(current_position + check_delta_1, current_position + check_delta_2):
                        is_valid = False
            else:
                # This isn't a move by 1, and it isn't a jump, so it's invalid.
                is_valid = False

        elif isinstance(action, WallAction):
            is_valid = self.board.can_place_wall(action.position, action.orientation)

        else:
            raise ValueError("Invalid action type")

        return is_valid

    def set_current_player(self, player: int):
        if player not in [0, 1]:
            raise ValueError(f"Invalid player {player}")
        self.current_player = player

    def get_current_player(self) -> int:
        return self.current_player

    def __str__(self):
        return str(self.board)


def create_jump_checks():
    """
    Pre-generate a sequence of checks for possible jumps.

    Pre-generating the checks saves some computation time checking validity of move actions later.

    There are 2 types of checks:
    - nowall: There must not be a wall between each of these pairs of positions.
    - wall: There must be a wall between each of these pairs of positions.

    Positions are relative to the player's current position.
    """
    # Checks for the case where we are moving to (row + 1, col) or jumping over
    # an opponent at (row + 1, col)
    checks_for_one_direction = {
        (2, 0): {
            "wall": [],
            "nowall": [((0, 0), (1, 0)), ((1, 0), (2, 0))],
        },
        (1, 1): {
            "wall": [((1, 0), (2, 0))],
            "nowall": [((0, 0), (1, 0)), ((1, 0), (1, 1))],
        },
        (1, -1): {
            "wall": [((1, 0), (2, 0))],
            "nowall": [((0, 0), (1, 0)), ((1, 0), (1, -1))],
        },
    }
    # Rotate the checks for all 4 possible directions and convert to numpy arrays.
    checks = {}
    for opponent_offset in np.array([(1, 0), (0, 1), (-1, 0), (0, -1)]):
        rotation_matrix = np.array((opponent_offset, -opponent_offset[::-1])).T
        for move in checks_for_one_direction:
            rotated_move = np.dot(rotation_matrix, move)  # Tuple so that we can use it in the key.
            lookup_key = (tuple(opponent_offset), tuple(rotated_move))
            checks[lookup_key] = {}
            for check_type in checks_for_one_direction[move]:
                checks[lookup_key][check_type] = []
                for wall_start, wall_end in checks_for_one_direction[move][check_type]:
                    rotated_wall = (np.dot(rotation_matrix, wall_start), np.dot(rotation_matrix, wall_end))
                    checks[lookup_key][check_type].append(rotated_wall)
    return checks


if __name__ == "__main__":
    game = Quoridor(Board(board_size=9, max_walls=10))
    game.step(MoveAction((1, 4)))
    game.step(WallAction((0, 4), WallOrientation.HORIZONTAL))
    game.step(WallAction((4, 2), WallOrientation.VERTICAL))
    print(str(game))
