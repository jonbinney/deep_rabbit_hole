import copy
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import TypeAlias

import numpy as np

Position: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint8]]  # (row, col)


@unique
class Player(IntEnum):
    ONE = 0
    TWO = 1


@unique
class WallOrientation(IntEnum):
    VERTICAL = 0
    HORIZONTAL = 1


class Action:
    pass


@dataclass
class MoveAction(Action):
    def __init__(self, destination: Position):
        assert is_valid_position_type(destination)
        self.destination = destination

    destination: Position


@dataclass
class WallAction(Action):
    def __init__(self, position: Position, orientation: WallOrientation):
        assert is_valid_position_type(position)
        assert isinstance(orientation, WallOrientation)
        self.position = np.array((position))
        self.orientation = orientation

    position: Position  # Start of wall
    orientation: WallOrientation


def is_valid_position_type(position: Position) -> bool:
    return isinstance(position, np.ndarray) and position.shape == (2,)


class Board:
    # Possible values for each cell in the grid.
    OUT_OF_BOUNDS = -2  # Returned if a move is off of the board.
    FREE = -1
    # The player numbers start at 0 so that they can also be used as indices into the player positions array.
    PLAYER1 = 0
    PLAYER2 = 1
    WALL = 10

    def __init__(self, board_size: int = 9, max_walls: int = 10):
        self.board_size = board_size
        self.max_walls = max_walls

        # We represent the board as a grid of cells alternating between wall cells and odd rows are player cells.
        # To make some checks easier, we add a double border of walls around the grid.
        self._grid = np.full((board_size * 2 + 3, board_size * 2 + 3), Board.FREE, dtype=np.int8)
        self._grid[:2, :] = Board.WALL
        self._grid[-2:, :] = Board.WALL
        self._grid[:, :2] = Board.WALL
        self._grid[:, -2:] = Board.WALL

        self._players = [Player.ONE, Player.TWO]
        self._player_positions = [np.array([0, board_size // 2]), np.array([board_size - 1, board_size // 2])]
        for player, position in zip(self._players, self._player_positions):
            self._grid[*(position * 2 + 2)] = player
        self._walls_remaining = [self.max_walls, self.max_walls]

        self._potential_wall_neighbors = {
            WallOrientation.VERTICAL: [
                np.array([(-1, -1), (-2, 0), (-1, 1)]),
                np.array([(1, -1), (1, 1)]),
                np.array([(3, -1), (4, 0), (3, 1)]),
            ],
            WallOrientation.HORIZONTAL: [
                np.array([(-1, -1), (0, -2), (1, -1)]),
                np.array([(-1, 1), (1, 1)]),
                np.array([(-1, 3), (0, 4), (1, 3)]),
            ],
        }

    def get_player_position(self, player: Player) -> Position:
        """
        Get the position of the player's pawn.
        """
        assert isinstance(player, Player)
        return self._player_positions[player]

    def move_player(self, player: Player, new_position: Position):
        """
        Move the player's pawn. Doesn't check if the move is valid according to game rules.
        """
        assert isinstance(player, Player)
        assert is_valid_position_type(new_position)

        if not self.is_position_on_board(new_position):
            raise ValueError(f"Position {new_position} is out of bounds")

        old_position = self._player_positions[player]
        self._grid[*old_position * 2 + 2] = Board.FREE
        self._grid[*new_position * 2 + 2] = player
        self._player_positions[player] = new_position

    def get_player_cell(self, position: Position) -> int:
        """
        Get the value of a "player cell" in the grid.
        """
        assert is_valid_position_type(position)

        if not self.is_position_on_board(position):
            raise ValueError(f"Position {position} is out of bounds")

        return self._grid[*position * 2 + 2]

    def add_wall(self, player: Player, position: Position, orientation: WallOrientation):
        """
        Mark the grid cells corresponding to the wall as occupied.
        """
        assert isinstance(player, Player)
        assert is_valid_position_type(position)
        assert isinstance(orientation, WallOrientation)

        if self._walls_remaining[player] < 1:
            raise ValueError("Cannot place wall, no walls remaining for player {player}")

        if self.can_place_wall(position, orientation):
            self._grid[self._get_wall_slice(position, orientation)] = Board.WALL
            self._walls_remaining[player] -= 1
        else:
            raise ValueError("Cannot place wall at position {position} with orientation {orientation}")

    def remove_wall(self, player: Player, position: Position, orientation: WallOrientation):
        assert isinstance(player, Player)
        assert is_valid_position_type(position)
        assert isinstance(orientation, WallOrientation)

        self._grid[self._get_wall_slice(position, orientation)] = Board.FREE
        self._walls_remaining[player] += 1

    def can_place_wall(self, position: Position, orientation: WallOrientation) -> bool:
        """
        Returns True if the wall can be placed at the given position and orientation.

        Checks that the player has walls remaining, the wall is within the bounds of the board, and doesn't overlap with other walls.
        """
        assert is_valid_position_type(position)
        assert isinstance(orientation, WallOrientation)

        try:
            return (self._grid[self._get_wall_slice(position, orientation)] == Board.FREE).all()
        except IndexError:
            return False

    def is_wall_between(self, position1: Position, position2: Position) -> bool:
        """
        Check if there is a wall between two positions.
        """
        assert is_valid_position_type(position1)
        assert is_valid_position_type(position2)
        assert np.sum(np.abs(position1 - position2)) == 1, "Positions must be adjacent"

        position1_on_board = self.is_position_on_board(position1)
        position2_on_board = self.is_position_on_board(position2)
        assert position1_on_board or position2_on_board, "At least one position must be on the board"

        if position1_on_board and position2_on_board:
            wall_position = position1 + position2 + 2
            return self._grid[*wall_position] == Board.WALL
        else:
            # By convention we treat the border as a "wall". This is makes checking jumps more convenient, since
            # players are allowed to jump diagonally if they are adjacent to another player and the border of the
            # board is on the other side of that player.
            return True

    def is_position_on_board(self, position: Position) -> bool:
        assert is_valid_position_type(position)

        return (
            (position[0] >= 0)
            and (position[0] < self.board_size)
            and (position[1] >= 0)
            and (position[1] < self.board_size)
        )

    def _wall_position_to_grid_index(self, position: Position, orientation: WallOrientation) -> tuple[int, int]:
        """
        Returns the grid index of the topmost/leftmost cell of the wall.
        """
        assert is_valid_position_type(position)
        assert isinstance(orientation, WallOrientation)

        if orientation == WallOrientation.VERTICAL:
            return (position[0] * 2 + 2, position[1] * 2 + 3)
        elif orientation == WallOrientation.HORIZONTAL:
            return (position[0] * 2 + 3, position[1] * 2 + 2)

        raise ValueError("Invalid wall orientation")

    def _get_wall_slice(self, position: Position, orientation: WallOrientation) -> tuple[slice, slice]:
        """
        Get a tuple of slices that correspond to the wall's cells in the grid.
        """
        assert is_valid_position_type(position)
        assert isinstance(orientation, WallOrientation)

        if orientation == WallOrientation.VERTICAL:
            wall_slice = (slice(position[0] * 2 + 2, position[0] * 2 + 5), position[1] * 2 + 3)
        elif orientation == WallOrientation.HORIZONTAL:
            wall_slice = (position[0] * 2 + 3, slice(position[1] * 2 + 2, position[1] * 2 + 5))

        return wall_slice

    def _is_wall_potential_block(self, position, orientation):
        wall_start_index = self._wall_position_to_grid_index(position, orientation)

        touches = 0
        for neighbor_offsets in self._potential_wall_neighbors[orientation]:
            neighbors = wall_start_index + neighbor_offsets
            if (self._grid[neighbors[:, 0], neighbors[:, 1]] == Board.WALL).any():
                touches += 1

        return touches > 1

    def _dfs(self, start_position: Position, target_row: int, visited: np.ndarray, any_path=True):
        """
        Performs a depth-first search to find whether the pawn can reach the target row.

        Args:
            start_position : The current row of the pawn
            target_row: The target row to reach
            visited: A 2D boolean array with the same shape as the board,
                indicating which positions have been visited
            If any_path is set to true, the first path to the target row will be returned (faster).
            Otherwise, the shortest path will be returned (potentially slower)

        Returns:
            int: Number of steps to reach the target or -1 if it's unreachable
        """
        if start_position[0] == target_row:
            return 0

        visited[*start_position] = True

        # Find out the forward direction to try it first and maybe get to the target faster
        fwd = 1 if target_row > start_position[0] else -1

        moves = [np.array(start_position + offset) for offset in [(fwd, 0), (0, -1), (0, 1), (-fwd, 0)]]
        best = -1
        for new_position in moves:
            if (
                self.is_position_on_board(new_position)
                and not self.is_wall_between(start_position, new_position)
                and not visited[*new_position]
            ):
                dfs = self._dfs(new_position, target_row, visited)
                if dfs != -1:
                    if any_path:
                        return dfs + 1
                    if best == -1 or dfs + 1 < best:
                        best = dfs + 1

        return best

    def __str__(self):
        """
        Return a pretty-printed string representation of the grid.
        """
        grid_without_border = self._grid[2:-2, 2:-2]
        display_grid = np.full(grid_without_border.shape, " ", dtype=str)
        display_grid[::2, ::2] = "."
        display_grid[grid_without_border == Board.WALL] = "#"
        for player, _ in enumerate(self._player_positions):
            display_grid[*self._player_positions[player] * 2] = str(player + 1)
        for idx, value in np.ndenumerate(grid_without_border):
            if value >= 5 and value <= 9:
                display_grid[idx] = str(value)  # Useful for debugging.
        return "\n".join([" ".join(row) for row in display_grid]) + "\n"


class Quoridor:
    def __init__(self, board: Board):
        """
        If you want to start from the initial game state, pass in board=Board(board_size, max_walls).
        """
        self.board = board
        self.current_player = Player.ONE

        self._goal_rows = {Player.ONE: self.board.board_size - 1, Player.TWO: 0}
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

        self.go_to_next_player()

    def is_action_valid(self, action: Action):
        """
        Check whether the given action is valid given the current game state.
        """
        player = self.get_current_player()
        opponent = Player(1 - player)
        is_valid = True
        if isinstance(action, MoveAction):
            assert is_valid_position_type(action.destination)
            current_position = self.board.get_player_position(player)
            opponent_position = self.board.get_player_position(opponent)
            opponent_offset = opponent_position - current_position
            position_delta = tuple(action.destination - current_position)  # Tuple so we can use it as a key.

            # Destination cell must be free
            if self.board.get_player_cell(action.destination) != Board.FREE:
                is_valid = False

            elif np.sum(np.abs(position_delta)) == 1:
                # Moving to an adjacent cell, just make sure no wall is in the way.
                if self.board.is_wall_between(current_position, action.destination):
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
            if self.board.can_place_wall(action.position, action.orientation):
                if self.board._is_wall_potential_block(action.position, action.orientation):
                    # Temporarily place the wall so that we can check player paths to their goals.
                    self.board.add_wall(self.current_player, action.position, action.orientation)
                    for p in [Player.ONE, Player.TWO]:
                        if not self.can_reach(self.board.get_player_position(p), self.get_goal_row(p)):
                            is_valid = False
                            break
                    # Restore the board to its previous state
                    self.board.remove_wall(self.current_player, action.position, action.orientation)
            else:
                is_valid = False
        else:
            raise ValueError("Invalid action type")

        return is_valid

    def go_to_next_player(self):
        return Player(1 - self.current_player)

    def get_current_player(self) -> int:
        return self.current_player

    def get_goal_row(self, player: Player) -> int:
        return self._goal_rows[player]

    def can_reach(self, start_position: Position, target_row: int):
        return self.distance_to_target(start_position, target_row, True) != -1

    def distance_to_target(self, start_position, target_row, any_path=False):
        """
        Returns the approximate number of moves it takes to reach the target row, or -1 if it's not reachable.
        If any_path is set to true, the first path to the target row will be returned (faster).
        Otherwise, the shortest path will be returned (potentially slower)
        """
        visited = np.zeros((self.board.board_size, self.board.board_size), dtype="bool")
        return self.board._dfs(start_position, target_row, visited, any_path)

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
    game.step(MoveAction(np.array((1, 4))))
    game.step(WallAction(np.array((0, 4)), WallOrientation.HORIZONTAL))
    game.step(WallAction(np.array((4, 2)), WallOrientation.VERTICAL))
    print(str(game))
