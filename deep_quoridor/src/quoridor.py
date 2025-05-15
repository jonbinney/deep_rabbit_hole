import copy
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Optional, Sequence

import numpy as np
import qgrid


@unique
class Player(IntEnum):
    ONE = 0
    TWO = 1


@unique
class WallOrientation(IntEnum):
    VERTICAL = qgrid.WALL_ORIENTATION_VERTICAL
    HORIZONTAL = qgrid.WALL_ORIENTATION_HORIZONTAL


class Action:
    pass


@dataclass(frozen=True)  # Frozen to make it hashable.
class MoveAction(Action):
    destination: tuple[int, int]  # Destination cell (row, col)


@dataclass(frozen=True)  # Frozen to make it hashable.
class WallAction(Action):
    position: tuple[int, int]  # Start position of the wall (row, col)
    orientation: WallOrientation


class ActionEncoder:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.wall_size = board_size - 1

    def action_to_index(self, action) -> int:
        """
        Converts an action object to an action index
        """
        if isinstance(action, MoveAction):
            return action.destination[0] * self.board_size + action.destination[1]
        elif isinstance(action, WallAction) and action.orientation == WallOrientation.VERTICAL:
            return self.board_size**2 + action.position[0] * self.wall_size + action.position[1]
        elif isinstance(action, WallAction) and action.orientation == WallOrientation.HORIZONTAL:
            return self.board_size**2 + self.wall_size**2 + action.position[0] * self.wall_size + action.position[1]
        else:
            raise ValueError(f"Invalid action type: {action}")

    def index_to_action(self, idx) -> Action:
        """
        Converts an action index to an action object.
        """
        action = None
        if idx < self.board_size**2:  # Pawn movement
            action = MoveAction(divmod(idx, self.board_size))
        elif idx < self.board_size**2 + self.wall_size**2:
            action = WallAction(divmod(idx - self.board_size**2, self.wall_size), WallOrientation.VERTICAL)
        elif idx < self.board_size**2 + (self.wall_size**2) * 2:
            action = WallAction(
                divmod(idx - self.board_size**2 - self.wall_size**2, self.wall_size),
                WallOrientation.HORIZONTAL,
            )
        else:
            raise ValueError(f"Invalid action index: {idx}")

        return action


class Board:
    # Possible values for each cell in the grid.
    FREE = qgrid.CELL_FREE
    # The player numbers start at 0 so that they can also be used as indices into the player positions array.
    PLAYER1 = qgrid.CELL_PLAYER1
    PLAYER2 = qgrid.CELL_PLAYER2
    WALL = qgrid.CELL_WALL

    # When we check whether a new wall could potentially block all routes to the goal for a player,
    # we first check to see whether it spans between two walls already on the board, or between
    # one wall on the board and the border of the board. There are three possible places a new wall
    # could touch an existing wall (or the border). At the start of the new wall, in the middle of the
    # new wall, or at the end of the new wall. To simplify these checks, we create this list of wall
    # grid cells that touch each of those three places on a new wall. THe positions here are relative
    # to the starting position of the new wall.
    _potential_wall_neighbors = {
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

    def __init__(self, board_size: int = 9, max_walls: int = 10):
        self.board_size = board_size
        self.max_walls = max_walls

        # We represent the board as a grid of cells alternating between wall cells and odd rows are player cells.
        # To make some checks easier, we add a double border of walls around the grid.
        self._grid = np.full((self.board_size * 2 + 3, self.board_size * 2 + 3), Board.FREE, dtype=np.int8)
        self._grid[:2, :] = Board.WALL
        self._grid[-2:, :] = Board.WALL
        self._grid[:, :2] = Board.WALL
        self._grid[:, -2:] = Board.WALL

        # We also keep track of the walls in a (row x col x orientation) array so that we can give that
        # representation to agents that expect it.
        self._old_style_walls = np.zeros((self.board_size - 1, self.board_size - 1, 2), dtype=np.int8)

        self._players = [Player.ONE, Player.TWO]
        self._player_positions = np.array([(0, self.board_size // 2), (self.board_size - 1, self.board_size // 2)])
        self._walls_remaining = np.array([self.max_walls, self.max_walls])
        for player, position in zip(self._players, self._player_positions):
            self.set_player_cell(position, player)

    def get_player_position(self, player: Player) -> tuple[int, int]:
        """
        Get the position of the player's pawn.
        """
        return tuple(self._player_positions[player])

    def move_player(self, player: Player, new_position: tuple[int, int]):
        """
        Move the player's pawn. Doesn't check if the move is valid according to game rules.
        """
        if not self.is_position_on_board(new_position):
            raise ValueError(f"Position {new_position} is out of bounds")

        old_position = self._player_positions[player]
        self.set_player_cell(old_position, Board.FREE)
        self.set_player_cell(new_position, player)
        self._player_positions[player] = new_position

    def get_player_cell(self, position: tuple[int, int]) -> int:
        """
        Get the value of a "player cell" in the grid.
        """
        assert self.is_position_on_board(position)

        return self._grid[position[0] * 2 + 2, position[1] * 2 + 2]

    def set_player_cell(self, position: tuple[int, int], value: int):
        self._grid[position[0] * 2 + 2, position[1] * 2 + 2] = value

    def get_walls_remaining(self, player: Player) -> int:
        """
        Get the number of walls remaining for the player.
        """
        return self._walls_remaining[player]

    def set_walls_remaining(self, player: Player, walls_remaining: int):
        """
        Set the number of walls remaining for the player.
        """
        self._walls_remaining[player] = walls_remaining

    def add_wall(self, player: Player, position: tuple[int, int], orientation: WallOrientation, check_if_valid=True):
        """
        Mark the grid cells corresponding to the wall as occupied.
        """
        assert (not check_if_valid) or self.can_place_wall(player, position, orientation)

        self._grid[self._get_wall_slice(position, orientation)] = Board.WALL
        self._walls_remaining[player] -= 1

        # Update the old-style wall representation.
        self._old_style_walls[*position, orientation] = 1

    def remove_wall(self, player: Player, position: tuple[int, int], orientation: WallOrientation):
        self._grid[self._get_wall_slice(position, orientation)] = Board.FREE
        self._walls_remaining[player] += 1

        # Update the old-style wall representation.
        self._old_style_walls[*position, orientation] = 0

    def can_place_wall(self, player: Player, position: tuple[int, int], orientation: WallOrientation) -> bool:
        """
        Returns True if the wall can be placed at the given position and orientation.

        Checks that the player has walls remaining, the wall is within the bounds of the board, and doesn't overlap with other walls.
        """
        if self._walls_remaining[player] < 1:
            return False

        return qgrid.are_wall_cells_free(self._grid, position[0], position[1], int(orientation))

    def _are_adjacent(self, position1: tuple[int, int], position2: tuple[int, int]) -> bool:
        dr = position1[0] - position2[0]
        dc = position1[1] - position2[1]
        return ((dr == 1 or dr == -1) and dc == 0) or ((dc == 1 or dc == -1) and dr == 0)

    def is_wall_between(self, position1: tuple[int, int], position2: tuple[int, int]) -> bool:
        """
        Check if there is a wall between two positions.
        """
        assert self._are_adjacent(position1, position2), "Positions must be adjacent"

        wall_position = (position1[0] + position2[0] + 2, position1[1] + position2[1] + 2)
        if 0 <= wall_position[0] < self._grid.shape[0] and 0 <= wall_position[1] < self._grid.shape[1]:
            return self._grid[*wall_position] == Board.WALL

        # By convention we treat the border as a "wall". This is makes checking jumps more convenient, since
        # players are allowed to jump diagonally if they are adjacent to another player and the border of the
        # board is on the other side of that player.
        return True

    def get_old_style_walls(self):
        return copy.copy(self._old_style_walls)

    def is_position_on_board(self, position: tuple[int, int]) -> bool:
        return 0 <= position[0] < self.board_size and 0 <= position[1] < self.board_size

    def _wall_position_to_grid_index(self, position: tuple[int, int], orientation: WallOrientation) -> tuple[int, int]:
        """
        Returns the grid index of the topmost/leftmost cell of the wall.
        """
        if orientation == WallOrientation.VERTICAL:
            return (position[0] * 2 + 2, position[1] * 2 + 3)
        elif orientation == WallOrientation.HORIZONTAL:
            return (position[0] * 2 + 3, position[1] * 2 + 2)

        raise ValueError("Invalid wall orientation")

    def _get_wall_slice(self, position: tuple[int, int], orientation: WallOrientation) -> tuple[slice, slice]:
        """
        Get a tuple of slices that correspond to the wall's cells in the grid.
        """
        r, c = self._wall_position_to_grid_index(position, orientation)
        if orientation == WallOrientation.VERTICAL:
            wall_slice = (slice(r, r + 3), c)
        elif orientation == WallOrientation.HORIZONTAL:
            wall_slice = (r, slice(c, c + 3))

        return wall_slice

    def _is_wall_potential_block(self, position, orientation):
        return qgrid.is_wall_potential_block(self._grid, position[0], position[1], int(orientation))

    def __str__(self):
        """
        Return a pretty-printed string representation of the grid.
        """
        grid_without_border = self._grid[2:-2, 2:-2]
        display_grid = np.full(grid_without_border.shape, " ", dtype=str)
        display_grid[::2, ::2] = "."
        display_grid[grid_without_border == Board.WALL] = "#"
        for player, (row, col) in enumerate(self._player_positions):
            display_grid[row * 2, col * 2] = str(player + 1)
        for idx, value in np.ndenumerate(grid_without_border):
            if value >= 5 and value <= 9:
                display_grid[idx] = str(value)  # Useful for debugging.
        return "\n".join([" ".join(row) for row in display_grid]) + "\n"


class Quoridor:
    def __init__(self, board: Board, current_player: Player = Player.ONE):
        """
        If you want to start from the initial game state, pass in board=Board(board_size, max_walls).
        """
        self.board = board
        self.current_player = current_player
        self.action_encoder = ActionEncoder(board.board_size)

        self._goal_rows = np.array([self.board.board_size - 1, 0])
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

    def is_action_valid(self, action: Action, for_player: Optional[Player] = None) -> bool:
        """
        Check whether the given action is valid given the current game state.
        """
        if for_player is None:
            for_player = self.get_current_player()

        is_valid = True
        if isinstance(action, MoveAction):
            return qgrid.is_move_action_valid(
                self.board._grid,
                self.board._player_positions,
                int(for_player),
                action.destination[0],
                action.destination[1],
            )

        elif isinstance(action, WallAction):
            is_valid = qgrid.is_wall_action_valid(
                self.board._grid,
                self.board._player_positions,
                self.board._walls_remaining,
                self._goal_rows,
                int(for_player),
                action.position[0],
                action.position[1],
                int(action.orientation),
            )
        else:
            raise ValueError("Invalid action type")

        return is_valid

    def get_valid_move_actions(self, player: Optional[Player] = None) -> list[MoveAction]:
        if player is None:
            player = self.get_current_player()

        action_mask = np.zeros(self.action_encoder.board_size**2, dtype=bool)
        qgrid.compute_move_action_mask(
            self.board._grid,
            self.board._player_positions,
            int(player),
            action_mask,
        )
        actions = [self.action_encoder.index_to_action(action_i) for action_i in np.flatnonzero(action_mask)]
        return actions

    def get_valid_wall_actions(self, player: Optional[Player] = None) -> list[WallAction]:
        if player is None:
            player = self.get_current_player()

        action_mask = np.zeros(2 * self.action_encoder.wall_size**2, dtype=bool)
        qgrid.compute_wall_action_mask(
            self.board._grid,
            self.board._player_positions,
            self.board._walls_remaining,
            self._goal_rows,
            int(player),
            action_mask,
        )
        actions = [
            self.action_encoder.index_to_action(self.action_encoder.board_size**2 + action_i)
            for action_i in np.flatnonzero(action_mask)
        ]
        return actions

    def get_valid_actions(self, player: Optional[Player] = None) -> Sequence[Action]:
        return self.get_valid_move_actions(player) + self.get_valid_wall_actions(player)

    def go_to_next_player(self):
        self.current_player = Player(1 - self.current_player)

    def get_current_player(self) -> Player:
        return self.current_player

    def set_current_player(self, player: Player):
        self.current_player = player

    def get_goal_row(self, player: Player) -> int:
        return self._goal_rows[player]

    def check_win(self, player):
        row, _ = self.board.get_player_position(player)
        return row == self.get_goal_row(player)

    def can_reach(self, start_position: tuple[int, int], target_row: int):
        d = qgrid.distance_to_row(self.board._grid, start_position[0], start_position[1], target_row)
        return d != -1

    def distance_to_target(self, start_position, target_row):
        """
        Returns the minimum number of moves it takes to reach the target row, or -1 if it's not reachable.
        """
        return qgrid.distance_to_row(self.board._grid, start_position[0], start_position[1], target_row)

    def player_distance_to_target(self, player: Player):
        start_position = self.board.get_player_position(player)
        target_row = self.get_goal_row(player)
        return self.distance_to_target(start_position, target_row)

    def is_game_over(self):
        return self.check_win(Player.ONE) or self.check_win(Player.TWO)

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
    # Rotate the checks for all 4 possible directions
    checks = {}
    for opponent_offset in np.array([(1, 0), (0, 1), (-1, 0), (0, -1)]):
        rotation_matrix = np.array((opponent_offset, -opponent_offset[::-1])).T
        for move in checks_for_one_direction:
            rotated_move = np.dot(rotation_matrix, move)
            lookup_key = (tuple(opponent_offset), tuple(rotated_move))
            checks[lookup_key] = {}
            for check_type in checks_for_one_direction[move]:
                checks[lookup_key][check_type] = []
                for wall_start, wall_end in checks_for_one_direction[move][check_type]:
                    rotated_wall = (
                        tuple(np.dot(rotation_matrix, wall_start)),
                        tuple(np.dot(rotation_matrix, wall_end)),
                    )
                    checks[lookup_key][check_type].append(rotated_wall)
    return checks


def construct_game_from_observation(observation: dict, player_id: str) -> tuple[Quoridor, Player, Player]:
    if player_id == "player_0":
        player = Player.ONE
        opponent = Player.TWO
    elif player_id == "player_1":
        player = Player.TWO
        opponent = Player.ONE
    else:
        raise ValueError(f"Invalid player ID: {player_id}")

    if observation["my_turn"]:
        current_player = player
    else:
        current_player = opponent

    # Hack: we set max walls very high because the observation doesn't actually tell us the maximum
    # number of walls. After we add the walls from the observation to the board, we set the walls_remaining
    # values to match those in the observation.
    board = Board(board_size=observation["board"].shape[0], max_walls=99999)

    player_positions = np.argwhere(observation["board"] > 0)
    assert len(player_positions) == 2, "There should be exactly two players on the board."

    for row, col in player_positions:
        player_on_board = Player(
            observation["board"][row, col] - 1
        )  # Players are 1 and 2 on the board, but we use 0 and 1.
        if player_on_board == Player.ONE:
            player_on_board = player
        elif player_on_board == Player.TWO:
            player_on_board = opponent
        board.move_player(player_on_board, (row, col))

    for row, col, orientation in np.argwhere(observation["walls"] == 1):
        board.add_wall(
            player,
            (row, col),
            WallOrientation(orientation),
            check_if_valid=False,
        )

    board.set_walls_remaining(player, observation["my_walls_remaining"])
    board.set_walls_remaining(opponent, observation["opponent_walls_remaining"])

    return Quoridor(board, current_player), player, opponent


if __name__ == "__main__":
    game = Quoridor(Board(board_size=9, max_walls=10))
    game.step(MoveAction((1, 4)))
    game.step(WallAction((0, 4), WallOrientation.HORIZONTAL))
    game.step(WallAction((4, 2), WallOrientation.VERTICAL))
    print(str(game))

    jump_checks = create_jump_checks()
    for key, value in jump_checks.items():
        print(f"{key}: {value}")
