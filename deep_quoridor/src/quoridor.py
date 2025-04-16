import copy
from dataclasses import dataclass
from enum import IntEnum, unique

import numpy as np


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


@dataclass(frozen=True)  # Frozen to make it hashable.
class MoveAction(Action):
    destination: tuple[int, int]  # Destination cell (row, col)


@dataclass(frozen=True)  # Frozen to make it hashable.
class WallAction(Action):
    position: tuple[int, int]  # Start position of the wall (row, col)
    orientation: WallOrientation


class ActionEncoder:
    def __init__(self, board_size: int):
        self._board_size = board_size
        self._wall_size = board_size - 1

    def action_to_index(self, action) -> int:
        """
        Converts an action object to an action index
        """
        if isinstance(action, MoveAction):
            return action.destination[0] * self._board_size + action.destination[1]
        elif isinstance(action, WallAction) and action.orientation == WallOrientation.VERTICAL:
            return self._board_size**2 + action.position[0] * self._wall_size + action.position[1]
        elif isinstance(action, WallAction) and action.orientation == WallOrientation.HORIZONTAL:
            return self._board_size**2 + self._wall_size**2 + action.position[0] * self._wall_size + action.position[1]
        else:
            raise ValueError(f"Invalid action type: {action}")

    def index_to_action(self, idx) -> Action:
        """
        Converts an action index to an action object.
        """
        action = None
        if idx < self._board_size**2:  # Pawn movement
            action = MoveAction(divmod(idx, self._board_size))
        elif idx >= self._board_size**2 and idx < self._board_size**2 + self._wall_size**2:
            action = WallAction(divmod(idx - self._board_size**2, self._wall_size), WallOrientation.VERTICAL)
        elif idx >= self._board_size**2 + self._wall_size**2 and idx < self._board_size**2 + (self._wall_size**2) * 2:
            action = WallAction(
                divmod(idx - self._board_size**2 - self._wall_size**2, self._wall_size),
                WallOrientation.HORIZONTAL,
            )
        else:
            raise ValueError(f"Invalid action index: {idx}")

        return action


class Board:
    # Possible values for each cell in the grid.
    FREE = -1
    # The player numbers start at 0 so that they can also be used as indices into the player positions array.
    PLAYER1 = 0
    PLAYER2 = 1
    WALL = 10

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

    def __init__(self, board_size: int = 9, max_walls: int = 10, from_observation: dict = None):
        if from_observation is None:
            self.board_size = board_size
        else:
            self.board_size = from_observation["board"].shape[0]
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
        self._player_positions = [
            (0, self.board_size // 2),
            (self.board_size - 1, self.board_size // 2),
        ]
        for player, position in zip(self._players, self._player_positions):
            self._grid[position[0] * 2 + 2, position[1] * 2 + 2] = player

        if from_observation is None:
            self._walls_remaining = [max_walls, max_walls]
        else:
            self.init_from_observation(from_observation)

    def init_from_observation(self, observation: dict):
        for row, col in np.argwhere(observation["board"] > 0):
            player = observation["board"][row, col] - 1  # Players are 1 and 2 on the board, but we use 0 and 1.
            self._player_positions[player] = (row, col)
            self._grid[row * 2 + 2, col * 2 + 2] = player

        self._old_style_walls = observation["walls"].copy()
        for row, col, orientation in np.argwhere(observation["walls"] == 1):
            self._grid[self._get_wall_slice((row, col), WallOrientation(orientation))] = Board.WALL

        self._walls_remaining = [observation["my_walls_remaining"], observation["opponent_walls_remaining"]]

    def get_player_position(self, player: Player) -> tuple[int, int]:
        """
        Get the position of the player's pawn.
        """
        return self._player_positions[player]

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

        return (self._grid[self._get_wall_slice(position, orientation)] == Board.FREE).all()

    def is_wall_between(self, position1: tuple[int, int], position2: tuple[int, int]) -> bool:
        """
        Check if there is a wall between two positions.
        """
        assert np.sum(np.abs(np.subtract(position1, position2))) == 1, "Positions must be adjacent"

        position1_on_board = self.is_position_on_board(position1)
        position2_on_board = self.is_position_on_board(position2)
        assert position1_on_board or position2_on_board, "At least one position must be on the board"

        if position1_on_board and position2_on_board:
            wall_position = (position1[0] + position2[0] + 2, position1[1] + position2[1] + 2)
            return self._grid[*wall_position] == Board.WALL
        else:
            # By convention we treat the border as a "wall". This is makes checking jumps more convenient, since
            # players are allowed to jump diagonally if they are adjacent to another player and the border of the
            # board is on the other side of that player.
            return True

    def get_old_style_walls(self):
        return copy.copy(self._old_style_walls)

    def is_position_on_board(self, position: tuple[int, int]) -> bool:
        return (
            (position[0] >= 0)
            and (position[0] < self.board_size)
            and (position[1] >= 0)
            and (position[1] < self.board_size)
        )

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
        if orientation == WallOrientation.VERTICAL:
            wall_slice = (slice(position[0] * 2 + 2, position[0] * 2 + 5), position[1] * 2 + 3)
        elif orientation == WallOrientation.HORIZONTAL:
            wall_slice = (position[0] * 2 + 3, slice(position[1] * 2 + 2, position[1] * 2 + 5))

        return wall_slice

    def _is_wall_potential_block(self, position, orientation):
        wall_start_index = self._wall_position_to_grid_index(position, orientation)

        touches = 0
        for neighbor_offsets in Board._potential_wall_neighbors[orientation]:
            neighbors = wall_start_index + neighbor_offsets
            if (self._grid[neighbors[:, 0], neighbors[:, 1]] == Board.WALL).any():
                touches += 1

        return touches > 1

    def _dfs(self, start_position: tuple[int, int], target_row: int, visited: np.ndarray, any_path=True):
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

        moves = [
            (start_position[0] + offset[0], start_position[1] + offset[1])
            for offset in [(fwd, 0), (0, -1), (0, 1), (-fwd, 0)]
        ]
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
        for player, (row, col) in enumerate(self._player_positions):
            display_grid[row * 2, col * 2] = str(player + 1)
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

    def is_action_valid(self, action: Action, for_player: Player = None) -> bool:
        """
        Check whether the given action is valid given the current game state.
        """
        if for_player is None:
            player = self.get_current_player()
        else:
            player = for_player
        opponent = Player(1 - player)

        is_valid = True
        if isinstance(action, MoveAction):
            current_position = self.board.get_player_position(player)
            opponent_position = self.board.get_player_position(opponent)
            opponent_offset = (opponent_position[0] - current_position[0], opponent_position[1] - current_position[1])
            position_delta = (action.destination[0] - current_position[0], action.destination[1] - current_position[1])

            # Destination cell must be free
            if self.board.get_player_cell(action.destination) != Board.FREE:
                is_valid = False

            elif np.sum(np.abs(position_delta)) == 1:
                # Moving to an adjacent cell, just make sure no wall is in the way.
                if self.board.is_wall_between(current_position, action.destination):
                    is_valid = False

            elif (opponent_offset, position_delta) in self._jump_checks:
                jump_checks = self._jump_checks[(opponent_offset, position_delta)]

                for check_delta_1, check_delta_2 in jump_checks["wall"]:
                    if not self.board.is_wall_between(
                        (current_position[0] + check_delta_1[0], current_position[1] + check_delta_1[1]),
                        (current_position[0] + check_delta_2[0], current_position[1] + check_delta_2[1]),
                    ):
                        is_valid = False
                for check_delta_1, check_delta_2 in jump_checks["nowall"]:
                    if self.board.is_wall_between(
                        (current_position[0] + check_delta_1[0], current_position[1] + check_delta_1[1]),
                        (current_position[0] + check_delta_2[0], current_position[1] + check_delta_2[1]),
                    ):
                        is_valid = False
            else:
                # This isn't a move by 1, and it isn't a jump, so it's invalid.
                is_valid = False

        elif isinstance(action, WallAction):
            if self.board.can_place_wall(player, action.position, action.orientation):
                if self.board._is_wall_potential_block(action.position, action.orientation):
                    # Temporarily place the wall so that we can check player paths to their goals.
                    self.board.add_wall(self.current_player, action.position, action.orientation, check_if_valid=False)
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

    def get_valid_move_actions(self, player: Player = None) -> set[MoveAction]:
        if player is None:
            player = self.get_current_player()

        position = self.board.get_player_position(player)

        valid_moves = set()
        for delta_row in range(-2, 3):
            for delta_col in range(-2, 3):
                destination = (position[0] + delta_row, position[1] + delta_col)
                if self.board.is_position_on_board(destination):
                    move_action = MoveAction(destination)
                    if self.is_action_valid(move_action, player):
                        valid_moves.add(move_action)
        return valid_moves

    def get_valid_wall_actions(self, player: Player = None) -> set[WallAction]:
        if player is None:
            player = self.get_current_player()

        valid_wall_actions = set()
        for row in range(self.board.board_size - 1):
            for col in range(self.board.board_size - 1):
                for orientation in [WallOrientation.VERTICAL, WallOrientation.HORIZONTAL]:
                    wall_action = WallAction((row, col), WallOrientation.VERTICAL)
                    if self.is_action_valid(wall_action, player):
                        valid_wall_actions.add(wall_action)
        return valid_wall_actions

    def get_valid_actions(self, player: Player = None) -> set[Action]:
        return self.get_valid_move_actions(player) | self.get_valid_wall_actions(player)

    def go_to_next_player(self):
        self.current_player = Player(1 - self.current_player)

    def get_current_player(self) -> int:
        return self.current_player

    def get_goal_row(self, player: Player) -> int:
        return self._goal_rows[player]

    def check_win(self, player):
        row, _ = self.board.get_player_position(player)
        return row == self.get_goal_row(player)

    def can_reach(self, start_position: tuple[int, int], target_row: int):
        return self.distance_to_target(start_position, target_row, True) != -1

    def distance_to_target(self, start_position, target_row, any_path=False):
        """
        Returns the approximate number of moves it takes to reach the target row, or -1 if it's not reachable.
        If any_path is set to true, the first path to the target row will be returned (faster).
        Otherwise, the shortest path will be returned (potentially slower)
        """
        visited = np.zeros((self.board.board_size, self.board.board_size), dtype="bool")
        return self.board._dfs(start_position, target_row, visited, any_path)

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


if __name__ == "__main__":
    game = Quoridor(Board(board_size=9, max_walls=10))
    game.step(MoveAction((1, 4)))
    game.step(WallAction((0, 4), WallOrientation.HORIZONTAL))
    game.step(WallAction((4, 2), WallOrientation.VERTICAL))
    print(str(game))
