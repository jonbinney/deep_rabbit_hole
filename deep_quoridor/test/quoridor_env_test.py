import pytest

from deep_quoridor.src.quoridor import Quoridor, MoveAction, WallAction, WallOrientation


def parse_board(board):
    """
    Parse a string representation of a Quoridor board (of any size) and convert it into a Quoridor instance.

    The board should contain N rows for the cells, which are composed by:
    - '1': Position of player 1
    - '2': Position of player 2
    - '.': Empty cell
    - '*': Potential move positions for player 1
    - '|': Vertical walls
    - 'v': A vertical wall cannot be placed here (for testing wall placement)
    - ' ': Empty spaces (just for formatting)
    - '+': Vertical wall continuation (just for formatting)

    Additionally, between 2 rows for the cells, a row with horizontal walls can be included.
    The position of the wall will be with respect to the row above.  The row is composed by:
    - '-': Horizontal walls====================================================================
    - '>': A horizontal wall cannot be placed here (for testing wall placement)
    - ' ': Empty spaces (just for formatting)
    - '+': Horizontal wall continuation (just for formatting)

    Returns:
        tuple: A tuple containing three elements:
            - Quoridor: A game instance representing the parsed board state
            - list: A list of tuples (row, col) representing potential move positions
            - list: A list of tuples (row, col, orientation) representing where the walls are forbidden to be placed

    """
    rows = [r for r in board.split("\n") if r.strip()]
    size = len([ch for ch in rows[0] if ch in "12.*"])
    game = Quoridor(size, 10)
    potential_moves = []
    forbidden_walls = []

    row_n = 0
    col_positions = {}  # map from position in the string to column number

    for row in rows:
        col_n = 0

        is_cell_row = set(row) <= set("12.*| +v")
        is_h_wall_row = set(row) <= set("- +>")
        assert is_cell_row or is_h_wall_row, f"Row contains invalid characters {row}"

        if is_h_wall_row:
            for i, col_n in col_positions.items():
                if i >= len(row):
                    break

                ch = row[i]
                if ch == "-":
                    game.step(WallAction((row_n - 1, col_n), WallOrientation.HORIZONTAL))

                if ch == ">" or ch == "-":
                    forbidden_walls.append((row_n - 1, col_n, 1))

            col_positions = {}

        else:
            for i, ch in enumerate(row):
                if ch == " " or ch == "+":
                    continue

                if ch == "v" or ch == "|":
                    forbidden_walls.append((row_n, col_n - 1, 0))
                    if ch == "|":
                        game.step(WallAction((row_n, col_n - 1), WallOrientation.VERTICAL))
                    continue

                if ch == "*":
                    potential_moves.append((row_n, col_n))
                elif ch == "1":
                    game.player_positions[0] = (row_n, col_n)
                elif ch == "2":
                    game.player_positions[1] = (row_n, col_n)

                col_positions[i] = col_n
                col_n += 1

            assert len(col_positions) == size, f"The row in the board doesn't contain {size} elements: {row}"
            row_n += 1

    assert row_n == size, f"Was expecting {size} rows, but found {row_n}"
    return game, potential_moves, forbidden_walls


class TestQuoridor:
    def _test_pawn_movements(self, s):
        game, potential_moves, _ = parse_board(s)

        game_moves = []
        for row in range(0, game.board_size):
            for col in range(0, game.board_size):
                if game.is_action_valid(MoveAction((row, col))):
                    game_moves.append((row, col))

        assert game_moves == potential_moves

    def _test_wall_placements(self, s):
        game, _, forbidden_walls = parse_board(s)
        print(str(game))

        game_walls = []
        for orientation in [WallOrientation.HORIZONTAL, WallOrientation.VERTICAL]:
            for row in range(0, game.board_size - 1):
                for col in range(0, game.board_size - 1):
                    if game.is_action_valid(WallAction((row, col), orientation)):
                        game_walls.append((row, col, orientation))
                game_walls.append((row, col, orientation))

        assert set(game_walls) == set(forbidden_walls)

    def test_corner_movements(self):
        self._test_pawn_movements("""
            1 * .
            * . .
            . . 2
        """)

        self._test_pawn_movements("""
            . * 1
            . . *
            . . 2
        """)

        self._test_pawn_movements("""
            2 . .
            . . *
            . * 1
        """)

        self._test_pawn_movements("""
            2 . .
            * . .
            1 * .
        """)

    def test_edge_movements(self):
        self._test_pawn_movements("""
            * . .
            1 * .
            * . 2
        """)

        self._test_pawn_movements("""
            * 1 *
            . * .
            . . 2
        """)

        self._test_pawn_movements("""
            . . *
            2 * 1
            . . *
        """)

        self._test_pawn_movements("""
            . . .
            2 * 1
            * 1 *
        """)

    def test_center_movements(self):
        self._test_pawn_movements("""
            . * .
            * 1 *
            . * 2
        """)

    def test_simple_jumps(self):
        self._test_pawn_movements("""
            * . .
            1 2 *
            * . .
        """)

        self._test_pawn_movements("""
            * 1 *
            . 2 .
            . * .
        """)

        self._test_pawn_movements("""
            . . *
            * 2 1
            . . *
        """)

        self._test_pawn_movements("""
            . * .
            . 2 .
            * 1 *
        """)

    def test_jumps_with_walls(self):
        self._test_pawn_movements("""
            * *|.
            1 2|.
            * * .
        """)

        self._test_pawn_movements("""
           . * *|.
           * 1 2|.
             - -
           . . . .
           . . . .
        """)

        self._test_pawn_movements("""
           . . *|.
           -+-  +
           * 1 2|.
             -+-
           . . . .
           . . . .
        """)

    def test_forbidden_walls(self):
        self._test_wall_placements("""
            . . 1 . .
            . . .v. .
            . . .|. .
                >
            . . .|. .
            . . 2 . .
        """)

        self._test_wall_placements("""
            . . 1 . .
            . . .v. .
              > - -
            . . . . .
            . . . . .
            . . 2 . .
        """)

        self._test_wall_placements("""
            . .v1 .v.
            . .|.v.|.
              >+-+-+
            . .|. .|.
            . . . . .
            . . 2 . .
        """)

    def test_forbidden_walls_due_to_blocking(self):
        self._test_wall_placements("""
            . .|1|. .
              > >
            . .|.|. .
              > >
            . . . . .
            . . . . .
            . . 2 . .
        """)

        self._test_wall_placements("""
            . . 1v.v.
            .v. .v. .
            -+- -+-
            . . . . .
            . . . . .
            . . 2 . .
        """)

        self._test_wall_placements("""
            . . 1 . . .
            . . . . . .
            .v. .v. . .
            -+- -+- >
            . . . . . .
            . . . . . .
            . . 2 . . .
        """)
