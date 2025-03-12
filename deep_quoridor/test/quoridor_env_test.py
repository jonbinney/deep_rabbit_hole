import pytest

from deep_quoridor.src.quoridor_env import env as quoridor_env


def parse_board(board):
    """
    Parse a string representation of a Quoridor board (of any size) and convert it into a QuoridorEnv instance.

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
    - '-': Horizontal walls
    - 'v': A horizontal wall cannot be placed here (for testing wall placement)
    - ' ': Empty spaces (just for formatting)
    - '+': Horizontal wall continuation (just for formatting)

    Returns:
        tuple: A tuple containing two elements:
            - QuoridorEnv: An environment instance representing the parsed board state
            - list: A list of tuples (row, col) representing potential move positions
            - list: A list of tuples (row, col, orientation) represeting where the walls are forbidden to be placed

    """
    rows = [r for r in board.split("\n") if r.strip()]
    size = len([ch for ch in rows[0] if ch in "12.*"])
    env = quoridor_env(size)
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
                if ch == "-" and not env.is_wall_between(row_n - 1, col_n, row_n, col_n):
                    env.place_wall("player_0", (row_n - 1, col_n), 1)

                if ch == ">" or ch == "-":
                    forbidden_walls.append((row_n - 1, col_n, 1))

            col_positions = {}

        else:
            for i, ch in enumerate(row):
                if ch == " " or ch == "+":
                    continue

                if ch == "v" or ch == "|":
                    forbidden_walls.append((row_n, col_n - 1, 0))
                    if ch == "|" and not env.is_wall_between(row_n, col_n - 1, row_n, col_n):
                        env.place_wall("player_0", (row_n, col_n - 1), 0)
                    continue

                if ch == "*":
                    potential_moves.append((row_n, col_n))
                elif ch == "1":
                    env.positions["player_0"] = row_n, col_n
                elif ch == "2":
                    env.positions["player_1"] = row_n, col_n

                col_positions[i] = col_n
                col_n += 1

            assert len(col_positions) == size, f"The row in the board doesn't contain {size} elements: {row}"
            row_n += 1

    assert row_n == size, f"Was expecting {size} rows, but found {row_n}"
    return env, potential_moves, forbidden_walls


class TestQuoridorEnv:
    def _test_pawn_movements(self, s):
        env, potential_moves, _ = parse_board(s)
        N = env.board_size
        action_mask = env.observe("player_0")["action_mask"]

        env_moves = []
        for i, value in enumerate(action_mask[: N**2]):
            if value == 1:
                env_moves.append(divmod(i, N))

        assert env_moves == potential_moves

    def _test_wall_placements(self, s):
        env, _, forbidden_walls = parse_board(s)
        N = env.board_size
        env.render()
        action_mask = env.observe("player_0")["action_mask"]

        env_walls = []
        for i in range(N**2, len(action_mask)):
            if action_mask[i] == 0:
                row, col, action_type = env.action_index_to_params(i)

                env_walls.append((row, col, action_type - 1))

        assert set(env_walls) == set(forbidden_walls)

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
