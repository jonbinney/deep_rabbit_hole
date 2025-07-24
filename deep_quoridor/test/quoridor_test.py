import copy

import numpy as np
from quoridor import (
    Action,
    ActionEncoder,
    Board,
    MoveAction,
    Player,
    Quoridor,
    WallAction,
    WallOrientation,
)
from quoridor_env import env as quoridor_env


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
    - '-': Horizontal walls
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
    board = Board(size, 10)
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
                    # HACK: Since each wall has two "|" characters, this tries to place each wall twice, and
                    # the second time is off by one position. Checking whether we can place it stops us
                    # placing the second wall.
                    if board.can_place_wall(Player.ONE, (row_n - 1, col_n), WallOrientation.HORIZONTAL):
                        board.add_wall(Player.ONE, (row_n - 1, col_n), WallOrientation.HORIZONTAL)

                if ch == ">" or ch == "-":
                    forbidden_walls.append((row_n - 1, col_n, WallOrientation.HORIZONTAL))

            col_positions = {}

        else:
            for i, ch in enumerate(row):
                if ch == " " or ch == "+":
                    continue

                if ch == "v" or ch == "|":
                    forbidden_walls.append((row_n, col_n - 1, WallOrientation.VERTICAL))
                    if ch == "|":
                        # We arbitrarily use player 1 to place the walls since it doesn't matter for these tests.
                        # HACK: Since each wall has two "|" characters, this tries to place each wall twice, and
                        # the second time is off by one position. Checking whether we can place it stops us
                        # placing the second wall.
                        if board.can_place_wall(Player.ONE, (row_n, col_n - 1), WallOrientation.VERTICAL):
                            board.add_wall(Player.ONE, (row_n, col_n - 1), WallOrientation.VERTICAL)
                    continue

                if ch == "*":
                    potential_moves.append((row_n, col_n))
                elif ch == "1":
                    board.move_player(Player.ONE, (row_n, col_n))
                elif ch == "2":
                    board.move_player(Player.TWO, (row_n, col_n))

                col_positions[i] = col_n
                col_n += 1

            assert len(col_positions) == size, f"The row in the board doesn't contain {size} elements: {row}"
            row_n += 1

    game = Quoridor(board)
    assert row_n == size, f"Was expecting {size} rows, but found {row_n}"
    return game, potential_moves, forbidden_walls


class TestQuoridor:
    def _test_pawn_movements(self, s):
        game, potential_moves, _ = parse_board(s)

        game_moves = []
        for row in range(0, game.board.board_size):
            for col in range(0, game.board.board_size):
                position = (row, col)
                if game.is_action_valid(MoveAction(position)):
                    game_moves.append(position)

        assert game_moves == potential_moves

        env = quoridor_env(board_size=game.board.board_size, max_walls=game.board.max_walls, game_start_state=game)
        N = game.board.board_size
        action_mask = env.observe("player_0")["action_mask"]

        env_moves = []
        for i, value in enumerate(action_mask[: N**2]):
            if value == 1:
                env_moves.append(divmod(i, N))

        assert env_moves == potential_moves

    def _test_wall_placements(self, s, just_highlighted=False):
        game, _, forbidden_walls = parse_board(s)
        N = game.board.board_size
        action_encoder = ActionEncoder(N)
        print(str(game))

        game_walls = []
        for orientation in [WallOrientation.HORIZONTAL, WallOrientation.VERTICAL]:
            for row in range(0, game.board.board_size - 1):
                for col in range(0, game.board.board_size - 1):
                    if not game.is_action_valid(WallAction((row, col), orientation)):
                        game_walls.append((row, col, orientation))

        assert set(game_walls) == set(forbidden_walls)

        env = quoridor_env(board_size=game.board.board_size, max_walls=game.board.max_walls, game_start_state=game)
        action_mask = env.observe("player_0")["action_mask"]

        env_walls = []
        for i in range(N**2, len(action_mask)):
            if action_mask[i] == 0:
                wall_action = action_encoder.index_to_action(i)

                env_walls.append((wall_action.position[0], wall_action.position[1], wall_action.orientation))

        if just_highlighted:
            diff = set(forbidden_walls).difference(set(env_walls))
            assert not diff
        else:
            assert set(env_walls) == set(forbidden_walls)

    def _test_distance_to_target(self, s, moves_p1, moves_p2):
        game, _, _ = parse_board(s)
        N = game.board.board_size
        player_one_position = game.board.get_player_position(Player.ONE)
        assert moves_p1 == game.distance_to_target(player_one_position, N - 1)
        player_two_position = game.board.get_player_position(Player.TWO)
        assert moves_p2 == game.distance_to_target(player_two_position, 0)

    def _test_board_rotation(self, s, s2):
        game, _, _ = parse_board(s)
        game2, _, _ = parse_board(s2)

        game.rotate_board()
        assert np.all(game.board._grid == game2.board._grid)
        assert game.get_current_player() == game2.get_current_player()
        for player in [Player.ONE, Player.TWO]:
            assert game.board.get_player_position(player) == game2.board.get_player_position(player)

        # Goal rows are swapped compared to a non-rotated board.
        assert game.get_goal_row(Player.ONE) == 0
        assert game.get_goal_row(Player.TWO) == game.board.board_size - 1

    def _test_board_and_action_rotation(self, game: Quoridor, action: Action):
        """
        Applying an action to the game should give the same result as rotating the board,
        applying the rotated version of the action, and rotating the board back.
        """
        g1 = copy.deepcopy(game)
        g1.step(action)

        g2 = copy.deepcopy(game)
        rotated_action = g2.rotate_action(action)
        g2.rotate_board()
        g2.step(rotated_action)
        g2.rotate_board()

        assert g2 == g1

    def test_distance_to_goal(self):
        self._test_distance_to_target(
            """
            1 . .
            . . .
            . . 2
        """,
            2,
            2,
        )

        self._test_distance_to_target(
            """
            . * .
            . . .
            2 . 1
        """,
            0,
            2,
        )

        self._test_distance_to_target(
            """
            1 . .
            - -
            . . .
            . . 2
        """,
            4,
            2,
        )

        self._test_distance_to_target(
            """
            1 .|.
            - -+
            . .|.
            . . 2
        """,
            -1,
            2,
        )

        self._test_distance_to_target(
            """
            1 . .
            - -
            . .|.
            . .|2
        """,
            4,
            2,
        )

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

        self._test_pawn_movements("""
            . * .
            * 1 *
            * 2 *
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

    def test_board_rotation(self):
        self._test_board_rotation(
            """
           . 1 .|.
           -+-  +
           . . 2|.
           . . . .
           . . . .
        """,
            """
           . . . .
           . . . .
           .|2 . .
            +  -+-
           .|. 1 .
        """,
        )

    def test_move_action_rotation(self):
        game = Quoridor(Board(board_size=5, max_walls=10))

        m1 = MoveAction((1, 2))
        m1_rotated = MoveAction((3, 2))
        assert game.rotate_action(m1) == m1_rotated
        self._test_board_and_action_rotation(game, m1)

    def test_wall_action_rotation(self):
        game = Quoridor(Board(board_size=5, max_walls=10))

        w1 = WallAction((0, 0), WallOrientation.VERTICAL)
        w1_rotated = WallAction((3, 3), WallOrientation.VERTICAL)
        assert game.rotate_action(w1) == w1_rotated
        self._test_board_and_action_rotation(game, w1)

        w2 = WallAction((0, 0), WallOrientation.HORIZONTAL)
        w2_rotated = WallAction((3, 3), WallOrientation.HORIZONTAL)
        assert game.rotate_action(w2) == w2_rotated
        self._test_board_and_action_rotation(game, w2)

        self._test_board_and_action_rotation(game, WallAction((1, 2), WallOrientation.VERTICAL))
        self._test_board_and_action_rotation(game, WallAction((1, 2), WallOrientation.HORIZONTAL))


class TestActionEncoder:
    def test_action_encoder(self):
        a5 = ActionEncoder(5)
        a5_again = ActionEncoder(5)
        assert a5 is a5_again

        a9 = ActionEncoder(9)
        assert a5 is not a9

        action_a = a5.index_to_action(10)
        action_b = a5_again.index_to_action(10)
        assert action_a is action_b

        action_c = a9.index_to_action(10)
        assert action_a is not action_c
