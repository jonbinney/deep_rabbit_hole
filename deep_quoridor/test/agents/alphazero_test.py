import copy

import numpy as np
from agents.alphazero.nn_evaluator import NNEvaluator
from quoridor import ActionEncoder, Board, MoveAction, Player, Quoridor, WallAction, WallOrientation
from utils import my_device


def test_evaluator_game_rotation():
    board_size = 3
    max_walls = 1

    action_encoder = ActionEncoder(board_size)
    evaluator = NNEvaluator(action_encoder, my_device())

    # Initial game evaluation should be the same, regardless of whose turn it is.
    g1 = Quoridor(Board(board_size, max_walls))
    g2 = Quoridor(Board(board_size, max_walls))
    g2.go_to_next_player()
    v1, p1 = evaluator.evaluate(g1)
    v2, p2 = evaluator.evaluate(g2)
    assert v1 == v2
    assert p1 == p2


test_evaluator_game_rotation()
