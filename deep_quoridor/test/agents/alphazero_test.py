import numpy as np
from agents.alphazero.nn_evaluator import NNEvaluator
from agents.core import rotation
from quoridor import ActionEncoder, Board, Quoridor
from utils import my_device


def _assert_policy_and_rotated_policy_equivalent(
    action_encoder: ActionEncoder, policy: np.ndarray, rotated_policy: np.ndarray
) -> None:
    assert policy.shape == (action_encoder.num_actions,)
    assert rotated_policy.shape == (action_encoder.num_actions,)

    for action_index in range(len((policy))):
        roated_action_index = rotation.convert_original_action_index_to_rotated(action_encoder.board_size, action_index)
        assert policy[action_index] == rotated_policy[roated_action_index]


def _rotation_test_setup(board_size: int, max_walls) -> tuple[ActionEncoder, NNEvaluator, Quoridor]:
    board_size = 3
    max_walls = 1

    action_encoder = ActionEncoder(board_size)
    evaluator = NNEvaluator(action_encoder, my_device())
    game = Quoridor(Board(board_size, max_walls))

    return action_encoder, evaluator, game


def test_evaluator_game_rotation():
    action_encoder, evaluator, game_1 = _rotation_test_setup(board_size=3, max_walls=1)

    game_2 = game_1.copy()
    game_2.go_to_next_player()
    value_1, policy_1 = evaluator.evaluate(game_1)
    value_2, policy_2 = evaluator.evaluate(game_2)
    assert value_1 == value_2
    _assert_policy_and_rotated_policy_equivalent(action_encoder, policy_1, policy_2)
