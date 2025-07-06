import numpy as np
from agents.alphazero.nn_evaluator import NNEvaluator
from agents.core import rotation
from quoridor import ActionEncoder, Board, MoveAction, Quoridor, WallAction, WallOrientation
from utils import my_device


def _assert_policy_and_rotated_policy_equivalent(
    action_encoder: ActionEncoder, policy: np.ndarray, rotated_policy: np.ndarray
) -> None:
    assert policy.shape == (action_encoder.num_actions,)
    assert rotated_policy.shape == (action_encoder.num_actions,)

    for action_index in range(len((policy))):
        roated_action_index = rotation.convert_original_action_index_to_rotated(action_encoder.board_size, action_index)
        assert policy[action_index] == rotated_policy[roated_action_index]


def _evaluator_rotation_test_setup(board_size: int, max_walls) -> tuple[ActionEncoder, NNEvaluator, Quoridor]:
    action_encoder = ActionEncoder(board_size)
    evaluator = NNEvaluator(action_encoder, my_device())
    game = Quoridor(Board(board_size, max_walls))

    return action_encoder, evaluator, game


def test_evaluator_rotation_on_initial_game():
    action_encoder, evaluator, game_1 = _evaluator_rotation_test_setup(board_size=3, max_walls=0)

    game_2 = game_1.copy()
    game_2.go_to_next_player()
    value_1, policy_1 = evaluator.evaluate(game_1)
    value_2, policy_2 = evaluator.evaluate(game_2)
    assert value_1 == value_2
    _assert_policy_and_rotated_policy_equivalent(action_encoder, policy_1, policy_2)


def test_evaluator_rotation_with_players_in_symmetrical_positions():
    action_encoder, evaluator, game_1 = _evaluator_rotation_test_setup(board_size=5, max_walls=0)

    game_1.step(MoveAction((1, 0)), validate=False)
    game_1.step(MoveAction((3, 4)), validate=False)

    game_2 = game_1.copy()
    game_2.go_to_next_player()
    value_1, policy_1 = evaluator.evaluate(game_1)
    value_2, policy_2 = evaluator.evaluate(game_2)
    assert value_1 == value_2
    _assert_policy_and_rotated_policy_equivalent(action_encoder, policy_1, policy_2)


def test_evaluator_rotation_with_players_and_walls_in_symmetrical_positions():
    action_encoder, evaluator, game_1 = _evaluator_rotation_test_setup(board_size=5, max_walls=2)

    game_1.step(MoveAction((1, 0)), validate=False)
    game_1.step(MoveAction((3, 4)), validate=False)
    game_1.step(WallAction((0, 1), WallOrientation.HORIZONTAL))
    game_1.step(WallAction((3, 2), WallOrientation.HORIZONTAL))
    game_1.step(WallAction((2, 0), WallOrientation.VERTICAL))
    game_1.step(WallAction((1, 3), WallOrientation.VERTICAL))

    game_2 = game_1.copy()
    game_2.go_to_next_player()
    value_1, policy_1 = evaluator.evaluate(game_1)
    value_2, policy_2 = evaluator.evaluate(game_2)
    assert value_1 == value_2
    _assert_policy_and_rotated_policy_equivalent(action_encoder, policy_1, policy_2)


def test_evaluator_training_with_fake_data():
    board_size = 3
    max_walls = 0
    learning_rate = 0.001
    batch_size = 10
    optimizer_iterations = 20
    desired_action = MoveAction((0, 0))

    action_encoder = ActionEncoder(board_size)
    evaluator = NNEvaluator(action_encoder, my_device())

    game = Quoridor(Board(board_size, max_walls))

    target_policy = np.zeros(action_encoder.num_actions, dtype=np.float32)
    desired_action_index = action_encoder.action_to_index(desired_action)
    target_policy[desired_action_index] = 1.0

    replay_buffer = []
    for _ in range(100):
        replay_buffer.append({"game": game, "mcts_policy": target_policy, "value": 1.0})

    evaluator.train_network(replay_buffer, learning_rate, batch_size, optimizer_iterations)

    evaluator.evaluate(game)
