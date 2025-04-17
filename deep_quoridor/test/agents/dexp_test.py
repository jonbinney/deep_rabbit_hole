import numpy as np
import pytest
import torch
from agents.dexp import DExpAgent, DExpAgentParams


@pytest.fixture
def mock_observation():
    return {
        "board": np.array([[0, 0, 0], [1, 0, 0], [0, 0, 2]]),
        "walls": np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]),
        "my_walls_remaining": 5,
        "opponent_walls_remaining": 4,
        "my_turn": True,
    }


@pytest.fixture
def base_agent():
    params = DExpAgentParams(rotate=False, turn=False, split=False, training_mode=True)
    agent = DExpAgent(params=params, board_size=3, max_walls=5)
    agent.device = torch.device("cpu")
    return agent


@pytest.fixture
def rotate_agent():
    params = DExpAgentParams(rotate=True, turn=False, split=False, training_mode=True)
    agent = DExpAgent(params=params, board_size=3, max_walls=5)
    agent.device = torch.device("cpu")
    return agent


@pytest.fixture
def split_agent():
    params = DExpAgentParams(rotate=False, turn=False, split=True, training_mode=True)
    agent = DExpAgent(params=params, board_size=3, max_walls=5)
    agent.device = torch.device("cpu")
    return agent


@pytest.fixture
def target_as_source_agent():
    params = DExpAgentParams(
        rotate=True, turn=False, split=False, target_as_source_for_opponent=True, training_mode=True
    )
    agent = DExpAgent(params=params, board_size=3, max_walls=5)
    agent.device = torch.device("cpu")
    return agent


def test_base_observation_conversion(base_agent, mock_observation):
    result = base_agent.observation_to_tensor(mock_observation, obs_player_id="player_0")

    # Expected tensor size: board (9) + walls (8) + wall counts (2) = 19
    assert result.shape == torch.Size([19])
    assert isinstance(result, torch.FloatTensor)

    # Verify board state conversion
    board_part = result[:9].numpy()
    expected_board = mock_observation["board"].flatten()
    np.testing.assert_array_almost_equal(board_part, expected_board)


def test_rotation_for_player1(rotate_agent, mock_observation):
    result = rotate_agent.observation_to_tensor(mock_observation, obs_player_id="player_1")

    # For player_1 with rotation, the board should be rotated 180 degrees
    board_part = result[:9].numpy()
    expected_board = np.rot90(mock_observation["board"], k=2)
    np.testing.assert_array_almost_equal(board_part, expected_board.flatten())


def test_split_board_representation(split_agent, mock_observation):
    result = split_agent.observation_to_tensor(mock_observation, obs_player_id="player_0")

    # Expected tensor size: split board (18) + walls (8) + wall counts (2) = 28
    assert result.shape == torch.Size([28])

    # Check player and opponent channels are correctly split
    player_channel = result[:9].numpy()
    opponent_channel = result[9:18].numpy()

    expected_player = (mock_observation["board"] == 1).astype(np.float32).flatten()
    expected_opponent = (mock_observation["board"] == 2).astype(np.float32).flatten()

    np.testing.assert_array_almost_equal(player_channel, expected_player)
    np.testing.assert_array_almost_equal(opponent_channel, expected_opponent)


def test_state_conversion_with_target_as_source(target_as_source_agent, mock_observation):
    # Test when it's not my turn (target state)
    mock_observation["my_turn"] = False

    # When it's not my turn and target_as_source_for_opponent is True,
    # the board should be from opponent's perspective

    # As player_0
    result = target_as_source_agent.observation_to_tensor(mock_observation, obs_player_id="player_0")
    wall_counts = result[-2:].numpy()
    # Wall counts should be swapped because this is a target state
    assert wall_counts[0] == mock_observation["opponent_walls_remaining"]
    assert wall_counts[1] == mock_observation["my_walls_remaining"]
