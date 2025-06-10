from agents.alphazero_os import AlphaZeroOSAgent, AlphaZeroOSParams
from gymnasium import spaces
from quoridor_env import env as quoridor_env

from deep_quoridor.test.quoridor_test import parse_board


class TestOpenSpielPettingZooConversion:
    """
    Test suite for testing the conversion between OpenSpiel and PettingZoo actions
    in the Quoridor game.
    """

    def setup_env_from_board(self, board_str):
        # Parse the board using the parse_board function from quoridor_test.py
        quoridor, _, _ = parse_board(board_str)

        # Create a PettingZoo environment
        env = quoridor_env(
            board_size=quoridor.board.board_size, max_walls=quoridor.board.max_walls, game_start_state=quoridor
        )

        # Create an AlphaZero agent instance to match the parsed parameters
        agent = AlphaZeroOSAgent(
            AlphaZeroOSParams(checkpoint_path=None),
            board_size=quoridor.board.board_size,
            max_walls=quoridor.board.max_walls,
            action_space=spaces.Discrete(quoridor.board.board_size**2 + (quoridor.board.board_size - 1) ** 2 * 2),
            observation_space=spaces.Dict(
                {
                    "board": spaces.Box(low=0, high=2, shape=(quoridor.board.board_size, quoridor.board.board_size)),
                    "walls": spaces.Box(
                        low=0, high=1, shape=(quoridor.board.board_size - 1, quoridor.board.board_size - 1, 2)
                    ),
                    "my_walls_remaining": spaces.Discrete(quoridor.board.max_walls),
                    "opponent_walls_remaining": spaces.Discrete(quoridor.board.max_walls),
                    "my_turn": spaces.Discrete(2),
                }
            ),
        )

        return env, agent

    def test_gym_to_os_3by3(self):
        """
        Test action conversions between PettingZoo and OpenSpiel.
        """
        # Setup environment from a simple board with player positions
        board_str = """
            . . 1
            . . .
            2 . .
        """

        # Hand crafted action pairs to check (gym action, expected OpenSpiel action)
        movement_pairs = [(3, 2), (7, 14)]
        wall_placement_pairs = [(9, 1), (10, 3), (11, 11), (12, 13), (13, 5)]

        env, agent = self.setup_env_from_board(board_str)

        obs = env.observe("player_1")["observation"]
        action_mask = env.observe("player_1")["action_mask"]

        for gym_move, expected_os_move in movement_pairs:
            assert action_mask[gym_move] == 1, f"Move {gym_move} is not enabled by the action mask"

            os_move = agent._convert_gym_action_to_openspiel(gym_move, obs)

            assert os_move == expected_os_move

        for gym_wall_placement, expected_os_wall_placement in wall_placement_pairs:
            assert (
                action_mask[gym_wall_placement] == 1
            ), f"Wall placement {gym_wall_placement} is not enabled by the action mask"

            os_wall_placement = agent._convert_gym_action_to_openspiel(gym_wall_placement, obs)

            assert os_wall_placement == expected_os_wall_placement

    def test_os_to_gym_3by3(self):
        """
        Test OpenSpiel to PettingZoo movement action conversions.
        """
        # Setup environment from a simple board with player positions
        board_str = """
            . . 1
            . . .
            2 . .
        """

        # Hand crafted movement pairs to check (gym action, expected OpenSpiel action)
        movement_pairs = [(3, 2), (7, 14)]
        wall_placement_pairs = [(9, 1), (10, 3), (11, 11), (12, 13), (13, 5)]

        env, agent = self.setup_env_from_board(board_str)

        obs = env.observe("player_1")["observation"]
        action_mask = env.observe("player_1")["action_mask"]

        for expected_gym_move, os_move in movement_pairs:
            gym_move = agent._convert_openspiel_action_to_gym(os_move, obs)

            assert gym_move == expected_gym_move
            assert action_mask[gym_move] == 1

        for expected_gym_wall_placement, os_wall_placement in wall_placement_pairs:
            gym_wall_placement = agent._convert_openspiel_action_to_gym(os_wall_placement, obs)

            assert action_mask[gym_wall_placement] == 1
            assert gym_wall_placement == expected_gym_wall_placement

    def test_jump(self):
        """
        Test OpenSpiel to PettingZoo jump movement actions.
        """
        # Setup environment from a simple board with player positions
        board_str = """
            . . .
            . 1 .
            . 2 .
        """

        # Hand crafted movement pairs to check (gym action, expected OpenSpiel action)
        movement_pairs = [(1, 2), (6, 10), (8, 14)]

        env, agent = self.setup_env_from_board(board_str)

        obs = env.observe("player_1")["observation"]
        action_mask = env.observe("player_1")["action_mask"]

        for expected_gym_move, os_move in movement_pairs:
            gym_move = agent._convert_openspiel_action_to_gym(os_move, obs)

            assert gym_move == expected_gym_move
            assert action_mask[gym_move] == 1

        for gym_move, expected_os_move in movement_pairs:
            os_move = agent._convert_gym_action_to_openspiel(gym_move, obs)

            assert os_move == expected_os_move
