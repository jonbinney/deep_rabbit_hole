from agents import Agent
from agents.core.agent import AgentRegistry
from arena import Arena, PlayMode
from arena_utils import GameResult
from quoridor_env import env
from renderers.match_results import MatchResultsRenderer
from utils.misc import compute_elo, get_opponent_player_id


class Metrics:
    """
    Computes metrics for an agent by making it play against other agents.
    """

    def __init__(self, board_size: int, max_walls: int, benchmarks: list[str | Agent] = []):
        self.board_size = board_size
        self.max_walls = max_walls
        self.stored_elos = {}
        self.benchmarks = benchmarks

    def _win_perc(self, results: list[GameResult], agent_name: str):
        played = 0
        won = 0
        for result in results:
            if result.player1 == agent_name or result.player2 == agent_name:
                played += 1

            if result.winner == agent_name:
                won += 1

        return 100.0 * won / played if played > 1 else 0.0

    def _compute_relative_elo(self, elo_table: dict[str, float], agent_name: str) -> int:
        agent_rating = elo_table[agent_name]
        best_opponent = 0
        for name, elo in elo_table.items():
            if name != agent_name:
                best_opponent = max(best_opponent, elo)

        return int(agent_rating - best_opponent)

    def compute(self, agent_encoded_name: str) -> tuple[int, dict[str, float], int, float, int, int]:
        """
        Evaluates the performance of a given agent by running it against a set of predefined opponents and computing its Elo rating and win percentage.

        Args:
            agent_encoded_name (str): The encoded name of the agent to evaluate.  If there are training params, they will be ignored

        Returns:
            tuple[int, dict[str, float], int, float]: A tuple containing:
                - VERSION (int): The version number of the scoring method.
                - elo_table (dict[str, float]): A dictionary mapping agent names to their computed Elo ratings.
                - relative_elo (int): The evaluated agent's Elo rating minus the elo for the best opponent.
                - win_perc (float): The win percentage of the evaluated agent against the opponents.
                - dumb_score (int): A score between 0 (perfect) and 100 (always wrong) on how the agent performs in certain basic situations

        Notes:
            - The method disables training mode for trainable agents during evaluation and restores it afterward.
            - Opponent Elo ratings are cached to avoid redundant computations.
            - The evaluation is performed using a fixed set of baseline agents and a specified number of games.
        """
        # Bump if there's any change in the scoring
        VERSION = 1
        times = 10

        players: list[str | Agent] = (
            [
                "greedy",
                "greedy:p_random=0.1,nick=greedy-01",
                "greedy:p_random=0.3,nick=greedy-03",
                "greedy:p_random=0.5,nick=greedy-05",
                "cnn3c:wandb_alias=best,nick=cnn3c",
                "dexp:wandb_alias=best,nick=dexp",
                "simple",
            ]
            if self.benchmarks is None
            else self.benchmarks
        )
        arena = Arena(self.board_size, self.max_walls, max_steps=200, renderers=[MatchResultsRenderer()])

        agent = AgentRegistry.create_from_encoded_name(
            agent_encoded_name,
            arena.game,
            remove_training_args=True,
            keep_args={"model_filename"},
        )

        # We store the elos of the opponents playing against each other so we don't have to play those matches
        # every time
        if not self.stored_elos:
            results = arena._play_games(players, times, PlayMode.ALL_VS_ALL)
            self.stored_elos = compute_elo(results)

        results = arena._play_games([agent] + players, times, PlayMode.FIRST_VS_RANDOM)

        elo_table = compute_elo(results, initial_elos=self.stored_elos.copy())
        relative_elo = self._compute_relative_elo(elo_table, agent.name())

        win_perc = self._win_perc(results, agent.name())
        absolute_elo = elo_table[agent.name()]

        dumb_score = self.dumb_score(agent)

        return VERSION, elo_table, relative_elo, win_perc, int(absolute_elo), dumb_score

    def dumb_score(self, agent: Agent, verbose: bool = False):
        def print_fail(initial: str, current: str):
            print("Initial position:")
            print(initial)
            print("Moved:")
            print(current)

        def dumb_score_before_goal(quoridor, agent: Agent, agent_id: str, verbose: bool = False) -> tuple[int, int]:
            # The agent is right before the goal (no walls blocking), and the opponent in the goal row in the middle column.
            # Moving forward (or forward and left or right in the middle) will lead it to winning.
            # E.g., the agent is player 2 and moving up:
            # . 1 .    . 1 .   . 1 .
            # 2 . .    . 2 .   . . 2
            dumb_score = 0

            for i in range(quoridor.board_size):
                agent.start_game(quoridor, agent_id)
                quoridor.reset()
                row = quoridor.get_goal_row(agent_id)
                row = 1 if row == 0 else row - 1  # Move 1 away from the goal
                quoridor.set_player_position(agent_id, (row, i))
                quoridor.set_current_player(agent_id)
                initial_pos = str(quoridor.game)

                action = agent.get_action(quoridor.observe(agent_id))
                quoridor.step(action)

                if quoridor.winner() != quoridor.agent_order.index(agent_id):
                    dumb_score += 1
                    if verbose:
                        print("Agent was expected to win but did something else.")
                        print_fail(initial_pos, str(quoridor.game))

            return quoridor.board_size, dumb_score

        def dumb_score_before_goal_with_jump(
            quoridor, agent: Agent, agent_id: str, opponent_id: str, verbose: bool = False
        ) -> tuple[int, int]:
            # The agent needs to jump over the opponent and win
            # E.g., the agent is player 2 and moving up:
            # . . .    . . .   . . .
            # 1 . .    . 1 .   . . 1
            # 2 . .    . 2 .   . . 2
            dumb_score = 0
            for i in range(quoridor.board_size):
                agent.start_game(quoridor, agent_id)
                quoridor.reset()
                goal_row = quoridor.get_goal_row(agent_id)
                agent_row = 2 if goal_row == 0 else goal_row - 2  # Our agent is 2 away from the goal
                opp_row = 1 if goal_row == 0 else goal_row - 1  # The opponent is 1 away from the goal

                quoridor.set_player_position(opponent_id, (opp_row, i))
                quoridor.set_player_position(agent_id, (agent_row, i))
                quoridor.set_current_player(agent_id)
                initial_pos = str(quoridor.game)

                action = agent.get_action(quoridor.observe(agent_id))
                quoridor.step(action)

                if quoridor.winner() != quoridor.agent_order.index(agent_id):
                    dumb_score += 1
                    if verbose:
                        print("Agent was expected to win but did something else.")
                        print_fail(initial_pos, str(quoridor.game))

            return quoridor.board_size, dumb_score

        def dumb_score_block_opponent(
            quoridor, agent: Agent, agent_id: str, opponent_id: str, verbose: bool = False
        ) -> tuple[int, int]:
            # The opponent is about to win and we're 2 away, so the agent should place a wall to block it.
            # This will be run only if the board is 5 or plus and we're playing with walls.
            # E.g., the agent is player 2 and moving up:
            # . . . . .
            # . . . . .
            # . . 2 . .
            # . . 1 . .
            # . . . . .
            if quoridor.board_size < 5 or quoridor.max_walls == 0:
                return 0, 0

            dumb_score = 0
            for i in range(quoridor.board_size):
                agent.start_game(quoridor, agent_id)
                quoridor.reset()
                goal_row = quoridor.get_goal_row(agent_id)
                agent_row = 2 if goal_row == 0 else goal_row - 2  # Our agent is 2 away from the goal

                opp_goal_row = quoridor.get_goal_row(opponent_id)
                opp_row = 1 if opp_goal_row == 0 else opp_goal_row - 1  # The opponent is 1 away from the goal

                quoridor.set_player_position(opponent_id, (opp_row, i))
                quoridor.set_player_position(agent_id, (agent_row, i))
                quoridor.set_current_player(agent_id)
                initial_pos = str(quoridor.game)

                action = agent.get_action(quoridor.observe(agent_id))
                quoridor.step(action)

                if not quoridor.game.board.is_wall_between((opp_row, i), (opp_goal_row, i)):
                    dumb_score += 1
                    if verbose:
                        print("Agent was expected to block the opponent but did something else.")
                        print_fail(initial_pos, str(quoridor.game))

            return quoridor.board_size, dumb_score

        quoridor = env(board_size=self.board_size, max_walls=self.max_walls)
        dumb_score = 0
        count = 0

        for agent_id in quoridor.agents:
            opponent_id = get_opponent_player_id(agent_id)
            c, ds = dumb_score_before_goal(quoridor, agent, agent_id, verbose)
            count += c
            dumb_score += ds

            c, ds = dumb_score_before_goal_with_jump(quoridor, agent, agent_id, opponent_id, verbose)
            count += c
            dumb_score += ds

            c, ds = dumb_score_block_opponent(quoridor, agent, agent_id, opponent_id, verbose)
            count += c
            dumb_score += ds

        return int((100.0 * dumb_score) / count)
