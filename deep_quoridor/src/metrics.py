from dataclasses import dataclass

import torch
from agents import Agent
from agents.core.agent import AgentRegistry
from arena import Arena, PlayMode
from arena_utils import GameResult
from quoridor_env import env
from utils.misc import compute_elo, get_opponent_player_id


@dataclass
class MatchupStats:
    wins: int = 0
    losses: int = 0
    ties: int = 0

    def total(self) -> int:
        """
        Total number of games played.
        """
        return self.wins + self.losses + self.ties

    def __add__(self, other: "MatchupStats"):
        return MatchupStats(self.wins + other.wins, self.losses + other.losses, self.ties + other.ties)


class Metrics:
    """
    Computes metrics for an agent by making it play against other agents.
    """

    def __init__(
        self,
        board_size: int,
        max_walls: int,
        benchmarks: list[str] = [],
        benchmarks_t: int = 10,
        max_steps=200,
        num_workers=0,
    ):
        self.board_size = board_size
        self.max_walls = max_walls
        self.stored_elos = {}
        self.benchmarks = benchmarks
        self.benchmarks_t = benchmarks_t
        self.max_steps = max_steps
        self.num_workers = num_workers

    def _compute_stats(
        self, results: list[GameResult], agent_name: str
    ) -> tuple[float, dict[str, MatchupStats], dict[str, MatchupStats]]:
        played = 0
        won = 0

        all_player_names = set()
        for result in results:
            all_player_names.update([result.player1, result.player2])

        # Count up the wins/losses/ties for each opponent when our agent is playing
        # as P1, and when it is playing as P2.
        # many times wins/loss
        p1_stats: dict[str, MatchupStats] = {}
        p2_stats: dict[str, MatchupStats] = {}
        for result in results:
            if result.player1 == agent_name:
                stats = p1_stats
                opponent_name = result.player2
            elif result.player2 == agent_name:
                stats = p2_stats
                opponent_name = result.player1
            else:
                continue

            if opponent_name not in stats:
                stats[opponent_name] = MatchupStats()

            played += 1
            if result.winner == agent_name:
                stats[opponent_name].wins += 1
                won += 1
            elif result.winner == opponent_name:
                stats[opponent_name].losses += 1
            else:
                # Neither player won; assume this was a tie.
                stats[opponent_name].ties += 1

        overall_win_percentage = 100.0 * won / played if played > 1 else 0.0

        return overall_win_percentage, p1_stats, p2_stats

    def _compute_relative_elo(self, elo_table: dict[str, float], agent_name: str) -> int:
        agent_rating = elo_table[agent_name]
        best_opponent = 0
        for name, elo in elo_table.items():
            if name != agent_name:
                best_opponent = max(best_opponent, elo)

        return int(agent_rating - best_opponent)

    def compute(
        self, agent_encoded_name: str
    ) -> tuple[int, dict[str, float], int, float, dict[str, MatchupStats], dict[str, MatchupStats], int, int]:
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
                - p1_stats (dict[str, MatchupStats]): Number of wins, losses, and ties as P1 against each opponent
                - p2_stats (dict[str, MatchupStats]): Number of wins, losses, and ties as P1 against each opponent
                - absolute_elo (int): ELO rating obtained during the tournament
                - dumb_score (int): A score between 0 (perfect) and 100 (always wrong) on how the agent performs in certain basic situations

        Notes:
            - The method disables training mode for trainable agents during evaluation and restores it afterward.
            - Opponent Elo ratings are cached to avoid redundant computations.
            - The evaluation is performed using a fixed set of baseline agents and a specified number of games.
        """
        # Bump if there's any change in the scoring
        VERSION = 1

        players: list[str] = (
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
        arena = Arena(self.board_size, self.max_walls, max_steps=self.max_steps)

        play_encoded_name = AgentRegistry.training_encoded_name_to_playing_encoded_name(agent_encoded_name)
        agent = AgentRegistry.create_from_encoded_name(play_encoded_name, arena.game)

        # We store the elos of the opponents playing against each other so we don't have to play those matches
        # every time
        if not self.stored_elos:
            results = arena._play_games(players, self.benchmarks_t, PlayMode.ALL_VS_ALL, num_workers=self.num_workers)
            self.stored_elos = compute_elo(results)

        results = arena._play_games(
            [play_encoded_name] + players, self.benchmarks_t, PlayMode.FIRST_VS_ALL, num_workers=self.num_workers
        )

        elo_table = compute_elo(results, initial_elos=self.stored_elos.copy())
        relative_elo = self._compute_relative_elo(elo_table, agent.name())

        overall_win_percentage, p1_stats, p2_stats = self._compute_stats(results, agent.name())
        absolute_elo = elo_table[agent.name()]

        dumb_score = self.dumb_score(agent)

        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (
            VERSION,
            elo_table,
            relative_elo,
            overall_win_percentage,
            p1_stats,
            p2_stats,
            int(absolute_elo),
            dumb_score,
        )

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
                agent.end_game(quoridor)

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
                agent.end_game(quoridor)

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
                agent.end_game(quoridor)

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
