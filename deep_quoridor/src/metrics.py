from agents import Agent
from agents.core.agent import AgentRegistry
from arena import Arena, PlayMode
from arena_utils import GameResult
from utils.misc import compute_elo


class Metrics:
    """
    Computes metrics for an agent by making it play against other agents.
    """

    def __init__(self, board_size: int, max_walls: int, observation_space, action_space):
        self.board_size = board_size
        self.max_walls = max_walls
        self.stored_elos = {}
        self.observation_space = observation_space
        self.action_space = action_space

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

    def compute(self, agent_encoded_name: str) -> tuple[int, dict[str, float], int, float]:
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

        Notes:
            - The method disables training mode for trainable agents during evaluation and restores it afterward.
            - Opponent Elo ratings are cached to avoid redundant computations.
            - The evaluation is performed using a fixed set of baseline agents and a specified number of games.
        """
        # Bump if there's any change in the scoring
        VERSION = 1
        times = 20

        players: list[str | Agent] = [
            "greedy",
            "greedy:p_random=0.1,nick=greedy-01",
            "greedy:p_random=0.3,nick=greedy-03",
            "greedy:p_random=0.5,nick=greedy-05",
            # "dexp:wandb_alias=best",
            "simple",
        ]

        agent = AgentRegistry.create_from_encoded_name(
            agent_encoded_name,
            board_size=self.board_size,
            max_walls=self.max_walls,
            observation_space=self.observation_space,
            action_space=self.action_space,
            remove_training_args=True,
            keep_args={"model_filename"},
        )

        arena = Arena(self.board_size, self.max_walls)

        # We store the elos of the opponents playing against each other so we don't have to play those matches
        # every time
        if not self.stored_elos:
            results = arena._play_games(players, times, PlayMode.ALL_VS_ALL)
            self.stored_elos = compute_elo(results)

        m = arena.max_steps
        arena.max_steps = 200
        results = arena._play_games([agent] + players, times, PlayMode.FIRST_VS_RANDOM)
        arena.max_steps = m

        elo_table = compute_elo(results, initial_elos=self.stored_elos.copy())
        relative_elo = self._compute_relative_elo(elo_table, agent.name())

        win_perc = self._win_perc(results, agent.name())
        absolute_elo = elo_table[agent.name()]

        return VERSION, elo_table, relative_elo, win_perc, absolute_elo
