from dataclasses import dataclass

from agents.core.agent import AgentRegistry
from arena import Arena, PlayMode
from arena_utils import ArenaPlugin
from renderers.match_results import MatchResultsRenderer
from utils.misc import compute_elo
from utils.subargs import SubargsBase


@dataclass
class AgentEvolutionTournamentParams(SubargsBase):
    # How many agents are kept in the tournament
    top_n: int = 5

    # How many games are played between two agents
    t: int = 10


class AgentEvolutionTournament:
    """
    Manages an evolutionary tournament for agents in the Quoridor game environment.

    This class facilitates the addition of new agents, evaluates their performance against existing agents,
    and maintains a leaderboard using Elo ratings. Agents compete in an arena, and only the top N agents
    (as specified by parameters) are retained in the tournament.
    """

    def __init__(
        self,
        board_size: int,
        max_walls: int,
        max_steps: int = 200,
        num_workers: int = 0,
        params: AgentEvolutionTournamentParams = AgentEvolutionTournamentParams(),
        verbose: bool = True,
    ):
        self.agents = {}
        self.elos = {}
        self.params = params
        self.num_workers = num_workers
        renderers: list[ArenaPlugin] = [MatchResultsRenderer()] if verbose else []
        self.arena = Arena(board_size, max_walls, max_steps=max_steps, renderers=renderers, verbose=verbose)

    def add_agent_and_compute(self, agent_encoded_name: str):
        play_encoded_name = AgentRegistry.training_encoded_name_to_playing_encoded_name(agent_encoded_name)

        agents_playing = [play_encoded_name] + list(self.agents.values())

        results = self.arena._play_games(agents_playing, self.params.t, PlayMode.FIRST_VS_ALL, self.num_workers)
        self.elos = compute_elo(results, initial_elos=self.elos)
        all_elos = self.elos.copy()

        nick = AgentRegistry.nick_from_encoded_name(play_encoded_name)
        self.agents[nick] = play_encoded_name

        if len(self.elos) > self.params.top_n:
            lowest_agent = min(self.elos, key=lambda k: self.elos[k])

            self.elos.pop(lowest_agent)
            self.agents.pop(lowest_agent)

        return all_elos
