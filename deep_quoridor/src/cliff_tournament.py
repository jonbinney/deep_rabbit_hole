from dataclasses import dataclass

from agents.core.agent import Agent, AgentRegistry
from arena import Arena, PlayMode
from renderers.match_results import MatchResultsRenderer
from utils.misc import compute_elo
from utils.subargs import SubargsBase


@dataclass
class CliffTournamentParams(SubargsBase):
    # How many agents are kept in the tournament
    top_n: int = 5

    # How many games are played between two agents
    t: int = 10


class CliffTournament:
    def __init__(
        self, board_size: int, max_walls: int, max_steps=200, params: CliffTournamentParams = CliffTournamentParams()
    ):
        self.agents = {}
        self.elos = {}
        self.params = params
        self.arena = Arena(board_size, max_walls, max_steps=max_steps, renderers=[MatchResultsRenderer()])

    def add_agent_and_compute(self, agent_encoded_name: str):
        agent = AgentRegistry.create_from_encoded_name(
            agent_encoded_name,
            self.arena.game,
            remove_training_args=True,
            keep_args={"model_filename"},
        )

        agents_playing: list[str | Agent] = [agent]
        agents_playing.extend(list(self.agents.values()))

        results = self.arena._play_games(agents_playing, self.params.t, PlayMode.FIRST_VS_ALL)
        self.elos = compute_elo(results, initial_elos=self.elos)
        self.agents[agent.name()] = agent
        if len(self.elos) > self.params.top_n:
            lowest_agent = min(self.elos, key=self.elos.get)
            self.elos.pop(lowest_agent)
            self.agents.pop(lowest_agent)

        return self.elos
